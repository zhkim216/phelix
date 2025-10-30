import json
import logging
import random
import time
from pathlib import Path
from typing import Literal, override

import atomworks.enums as aw_enums
import atomworks.ml.preprocessing.constants as aw_const
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from atomworks.ml.example_id import generate_example_id
from atomworks.ml.datasets.datasets import MolecularDataset
from atomworks.ml.datasets.parsers import GenericDFParser
from atomworks.ml.samplers import DistributedMixedSampler
from atomworks.ml.utils.io import read_parquet_with_metadata
from atomworks.ml.samplers import LazyWeightedRandomSampler
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader

# from allatom_design.data.sampler import Sampler
from allatom_design.data.transform.pad import pad_to_max
from allatom_design.data.transform import sd_featurizer

logger = logging.getLogger(__name__)

class AtomworksSDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path
        self._train_set = SDDataset(cfg, phase="train")
        self._val_set = SDDataset(cfg, phase="val")
        self.prefetch_buffer_size = cfg.prefetch_buffer_size                
        
    def train_dataloader(self) -> DataLoader:
        weights = torch.as_tensor(self._train_set.get_sampling_weights(), dtype=torch.float32)        
        
        if self.cfg.samples_per_epoch is not None:
            num_samples = self.cfg.samples_per_epoch
        else:
            num_samples = len(self._train_set)
        
        base_sampler = LazyWeightedRandomSampler(weights, num_samples=num_samples, \
            replacement=True, prefetch_buffer_size=self.prefetch_buffer_size)
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        #Todo (JH): Extend to multiple datasets
        sampler = DistributedMixedSampler(
                    datasets_info=[{"dataset": self._train_set, "sampler": base_sampler, "probability": 1.0}],
                    num_replicas=world_size,
                    rank=rank,
                    n_examples_per_epoch=num_samples,
                    shuffle=True,
                    drop_last=True,
                )
                
        loader = DataLoader(dataset=self._train_set,
                            sampler=sampler,
                            batch_size=self.cfg.batch_size,
                            num_workers=self.cfg.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=sd_collator,
                            persistent_workers=(self.cfg.num_workers > 0),
                            worker_init_fn=worker_init_fn)

        self._train_sampler = sampler

        return loader
        

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(self._val_set,
                                batch_size=self.cfg.batch_size,
                                num_workers=self.cfg.num_workers,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False, #! (JH) changed 251003
                                collate_fn=sd_collator,
                                worker_init_fn=worker_init_fn)

        return val_loader

def worker_init_fn(_):
    """Initialize per-worker global random number generators."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SDDataset(MolecularDataset):
    def __init__(self, cfg: DictConfig, phase: Literal["train", "val"]):
        super().__init__(name=f"sd_dataset::{phase}", transform=None)
            
        self.cfg = cfg
        self.phase = phase
        self.save_failed_examples_to_dir = cfg.save_failed_examples_to_dir        
        self._rng = None # For fallback sampling

        # Initialize featurizer
        # Note: We remove INFERENCE_ONLY_KEYS to avoid cuda initialization issues during training.
        self.featurizer = sd_featurizer.sd_featurizer(**cfg.featurizer_cfg,
                                                      remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS,
                                                      ) #! (JH) changed
                            
        # Link featurizer to transform
        self.transform = self.featurizer
    
        # Read in chain metadata parquet        
        self.chain_df, self.dummy_chain_df = self._process_chain_df()

        # Build interface df from contacts in chain df
        self.interface_df = self._process_interface_df()

        # Parse dfs into a common format and concatenate
        self.parsed_df = self._parse_dfs()
        self.data = self.parsed_df

        # Prepare fallback probabilities
        # Todo: Change to FallbackDatasetWrapper + FallbackSamplerWrapper, when using DDP
        self._fallback_probs = self.get_sampling_weights().astype(np.float64)
        self._fallback_probs /= self._fallback_probs.sum()
        

    @override
    def __getitem__(self, idx: int):       
        self._ensure_worker_rng() # Prepare per-worker random number generator for fallback
        
        # Load cached example.        
        example_id = self.idx_to_id(idx)                            
        parsed_row = self.parsed_df.loc[example_id]
                            
        try:
            example = self._load_cached_example(parsed_row["extra_info"]["pdb_id"])
        except FileNotFoundError:
            logger.warning(f"Cached example for {parsed_row['extra_info']['pdb_id']} not found in {self.cfg.pdb_path}/cached_examples in {self.phase} dataset, skipping...")            
            if self.phase == "train":
                # Fallback to next example, based on fallback probabilities                        
                fallback_idx = self._rng.choice(len(self.parsed_df), p=self._fallback_probs)
                # logger.warning(f"Falling back to next example {fallback_idx} based on fallback probabilities in {self.phase} dataset...")
                return self.__getitem__(fallback_idx)
            else:
                idx = idx + 1
                # logger.warning(f"Falling back to next example {idx} in {self.phase} dataset...")                
                return self.__getitem__(idx)
            
        example.update(parsed_row)  # add in query_pn_unit_iids
                    
        # Apply train-time transforms.
        try:
            feats = self._apply_transform(example, example_id=example_id, idx=idx)            
        except Exception as e:
            logger.error(f"Error applying train-time transforms to example {example_id} in {self.phase} dataset: {e}")
            if self.phase == "train":
                # Fallback to next example, based on fallback probabilities                        
                fallback_idx = self._rng.choice(len(self.parsed_df), p=self._fallback_probs)
                # logger.warning(f"Falling back to next example {fallback_idx} based on fallback probabilities in {self.phase} dataset...")
                return self.__getitem__(fallback_idx)
            else:
                idx = idx + 1
                # logger.warning(f"Falling back to next example {idx} in {self.phase} dataset...")                
                return self.__getitem__(idx)            

        return feats

    def _ensure_worker_rng(self):
        """Ensure that each worker has a unique random number generator."""
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)        

    def _process_chain_df(self) -> pd.DataFrame:
        """
        Processes the chain dataframe. Adds chain counts info and sampling weights, and applies filters.
        """
        
        # Read in chain parquet        
        chain_df = read_parquet_with_metadata(self.cfg.parquet_path)
        
        chain_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        chain_df["q_pn_unit_contacting_pn_unit_iids"] = chain_df["q_pn_unit_contacting_pn_unit_iids"].apply(json.loads)
    
        # Load in validation IDs and hold out based on phase. Case insensitive, no extension.    
        with open(self.cfg.validation_ids_txt, "r") as f:
            logger.info(f"Loading in validation IDs from {self.cfg.validation_ids_txt}...")
            val_split = {x.lower().split(".")[0] for x in f.read().splitlines()}
                                                
        chain_df.loc[~chain_df["pdb_id"].str.lower().isin(val_split), "phase"] = "train"
        chain_df.loc[chain_df["pdb_id"].str.lower().isin(val_split), "phase"] = "val"
        
        if self.cfg.exclude_val_cluster:
            val_cluster_ids = list(chain_df[chain_df["phase"] == "val"]["q_pn_unit_cluster_id"])
            
        chain_df = chain_df[chain_df["phase"] == self.phase]
        
        if self.cfg.exclude_val_cluster and self.phase == "train":
            chain_df = chain_df[~chain_df["q_pn_unit_cluster_id"].isin(val_cluster_ids)]
                
        if self.cfg.debug:
            if self.cfg.debug_num_rows is None:
                self.cfg.debug_num_rows = len(chain_df)
            else:
                chain_df = chain_df.iloc[:self.cfg.debug_num_rows]
        
        # Add chain counts info and sampling weights
        if self.cfg.debug: 
            t0 = time.perf_counter()
        # Todo: faster way for add_chain_counts_info
        chain_df = add_chain_counts_info(chain_df,
                                         chain_type_cols=["q_pn_unit_type"],
                                         seq_length_cols=["q_pn_unit_sequence_length"],
                                         is_metal_cols=["q_pn_unit_is_metal"]) #! (JH) changed 250925
        
        if self.cfg.debug:
            t1 = time.perf_counter()
            print(f"{t1-t0}s passed in add_chain_counts_info with {self.cfg.debug_num_rows} rows")
            
        if not self.cfg.task == "lc_seq_des":
            alphas = self.cfg.sampling_weights["alphas"]
        else:
            alphas = self.cfg.sampling_weights["alphas_chain"] #! (JH) changed 250925
        
        chain_df = add_sampling_weights_info(chain_df,
                                             alphas=alphas,
                                             beta=self.cfg.sampling_weights["betas"]["beta_chain"],
                                             cluster_cols=["q_pn_unit_cluster_id"])
        
        # Apply chain filters
        if not self.cfg.task == "lc_seq_des":
            filters = self.cfg.train_filters.chain if self.phase == "train" else self.cfg.val_filters.chain
            chain_df = self._apply_filters(filters, chain_df)    
            return chain_df, None

        else: # ligand-cond
            chain_filter1 = self.cfg.train_filters.chain_filter1 if self.phase == "train" else self.cfg.val_filters.chain_filter1
            dummy_chain_df = self._apply_filters(chain_filter1, chain_df)
            chain_filter2 = self.cfg.train_filters.chain_filter2 if self.phase == "train" else self.cfg.val_filters.chain_filter2
            chain_df = self._apply_filters(chain_filter2, dummy_chain_df)
            return chain_df, dummy_chain_df
        

    def _process_interface_df(self) -> pd.DataFrame:
        """
        Processes the interface dataframe based on the filtered chain dataframe. Adds chain counts info and sampling weights.
        """        
        if not self.cfg.task == "lc_seq_des":
            interface_df = build_interface_df(self.chain_df, dataset_name=Path(self.cfg.parquet_path).parent.name)
        else:
            interface_df = build_interface_df(self.dummy_chain_df, dataset_name=Path(self.cfg.parquet_path).parent.name)
                    
        interface_df = add_chain_counts_info(interface_df,
                                             chain_type_cols=["q_pn_unit_type_1", "q_pn_unit_type_2"],
                                             seq_length_cols=["q_pn_unit_sequence_length_1", "q_pn_unit_sequence_length_2"],
                                             is_metal_cols=["q_pn_unit_is_metal_1", "q_pn_unit_is_metal_2"]) #! (JH) changed 250925
        
        if not self.cfg.task == "lc_seq_des":
            alphas = self.cfg.sampling_weights["alphas"]
        else:
            alphas = self.cfg.sampling_weights["alphas_interface"] #! (JH) changed 250925
            
        interface_df = add_sampling_weights_info(interface_df,
                                                 alphas=alphas,
                                                 beta=self.cfg.sampling_weights["betas"]["beta_interface"],
                                                 cluster_cols=["q_pn_unit_cluster_id_1", "q_pn_unit_cluster_id_2"])

        if self.cfg.task == "lc_seq_des":
            interface_df = self._apply_filters(self.cfg.train_filters.interface if self.phase == "train" else self.cfg.val_filters.interface, interface_df)            
        
        return interface_df
            
    def _parse_dfs(self) -> pd.DataFrame:
        """
        Parses the chain and interface dataframes into a common format and concatenates them.
        """
        chain_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid"])
        interface_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid_1", "q_pn_unit_iid_2"])

        logger.info(f"Final {self.phase} dataset contains {len(self.chain_df)} chains and {len(self.interface_df)} interfaces")
        
        parsed_df = pd.concat([
            self.chain_df.apply(chain_parser.parse, axis=1),
            self.interface_df.apply(interface_parser.parse, axis=1)
        ], axis=0)

        return parsed_df


    def get_sampling_weights(self) -> np.ndarray:
        return self.parsed_df.apply(lambda x: x["extra_info"]["sampling_weight"]).to_numpy()


    @override
    def __len__(self) -> int:
        if self.phase == "train":
            return self.cfg.samples_per_epoch
        return len(self.parsed_df)


    @override
    def __contains__(self, example_id: str) -> bool:
        return example_id in self.parsed_df.index


    @override
    def id_to_idx(self, example_id: str) -> int:
        return self.parsed_df.index.get_loc(example_id)


    @override
    def idx_to_id(self, idx: int) -> str:
        return self.parsed_df.index[idx]


    def _load_cached_example(self, pdb_id: str) -> dict[str, torch.Tensor]:
        cached_example_path = f"{self.cfg.pdb_path}/cached_examples/{pdb_id}.pt"
        return torch.load(cached_example_path, map_location="cpu", weights_only=False)


    def _apply_filters(self, filters: list[str] | None, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filters to the data based on the provided list of query strings.
        For documentation on pandas query syntax, see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html

        Args:
            filters (list[str]): List of query strings to apply to the data.

        Raises:
            ValueError: If the data is not initialized or if a query removes all rows.
            Warning: If a query does not remove any rows.

        Exampleelse:
            logger.info(
                f"Query '{query}' filtered dataset from {original_num_rows:,} to {filtered_num_rows:,} rows (dropped {original_num_rows - filtered_num_rows:,} rows)"
            ):
            queries = [
                "deposition_date < '2020-01-01'",
                "resolution < 2.5 and ~method.str.contains('NMR')",
                "cluster.notnull()",
                "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']"
            ]
        """
        if filters is None:
            return df

        # Apply queries one by one, confirming the impact of each
        for query in filters:
            df = self._apply_query(query, df)

        return df


    def _apply_query(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a single query to the data.

        Args:
            query (str): A query string to apply to the data.
        """
        # Filter using query and validate impact
        original_num_rows = len(df)
        df = df.query(query)
        filtered_num_rows = len(df)
        self._validate_filter_impact(query, original_num_rows, filtered_num_rows)
        return df


    def _validate_filter_impact(self, query: str, original_num_rows: int, filtered_num_rows: int) -> None:
        """
        Validate the impact of the filter.

        Args:
            query (str): The query string that was applied.
            original_num_rows (int): The number of rows before applying the filter.
            filtered_num_rows (int): The number of rows after applying the filter.

        Raises:
            Warning: If the filter did not remove any rows.
            ValueError: If the filter removed all rows.
        """
        rows_removed = original_num_rows - filtered_num_rows
        percent_removed = (rows_removed / original_num_rows) * 100
        percent_remaining = (filtered_num_rows / original_num_rows) * 100

        if filtered_num_rows == original_num_rows:
            logger.warning(f"Query '{query}' on dataset did not remove any rows.")
        elif filtered_num_rows == 0:
            raise ValueError(f"Query '{query}' on dataset removed all rows.")
        else:
            logger.info(
                f"\n+-------------------------------------------+\n"
                f"Query '{query}' on dataset:\n"
                f"  - Started with: {original_num_rows:,} rows\n"
                f"  - Removed: {rows_removed:,} rows ({percent_removed:.2f}%)\n"
                f"  - Remaining: {filtered_num_rows:,} rows ({percent_remaining:.2f}%)\n"
                f"+-------------------------------------------+\n"
            )
    

def sd_collator(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate sequence denoiser features into a batch.

    Parameters
    ----------
    data : list[dict[str, torch.Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, torch.Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
 
    for key in keys:
        values = [d[key] for d in data]

        if key not in ["example_id", *sd_featurizer.INFERENCE_ONLY_KEYS]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)
        
        # Stack the values
        collated[key] = values
    return collated


def build_interface_df(chain_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    # Bring example_id into a column if it's the index
    chain_df = chain_df.reset_index(drop=True)

    # Get columns we'll need from the source df
    chain_specific_cols = ["q_pn_unit_iid", "q_pn_unit_type", "q_pn_unit_sequence_length", "q_pn_unit_cluster_id", "q_pn_unit_is_metal"]  #! columns we need for each chain (JH) changed 250925    
    base_cols = [
        "example_id", "pdb_id", "assembly_id", "path", "q_pn_unit_contacting_pn_unit_iids",
        *chain_specific_cols,
    ]
    interface_df = chain_df[base_cols].copy()

    # Explode interface contacts
    interface_df = interface_df.explode("q_pn_unit_contacting_pn_unit_iids", ignore_index=True)
    interface_df = interface_df.dropna(subset=["q_pn_unit_contacting_pn_unit_iids"])  # drop pn_units without interface contacts

    # Extract the contacted iid
    interface_df["q_pn_unit_iid_2"] = interface_df["q_pn_unit_contacting_pn_unit_iids"].map(
        lambda d: d.get("pn_unit_iid") if isinstance(d, dict) else None
    )
    interface_df = interface_df.dropna(subset=["q_pn_unit_iid_2"])

    # Join back to get chain info for chain_2
    right = chain_df[["pdb_id", "assembly_id"] + chain_specific_cols].rename(
                    columns={f"{c}": f"{c}_2" for c in chain_specific_cols})
    interface_df = interface_df.merge(
        right, on=["pdb_id", "assembly_id", "q_pn_unit_iid_2"], how="inner", validate="many_to_one"  # inner join gets rid of interfaces where chain_2 was not in the input chain df
    )

    # Canonicalize pair ordering to dedupe (A_1, B_1) == (B_1, A_1)
    interface_df = _canonicalize_pair_columns(interface_df, order_by="q_pn_unit_iid", paired_cols=chain_specific_cols)

    # Drop exact duplicate interfaces within (pdb_id, assembly_id)
    interface_df = interface_df.drop_duplicates(subset=["pdb_id", "assembly_id", "q_pn_unit_iid_1", "q_pn_unit_iid_2"], keep="first")

    # Build example_id for interfaces by appending 'interfaces' to the source dataset_names
    def _get_interface_example_id(row):
        dataset_names = [dataset_name, "interfaces"]
        query_pn_unit_iids = [row["q_pn_unit_iid_1"], row["q_pn_unit_iid_2"]]
        return generate_example_id(dataset_names, row["pdb_id"], row["assembly_id"], query_pn_unit_iids)

    interface_df["example_id"] = interface_df.apply(_get_interface_example_id, axis=1)

    # Final selection / order of columns
    interface_df = interface_df[
        ["example_id", "pdb_id", "assembly_id", "path"] + [f"{c}_1" for c in chain_specific_cols] + [f"{c}_2" for c in chain_specific_cols]
    ].reset_index(drop=True)
    interface_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)

    return interface_df

# Todo (JH): Exclude the pockets with both small molecules and metals. Need to consider three chains at once.


def _canonicalize_pair_columns(
    df: pd.DataFrame,
    order_by: str = "q_pn_unit_iid",
    paired_cols: list[str] = (
        "q_pn_unit_iid",
        "q_pn_unit_type",
        "q_pn_unit_sequence_length",
        "q_pn_unit_cluster_id",
    ),
    right_suffix: str = "_2",
    out_suffixes: tuple[str, str] = ("_1", "_2"),
) -> pd.DataFrame:
    """
    Create canonical *_1/*_2 columns for each entry in `paired_cols` by
    ordering rows so that order_by_1 <= order_by_2 (lexicographic).
    Does not mutate inputs; returns a new DataFrame with added *_1/*_2 columns.
    """
    out = df.copy()

    # mask: True -> swap left/right
    swap = out[order_by].to_numpy() > out[f"{order_by}{right_suffix}"].to_numpy()

    for base in paired_cols:
        a = out[f"{base}"].to_numpy()
        b = out[f"{base}{right_suffix}"].to_numpy()

        out[f"{base}{out_suffixes[0]}"] = np.where(swap, b, a)
        out[f"{base}{out_suffixes[1]}"] = np.where(swap, a, b)

    return out


def add_chain_counts_info(df: pd.DataFrame, chain_type_cols: list[str], \
                        seq_length_cols: list[str], is_metal_cols: list[str]) -> pd.DataFrame:
    """
    Add chain type and sequence length columns to the dataframe.
    Modifies the dataframe in place and returns it.
    # TODO (JH): faster way?
    """
    # Compute chain type counts
    chain_count_cols = ["n_prot", "n_nuc", "n_peptide", "n_small_molecule", "n_metal", "n_loi"]
    df["chain_types"] = df[chain_type_cols].apply(lambda x: tuple(x), axis=1)
    df["seq_lengths"] = df[seq_length_cols].apply(lambda x: tuple(x), axis=1)
    df["is_metal"] = df[is_metal_cols].apply(lambda x: tuple(x), axis=1) #! (JH) changed 250925

    def _get_chain_type_counts(row) -> dict[str, int]:
        chain_types: tuple[str] = row["chain_types"]
        seq_lengths: tuple[int] = row["seq_lengths"]        
        is_metal: tuple[bool] = row["is_metal"]
        chain_type_counts = {c: 0 for c in chain_count_cols}

        for t, l, m in zip(chain_types, seq_lengths, is_metal):
            if t in aw_enums.ChainTypeInfo.PROTEINS:
                if l < aw_const.PEPTIDE_MAX_RESIDUES:
                    chain_type_counts["n_peptide"] += 1
                else:
                    chain_type_counts["n_prot"] += 1
            elif t in aw_enums.ChainTypeInfo.NUCLEIC_ACIDS:
                chain_type_counts["n_nuc"] += 1
            else:
                if m:
                    chain_type_counts["n_metal"] += 1
                else:
                    chain_type_counts["n_small_molecule"] += 1
                        
        return pd.Series(chain_type_counts)
    
    df[chain_count_cols] = df.apply(_get_chain_type_counts, axis=1)    

    # Delete intermediate columns
    del df["chain_types"]
    del df["seq_lengths"]
    del df["is_metal"] #! (JH) changed 250925
    return df


def add_sampling_weights_info(df: pd.DataFrame,
                              alphas: dict[str, float],
                              beta: float,
                              cluster_cols: list[str]) -> pd.DataFrame:
    """
    Based on the cluster ID in cluster_col and chain counts info, add a sampling weights column to the dataframe.
    Modifies the dataframe in place and returns it.
    """
    assert all(col in df.columns for col in ["n_prot", "n_peptide", "n_nuc", "n_small_molecule", "n_metal", "n_loi"]), "Need to add chain counts info before computing sampling weights"

    # Get cluster size
    df["clusters"] = df[cluster_cols].apply(lambda x: tuple(sorted(tuple(x))), axis=1)  # sort cluster ids to dedupe
    cluster_id_to_size = df["clusters"].value_counts()
    df["cluster_size"] = df["clusters"].map(cluster_id_to_size)

    # Compute weights
    missing_alphas = set(alphas.keys()) - {"a_prot", "a_peptide", "a_nuc", "a_small_molecule", "a_metal", "a_loi"}
    missing_counts = {"n_prot", "n_peptide", "n_nuc", "n_small_molecule", "n_metal", "n_loi"} - set(df.columns)

    if missing_alphas:
        logger.warning(f"Missing alphas from configuration file: {missing_alphas}; defaulting to 0")
    if missing_counts:
        logger.warning(f"Missing chain within dataframe counts: {missing_counts}; defaulting to 0")
        logger.warning(f"Columns in dataframe: {df.columns}")

    logger.info(f"Calculating weights for AF-3 examples using alphas={alphas}, beta={beta}")

    weights = (beta / df["cluster_size"]) * (
        alphas.get("a_prot", 0) * df["n_prot"]
        + alphas.get("a_peptide", 0) * df["n_peptide"]
        + alphas.get("a_nuc", 0) * df["n_nuc"]
        + alphas.get("a_small_molecule", 0) * df["n_small_molecule"]
        + alphas.get("a_metal", 0) * df["n_metal"]
        + alphas.get("a_loi", 0) * df["n_loi"]  # always 0 for now
    )

    df["sampling_weight"] = weights
    return df
