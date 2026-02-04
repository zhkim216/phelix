import json
import logging
import random
import time
from pathlib import Path
from typing import Literal, override
import ast

import atomworks.enums as aw_enums
import atomworks.ml.preprocessing.constants as aw_const
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from atomworks.ml.example_id import generate_example_id
from atomworks.ml.datasets import MolecularDataset
from atomworks.ml.datasets.parsers import GenericDFParser
from atomworks.ml.samplers import DistributedMixedSampler
from atomworks.ml.utils.io import read_parquet_with_metadata
from atomworks.ml.samplers import LazyWeightedRandomSampler
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader

from allatom_design.data.sampler import Sampler
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
                
    def train_dataloader(self) -> DataLoader:
                                
        train_loader = DataLoader(dataset=self._train_set,                            
                            batch_size=self.cfg.batch_size,
                            num_workers=self.cfg.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=sd_collator,                            
                            worker_init_fn=worker_init_fn)
                            
        
        return train_loader
        

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(dataset=self._val_set,
                                batch_size=self.cfg.batch_size,
                                num_workers=self.cfg.num_workers,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True,
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

        self.scheme = cfg.get("grouping_scheme", "all")

        # Initialize featurizer
        # Note: We remove INFERENCE_ONLY_KEYS to avoid cuda initialization issues during training.
        self.featurizer = sd_featurizer.sd_featurizer(**cfg.featurizer_cfg,
                                                      remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS,
                                                      ) #! (JH) changed
        
        # Process dataframes for training
        if self.phase == "train":
            self.metadata_path = self.cfg.train_metadata_path
            # Initialize metadata df
            self.metadata_df = self._process_metadata_df(metadata_path=self.metadata_path)        
            
            # Process protein chain df
            self.protein_monomer_chain_df = self._process_protein_monomer_chain_df(dataset_name=Path(self.metadata_path).parent.name)                                                                
        
            # Process interface df
            self.interface_df = self._process_interface_df(metadata_path=self.metadata_path)
        
            # # Process complex df
            # self.complex_df = self._process_complex_df(filters=self.cfg.train_filters.complex_filter, dataset_name=Path(self.cfg.train_metadata_path).parent.name)
            
            # Parse dfs into a common format and concatenate
            self.parsed_df = self._parse_dfs()        

        elif self.phase == "val":
            self.metadata_path = self.cfg.val_metadata_path
            # Initialize metadata df
            self.metadata_df = pd.read_parquet(self.metadata_path)
            
            self.metadata_df["query_pn_unit_iids"] = self.metadata_df["query_pn_unit_iids"].apply(ast.literal_eval)
            # self.metadata_df = self._process_metadata_df(self.cfg.val_metadata_path)        
            self.parsed_df = self._parse_dfs()
                    
        # Initialize per-worker random number generator
        if phase == "train":
            self._sampler = Sampler(self.get_sampling_weights())
            self._rng, self._samples = None, None

        

    @override
    def __getitem__(self, idx: int):       
        if self.phase == "train":
            # For training, draw from infinite sampler.
            self._ensure_worker_rng()
            idx = next(self._samples)
        
        # Load cached example.                
        example_id = self.idx_to_id(idx)                                            
        parsed_row = self.parsed_df.loc[example_id]        
                
        try:
            example = self._load_cached_example(parsed_row["extra_info"]["pdb_id"])
        except FileNotFoundError:
            logger.warning(f"Cached example for {parsed_row['extra_info']['pdb_id']} not found in {self.cfg.pdb_path}/cached_examples in {self.phase} dataset, skipping...")                        
            return self.__getitem__(idx + 1)
        
        # if self.phase == "train":
        #     s = parsed_row["example_id"]
        #     import re
        #     match = re.search(r"\['[^']+',\s*'([^']+)'\]", s)
        #     if match:
        #         dataset_type = match.group(1)  # Either of protein_monomer_chain or interface
        #         # if dataset_type == "protein_monomer_chain":
        #         #     print(1)
        #         if dataset_type == "interface":
        #             if len(example["chain_info"].keys()) >= 2:
        #                 print(1)
                           
        # Add metadata info and phase info     
        example.update(parsed_row)                                  
        example["phase"] = self.phase
                                            
        # Apply train-time transforms.
        try:
            feats = self.featurizer(example)            
        except Exception as e:
            logger.error(f"Error applying train-time transforms to example {example_id} in {self.phase} dataset: {e}")
            return self.__getitem__(idx + 1)

        return feats

    def _ensure_worker_rng(self):
        """Ensure that each worker has a unique random number generator."""
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._samples = self._sampler.sample(self._rng)
    
    def _process_metadata_df(self, metadata_path: str = None) -> pd.DataFrame:
        """
        Initial processing of the metadata dataframe. Adds phase info and validation IDs.
        """                                                    
        metadata_df = read_parquet_with_metadata(metadata_path)
        
        # Add q_pn_unit_is_nuc & q_pn_unit_is_small_molecule columns
        nuc_chain_type_enums = [chain_type.value for chain_type in aw_enums.ChainType.get_nucleic_acids()]
        metadata_df["q_pn_unit_is_nuc"] = metadata_df["q_pn_unit_is_polymer"].astype(bool) & (metadata_df["q_pn_unit_type"].isin(nuc_chain_type_enums))            
        metadata_df["q_pn_unit_is_small_molecule"] = (~metadata_df["q_pn_unit_is_polymer"].astype(bool)) & (~metadata_df["q_pn_unit_is_metal"].astype(bool))
        
        # Set index to example_id        
        metadata_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        
        # Convert q_pn_unit_contacting_pn_unit_iids to list. It was saved as a json string.
        if self.phase == "train":
            metadata_df["q_pn_unit_contacting_pn_unit_iids"] = metadata_df["q_pn_unit_contacting_pn_unit_iids"].apply(json.loads)
    
        # Load in validation IDs and hold out based on phase. Case insensitive, no extension.                    
        with open(self.cfg.validation_ids_file, "r") as f:                
            val_split = {x.lower().split(".")[0] for x in f.read().splitlines()}        
        logger.info(f"Loading in validation IDs from {self.cfg.validation_ids_file}...")
            
        if self.cfg.debug:
            debug_pdb_list = np.random.choice(metadata_df['pdb_id'].unique().tolist(), size=self.cfg.debug_num_ids, replace=False)
            debug_train_pdb_list = debug_pdb_list[:3*self.cfg.debug_num_ids//4]
            debug_val_pdb_list = debug_pdb_list[3*self.cfg.debug_num_ids//4:]
            metadata_df.loc[metadata_df["pdb_id"].isin(debug_train_pdb_list), "phase"] = "train"                        
            metadata_df.loc[metadata_df["pdb_id"].isin(debug_val_pdb_list), "phase"] = "val"            
        else:                                            
            metadata_df.loc[~metadata_df["pdb_id"].str.lower().isin(val_split), "phase"] = "train"
            metadata_df.loc[metadata_df["pdb_id"].str.lower().isin(val_split), "phase"] = "val"        
            
        if self.cfg.exclude_val_cluster: #Todo: This is a strategy used in ligandmpnn, need to be revisited later (JH)
            self.val_cluster_ids = list(set(metadata_df[(metadata_df['q_pn_unit_is_protein'] == True) & (metadata_df['phase'] == 'val')]['q_pn_unit_cluster_id']))
        
        # Subset metadata_df to the current phase
        metadata_df = metadata_df[metadata_df["phase"] == self.phase]               
        
        # Apply metadata filters
        metadata_df = self._apply_filters(self.cfg.train_filters.metadata_filter if self.phase == "train" else self.cfg.val_filters.metadata_filter, metadata_df)        
                                                
        return metadata_df
                                    

    def _process_protein_monomer_chain_df(self, dataset_name: str = None) -> pd.DataFrame:
        """
        Processes the protein monomer chain dataframe. Adds chain counts info and sampling weights, and applies filters.
        """           
        
        metadata_df = self.metadata_df.copy()                                         

        protein_monomer_chain_df = self._apply_filters(self.cfg.train_filters.protein_monomer_chain_filter, metadata_df)                        
        
        if self.cfg.exclude_val_cluster:
            prev_len = len(protein_monomer_chain_df)
            protein_monomer_chain_df = protein_monomer_chain_df[~(protein_monomer_chain_df['q_pn_unit_cluster_id'].isin(self.val_cluster_ids))]
            current_len = len(protein_monomer_chain_df)
            logger.info(f"Excluded {prev_len - current_len} chains in {dataset_name} protein monomer chain dataset, because of cluster exclusion")
                                                                        
        # Add chain counts info
        protein_monomer_chain_df = add_chain_counts_info(protein_monomer_chain_df)
        
        # Add sampling weights
        alphas = self.cfg.sampling_weights["alphas_protein_monomer_chain"] #! (JH) changed 250925
        protein_monomer_chain_df = add_sampling_weights_info(protein_monomer_chain_df,
                                             alphas=alphas,
                                             beta=self.cfg.sampling_weights["betas"]["beta_protein_monomer_chain"],
                                             cluster_cols=["q_pn_unit_cluster_id"])                                        
        
        def _get_protein_monomer_chain_example_id(row):
            dataset_names = [dataset_name, "protein_monomer_chain"]
            pdb_id = row["pdb_id"]
            assembly_id = row["assembly_id"]
            query_pn_unit_iids = row["q_pn_unit_iid"]
            return generate_example_id(dataset_names, pdb_id, assembly_id, query_pn_unit_iids)
        
        protein_monomer_chain_df["example_id"] = protein_monomer_chain_df.apply(_get_protein_monomer_chain_example_id, axis=1)
        protein_monomer_chain_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        
        return protein_monomer_chain_df
        
    def _process_complex_df(self, filters = None, dataset_name: str = None) -> pd.DataFrame:
        """
        Processes the dataframe for complexes. In L-caliby, we give all the ligands to the model, so we don't need to build interface df.
        After processing, each row represents one complex (pdb_id + assembly_id), not one chain.
        """
        
        # Apply the general filters for cif files first            
        complex_df = self._apply_filters(filters["1"], self.metadata_df)
        
        if self.cfg.exclude_val_cluster:
            prev_len = len(complex_df)            
            complex_df = complex_df[~(complex_df['q_pn_unit_cluster_id'].isin(self.val_cluster_ids))]
            
            remaining_chains = complex_df.groupby(['pdb_id', 'assembly_id'])['q_pn_unit_iid'].apply(set).to_dict()
            def update_pn_unit_iids(row):
                key = (row['pdb_id'], row['assembly_id'])
                remaining = remaining_chains.get(key, set())
                original_iids = json.loads(row['all_pn_unit_iids_after_processing'])
                filtered_iids = [iid for iid in original_iids if iid in remaining]
                return json.dumps(filtered_iids)
            complex_df['all_pn_unit_iids_after_processing'] = complex_df.apply(update_pn_unit_iids, axis=1)
        
            current_len = len(complex_df)
            logger.info(f"Excluded {prev_len - current_len} chains in {dataset_name} dataset, because of cluster exclusion")
                

        # Aggregate rows by (pdb_id, assembly_id) so each complex is one row
        complex_df = self._aggregate_complex_df(complex_df, dataset_name)

        # Calculate the chain counts for the complex (now operating on aggregated df)
        complex_df = add_chain_counts_info_aggregated(complex_df)

        # Apply the specific filters for the complex                          
        complex_df = self._apply_filters(filters["2"], complex_df)
                        
        complex_df = add_sampling_weights_info_aggregated(
            df=complex_df,
            alphas=self.cfg.sampling_weights["alphas_complex"],
            beta=self.cfg.sampling_weights["betas"]["beta_complex"],
        )
        
        def _get_complex_example_id(row):
            dataset_names = [dataset_name, "complexes"]
            pdb_id = row["pdb_id"]
            assembly_id = row["assembly_id"]            
            query_pn_unit_iids = row["q_pn_unit_iid"]  # Now a list like ['A_1', 'B_1']
            return generate_example_id(dataset_names, pdb_id, assembly_id, query_pn_unit_iids)
        
        complex_df["example_id"] = complex_df.apply(_get_complex_example_id, axis=1)
        complex_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        
        return complex_df

    def _aggregate_complex_df(self, complex_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Aggregate complex_df so that each (pdb_id, assembly_id) becomes a single row.
        
        - Columns that are the same across all chains in a complex are kept as-is (first value).
        - Chain-specific columns (q_pn_unit_*) are aggregated into lists.
        - The order of chains in lists follows alphabetical order of q_pn_unit_iid.
        """
        # Columns that should remain as single values (same across all chains in a complex)
        keep_first_cols = [
            'pdb_id', 'assembly_id', 'clash_severity', 'resolution', 
            'deposition_date', 'release_date', 'method', 'num_polymer_pn_units',
            'num_resolved_atoms_in_processed_assembly', 'total_num_atoms_in_unprocessed_assembly',
            'all_pn_unit_iids_after_processing', 'path', 'rel_path', 'phase'
        ]
        
        # Columns that should be aggregated into lists (chain-specific)
        list_agg_cols = [col for col in complex_df.columns 
                         if col.startswith('q_pn_unit_') and col in complex_df.columns]
        
        # Sort by q_pn_unit_iid within each group to ensure consistent ordering
        complex_df = complex_df.sort_values(['pdb_id', 'assembly_id', 'q_pn_unit_iid'])
        
        # Build aggregation dictionary
        agg_dict = {}
        for col in keep_first_cols:
            if col in complex_df.columns:
                agg_dict[col] = 'first'
        for col in list_agg_cols:
            if col in complex_df.columns:
                agg_dict[col] = list
        
        # Group by (pdb_id, assembly_id) and aggregate
        aggregated_df = complex_df.groupby(['pdb_id', 'assembly_id'], as_index=False).agg(agg_dict)
        
        logger.info(f"Aggregated {len(complex_df)} chain rows into {len(aggregated_df)} complex rows in {dataset_name} dataset")
        
        return aggregated_df

    def _process_interface_df(self, metadata_path: str = None,
                              dataset_name: str = None) -> pd.DataFrame:
        """
        Processes the interface dataframe based on the filtered chain dataframe. Adds chain counts info and sampling weights.
        """                
        # Copy the metadata dataframe to avoid modifying the original dataframe
        metadata_df = self.metadata_df.copy()
        
        # Apply the general filters for interface df first
        metadata_df = self._apply_filters(self.cfg.train_filters.interface_filter["1"], metadata_df)
        
        # Convert all_pn_unit_iids_after_processing to list
        metadata_df["all_pn_unit_iids_after_processing"] = metadata_df["all_pn_unit_iids_after_processing"].apply(json.loads)
        
        # Delete excluded chains by filters in all_pn_unit_iids_after_processing
        iids_by_pdb = metadata_df.groupby(['pdb_id', 'assembly_id'])['q_pn_unit_iid'].apply(list).to_dict()
        metadata_df['all_pn_unit_iids_after_processing'] = metadata_df.apply(lambda row: iids_by_pdb[(row['pdb_id'], row['assembly_id'])], axis=1)
                                    
        # Build interface df
        interface_df = build_interface_df(metadata_df=metadata_df, dataset_name=Path(metadata_path).parent.name)
                        
        # Filter out invalid iids in interface df based on cluster exclusion
        if self.cfg.exclude_val_cluster:                        
            # Filter out invalid iids in all_pn_unit_iids_after_processing
            iid_to_cluster = metadata_df.set_index(['pdb_id', 'assembly_id', 'q_pn_unit_iid'])['q_pn_unit_cluster_id'].to_dict()            
            def filter_valid_iids(row):                
                pdb_id = row['pdb_id']
                assembly_id = row['assembly_id']
                iids = row['all_pn_unit_iids_after_processing']
                
                filtered_iids = []
                for iid in iids:
                    if iid_to_cluster[(pdb_id, assembly_id, iid)] not in self.val_cluster_ids:
                        filtered_iids.append(iid)
                                
                return filtered_iids
            
            interface_df['all_pn_unit_iids_after_processing'] = interface_df.apply(filter_valid_iids, axis=1)
            
            # Filter out interfaces that have invalid iids
            prev_len = len(interface_df)
            interface_df = interface_df[~(interface_df['q_pn_unit_cluster_id_1'].isin(self.val_cluster_ids))]
            interface_df = interface_df[~(interface_df['q_pn_unit_cluster_id_2'].isin(self.val_cluster_ids))]
            current_len = len(interface_df)
            logger.info("--------------------------------")
            logger.info(f"Started with: {prev_len} interfaces")
            logger.info(f"Excluded {prev_len - current_len} interfaces in {dataset_name} interface dataset, because of cluster exclusion")
            logger.info(f"Ended with: {current_len} interfaces")
            logger.info("--------------------------------")
            
            # prev_len = len(metadata_df)
            # metadata_df = metadata_df[~metadata_df['q_pn_unit_cluster_id'].isin(self.val_cluster_ids)]
            # current_len = len(metadata_df)
            # logger.info(f"Excluded {prev_len - current_len} chains in {dataset_name} interface dataset, because of cluster exclusion")
                    
        interface_df = add_chain_counts_info(interface_df)
        
        # Apply the specific filters for the interface                          
        interface_df = self._apply_filters(self.cfg.train_filters.interface_filter["2"] if self.phase == "train" else self.cfg.val_filters.interface_filter["2"], interface_df)            
                
        alphas = self.cfg.sampling_weights["alphas_interface"] 
            
        interface_df = add_sampling_weights_info(interface_df,
                                                 alphas=alphas,
                                                 beta=self.cfg.sampling_weights["betas"]["beta_interface"],
                                                 cluster_cols=["q_pn_unit_cluster_id_1", "q_pn_unit_cluster_id_2"])                        
                                            
        return interface_df
            
    def _parse_dfs(self) -> pd.DataFrame:
        """
        Parses the chain and interface dataframes into a common format and concatenates them.
        """                
                        
        if self.phase == "train":
            chain_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid"])
            # if self.complex_df is not None:
            #     complex_parser = GenericDFParser(pn_unit_iid_colnames=['all_pn_unit_iids_after_processing'])
            #     n_complexes = len(self.complex_df)
            #     n_chains_in_complexes = self.complex_df['n_chains'].sum() if 'n_chains' in self.complex_df.columns else n_complexes
            #     logger.info(f"Final {self.phase} dataset contains {len(self.protein_chain_df)} protein monomer chains and {n_chains_in_complexes} chains in {n_complexes} complexes")                
            #     parsed_df = pd.concat([
            #         self.protein_chain_df.apply(chain_parser.parse, axis=1),
            #         self.complex_df.apply(complex_parser.parse, axis=1)
            #     ], axis=0)
            
            if self.scheme == "all":
                interface_parser = GenericDFParser(pn_unit_iid_colnames=['all_pn_unit_iids_after_processing'])
            elif self.scheme == "interface":
                interface_parser = GenericDFParser(pn_unit_iid_colnames=['q_pn_unit_iid_1', 'q_pn_unit_iid_2'])                          
                
            
            parsed_df = pd.concat([
                self.protein_monomer_chain_df.apply(chain_parser.parse, axis=1),
                self.interface_df.apply(interface_parser.parse, axis=1)
            ], axis=0)

        else: 
            val_parser = GenericDFParser(pn_unit_iid_colnames=['query_pn_unit_iids'])
            parsed_df = self.metadata_df.apply(val_parser.parse, axis=1)
            logger.info(f"Final {self.phase} dataset contains {len(self.metadata_df['pdb_id'].unique().tolist())} pdbs")                

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


def build_interface_df(metadata_df: pd.DataFrame = None, dataset_name: str = None) -> pd.DataFrame:
    # Bring example_id into a column if it's the index
    metadata_df = metadata_df.reset_index(drop=True)

    # Get columns we'll need from the source df    
    chain_specific_cols = ['q_pn_unit_id', 'q_pn_unit_iid', 'q_pn_unit_type', 'q_pn_unit_sequence_length', 
                           'q_pn_unit_is_protein', 'q_pn_unit_is_peptide', 'q_pn_unit_is_nuc', 'q_pn_unit_is_small_molecule', 'q_pn_unit_is_metal', 
                           'q_pn_unit_is_loi', 'q_pn_unit_is_polymer', 'q_pn_unit_cluster_id']
        
    base_cols = [
        "example_id", "pdb_id", "assembly_id", "path", "all_pn_unit_iids_after_processing", "q_pn_unit_contacting_pn_unit_iids",
        *chain_specific_cols,
    ]
    interface_df = metadata_df[base_cols].copy()

    # Explode interface contacts
    interface_df = interface_df.explode("q_pn_unit_contacting_pn_unit_iids", ignore_index=True)
    interface_df = interface_df.dropna(subset=["q_pn_unit_contacting_pn_unit_iids"])  # drop pn_units without interface contacts
        
    interface_df["contact_min_distance"] = interface_df["q_pn_unit_contacting_pn_unit_iids"].map(
        lambda d: d.get("min_distance", np.inf) if isinstance(d, dict) else np.inf
    )
    interface_df["contact_num_contacts"] = interface_df["q_pn_unit_contacting_pn_unit_iids"].map(
        lambda d: d.get("num_contacts", 0) if isinstance(d, dict) else 0
    )

    # Extract the contacted iid
    interface_df["q_pn_unit_iid_2"] = interface_df["q_pn_unit_contacting_pn_unit_iids"].map(
        lambda d: d.get("pn_unit_iid") if isinstance(d, dict) else None
    )
    interface_df = interface_df.dropna(subset=["q_pn_unit_iid_2"])

    # Join back to get chain info for chain_2
    right = metadata_df[["pdb_id", "assembly_id"] + chain_specific_cols].rename(
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
        dataset_names = [dataset_name, "interface"]
        query_pn_unit_iids = [row["q_pn_unit_iid_1"], row["q_pn_unit_iid_2"]]
        return generate_example_id(dataset_names, row["pdb_id"], row["assembly_id"], query_pn_unit_iids)

    interface_df["example_id"] = interface_df.apply(_get_interface_example_id, axis=1)

    # Final selection / order of columns
    interface_df = interface_df[
        [
            "example_id",
            "pdb_id",
            "assembly_id",
            "path",
            "all_pn_unit_iids_after_processing",
            "contact_min_distance",
            "contact_num_contacts",
        ]
        + [f"{c}_1" for c in chain_specific_cols]
        + [f"{c}_2" for c in chain_specific_cols]
    ].reset_index(drop=True)
    interface_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)

    return interface_df

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



def add_chain_counts_info(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add chain type count columns to the dataframe, aggregated by pdb_id.
    Each row will have the total count of each chain type for its pdb_id.
    
    Handles both chain_df (columns without suffix) and compelex_df.
    If complex is True, counts are summed across all chains in the complex.
    """
    
    # Nucleic acid types
    is_interface = 'q_pn_unit_type_1' in df.columns
    
    if not is_interface:        
        df['n_prot'] = df.apply(lambda x: 1 if x['q_pn_unit_is_protein'] else 0, axis=1)
        df['n_nuc'] = df.apply(lambda x: 1 if x['q_pn_unit_is_nuc'] else 0, axis=1)
        df['n_peptide'] = df.apply(lambda x: 1 if x['q_pn_unit_is_peptide'] else 0, axis=1)
        df['n_small_molecule'] = df.apply(lambda x: 1 if x['q_pn_unit_is_small_molecule'] else 0, axis=1)
        df['n_metal'] = df.apply(lambda x: 1 if x['q_pn_unit_is_metal'] else 0, axis=1)
        df['n_loi'] = df.apply(lambda x: 1 if x['q_pn_unit_is_loi'] else 0, axis=1)
    else: 
        # Interface df: sum counts from both chains
        df['n_prot'] = df.apply(
            lambda x: (1 if x['q_pn_unit_is_protein_1'] else 0) + (1 if x['q_pn_unit_is_protein_2'] else 0), axis=1)
        df['n_nuc'] = df.apply(
            lambda x: (1 if x['q_pn_unit_is_nuc_1'] else 0) + (1 if x['q_pn_unit_is_nuc_2'] else 0), axis=1)
        df['n_peptide'] = df.apply(
            lambda x: (1 if x['q_pn_unit_is_peptide_1'] else 0) + (1 if x['q_pn_unit_is_peptide_2'] else 0), axis=1)
        df['n_small_molecule'] = df.apply(
            lambda x: (1 if x['q_pn_unit_is_small_molecule_1'] else 0) + 
                      (1 if x['q_pn_unit_is_small_molecule_2'] else 0), axis=1)
        df['n_metal'] = df.apply(
            lambda x: (1 if x['q_pn_unit_is_metal_1'] else 0) + (1 if x['q_pn_unit_is_metal_2'] else 0), axis=1)
        df['n_loi'] = df.apply(
            lambda x: (1 if x['q_pn_unit_is_loi_1'] else 0) + (1 if x['q_pn_unit_is_loi_2'] else 0), axis=1)
    
        # # First, compute per-row indicators (0 or 1)
        # df['_is_prot'] = df['q_pn_unit_is_protein'].astype(int)
        # df['_is_nuc'] = ((df['q_pn_unit_is_polymer']) & (df['q_pn_unit_type'].isin(nuc_chain_type_enums))).astype(int)
        # df['_is_peptide'] = df['q_pn_unit_is_peptide'].astype(int)
        # df['_is_small_molecule'] = ((~df['q_pn_unit_is_polymer']) & (~df['q_pn_unit_is_metal'])).astype(int)
        # df['_is_metal'] = df['q_pn_unit_is_metal'].astype(int)
        # df['_is_loi'] = df['q_pn_unit_is_loi'].astype(int)
    
        # # Aggregate by pdb_id
        # pdb_counts = df.groupby('pdb_id').agg({
        #     '_is_prot': 'sum',
        #     '_is_nuc': 'sum',
        #     '_is_peptide': 'sum',
        #     '_is_small_molecule': 'sum',
        #     '_is_metal': 'sum',
        #     '_is_loi': 'sum'
        # }).rename(columns={
        #     '_is_prot': 'n_prot',
        #     '_is_nuc': 'n_nuc',
        #     '_is_peptide': 'n_peptide',
        #     '_is_small_molecule': 'n_small_molecule',
        #     '_is_metal': 'n_metal',
        #     '_is_loi': 'n_loi'
        # })
        # # Map aggregated counts back to each row
        # df['n_prot'] = df['pdb_id'].map(pdb_counts['n_prot'])
        # df['n_nuc'] = df['pdb_id'].map(pdb_counts['n_nuc'])
        # df['n_peptide'] = df['pdb_id'].map(pdb_counts['n_peptide'])
        # df['n_small_molecule'] = df['pdb_id'].map(pdb_counts['n_small_molecule'])
        # df['n_metal'] = df['pdb_id'].map(pdb_counts['n_metal'])
        # df['n_loi'] = df['pdb_id'].map(pdb_counts['n_loi'])
            
        # # Drop temporary columns
        # df.drop(columns=['_is_prot', '_is_nuc', '_is_peptide', '_is_small_molecule', '_is_metal', '_is_loi'], inplace=True)
            
    return df


def add_sampling_weights_info(df: pd.DataFrame,
                              alphas: dict[str, float],
                              beta: float,
                              cluster_cols: list[str]) -> pd.DataFrame:
    """
    Based on the cluster ID in cluster_col and chain counts info, add a sampling weights column to the dataframe.
    Modifies the dataframe in place and returns it.
    """
    # Get cluster size
    df["clusters"] = df[cluster_cols].apply(lambda x: tuple(sorted(tuple(x))), axis=1)  
    #! No need to apply tuple here, as multi-ligand 
    cluster_id_to_size = df["clusters"].value_counts()
    df["cluster_size"] = df["clusters"].map(cluster_id_to_size)

    # Compute weights
    missing_alphas = set(alphas.keys()) - {"a_prot", "a_nuc", "a_peptide", "a_small_molecule", "a_metal", "a_loi"}
    missing_counts = {"n_prot", "n_nuc", "n_peptide", "n_small_molecule", "n_metal", "n_loi"} - set(df.columns)

    if missing_alphas:
        logger.warning(f"Missing alphas from configuration file: {missing_alphas}; defaulting to 0")
    if missing_counts:
        logger.warning(f"Missing chain within dataframe counts: {missing_counts}; defaulting to 0")
        logger.warning(f"Columns in dataframe: {df.columns}")

    logger.info(f"Calculating weights for AF-3 examples using alphas={alphas}, beta={beta}")

    weights = (beta / df["cluster_size"]) * (
        alphas.get("a_prot", 0) * df["n_prot"]        
        + alphas.get("a_nuc", 0) * df["n_nuc"]
        + alphas.get("a_peptide", 0) * df["n_peptide"]
        + alphas.get("a_small_molecule", 0) * df["n_small_molecule"]        
        + alphas.get("a_metal", 0) * df["n_metal"]
        + alphas.get("a_loi", 0) * df["n_loi"]  # always 0 for now
    )

    df["sampling_weight"] = weights
    return df


def add_chain_counts_info_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add chain type count columns to the aggregated complex dataframe.
    
    In the aggregated df, q_pn_unit_* columns are lists (one value per chain).
    We count how many chains of each type exist by summing the boolean lists.
    """
    # Nucleic acid types
    nuc_chain_type_enums = [chain_type.value for chain_type in aw_enums.ChainType.get_nucleic_acids()]
    
    # Count proteins: sum of True values in q_pn_unit_is_protein list
    df['n_prot'] = df['q_pn_unit_is_protein'].apply(lambda x: sum(x) if isinstance(x, list) else int(x))
    
    # Count nucleic acids: need to check both q_pn_unit_is_polymer and q_pn_unit_type
    def count_nuc(row):
        is_polymer_list = row['q_pn_unit_is_polymer']
        type_list = row['q_pn_unit_type']
        if isinstance(is_polymer_list, list):
            return sum(1 for is_poly, chain_type in zip(is_polymer_list, type_list) 
                      if is_poly and chain_type in nuc_chain_type_enums)
        else:
            return int(is_polymer_list and type_list in nuc_chain_type_enums)
    df['n_nuc'] = df.apply(count_nuc, axis=1)
    
    # Count peptides
    df['n_peptide'] = df['q_pn_unit_is_peptide'].apply(lambda x: sum(x) if isinstance(x, list) else int(x))
    
    # Count small molecules: not polymer and not metal
    def count_small_molecule(row):
        is_polymer_list = row['q_pn_unit_is_polymer']
        is_metal_list = row['q_pn_unit_is_metal']
        if isinstance(is_polymer_list, list):
            return sum(1 for is_poly, is_metal in zip(is_polymer_list, is_metal_list) 
                      if not is_poly and not is_metal)
        else:
            return int(not is_polymer_list and not is_metal_list)
    df['n_small_molecule'] = df.apply(count_small_molecule, axis=1)
    
    # Count metals
    df['n_metal'] = df['q_pn_unit_is_metal'].apply(lambda x: sum(x) if isinstance(x, list) else int(x))
    
    # Count LOI (ligand of interest)
    df['n_loi'] = df['q_pn_unit_is_loi'].apply(lambda x: sum(x) if isinstance(x, list) else int(x))
    
    # Count total chains
    df['n_chains'] = df['q_pn_unit_iid'].apply(lambda x: len(x) if isinstance(x, list) else 1)
    
    return df


def add_sampling_weights_info_aggregated(
    df: pd.DataFrame,
    alphas: dict[str, float],
    beta: float,
) -> pd.DataFrame:
    """
    Add sampling weights to the aggregated complex dataframe.
    
    In the aggregated df, q_pn_unit_cluster_id is a list of cluster IDs.
    We create a tuple of sorted cluster IDs to represent the complex's clusters.
    """
    # Create clusters tuple from the list of cluster IDs (keep duplicates, just sort)
    df["clusters"] = df['q_pn_unit_cluster_id'].apply(lambda x: tuple(sorted(x)) if isinstance(x, list) else (x,))
    
    # Count how many complexes have each unique cluster tuple
    cluster_id_to_size = df["clusters"].value_counts()
    df["cluster_size"] = df["clusters"].map(cluster_id_to_size)
    
    # Compute weights
    missing_alphas = set(alphas.keys()) - {"a_prot", "a_nuc", "a_peptide", "a_small_molecule", "a_metal", "a_loi"}
    missing_counts = {"n_prot", "n_nuc", "n_peptide", "n_small_molecule", "n_metal", "n_loi"} - set(df.columns)

    if missing_alphas:
        logger.warning(f"Missing alphas from configuration file: {missing_alphas}; defaulting to 0")
    if missing_counts:
        logger.warning(f"Missing chain within dataframe counts: {missing_counts}; defaulting to 0")
        logger.warning(f"Columns in dataframe: {df.columns}")

    logger.info(f"Calculating weights for aggregated complexes using alphas={alphas}, beta={beta}")

    # For aggregated complexes, we don't need to divide by n_chains since each row is already one complex
    weights = (beta / df["cluster_size"]) * (
        alphas.get("a_prot", 0) * df["n_prot"]        
        + alphas.get("a_nuc", 0) * df["n_nuc"]
        + alphas.get("a_peptide", 0) * df["n_peptide"]
        + alphas.get("a_small_molecule", 0) * df["n_small_molecule"]        
        + alphas.get("a_metal", 0) * df["n_metal"]
        + alphas.get("a_loi", 0) * df["n_loi"]
    )

    df["sampling_weight"] = weights
    return df
