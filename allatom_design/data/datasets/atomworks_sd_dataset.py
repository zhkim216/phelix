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
from allatom_design.data.transform import sd_featurizer_pocket_only

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
                                drop_last=False,
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
        self.pocket_only_training = cfg.get("pocket_only_training", False)
        self.interface_only_training = cfg.get("interface_only_training", False)
        if self.pocket_only_training and self.interface_only_training:
            logger.warning("interface_only_training is ignored when pocket_only_training=True")
            self.interface_only_training = False

        # Initialize featurizer
        # Note: We remove INFERENCE_ONLY_KEYS to avoid cuda initialization issues during training.
        if self.pocket_only_training:
            self.featurizer = sd_featurizer_pocket_only.sd_featurizer_pocket_only(
                **cfg.pocket_featurizer_cfg,
                remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS,
            )
        else:
            self.featurizer = sd_featurizer.sd_featurizer(**cfg.featurizer_cfg,
                                                          remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS,
                                                          ) #! (JH) changed

        
        # Process dataframes for training
        if self.phase == "train":
            self.metadata_path = self.cfg.train_metadata_path
            # Initialize metadata df
            self.metadata_df = self._process_metadata_df(metadata_path=self.metadata_path)        
            
            if not self.pocket_only_training:
                # Process interface df
                self.interface_df = self._process_interface_df(metadata_path=self.metadata_path, dataset_name=Path(self.metadata_path).parent.name)
                if self.interface_only_training:
                    # Keep API compatibility for downstream code paths.
                    self.protein_monomer_chain_df = self.metadata_df.iloc[0:0].copy()
                else:
                    # Process protein chain df (skipped in pocket-only/interface-only training)
                    self.protein_monomer_chain_df = self._process_protein_monomer_chain_df(dataset_name=Path(self.metadata_path).parent.name)                                                                
            
            else:
                self.pocket_df = self._process_pocket_df(metadata_path=self.metadata_path, dataset_name=Path(self.metadata_path).parent.name)
        
            # Compute sampling weights
            if not self.pocket_only_training:
                if self.interface_only_training:
                    # Preserve existing interface weighting logic while excluding monomer samples.
                    empty_monomer_df = self.metadata_df.iloc[0:0].copy()
                    _, self.interface_df = add_cluster_balanced_sampling_weights(
                        monomer_df=empty_monomer_df,
                        interface_df=self.interface_df,
                        alphas_interface=self.cfg.sampling_weights["alphas_interface"],
                        cluster_col="q_pn_unit_cluster_id",
                        k_percentile=self.cfg.sampling_weights["k_percentile"],
                    )
                else:
                    # Cluster-balanced sampling across both dataframes
                    self.protein_monomer_chain_df, self.interface_df = add_cluster_balanced_sampling_weights(
                        monomer_df=self.protein_monomer_chain_df,
                        interface_df=self.interface_df,
                        alphas_interface=self.cfg.sampling_weights["alphas_interface"],                
                        cluster_col="q_pn_unit_cluster_id",
                        k_percentile=self.cfg.sampling_weights["k_percentile"]
                    )            
            # Parse dfs into a common format and concatenate
            self.parsed_df = self._parse_dfs()        


        elif self.phase == "val":
            self.metadata_path = self.cfg.val_metadata_path
            # Initialize metadata df
            self.metadata_df = pd.read_parquet(self.metadata_path)
            if not self.pocket_only_training:                        
                self.metadata_df["query_pn_unit_iids"] = self.metadata_df["query_pn_unit_iids"].apply(ast.literal_eval)
                self.parsed_df = self._parse_dfs()                        
                
            else:
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
        
        # q_pn_unit_is_small_molecule: config option으로 biologically meaningful 버전 사용 가능
        if self.cfg.get("use_biologically_meaningful_small_molecule", False) and \
           "q_pn_unit_is_biologically_meaningful_small_molecule" in metadata_df.columns:
            metadata_df["q_pn_unit_is_small_molecule"] = metadata_df["q_pn_unit_is_biologically_meaningful_small_molecule"]
            logger.info("Using q_pn_unit_is_biologically_meaningful_small_molecule as q_pn_unit_is_small_molecule")
        else:
            metadata_df["q_pn_unit_is_small_molecule"] = (~metadata_df["q_pn_unit_is_polymer"].astype(bool)) & (~metadata_df["q_pn_unit_is_metal"].astype(bool))
        
        # Set index to example_id        
        metadata_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        
        # Convert q_pn_unit_contacting_pn_unit_iids to list. It was saved as a json string.
        if self.phase == "train":
            try:
                metadata_df["q_pn_unit_contacting_pn_unit_iids"] = metadata_df["q_pn_unit_contacting_pn_unit_iids"].apply(json.loads)
            except:
                logger.info("q_pn_unit_contacting_pn_unit_iids is already a list, skipping...")
    
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
        
        # Note: Sampling weights are computed later in add_sampling_weights_with_combined_clusters
        
        def _get_protein_monomer_chain_example_id(row):
            dataset_names = [dataset_name, "protein_monomer_chain"]
            pdb_id = row["pdb_id"]
            assembly_id = row["assembly_id"]
            query_pn_unit_iids = row["q_pn_unit_iid"]
            return generate_example_id(dataset_names, pdb_id, assembly_id, query_pn_unit_iids)
        
        protein_monomer_chain_df["example_id"] = protein_monomer_chain_df.apply(_get_protein_monomer_chain_example_id, axis=1)
        protein_monomer_chain_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        
        return protein_monomer_chain_df        

    def _process_interface_df(self, metadata_path: str = None,
                              dataset_name: str = None) -> pd.DataFrame:
        """
        Processes the interface dataframe based on the filtered chain dataframe. Adds chain counts info and sampling weights.
        """                
        # Copy the metadata dataframe to avoid modifying the original dataframe
        metadata_df = self.metadata_df.copy()
        
        # Apply the general filters for interface df first
        metadata_df = self._apply_filters(self.cfg.train_filters.interface_filter["1"], metadata_df)
                        
        # Exclude small molecules that are covalently linked to proteins
        if self.cfg.exclude_small_molecules_covalently_linked_to_protein and metadata_df.get("q_pn_unit_is_maybe_covalently_linked_to_protein", False).sum() > 0:
            len_before = len(metadata_df)
            metadata_df = metadata_df[~metadata_df['q_pn_unit_is_maybe_covalently_linked_to_protein']]        
            len_after = len(metadata_df)
            logger.info(f"Excluded {len_before - len_after} small molecules in {dataset_name} interface dataset, because of covalently linked to protein")
        
        ### Filter out small molecules with resolution ratio less than min_resolution_ratio
        min_resolution_ratio = self.cfg.get("min_resolution_ratio", 0.8)
        if min_resolution_ratio is not None:
            len_before = len(metadata_df)
            metadata_df = metadata_df[metadata_df['q_pn_unit_resolution_ratio'] >= min_resolution_ratio]
            len_after = len(metadata_df)
            logger.info(f"Excluded {len_before - len_after} small molecules in {dataset_name} interface dataset, because of resolution ratio less than {min_resolution_ratio}")
        
                        
        if self.scheme == "neighbor":
            # Remove chain iids from q_pn_unit_context_group_iids that were excluded by filters
            valid_iids_per_assembly = metadata_df.groupby(['pdb_id', 'assembly_id'])['q_pn_unit_iid'].apply(set).to_dict()
            metadata_df['q_pn_unit_context_group_iids'] = metadata_df.apply(
                lambda row: [
                    iid for iid in row['q_pn_unit_context_group_iids']
                    if iid in valid_iids_per_assembly.get((row['pdb_id'], row['assembly_id']), set())
                ] if row['q_pn_unit_context_group_iids'] is not None else None,
                axis=1
            )
                                    
        # Build interface df
        interface_df = build_interface_df(metadata_df=metadata_df, dataset_name=Path(metadata_path).parent.name)
                        
        # Filter out invalid iids in interface df based on cluster exclusion
        if self.cfg.exclude_val_cluster:                        
            # Filter out invalid iids in all_pn_unit_iids_after_processing and q_pn_unit_context_group_iids
            iid_to_cluster = metadata_df.set_index(['pdb_id', 'assembly_id', 'q_pn_unit_iid'])['q_pn_unit_cluster_id'].to_dict()            
            def filter_valid_iids(row, colname):                
                pdb_id = row['pdb_id']
                assembly_id = row['assembly_id']
                iids = row[colname]
                
                if iids is None:
                    return None
                
                filtered_iids = []
                for iid in iids:
                    if iid_to_cluster.get((pdb_id, assembly_id, iid)) not in self.val_cluster_ids:
                        filtered_iids.append(iid)
                                
                return filtered_iids
                                    
            if self.scheme == "neighbor":
                interface_df['q_pn_unit_context_group_iids'] = interface_df.apply(filter_valid_iids, axis=1, colname='q_pn_unit_context_group_iids')
            
            # Filter out interfaces that have invalid iids (always use cluster_id_1 and cluster_id_2)
            prev_len = len(interface_df)
            interface_df = interface_df[~(interface_df['q_pn_unit_cluster_id_1'].isin(self.val_cluster_ids))]
            interface_df = interface_df[~(interface_df['q_pn_unit_cluster_id_2'].isin(self.val_cluster_ids))]
            current_len = len(interface_df)
            logger.info("--------------------------------")
            logger.info(f"Started with: {prev_len} interfaces")
            logger.info(f"Excluded {prev_len - current_len} interfaces in {dataset_name} interface dataset, because of cluster exclusion")
            logger.info(f"Ended with: {current_len} interfaces")
            logger.info("--------------------------------")
                                
        interface_df = add_chain_counts_info(interface_df)
        
        # Apply the specific filters for the interface                          
        interface_df = self._apply_filters(self.cfg.train_filters.interface_filter["2"] if self.phase == "train" else self.cfg.val_filters.interface_filter["2"], interface_df)            
        
        # Note: Sampling weights are computed later in add_sampling_weights_with_combined_clusters
                                            
        return interface_df
    
    def _process_pocket_df(self, metadata_path: str = None,
                           dataset_name: str = None) -> pd.DataFrame:
        """
        Processes the pocket dataframe.
        """
        metadata_df = self.metadata_df.copy()
        metadata_df = self._apply_filters(self.cfg.train_filters.pocket_filter["1"], metadata_df)

        if self.cfg.exclude_val_cluster:
            prev_len = len(metadata_df)
            metadata_df = metadata_df[~(metadata_df['q_pn_unit_cluster_id'].isin(self.val_cluster_ids))]
            current_len = len(metadata_df)
            logger.info(f"Excluded {prev_len - current_len} pockets in {dataset_name} pocket dataset, because of cluster exclusion")
        
        if self.cfg.exclude_val_cluster:
            prev_len = len(metadata_df)
            metadata_df = metadata_df[~(metadata_df['q_pn_unit_cluster_id'].isin(self.val_cluster_ids))]
            current_len = len(metadata_df)
            logger.info(f"Excluded {prev_len - current_len} pockets in {dataset_name} pocket dataset, because of cluster exclusion")
        
        # Exclude small molecules that are covalently linked to proteins
        if self.cfg.exclude_small_molecules_covalently_linked_to_protein and metadata_df.get("q_pn_unit_is_maybe_covalently_linked_to_protein", False).sum() > 0:
            len_before = len(metadata_df)
            mask = (metadata_df['q_pn_unit_is_biologically_meaningful_small_molecule'] & ~metadata_df['q_pn_unit_is_maybe_covalently_linked_to_protein']) | (~metadata_df['q_pn_unit_is_biologically_meaningful_small_molecule'])
            metadata_df = metadata_df[mask]        
            len_after = len(metadata_df)
            logger.info(f"Excluded {len_before - len_after} small molecules in {dataset_name} interface dataset, because of covalently linked to protein")
            
        if self.scheme == "neighbor":
            # Remove chain iids from q_pn_unit_context_group_iids that were excluded by filters
            valid_iids_per_assembly = metadata_df.groupby(['pdb_id', 'assembly_id'])['q_pn_unit_iid'].apply(set).to_dict()
            metadata_df['q_pn_unit_context_group_iids'] = metadata_df.apply(
                lambda row: [
                    iid for iid in row['q_pn_unit_context_group_iids']
                    if iid in valid_iids_per_assembly.get((row['pdb_id'], row['assembly_id']), set())
                ] if row['q_pn_unit_context_group_iids'] is not None else None,
                axis=1
            )
        
        pocket_df = self._apply_filters(self.cfg.train_filters.pocket_filter["2"], metadata_df)  
        # Keep target ligand IDs at pn_unit granularity (e.g., "A_1,B_1"), not per-chain IDs.
        pocket_df['q_pn_unit_target_ligand_iids'] = pocket_df['q_pn_unit_iid'].apply(
            lambda x: [x] if isinstance(x, str) else (list(x) if isinstance(x, (list, tuple, np.ndarray)) else [])
        )
        # Exclude unexpected multi-target entries (e.g., ["A_1,B_1", "C_1,D_1"]).
        single_target_mask = pocket_df['q_pn_unit_target_ligand_iids'].apply(lambda x: len(x) == 1)
        if (~single_target_mask).any():
            n_excluded = int((~single_target_mask).sum())
            pocket_df = pocket_df[single_target_mask]
            logger.info(
                f"Excluded {n_excluded} pocket examples in {dataset_name} interface dataset, because of multiple target ligands"
            )
        
        #########################################################
        # Add sampling weights info
        #########################################################
        pocket_df["clusters"] = pocket_df[['q_pn_unit_cluster_id']].apply(lambda x: tuple(sorted(tuple(x))), axis=1) 
        cluster_id_to_size = pocket_df["clusters"].value_counts()
        pocket_df["cluster_size"] = pocket_df["clusters"].map(cluster_id_to_size)
        
        weights = 1 / pocket_df["cluster_size"]
        
        pocket_df["sampling_weight"] = weights
        
        return pocket_df
            
    def _parse_dfs(self) -> pd.DataFrame:
        """
        Parses the chain and interface dataframes into a common format and concatenates them.
        """                
                        
        if self.phase == "train":
            chain_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid"])
            
            if self.scheme == "neighbor":
                if self.pocket_only_training:
                    pocket_parser = GenericDFParser(pn_unit_iid_colnames=['q_pn_unit_context_group_iids'], target_ligand_iids_colname=['q_pn_unit_target_ligand_iids'])
                else:
                    interface_parser = GenericDFParser(pn_unit_iid_colnames=['q_pn_unit_context_group_iids'])
            elif self.scheme == "interface":
                interface_parser = GenericDFParser(pn_unit_iid_colnames=['q_pn_unit_iid_1', 'q_pn_unit_iid_2'])                            
            
            if self.pocket_only_training:
                # Pocket-only: only use interface_df (no monomer data)
                parsed_df = self.pocket_df.apply(pocket_parser.parse, axis=1)
            elif self.interface_only_training:
                parsed_df = self.interface_df.apply(interface_parser.parse, axis=1)
            else:
                parsed_df = pd.concat([
                    self.protein_monomer_chain_df.apply(chain_parser.parse, axis=1),
                    self.interface_df.apply(interface_parser.parse, axis=1)
                ], axis=0)

        else:           
            if not self.pocket_only_training:
                val_parser = GenericDFParser(pn_unit_iid_colnames=['query_pn_unit_iids'])
                parsed_df = self.metadata_df.apply(val_parser.parse, axis=1)
            else:
                val_parser = GenericDFParser(pn_unit_iid_colnames=['q_pn_unit_context_group_iids'], target_ligand_iids_colname=['q_pn_unit_target_ligand_iids'])
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


def build_interface_df(metadata_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    # Bring example_id into a column if it's the index
    metadata_df = metadata_df.reset_index(drop=True)

    # Get columns we'll need from the source df        
    chain_specific_cols = ['q_pn_unit_id', 'q_pn_unit_iid', 'q_pn_unit_type', 'q_pn_unit_sequence_length', 
                        'q_pn_unit_is_protein', 'q_pn_unit_is_peptide', 'q_pn_unit_is_nuc', 'q_pn_unit_is_small_molecule', 'q_pn_unit_is_metal', 
                        'q_pn_unit_is_loi', 'q_pn_unit_is_polymer', 'q_pn_unit_cluster_id']    
        
    base_cols = [
        "example_id", "pdb_id", "assembly_id", "path", "q_pn_unit_contacting_pn_unit_iids", "q_pn_unit_context_group_iids",
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
    # Sort so that rows with non-None q_pn_unit_context_group_iids come first,
    # ensuring dedup keeps the row with neighbor chain info (e.g., from the small molecule side)
    interface_df['_neighbor_is_none'] = interface_df['q_pn_unit_context_group_iids'].isna()
    interface_df = interface_df.sort_values('_neighbor_is_none', kind='stable')
    interface_df = interface_df.drop_duplicates(subset=["pdb_id", "assembly_id", "q_pn_unit_iid_1", "q_pn_unit_iid_2"], keep="first")
    interface_df = interface_df.drop(columns=['_neighbor_is_none'])

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
            "q_pn_unit_context_group_iids",
            "contact_min_distance",
            "contact_num_contacts",
        ]
        + [f"{c}_1" for c in chain_specific_cols]
        + [f"{c}_2" for c in chain_specific_cols]
    ].reset_index(drop=True)
    interface_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)

    # Add interface_type column for determining alpha weights
    # Types: protein_protein, protein_nuc, protein_peptide, protein_small_molecule, protein_metal
    def get_interface_type(row):
        is_protein_1 = row["q_pn_unit_is_protein_1"]
        is_protein_2 = row["q_pn_unit_is_protein_2"]
        
        if is_protein_1 and is_protein_2:
            return "protein_protein"
        
        # Determine the non-protein side
        if is_protein_1:
            # Chain 2 is the non-protein
            if row.get("q_pn_unit_is_nuc_2", False):
                return "protein_nuc"
            elif row.get("q_pn_unit_is_peptide_2", False):
                return "protein_peptide"
            elif row.get("q_pn_unit_is_metal_2", False):
                return "protein_metal"
            elif row.get("q_pn_unit_is_small_molecule_2", False):
                return "protein_small_molecule"
        else:
            # Chain 1 is the non-protein (chain 2 is protein)
            if row.get("q_pn_unit_is_nuc_1", False):
                return "protein_nuc"
            elif row.get("q_pn_unit_is_peptide_1", False):
                return "protein_peptide"
            elif row.get("q_pn_unit_is_metal_1", False):
                return "protein_metal"
            elif row.get("q_pn_unit_is_small_molecule_1", False):
                return "protein_small_molecule"
        
        # Default fallback
        return "protein_small_molecule"
    
    interface_df["interface_type"] = interface_df.apply(get_interface_type, axis=1)

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

def add_cluster_balanced_sampling_weights(
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    alphas_interface: dict[str, float],    
    cluster_col: str = "q_pn_unit_cluster_id",
    k_percentile: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute sampling weights for cluster-balanced sampling across monomer and interface dataframes.
    
    Ensures two levels of equalization:
    1. Interface pair cluster equalization: each (c1, c2) pair in interface_df is sampled equally
    2. Overall protein cluster equalization: each protein cluster C is sampled equally across monomer_df + interface_df
    
    Algorithm:
    - Step 1: Compute interface weights (pair cluster equalization)
    - Step 2: Compute each protein cluster's interface contribution
    - Step 3: Adjust monomer weights to achieve overall protein cluster equalization (auto-computed)
    
    Note: Monomer weights are automatically computed to ensure protein cluster equalization.
    k_percentile: Adjust this value to control the balance between interface and monomer sampling
          
    K Calculation:
    - K is the target total contribution per protein cluster
    - K = percentile(interface_contrib, k_percentile)
    - If k_percentile=100.0 (default): K = max(interface_contrib)
    - If k_percentile=80.0: K = 80th percentile of interface_contrib
    - Clusters with interface_contrib > K will have monomer_weight = 0 (interface-only sampling)
    
    Args:
        monomer_df: Protein monomer chain dataframe with cluster_col
        interface_df: Interface dataframe with q_pn_unit_cluster_id_1, q_pn_unit_cluster_id_2, interface_type
        alphas_interface: Dict with keys like a_protein_protein, a_protein_small_molecule, etc.        
        cluster_col: Column name for protein cluster ID in monomer_df
        k_percentile: Percentile of interface_contrib to use for K calculation (default: 100.0 = max)
    """
    # ===== Step 1: Compute interface weights (pair cluster equalization) =====
    
    # Compute pair cluster sizes for interface_df
    interface_df["pair_cluster"] = interface_df.apply(
        lambda row: tuple(sorted([row["q_pn_unit_cluster_id_1"], row["q_pn_unit_cluster_id_2"]])),
        axis=1
    )
    pair_cluster_sizes = interface_df["pair_cluster"].value_counts()
    interface_df["pair_cluster_size"] = interface_df["pair_cluster"].map(pair_cluster_sizes)
    
    # Compute alpha for each interface based on interface_type
    def get_interface_alpha(row):
        interface_type = row["interface_type"]
        base_alpha = alphas_interface.get(f"a_{interface_type}", 0.0)
        
        # Add a_protein_loi if the non-protein chain is loi
        loi_alpha = 0.0
        if row.get("q_pn_unit_is_loi_1", False) or row.get("q_pn_unit_is_loi_2", False):
            loi_alpha = alphas_interface.get("a_protein_loi", 0.0)
        
        return base_alpha + loi_alpha
    
    interface_df["alpha"] = interface_df.apply(get_interface_alpha, axis=1)
    
    # Interface weight: β_i × alpha / pair_cluster_size
    interface_df["sampling_weight"] = interface_df["alpha"] / interface_df["pair_cluster_size"]
    
    # ===== Step 2: Compute each protein cluster's interface contribution =====
    
    # For each interface, identify which protein clusters are involved
    # protein-protein: both c1 and c2 contribute
    # protein-X (where X is not protein): only the protein side contributes
    
    # Create a mapping: protein_cluster -> total interface contribution
    interface_contrib = {}
    
    for _, row in interface_df.iterrows():
        weight = row["sampling_weight"]
        c1, c2 = row["q_pn_unit_cluster_id_1"], row["q_pn_unit_cluster_id_2"]
        is_protein_1 = row.get("q_pn_unit_is_protein_1", False)
        is_protein_2 = row.get("q_pn_unit_is_protein_2", False)
        
        if is_protein_1:
            interface_contrib[c1] = interface_contrib.get(c1, 0.0) + weight
        if is_protein_2:
            interface_contrib[c2] = interface_contrib.get(c2, 0.0) + weight
    
    # ===== Step 3: Compute K and scale interface weights for equalization =====
    
    # Count monomer rows per cluster
    monomer_cluster_counts = monomer_df[cluster_col].value_counts().to_dict()
    
    # Get all protein clusters (from both monomer and interface)
    all_protein_clusters = set(monomer_df[cluster_col].unique())
    for c in interface_contrib.keys():
        all_protein_clusters.add(c)
    
    # Compute K: target total contribution per cluster
    # K = percentile(interface_contrib, k_percentile)
    if interface_contrib:
        contrib_values = list(interface_contrib.values())
        max_interface_contrib = max(contrib_values)
        # If all interface contributions are 0 (all alphas are 0), use default K
        # This ensures monomer sampling still works when interface sampling is disabled
        if max_interface_contrib == 0.0:
            K = 1.0
            n_clusters_exceeding_k = 0
        else:
            K = np.percentile(contrib_values, k_percentile)
            n_clusters_exceeding_k = sum(1 for v in contrib_values if v > K)
    else:
        max_interface_contrib = 0.0
        K = 1.0  # Default K for no interfaces
        n_clusters_exceeding_k = 0
    
    logger.info(
        f"Normalized sampling: K={K:.4f} (k_percentile={k_percentile}), "
        f"max_interface_contrib={max_interface_contrib:.4f}, "
        f"clusters_exceeding_K={n_clusters_exceeding_k}"
    )
    
    # ===== Step 3b: Scale down interface weights for clusters exceeding K =====
    # For clusters with interface_contrib > K, scale their interface weights
    # so that the total interface contribution becomes exactly K.
    # This ensures all protein clusters have equal total contribution.
    
    # Compute per-cluster scaling factors
    scaling_factors = {}
    for c, contrib in interface_contrib.items():
        if contrib > K:
            scaling_factors[c] = K / contrib
        # else: no scaling needed (factor = 1.0)
    
    if scaling_factors:
        logger.info(
            f"Scaling interface weights for {len(scaling_factors)} clusters exceeding K. "
            f"Min scaling factor: {min(scaling_factors.values()):.6f}, "
            f"Max scaling factor: {max(scaling_factors.values()):.6f}"
        )
        
        # Apply scaling to interface_df rows
        def scale_interface_weight(row):
            weight = row["sampling_weight"]
            if weight == 0.0:
                return 0.0
            
            c1, c2 = row["q_pn_unit_cluster_id_1"], row["q_pn_unit_cluster_id_2"]
            is_protein_1 = row.get("q_pn_unit_is_protein_1", False)
            is_protein_2 = row.get("q_pn_unit_is_protein_2", False)
            
            # Collect scaling factors from protein sides
            factors = []
            if is_protein_1 and c1 in scaling_factors:
                factors.append(scaling_factors[c1])
            if is_protein_2 and c2 in scaling_factors:
                factors.append(scaling_factors[c2])
            
            if factors:
                # Use min factor to ensure neither cluster exceeds K
                return weight * min(factors)
            return weight
        
        interface_df["sampling_weight"] = interface_df.apply(scale_interface_weight, axis=1)
        
        # Recompute interface_contrib after scaling
        interface_contrib = {}
        for _, row in interface_df.iterrows():
            weight = row["sampling_weight"]
            c1, c2 = row["q_pn_unit_cluster_id_1"], row["q_pn_unit_cluster_id_2"]
            is_protein_1 = row.get("q_pn_unit_is_protein_1", False)
            is_protein_2 = row.get("q_pn_unit_is_protein_2", False)
            
            if is_protein_1:
                interface_contrib[c1] = interface_contrib.get(c1, 0.0) + weight
            if is_protein_2:
                interface_contrib[c2] = interface_contrib.get(c2, 0.0) + weight
    
    # ===== Step 4: Compute monomer weights for overall protein cluster equalization =====
    
    # Compute monomer weight for each cluster
    # monomer_contrib[C] = K - interface_contrib[C]
    # weight per row = monomer_contrib[C] / monomer_count[C]
    
    def compute_monomer_weight(row):
        c = row[cluster_col]
        i_contrib = interface_contrib.get(c, 0.0)
        m_count = monomer_cluster_counts.get(c, 1)
        
        # Target monomer contribution for this cluster
        target_monomer_contrib = K - i_contrib
        
        # Ensure non-negative weight
        if target_monomer_contrib < 0:            
            target_monomer_contrib = 0.0
        
        # Weight per row (auto-computed for equalization)
        weight = target_monomer_contrib / m_count
        return weight
    
    monomer_df["sampling_weight"] = monomer_df.apply(compute_monomer_weight, axis=1)
    
    # Log statistics        
    protein_clusters_in_monomer_df = set(monomer_df[cluster_col].unique())
    protein_clusters_in_interface_df = set(interface_contrib.keys())
    protein_clusters_only_in_monomer_df = protein_clusters_in_monomer_df - protein_clusters_in_interface_df
    protein_clusters_only_in_interface_df = protein_clusters_in_interface_df - protein_clusters_in_monomer_df
    protein_clusters_in_both_df = protein_clusters_in_monomer_df & protein_clusters_in_interface_df
    
    
    n_monomer_clusters = len(protein_clusters_in_monomer_df)
    n_interface_clusters = len(interface_df["pair_cluster"].unique())
    
    logger.info(
        f"Combined cluster sampling weights:\n"
        f"  - Protein monomer df: {len(monomer_df)} samples, {n_monomer_clusters} unique protein clusters\n"
        f"  - Interface df: {len(interface_df)} samples, {len(pair_cluster_sizes)} unique pair clusters\n"
        f"  - Total protein clusters: {len(all_protein_clusters)} clusters in total\n"
        f"  - Protein clusters in both: {len(protein_clusters_in_both_df)} clusters in both\n"
        f"  - Protein clusters only in monomer df: {len(protein_clusters_only_in_monomer_df)} clusters only in monomer df\n"
        f"  - Protein clusters only in interface df: {len(protein_clusters_only_in_interface_df)} clusters only in interface df"
    )
    
    return monomer_df, interface_df

