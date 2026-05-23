import json
import logging
import random
import time
from pathlib import Path
from typing import Literal
import ast

import atomworks.enums as aw_enums
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
from typing_extensions import override

from allatom_design.data.sampler import Sampler
from allatom_design.data.transform.pad import pad_to_max
from allatom_design.data.transform import sd_featurizer
from allatom_design.utils.metadata_utils import split_components

logger = logging.getLogger(__name__)

# Map ligand-centered interface_type values to the
# corresponding alpha key in `alphas_interface`. Adding a new modality
# (e.g. metal-centered, nuc-centered, peptide-centered, PPI) requires
# only adding an entry here; the rest of the weighting code stays unchanged.
_INTERFACE_TYPE_TO_ALPHA_KEY: dict[str, str] = {
    "bmsm_protein": "alpha_protein_small_molecule",
    "bmm_protein": "alpha_protein_metal",
    "nuc_lig_protein": "alpha_protein_nuc_lig",
    # Future extensions:
    # "peptide_protein": "alpha_protein_peptide",
    # "protein_protein": "alpha_protein_protein",
}

class AtomworksSDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path
        self._train_set = SDDataset(cfg, phase="train")
        self._val_set = SDDataset(cfg, phase="val")

    def train_dataloader(self) -> DataLoader:
        num_workers = self.cfg.get("num_workers", 0)
        persistent_workers = num_workers > 0
        prefetch_factor = 4 if num_workers > 0 else None

        train_loader = DataLoader(dataset=self._train_set,
                            batch_size=self.cfg.batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=sd_collator,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor,
                            worker_init_fn=worker_init_fn)

        return train_loader


    def val_dataloader(self) -> DataLoader:
        num_workers = self.cfg.get("num_workers", 0)
        persistent_workers = num_workers > 0
        prefetch_factor = 4 if num_workers > 0 else None

        val_loader = DataLoader(dataset=self._val_set,
                                batch_size=self.cfg.batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=sd_collator,
                                persistent_workers=persistent_workers,
                                prefetch_factor=prefetch_factor,
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

        self.interface_only_training = cfg.get("interface_only_training", False)

        # Initialize featurizer
        # Note: We remove INFERENCE_ONLY_KEYS to avoid cuda initialization issues during training.
        self.featurizer = sd_featurizer.sd_featurizer(
            **cfg.featurizer_cfg,
            remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS,
        )


        # Process dataframes for training
        if self.phase == "train":
            self.metadata_path = self.cfg.train_metadata_path
            # Initialize metadata df
            self.metadata_df = self._process_metadata_df(metadata_path=self.metadata_path)

            # Process interface df
            self.interface_df = self._process_interface_df(metadata_path=self.metadata_path, dataset_name=Path(self.metadata_path).parent.name)
            if self.interface_only_training:
                # Keep API compatibility for downstream code paths.
                self.protein_monomer_chain_df = self.metadata_df.iloc[0:0].copy()
            else:
                # Process protein chain df (skipped in pocket-only/interface-only training)
                self.protein_monomer_chain_df = self._process_protein_monomer_chain_df(dataset_name=Path(self.metadata_path).parent.name)

            # Compute sampling weights
            ligand_cluster_col = self.cfg.sampling_weights.get("ligand_cluster_col", None)
            if self.interface_only_training:
                # Preserve existing interface weighting logic while excluding monomer samples.
                empty_monomer_df = self.metadata_df.iloc[0:0].copy()
                _, self.interface_df = add_cluster_balanced_sampling_weights(
                    monomer_df=empty_monomer_df,
                    interface_df=self.interface_df,
                    alphas_interface=self.cfg.sampling_weights["alphas_interface"],
                    cluster_col="q_pn_unit_cluster_id",
                    k_percentile=self.cfg.sampling_weights["k_percentile"],
                    ligand_cluster_col=ligand_cluster_col,
                )
            else:
                # Cluster-balanced sampling across both dataframes
                self.protein_monomer_chain_df, self.interface_df = add_cluster_balanced_sampling_weights(
                    monomer_df=self.protein_monomer_chain_df,
                    interface_df=self.interface_df,
                    alphas_interface=self.cfg.sampling_weights["alphas_interface"],
                    cluster_col="q_pn_unit_cluster_id",
                    k_percentile=self.cfg.sampling_weights["k_percentile"],
                    ligand_cluster_col=ligand_cluster_col,
                )

            # Parse dfs into a common format and concatenate
            self.parsed_df = self._parse_dfs()


        elif self.phase == "val":
            self.metadata_path = self.cfg.val_metadata_path
            # Initialize metadata df
            self.metadata_df = pd.read_parquet(self.metadata_path)
            if "example_id" not in self.metadata_df.columns and self.metadata_df.index.name == "example_id":
                self.metadata_df = self.metadata_df.reset_index()
            self.metadata_df["query_pn_unit_iids"] = self.metadata_df["query_pn_unit_iids"].apply(_parse_pn_unit_iids_value)
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

        # Define nucleic acid and small molecule columns.
        nuc_chain_type_enums = [chain_type.value for chain_type in aw_enums.ChainType.get_nucleic_acids()]
        metadata_df["q_pn_unit_is_nuc"] = metadata_df["q_pn_unit_is_polymer"].astype(bool) & (metadata_df["q_pn_unit_type"].isin(nuc_chain_type_enums))
        metadata_df["q_pn_unit_is_small_molecule"] = (
            (~metadata_df["q_pn_unit_is_polymer"].astype(bool))
            & (~metadata_df["q_pn_unit_is_metal"].astype(bool))
            & (~metadata_df["q_pn_unit_is_halide"].astype(bool))
        )

        # Set index to example_id
        metadata_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)

        # Convert q_pn_unit_contacting_pn_unit_iids to list. It was saved as a json string.
        if self.phase == "train":
            try:
                metadata_df["q_pn_unit_contacting_pn_unit_iids"] = metadata_df["q_pn_unit_contacting_pn_unit_iids"].apply(json.loads)
            except Exception:
                logger.info("q_pn_unit_contacting_pn_unit_iids is already a list, skipping...")


        # Narrow / derive biologically-meaningful flags. BMSM uses only protein
        # partner contacts so it is independent and runs once. BMSM_context / BMM
        # / BMH depend on each other through partner contact edges and are solved
        # by `_iterative_narrow_bm_flags` (Gauss-Seidel to fixed point). After
        # narrowing, `_drop_non_bm_modality_rows` removes non-covalent
        # sm/metal/halide rows that fail their narrowed BM* flag. Maybe-covalent
        # small molecules are retained in the BM PN-unit set, so downstream
        # interface-center logic must use the explicit BMSM flag.
        metadata_df = self._narrow_bmsm_flag(metadata_df)
        metadata_df = self._iterative_narrow_bm_flags(metadata_df)
        metadata_df = self._drop_non_bm_modality_rows(metadata_df)

        # Load in validation IDs and hold out based on phase. Case insensitive, no extension.
        with open(self.cfg.validation_ids_file, "r") as f:
            val_split = {x.lower().split(".")[0] for x in f.read().splitlines()}
        logger.info(f"Loading in validation IDs from {self.cfg.validation_ids_file}...")

        # Debug mode: sample a subset of pdbs for training and validation
        if self.cfg.debug:
            debug_pdb_list = np.random.choice(metadata_df['pdb_id'].unique().tolist(), size=self.cfg.debug_num_ids, replace=False)
            debug_train_pdb_list = debug_pdb_list[:3*self.cfg.debug_num_ids//4]
            debug_val_pdb_list = debug_pdb_list[3*self.cfg.debug_num_ids//4:]
            metadata_df.loc[metadata_df["pdb_id"].isin(debug_train_pdb_list), "phase"] = "train"
            metadata_df.loc[metadata_df["pdb_id"].isin(debug_val_pdb_list), "phase"] = "val"
        else: # Normal mode: use validation IDs to split train and val datasets
            metadata_df.loc[~metadata_df["pdb_id"].str.lower().isin(val_split), "phase"] = "train"
            metadata_df.loc[metadata_df["pdb_id"].str.lower().isin(val_split), "phase"] = "val"

        # Exclude clusters that appear in the val split
        if self.cfg.exclude_val_cluster:
            self.val_cluster_ids = list(set(metadata_df[(metadata_df['q_pn_unit_is_protein'] == True) & (metadata_df['phase'] == 'val')]['q_pn_unit_cluster_id']))

        # Subset metadata_df to the current phase
        metadata_df = metadata_df[metadata_df["phase"] == self.phase]

        # Apply metadata filters
        metadata_df = self._apply_filters(self.cfg.train_filters.metadata_filter if self.phase == "train" else self.cfg.val_filters.metadata_filter, metadata_df)
        metadata_df = self._add_biologically_meaningful_pn_unit_iids(metadata_df)

        return metadata_df

    def _narrow_bmsm_flag(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Narrow the BMSM flag with optional pocket gates and add derived contact columns.

        Always adds three derived columns (small cost; also used by downstream
        yaml filters that may reference them):
          * `num_contacting_protein_chains`
          * `q_pn_unit_n_contacting_protein_atoms` (v9 parquet only)
          * `q_pn_unit_contacting_protein_atom_ratio` (v9 parquet only)

        Then AND-masks `q_pn_unit_is_biologically_meaningful_small_molecule`
        against six optional gates from `cfg.biologically_meaningful_small_molecule`:
          * `exclude_covalently_linked_to_protein` (bool, default False)
          * `num_contacting_protein_chains` (int / inequality str / null)
          * `min_contacting_protein_atoms` (int / null)
          * `min_contacting_protein_atom_ratio` (float / null)
          * `min_resolution_ratio` (float / null)
          * `min_avg_occupancy_nonpolymer` (float / null)

        The flag column is updated in place; rows are not dropped and non-BMSM
        rows are unaffected. No-op when `use_biologically_meaningful_small_molecule`
        is falsy or the flag column is missing.
        """
        # Derived columns (always on; cheap and feed downstream gates / yaml queries).
        metadata_df = _add_num_contacting_protein_chains(metadata_df, distance_cutoff=5.0)
        metadata_df = _add_n_contacting_protein_atoms_and_ratio(metadata_df)

        flag_col = "q_pn_unit_is_biologically_meaningful_small_molecule"
        if not self.cfg.get("use_biologically_meaningful_small_molecule", False):
            return metadata_df
        if flag_col not in metadata_df.columns:
            return metadata_df

        bmsm_cfg = self.cfg.get("biologically_meaningful_small_molecule", {})

        # Universe of rows we are narrowing: raw BMSM flag intersected with small_molecule.
        bmsm_flag = (
            metadata_df[flag_col].fillna(False).astype(bool)
            & metadata_df["q_pn_unit_is_small_molecule"].astype(bool)
        )

        # Gate 1 — drop rows that are (maybe) covalently linked to protein.
        exclude_cov = bmsm_cfg.get("exclude_covalently_linked_to_protein", False)
        if exclude_cov:
            cov = metadata_df.get(
                "q_pn_unit_is_maybe_covalently_linked_to_protein",
                pd.Series(False, index=metadata_df.index),
            ).fillna(False).astype(bool)
            bmsm_flag = bmsm_flag & ~cov

        # Gate 2 — chain-count comparison (e.g. "== 1" or ">= 2"); see _build_num_contacting_query.
        n_chains_query = _build_num_contacting_query(
            bmsm_cfg.get("num_contacting_protein_chains", None)
        )
        if n_chains_query is not None:
            mask = metadata_df.eval(n_chains_query)
            if not isinstance(mask, pd.Series) or mask.dtype != bool:
                raise ValueError(
                    f"num_contacting_protein_chains query {n_chains_query!r} did not "
                    f"yield a boolean mask (got type={type(mask).__name__}, "
                    f"dtype={getattr(mask, 'dtype', None)})."
                )
            bmsm_flag = bmsm_flag & mask

        # Gate 3 — minimum protein heavy-atom contacts (atom-level pocket depth).
        min_atoms = bmsm_cfg.get("min_contacting_protein_atoms", None)
        if min_atoms is not None and "q_pn_unit_n_contacting_protein_atoms" in metadata_df.columns:
            bmsm_flag = bmsm_flag & (
                metadata_df["q_pn_unit_n_contacting_protein_atoms"].fillna(0)
                >= int(min_atoms)
            )

        # Gate 4 — minimum coverage ratio (contacting_atoms / expected_heavy_atoms_non_polymer).
        min_ratio = bmsm_cfg.get("min_contacting_protein_atom_ratio", None)
        if min_ratio is not None and "q_pn_unit_contacting_protein_atom_ratio" in metadata_df.columns:
            bmsm_flag = bmsm_flag & (
                metadata_df["q_pn_unit_contacting_protein_atom_ratio"].fillna(-np.inf)
                >= float(min_ratio)
            )

        # Gate 5 — minimum resolution ratio (crystallographic quality).
        min_res = bmsm_cfg.get("min_resolution_ratio", None)
        if min_res is not None and "q_pn_unit_resolution_ratio" in metadata_df.columns:
            bmsm_flag = bmsm_flag & (
                metadata_df["q_pn_unit_resolution_ratio"].fillna(-np.inf)
                >= float(min_res)
            )

        # Gate 6 — minimum non-polymer average occupancy (drop partial-occupancy ligands).
        min_occ = bmsm_cfg.get("min_avg_occupancy_nonpolymer", None)
        if min_occ is not None and "q_pn_unit_avg_occupancy_nonpolymer" in metadata_df.columns:
            bmsm_flag = bmsm_flag & (
                metadata_df["q_pn_unit_avg_occupancy_nonpolymer"].fillna(-np.inf)
                >= float(min_occ)
            )

        metadata_df[flag_col] = bmsm_flag
        logger.info(
            "BMSM narrowed flag "
            f"(exclude_cov={exclude_cov}, "
            f"n_chains_query={n_chains_query!r}, "
            f"min_contacting_protein_atoms={min_atoms}, "
            f"min_contacting_protein_atom_ratio={min_ratio}, "
            f"min_resolution_ratio={min_res}, "
            f"min_avg_occupancy_nonpolymer={min_occ}): "
            f"{int(bmsm_flag.sum()):,} rows pass"
        )
        return metadata_df

    def _iterative_narrow_bm_flags(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Iteratively narrow BMSM_context / BMM / BMH flags with partner filtering.

        Adds three derived boolean columns to `metadata_df` (in place):
          * `q_pn_unit_is_biologically_meaningful_small_molecule_context`
          * `q_pn_unit_is_biologically_meaningful_metal`
          * `q_pn_unit_is_biologically_meaningful_halide`

        Each derived flag is `is_X & static_gate & partner_gate`, where the
        partner gate counts only partners whose own modality flag passes:
          * sm-partner passes if BMSM (already-narrowed) or BMSM_context is True
          * metal-partner passes if BMM is True
          * halide-partner passes if BMH is True
          * protein / other partners always pass
          * partners absent from `metadata_df` (rare) are excluded

        Configurable gates:
          biologically_meaningful_small_molecule_context:
            min_neighboring_heavy_atoms          # SUM(count) of passing partners (sm-source)
            min_neighboring_heavy_atoms_ratio    # SUM(count) / expected_heavy_atoms_non_polymer
          biologically_meaningful_metal:
            min_n_coordination_partners          # COUNT(distinct passing partners) (metal-source)
            min_avg_occupancy_nonpolymer         # static, partner-independent
          biologically_meaningful_halide:
            min_n_coordination_partners          # COUNT(distinct passing partners) (halide-source)
            min_avg_occupancy_nonpolymer         # static

        Iteration: Gauss-Seidel order (bmsm_context -> bm_metal -> bm_halide),
        repeated until a full sweep produces no change. `cfg.max_narrow_loop_iter`
        caps the iteration count (null = run until convergence). Convergence is
        guaranteed because each narrow only flips True->False (monotonic).

        No-op fallback: when all three `use_*` toggles are False (or all required
        columns are missing), derived columns are populated with raw `is_X`.
        """
        use_bmsm_ctx = bool(self.cfg.get("use_biologically_meaningful_small_molecule_context", False))
        use_bmm = bool(self.cfg.get("use_biologically_meaningful_metal", False))
        use_bmh = bool(self.cfg.get("use_biologically_meaningful_halide", False))

        idx = metadata_df.index
        is_sm = metadata_df.get(
            "q_pn_unit_is_small_molecule", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        is_mt = metadata_df.get(
            "q_pn_unit_is_metal", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        is_hd = metadata_df.get(
            "q_pn_unit_is_halide", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)

        bmsm_ctx_col = "q_pn_unit_is_biologically_meaningful_small_molecule_context"
        bmm_col = "q_pn_unit_is_biologically_meaningful_metal"
        bmh_col = "q_pn_unit_is_biologically_meaningful_halide"

        # Initial flag state: most permissive (each derived flag = its raw is_X).
        # Subsequent narrowing is monotonic ↓, so this guarantees fixed-point convergence.
        bmsm_ctx = is_sm.copy()
        bmm = is_mt.copy()
        bmh = is_hd.copy()

        if not (use_bmsm_ctx or use_bmm or use_bmh):
            metadata_df[bmsm_ctx_col] = bmsm_ctx
            metadata_df[bmm_col] = bmm
            metadata_df[bmh_col] = bmh
            return metadata_df

        # BMSM was already narrowed independently (Step 4); merge into sm-partner-pass.
        bmsm_flag = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_small_molecule",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)

        edges = _build_partner_edges(metadata_df)

        bmsm_ctx_cfg = self.cfg.get("biologically_meaningful_small_molecule_context", {}) or {}
        bmm_cfg = self.cfg.get("biologically_meaningful_metal", {}) or {}
        bmh_cfg = self.cfg.get("biologically_meaningful_halide", {}) or {}

        min_neighbor_atoms = bmsm_ctx_cfg.get("min_neighboring_heavy_atoms", None)
        min_neighbor_ratio = bmsm_ctx_cfg.get("min_neighboring_heavy_atoms_ratio", None)
        bmm_allowed_ccds = bmm_cfg.get("allowed_ccd_codes", bmm_cfg.get("ccd_codes", None))
        bmm_min_coord = bmm_cfg.get("min_n_coordination_partners", None)
        bmm_min_protein_donors = bmm_cfg.get("min_coordinating_protein_donor_atoms", None)
        bmm_min_occ = bmm_cfg.get("min_avg_occupancy_nonpolymer", None)
        bmh_min_coord = bmh_cfg.get("min_n_coordination_partners", None)
        bmh_min_occ = bmh_cfg.get("min_avg_occupancy_nonpolymer", None)

        expected_atoms = metadata_df.get("q_pn_unit_expected_heavy_atoms_non_polymer")
        occupancy = metadata_df.get("q_pn_unit_avg_occupancy_nonpolymer")

        # Cap to avoid unbounded loops; convergence is theoretically finite but the
        # cap also defends against config bugs. The default cap (max_narrow_loop_iter=null)
        # is generous enough that any practical input will converge first.
        max_iter_cfg = self.cfg.get("max_narrow_loop_iter", None)
        max_iter_eff = int(max_iter_cfg) if max_iter_cfg is not None else 100

        logger.info(
            "Iterative BM flag narrow start "
            f"(use_bmsm_context={use_bmsm_ctx}, use_bmm={use_bmm}, use_bmh={use_bmh}, "
            f"max_narrow_loop_iter={max_iter_cfg}): "
            f"edges={len(edges):,}, init bmsm_context={int(bmsm_ctx.sum()):,}, "
            f"bm_metal={int(bmm.sum()):,}, bm_halide={int(bmh.sum()):,}"
        )

        def _narrow_bmsm_ctx(bmsm_ctx_in, bmm_in, bmh_in):
            if not use_bmsm_ctx:
                return bmsm_ctx_in
            new = is_sm.copy()
            if not edges.empty and (min_neighbor_atoms is not None or min_neighbor_ratio is not None):
                passes = _compute_partner_passes(edges, bmsm_flag | bmsm_ctx_in, bmm_in, bmh_in)
                sm_edges = edges.loc[(edges["source_kind"].values == "sm") & passes]
                sm_eff = sm_edges.groupby("source_idx")["count"].sum().reindex(idx, fill_value=0)
                if min_neighbor_atoms is not None:
                    new = new & (sm_eff >= int(min_neighbor_atoms))
                if min_neighbor_ratio is not None and expected_atoms is not None:
                    denom = expected_atoms.astype(float).where(expected_atoms.astype(float) > 0, np.nan)
                    ratio = (sm_eff.astype(float) / denom).fillna(-np.inf)
                    new = new & (ratio >= float(min_neighbor_ratio))
            return new

        def _narrow_bm_metal(bmsm_ctx_in, bmm_in, bmh_in):
            if not use_bmm:
                return bmm_in
            new = is_mt.copy()
            if bmm_allowed_ccds is not None:
                new = new & _series_has_any_exact_ccd(
                    metadata_df.get("q_pn_unit_non_polymer_res_names"),
                    bmm_allowed_ccds,
                    index=idx,
                )
            if (bmm_min_coord is not None or bmm_min_protein_donors is not None) and not edges.empty:
                passes = _compute_partner_passes(edges, bmsm_flag | bmsm_ctx_in, bmm_in, bmh_in)
                mt_edges = edges.loc[(edges["source_kind"].values == "metal") & passes]
                if bmm_min_coord is not None:
                    mt_eff = mt_edges.groupby("source_idx").size().reindex(idx, fill_value=0)
                    new = new & (mt_eff >= int(bmm_min_coord))
                if bmm_min_protein_donors is not None:
                    mt_protein_edges = mt_edges.loc[mt_edges["partner_kind"].values == "protein"]
                    mt_eff = mt_protein_edges.groupby("source_idx")["count"].sum().reindex(idx, fill_value=0)
                    new = new & (mt_eff >= int(bmm_min_protein_donors))
            if bmm_min_occ is not None and occupancy is not None:
                new = new & (occupancy.fillna(-np.inf) >= float(bmm_min_occ))
            return new

        def _narrow_bm_halide(bmsm_ctx_in, bmm_in, bmh_in):
            if not use_bmh:
                return bmh_in
            new = is_hd.copy()
            if bmh_min_coord is not None and not edges.empty:
                passes = _compute_partner_passes(edges, bmsm_flag | bmsm_ctx_in, bmm_in, bmh_in)
                hd_edges = edges.loc[(edges["source_kind"].values == "halide") & passes]
                hd_eff = hd_edges.groupby("source_idx").size().reindex(idx, fill_value=0)
                new = new & (hd_eff >= int(bmh_min_coord))
            if bmh_min_occ is not None and occupancy is not None:
                new = new & (occupancy.fillna(-np.inf) >= float(bmh_min_occ))
            return new

        converged = False
        last_it = 0
        for it in range(1, max_iter_eff + 1):
            last_it = it
            prev_ctx, prev_mt, prev_hd = bmsm_ctx.copy(), bmm.copy(), bmh.copy()

            # Gauss-Seidel: each step reads the latest flag values produced earlier
            # in the same iteration. Order is bmsm_context -> bm_metal -> bm_halide.
            bmsm_ctx = _narrow_bmsm_ctx(bmsm_ctx, bmm, bmh)
            bmm = _narrow_bm_metal(bmsm_ctx, bmm, bmh)
            bmh = _narrow_bm_halide(bmsm_ctx, bmm, bmh)

            d_ctx = int((bmsm_ctx != prev_ctx).sum())
            d_mt = int((bmm != prev_mt).sum())
            d_hd = int((bmh != prev_hd).sum())

            logger.info(
                f"  iter={it}: bmsm_context={int(bmsm_ctx.sum()):,} (Δ={d_ctx}), "
                f"bm_metal={int(bmm.sum()):,} (Δ={d_mt}), "
                f"bm_halide={int(bmh.sum()):,} (Δ={d_hd})"
            )

            if d_ctx == 0 and d_mt == 0 and d_hd == 0:
                converged = True
                break

        if converged:
            logger.info(f"Iterative BM flag narrow converged in {last_it} iteration(s).")
        else:
            logger.warning(
                f"Iterative BM flag narrow did not converge within max_narrow_loop_iter="
                f"{max_iter_cfg} (cap={max_iter_eff}); using last state."
            )

        metadata_df[bmsm_ctx_col] = bmsm_ctx
        metadata_df[bmm_col] = bmm
        metadata_df[bmh_col] = bmh
        return metadata_df

    def _drop_non_bm_modality_rows(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Drop sm/metal/halide rows whose narrowed BM* flag is False.

        After this call, raw ``q_pn_unit_is_metal|is_halide`` on metadata_df
        is exactly the BM* narrowed flag. Small molecules are broader because
        maybe-covalent-to-protein rows are retained in the BM PN-unit set even
        when they are excluded as BMSM interface centers.

        Keep policy:
          * small molecule row: keep iff BMSM OR BMSM_context OR maybe-covalent
            to protein is True
            (BMSM_context captures buried/chain-forming sm needed as ligand-side
            context and as a valid partner during BMM/BMH narrowing; maybe-covalent
            rows must remain attached to their protein context).
          * metal row:   keep iff ``is_biologically_meaningful_metal`` is True.
          * halide row:  keep iff ``is_biologically_meaningful_halide`` is True.

        Non-sm/metal/halide rows (protein, peptide, nucleic acid, polymer,
        other) are unaffected. Each modality is gated by its
        ``use_biologically_meaningful_*`` config toggle: when the toggle is
        falsy, the corresponding rows are not dropped (back-compat). When all
        toggles are off, ``metadata_df`` is returned unchanged.
        """
        idx = metadata_df.index
        n_before = len(metadata_df)

        is_sm = metadata_df.get(
            "q_pn_unit_is_small_molecule", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        is_mt = metadata_df.get(
            "q_pn_unit_is_metal", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        is_hd = metadata_df.get(
            "q_pn_unit_is_halide", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)

        bmsm = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_small_molecule",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        bmsm_ctx = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_small_molecule_context",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        bmm = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_metal",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        bmh = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_halide",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        maybe_coval_sm = (
            is_sm
            & metadata_df.get(
                "q_pn_unit_is_maybe_covalently_linked_to_protein",
                pd.Series(False, index=idx),
            ).fillna(False).astype(bool)
        )

        use_bmsm = bool(self.cfg.get("use_biologically_meaningful_small_molecule", False))
        use_bmm = bool(self.cfg.get("use_biologically_meaningful_metal", False))
        use_bmh = bool(self.cfg.get("use_biologically_meaningful_halide", False))

        # `keep_*` masks are True for rows that should remain. Non-target rows
        # (e.g. protein, peptide) are always kept regardless of the toggles.
        if use_bmsm:
            keep_sm = ~is_sm | (bmsm | bmsm_ctx | maybe_coval_sm)
        else:
            keep_sm = pd.Series(True, index=idx)
        if use_bmm:
            keep_mt = ~is_mt | bmm
        else:
            keep_mt = pd.Series(True, index=idx)
        if use_bmh:
            keep_hd = ~is_hd | bmh
        else:
            keep_hd = pd.Series(True, index=idx)

        keep = keep_sm & keep_mt & keep_hd
        if keep.all():
            logger.info(
                "Drop non-BM modality rows: 0 dropped "
                f"(use_bmsm={use_bmsm}, use_bmm={use_bmm}, use_bmh={use_bmh}); "
                f"{n_before:,} rows unchanged."
            )
            return metadata_df

        dropped_sm = int(((is_sm) & (~keep_sm)).sum())
        dropped_mt = int(((is_mt) & (~keep_mt)).sum())
        dropped_hd = int(((is_hd) & (~keep_hd)).sum())
        kept_coval_sm = int((is_sm & maybe_coval_sm & keep_sm).sum())
        out = metadata_df.loc[keep].copy()
        n_after = len(out)
        logger.info(
            "Drop non-BM modality rows: "
            f"dropped {dropped_sm:,} non-BMSM sm, "
            f"{dropped_mt:,} non-BMM metal, "
            f"{dropped_hd:,} non-BMH halide "
            f"(kept {kept_coval_sm:,} maybe-coval sm) "
            f"({n_before:,} -> {n_after:,} rows)."
        )
        return out

    def _add_biologically_meaningful_pn_unit_iids(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Attach per-assembly PN-unit keep lists for train-time featurization.

        Cached examples still contain the broad processed assembly. Interface
        training should crop from a biologically meaningful PN-unit set: all
        polymers plus narrowed BMSM/BMSM_context/BMM/BMH rows, while also
        keeping maybe-covalent small molecules attached to proteins.
        """
        metadata_df = metadata_df.copy()
        if metadata_df.empty:
            metadata_df["biologically_meaningful_pn_unit_iids"] = pd.Series(dtype=object)
            return metadata_df

        idx = metadata_df.index
        is_polymer = metadata_df.get(
            "q_pn_unit_is_polymer", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        is_sm = metadata_df.get(
            "q_pn_unit_is_small_molecule", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        is_mt = metadata_df.get(
            "q_pn_unit_is_metal", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        is_hd = metadata_df.get(
            "q_pn_unit_is_halide", pd.Series(False, index=idx),
        ).fillna(False).astype(bool)

        bmsm = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_small_molecule",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        bmsm_ctx = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_small_molecule_context",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        bmm = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_metal",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        bmh = metadata_df.get(
            "q_pn_unit_is_biologically_meaningful_halide",
            pd.Series(False, index=idx),
        ).fillna(False).astype(bool)
        maybe_coval_sm = (
            is_sm
            & metadata_df.get(
                "q_pn_unit_is_maybe_covalently_linked_to_protein",
                pd.Series(False, index=idx),
            ).fillna(False).astype(bool)
        )

        use_bmsm = bool(self.cfg.get("use_biologically_meaningful_small_molecule", False))
        use_bmm = bool(self.cfg.get("use_biologically_meaningful_metal", False))
        use_bmh = bool(self.cfg.get("use_biologically_meaningful_halide", False))

        keep_sm = is_sm & ((bmsm | bmsm_ctx | maybe_coval_sm) if use_bmsm else pd.Series(True, index=idx))
        keep_mt = is_mt & (bmm if use_bmm else pd.Series(True, index=idx))
        keep_hd = is_hd & (bmh if use_bmh else pd.Series(True, index=idx))
        keep_bm_pn_unit = is_polymer | keep_sm | keep_mt | keep_hd

        bm_pn_unit_iids_by_assembly = {}
        bm_pn_unit_df = metadata_df.loc[keep_bm_pn_unit, ["pdb_id", "assembly_id", "q_pn_unit_iid"]]
        for key, group in bm_pn_unit_df.groupby(["pdb_id", "assembly_id"], sort=False):
            bm_pn_unit_iids_by_assembly[(key[0], str(key[1]))] = list(dict.fromkeys(group["q_pn_unit_iid"].tolist()))

        metadata_df["biologically_meaningful_pn_unit_iids"] = [
            bm_pn_unit_iids_by_assembly.get((row.pdb_id, str(row.assembly_id)), [])
            for row in metadata_df.itertuples(index=False)
        ]
        logger.info(
            "Attached biologically_meaningful_pn_unit_iids "
            f"for {len(bm_pn_unit_iids_by_assembly):,} assemblies "
            f"(kept rows={int(keep_bm_pn_unit.sum()):,}/{len(metadata_df):,})."
        )
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

        # Build ligand-centered interface df. The contact list is already defined
        # at the 5 A interface cutoff in metadata, so no extra distance/contact
        # filter is applied here.
        ligand_cluster_col = self.cfg.sampling_weights.get("ligand_cluster_col", None)
        interface_df = build_interface_df(
            metadata_df=metadata_df,
            dataset_name=Path(metadata_path).parent.name,
            ligand_cluster_col=ligand_cluster_col,
        )

        # Filter out interfaces whose cluster appears in the val split.
        if self.cfg.exclude_val_cluster:
            prev_len = len(interface_df)
            val_cluster_ids = set(self.val_cluster_ids)
            interface_df = interface_df[
                ~interface_df["protein_cluster_multiset"].apply(
                    lambda clusters: any(c in val_cluster_ids for c in clusters)
                )
            ]
            current_len = len(interface_df)
            logger.info("--------------------------------")
            logger.info(f"Started with: {prev_len} interfaces")
            logger.info(f"Excluded {prev_len - current_len} interfaces in {dataset_name} interface dataset, because of cluster exclusion")
            logger.info(f"Ended with: {current_len} interfaces")
            logger.info("--------------------------------")

        interface_df = add_chain_counts_info(interface_df)

        # Keep the BMSM metadata contact-count policy and the constructed
        # interface row in sync. For a BMSM-centered row, `n_prot` is the number
        # of contacting protein chains.
        bmsm_cfg = self.cfg.get("biologically_meaningful_small_molecule", {}) or {}
        n_prot_query = _build_count_query(
            bmsm_cfg.get("num_contacting_protein_chains", None),
            count_col="n_prot",
        )
        if n_prot_query is not None:
            bmsm_mask = interface_df.get(
                "interface_type", pd.Series("", index=interface_df.index)
            ) == "bmsm_protein"
            bmsm_df = self._apply_query(n_prot_query, interface_df.loc[bmsm_mask])
            interface_df = pd.concat([bmsm_df, interface_df.loc[~bmsm_mask]], axis=0)

        # Apply the specific filters for the interface
        interface_df = self._apply_filters(self.cfg.train_filters.interface_filter["2"] if self.phase == "train" else self.cfg.val_filters.interface_filter["2"], interface_df)

        # Note: Sampling weights are computed later in add_cluster_balanced_sampling_weights

        return interface_df

    def _parse_dfs(self) -> pd.DataFrame:
        """Parse chain / interface dataframes into a common format.

        Interface rows carry a variable-length ``query_pn_unit_iids`` list:
        ligand pn_unit(s) first, followed by all contacting protein chains. The
        train interface path still keeps the full complex before spatial cropping.
        """
        if self.phase == "train":
            chain_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid"])
            interface_parser = GenericDFParser(pn_unit_iid_colnames=[])

            def parse_interface_row(row):
                parsed = interface_parser.parse(row)
                parsed["query_pn_unit_iids"] = list(row["query_pn_unit_iids"])
                ligand_pn_unit_iids = list(row.get("ligand_pn_unit_iids", [row["q_pn_unit_iid"]]))
                protein_pn_unit_iids = list(row["protein_pn_unit_iids"])
                parsed["ligand_pn_unit_iids"] = ligand_pn_unit_iids
                parsed["protein_pn_unit_iids"] = protein_pn_unit_iids
                parsed["crop_center_pn_unit_iids"] = [*ligand_pn_unit_iids, *protein_pn_unit_iids]
                if "biologically_meaningful_pn_unit_iids" in row:
                    parsed["biologically_meaningful_pn_unit_iids"] = list(row["biologically_meaningful_pn_unit_iids"])
                parsed["extra_info"].pop("query_pn_unit_iids", None)
                parsed["extra_info"].pop("ligand_pn_unit_iids", None)
                parsed["extra_info"].pop("protein_pn_unit_iids", None)
                parsed["extra_info"].pop("crop_center_pn_unit_iids", None)
                parsed["extra_info"].pop("biologically_meaningful_pn_unit_iids", None)
                return parsed

            if self.interface_only_training:
                parsed_df = self.interface_df.apply(parse_interface_row, axis=1)
            else:
                parsed_df = pd.concat([
                    self.protein_monomer_chain_df.apply(chain_parser.parse, axis=1),
                    self.interface_df.apply(parse_interface_row, axis=1),
                ], axis=0)
        else:
            val_parser = GenericDFParser(pn_unit_iid_colnames=["query_pn_unit_iids"])
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
        if original_num_rows == 0:
            logger.warning(f"Query '{query}' was applied to an empty dataset.")
            return

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


def _parse_contact_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(value, list):
        return []
    return value


def _parse_pn_unit_iids_value(value) -> list[str]:
    """Parse PN-unit iid cells stored as strings, arrays, or Python sequences."""
    if value is None:
        return []
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            value = stripped

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    return [str(value)] if str(value) else []


def _normalize_ligand_ccd_key(value):
    if value is None:
        return ("ccd", "unknown")
    if isinstance(value, float) and pd.isna(value):
        return ("ccd", "unknown")
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = [str(v).strip() for v in value if str(v).strip()]
    else:
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return ("ccd", "unknown")
    return ("ccd", ",".join(sorted(parts)))


def _normalize_ccd_codes(codes) -> set[str]:
    if codes is None:
        return set()
    if isinstance(codes, str):
        codes = [codes]
    return {str(code).strip().upper() for code in codes if str(code).strip()}


def _split_ccd_tokens(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_tokens = value
    else:
        raw_tokens = str(value).replace(";", ",").split(",")
    return [str(token).strip().upper() for token in raw_tokens if str(token).strip()]


def _series_has_any_exact_ccd(series, codes, index) -> pd.Series:
    code_set = _normalize_ccd_codes(codes)
    if not code_set:
        return pd.Series(True, index=index)
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).apply(lambda value: bool(code_set.intersection(_split_ccd_tokens(value))))


def build_interface_df(
    metadata_df: pd.DataFrame,
    dataset_name: str,
    ligand_cluster_col: str | None = None,
) -> pd.DataFrame:
    """Build ligand-centered interface rows.

    Each BMSM output row is one biologically-meaningful small molecule plus
    every protein pn_unit it contacts. Each nucleic-acid-ligand row is one
    4.5 A-connected small NA group plus every protein pn_unit contacted by any
    member at 5 A.
    """
    metadata_df = metadata_df.reset_index(drop=True)
    optional_bool_cols = [
        "q_pn_unit_is_biologically_meaningful_small_molecule",
        "q_pn_unit_is_biologically_meaningful_metal",
        "q_pn_unit_is_nuc_ligand",
    ]
    for col in optional_bool_cols:
        if col not in metadata_df.columns:
            metadata_df[col] = False

    required_cols = [
        "pdb_id",
        "assembly_id",
        "path",
        "q_pn_unit_iid",
        "q_pn_unit_contacting_pn_unit_iids",
        "q_pn_unit_is_protein",
        "q_pn_unit_is_biologically_meaningful_small_molecule",
        "q_pn_unit_cluster_id",
    ]
    missing = [c for c in required_cols if c not in metadata_df.columns]
    if missing:
        raise KeyError(f"build_interface_df missing required columns: {missing}")

    center_cols = [
        "q_pn_unit_id",
        "q_pn_unit_iid",
        "q_pn_unit_type",
        "q_pn_unit_sequence_length",
        "q_pn_unit_is_protein",
        "q_pn_unit_is_peptide",
        "q_pn_unit_is_nuc",
        "q_pn_unit_is_small_molecule",
        "q_pn_unit_is_metal",
        "q_pn_unit_is_polymer",
        "q_pn_unit_is_biologically_meaningful_small_molecule",
        "q_pn_unit_is_biologically_meaningful_metal",
        "q_pn_unit_is_nuc_ligand",
        "q_pn_unit_is_nuc_polymer",
        "q_pn_unit_nucleic_acid_group_id",
        "q_pn_unit_nucleic_acid_group_iids",
        "q_pn_unit_num_resolved_residues_in_nucleic_acid_group",
        "q_pn_unit_nucleic_acid_group_cluster_ids",
        "q_pn_unit_cluster_id",
        "q_pn_unit_non_polymer_res_names",
        "q_pn_unit_bmsm_ligand_cluster_id",
        "biologically_meaningful_pn_unit_iids",
    ]
    if ligand_cluster_col is not None:
        center_cols.append(ligand_cluster_col)
    center_cols = [c for c in dict.fromkeys(center_cols) if c in metadata_df.columns]

    protein_df = metadata_df[metadata_df["q_pn_unit_is_protein"].fillna(False).astype(bool)]
    protein_lookup = {
        (row.pdb_id, str(row.assembly_id), row.q_pn_unit_iid): row
        for row in protein_df.itertuples(index=False)
    }

    def _contact_within_cutoff(contact: dict, distance_cutoff: float | None) -> bool:
        if distance_cutoff is None:
            return True
        distance = contact.get("min_distance")
        if distance is None:
            return True
        try:
            return float(distance) <= distance_cutoff
        except (TypeError, ValueError):
            return False

    def _collect_contacted_proteins(source_rows, distance_cutoff: float | None = None):
        protein_rows = []
        seen_iids = set()
        for source in source_rows:
            contacts = _parse_contact_list(getattr(source, "q_pn_unit_contacting_pn_unit_iids"))
            for contact in contacts:
                if not isinstance(contact, dict) or not _contact_within_cutoff(contact, distance_cutoff):
                    continue
                raw_iid = contact.get("pn_unit_iid")
                if raw_iid is None:
                    continue
                candidate_iids = [str(raw_iid), *split_components(str(raw_iid))]
                for protein_iid in candidate_iids:
                    if protein_iid in seen_iids:
                        continue
                    key = (source.pdb_id, str(source.assembly_id), protein_iid)
                    protein = protein_lookup.get(key)
                    if protein is None:
                        continue
                    seen_iids.add(protein_iid)
                    protein_rows.append(protein)
        return sorted(protein_rows, key=lambda r: r.q_pn_unit_iid)

    def _protein_side(protein_rows):
        protein_iids = [r.q_pn_unit_iid for r in protein_rows]
        protein_cluster_multiset = tuple(r.q_pn_unit_cluster_id for r in protein_rows)
        return protein_iids, protein_cluster_multiset

    center_mask = (
        metadata_df["q_pn_unit_is_biologically_meaningful_small_molecule"]
        .fillna(False)
        .astype(bool)
    )
    if "q_pn_unit_is_small_molecule" in metadata_df.columns:
        center_mask = center_mask & metadata_df["q_pn_unit_is_small_molecule"].fillna(False).astype(bool)
    center_df = metadata_df[center_mask]

    rows = []
    for center in center_df.itertuples(index=False):
        protein_rows = _collect_contacted_proteins([center])
        if not protein_rows:
            continue

        protein_iids, protein_cluster_multiset = _protein_side(protein_rows)
        query_pn_unit_iids = [center.q_pn_unit_iid, *protein_iids]

        row = {
            "pdb_id": center.pdb_id,
            "assembly_id": center.assembly_id,
            "path": center.path,
            "query_pn_unit_iids": query_pn_unit_iids,
            "ligand_pn_unit_iids": (center.q_pn_unit_iid,),
            "protein_pn_unit_iids": tuple(protein_iids),
            "protein_cluster_multiset": protein_cluster_multiset,
            "ligand_ccd_key": _normalize_ligand_ccd_key(
                getattr(center, "q_pn_unit_non_polymer_res_names", None)
            ),
            "interface_type": "bmsm_protein",
        }
        for col in center_cols:
            row[col] = getattr(center, col)

        rows.append(row)

    bmm_center_mask = metadata_df["q_pn_unit_is_biologically_meaningful_metal"].fillna(False).astype(bool)
    if "q_pn_unit_is_metal" in metadata_df.columns:
        bmm_center_mask = bmm_center_mask & metadata_df["q_pn_unit_is_metal"].fillna(False).astype(bool)
    bmm_center_df = metadata_df[bmm_center_mask]

    for center in bmm_center_df.itertuples(index=False):
        protein_rows = _collect_contacted_proteins([center], distance_cutoff=5.0)
        if not protein_rows:
            continue

        protein_iids, protein_cluster_multiset = _protein_side(protein_rows)
        query_pn_unit_iids = [center.q_pn_unit_iid, *protein_iids]

        row = {
            "pdb_id": center.pdb_id,
            "assembly_id": center.assembly_id,
            "path": center.path,
            "query_pn_unit_iids": query_pn_unit_iids,
            "ligand_pn_unit_iids": (center.q_pn_unit_iid,),
            "protein_pn_unit_iids": tuple(protein_iids),
            "protein_cluster_multiset": protein_cluster_multiset,
            "ligand_ccd_key": _normalize_ligand_ccd_key(
                getattr(center, "q_pn_unit_non_polymer_res_names", None)
            ),
            "interface_type": "bmm_protein",
        }
        for col in center_cols:
            row[col] = getattr(center, col)

        rows.append(row)

    nuc_group_cols = {"q_pn_unit_is_nuc_ligand", "q_pn_unit_nucleic_acid_group_id"}
    if nuc_group_cols.issubset(metadata_df.columns):
        nuc_mask = (
            metadata_df["q_pn_unit_is_nuc_ligand"].fillna(False).astype(bool)
            & metadata_df["q_pn_unit_nucleic_acid_group_id"].notna()
        )
        nuc_center_df = metadata_df[nuc_mask]
        for _, group_df in nuc_center_df.groupby(
            ["pdb_id", "assembly_id", "q_pn_unit_nucleic_acid_group_id"],
            sort=False,
        ):
            group_rows = sorted(
                list(group_df.itertuples(index=False)),
                key=lambda r: r.q_pn_unit_iid,
            )
            if not group_rows:
                continue
            protein_rows = _collect_contacted_proteins(group_rows, distance_cutoff=5.0)
            if not protein_rows:
                continue

            ligand_iids = list(dict.fromkeys(r.q_pn_unit_iid for r in group_rows))
            protein_iids, protein_cluster_multiset = _protein_side(protein_rows)
            query_pn_unit_iids = [*ligand_iids, *protein_iids]
            center = group_rows[0]
            nuc_cluster_ids = tuple(sorted(
                {r.q_pn_unit_cluster_id for r in group_rows},
                key=lambda x: repr(x),
            ))

            row = {
                "pdb_id": center.pdb_id,
                "assembly_id": center.assembly_id,
                "path": center.path,
                "query_pn_unit_iids": query_pn_unit_iids,
                "ligand_pn_unit_iids": tuple(ligand_iids),
                "protein_pn_unit_iids": tuple(protein_iids),
                "protein_cluster_multiset": protein_cluster_multiset,
                "ligand_ccd_key": ("nuc_seq_cluster", nuc_cluster_ids),
                "interface_type": "nuc_lig_protein",
            }
            for col in center_cols:
                row[col] = getattr(center, col)

            rows.append(row)

    output_cols = [
        "example_id",
        "pdb_id",
        "assembly_id",
        "path",
        "query_pn_unit_iids",
        "ligand_pn_unit_iids",
        "protein_pn_unit_iids",
        "protein_cluster_multiset",
        "ligand_ccd_key",
        "interface_type",
        *center_cols,
    ]
    interface_df = pd.DataFrame(rows)
    if interface_df.empty:
        interface_df = pd.DataFrame(columns=output_cols)
    else:
        def _get_interface_example_id(row):
            dataset_names = [dataset_name, "interface"]
            return generate_example_id(
                dataset_names,
                row["pdb_id"],
                row["assembly_id"],
                row["query_pn_unit_iids"],
            )

        interface_df["example_id"] = interface_df.apply(_get_interface_example_id, axis=1)
        interface_df = interface_df[output_cols]

    interface_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
    type_counts = (
        interface_df["interface_type"].value_counts().to_dict()
        if "interface_type" in interface_df.columns
        else {}
    )
    logger.info(
        f"Built ligand-centered interface_df with {len(interface_df):,} rows "
        f"from {len(center_df):,} BMSM centers and {len(bmm_center_df):,} BMM centers; "
        f"type_counts={type_counts}"
    )
    return interface_df

def add_chain_counts_info(df: pd.DataFrame = None) -> pd.DataFrame:
    """Add chain-type count columns (``n_prot``, ``n_nuc``, ``n_peptide``,
    ``n_small_molecule``, ``n_metal``, ``n_loi``) to ``df`` in place.

    Two layouts are supported:

    * BMSM-centered interface_df: detected by the ``protein_cluster_multiset``
      column. ``n_prot`` is the number of contacting protein chains.
      ``n_small_molecule`` uses the explicit BMSM center flag so retained
      maybe-covalent retained small molecules do not satisfy interface filters.
    * Monomer chain_df: one row per pn_unit; counts are 1 if the row's raw
      ``q_pn_unit_is_*`` is True, else 0. Same raw==narrow guarantee applies.
    """
    df = df.copy()
    if "protein_cluster_multiset" in df.columns:
        df["n_prot"] = df["protein_cluster_multiset"].apply(lambda clusters: len(clusters))
        df["n_nuc"] = df.apply(lambda x: 1 if x.get("q_pn_unit_is_nuc", False) else 0, axis=1)
        df["n_peptide"] = df.apply(lambda x: 1 if x.get("q_pn_unit_is_peptide", False) else 0, axis=1)
        if "q_pn_unit_is_biologically_meaningful_small_molecule" in df.columns:
            df["n_small_molecule"] = df.apply(
                lambda x: 1 if x.get("q_pn_unit_is_biologically_meaningful_small_molecule", False) else 0,
                axis=1,
            )
        else:
            df["n_small_molecule"] = df.apply(
                lambda x: 1 if x.get("interface_type", None) == "bmsm_protein" else 0,
                axis=1,
            )
        df["n_metal"] = df.apply(lambda x: 1 if x.get("q_pn_unit_is_metal", False) else 0, axis=1)
        df["n_loi"] = 0
        return df

    df['n_prot'] = df.apply(lambda x: 1 if x['q_pn_unit_is_protein'] else 0, axis=1)
    df['n_nuc'] = df.apply(lambda x: 1 if x['q_pn_unit_is_nuc'] else 0, axis=1)
    df['n_peptide'] = df.apply(lambda x: 1 if x['q_pn_unit_is_peptide'] else 0, axis=1)
    df['n_small_molecule'] = df.apply(lambda x: 1 if x['q_pn_unit_is_small_molecule'] else 0, axis=1)
    df['n_metal'] = df.apply(lambda x: 1 if x['q_pn_unit_is_metal'] else 0, axis=1)
    df['n_loi'] = 0
    return df

def add_cluster_balanced_sampling_weights(
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    alphas_interface: dict[str, float],
    cluster_col: str = "q_pn_unit_cluster_id",
    k_percentile: float = 100.0,
    ligand_cluster_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute sampling weights for cluster-balanced sampling across monomer and interface dataframes.

    Ensures two levels of equalization:
    1. Interface bucket equalization: each ``pair_cluster`` (ligand cluster +
       sorted multiset of contacting protein clusters) is sampled equally.
    2. Overall protein cluster equalization: each protein cluster C is sampled
       equally across ``monomer_df`` and ``interface_df`` combined.

    Algorithm:
    - Step 1: Build pair_cluster, alpha (per-row interface_type lookup), and
      bucket-equalized interface weights.
    - Step 2: Sum per-protein-cluster interface contribution.
    - Step 3: Compute K (target total per cluster) and scale down interface
      weights for clusters whose interface_contrib exceeds K.
    - Step 4: Set monomer weights so each cluster's total contribution is K.

    K Calculation:
    - K = percentile(interface_contrib, k_percentile)
    - k_percentile=100.0 (default): K = max(interface_contrib).
    - k_percentile=80.0: K = 80th percentile of interface_contrib.
    - Clusters with interface_contrib > K have their interface weights scaled
      down to K and end up with monomer_weight = 0 (interface-only).

    Args:
        monomer_df: Protein monomer chain dataframe with ``cluster_col``.
        interface_df: ligand-centered interface dataframe with
            ``protein_cluster_multiset`` and ``interface_type``.
        alphas_interface: Dict with five keys
            ``alpha_protein_protein``, ``alpha_protein_small_molecule``,
            ``alpha_protein_metal``, ``alpha_protein_nuc_lig``,
            ``alpha_protein_peptide``. Missing keys default to 0.0.
        cluster_col: Column name for protein cluster ID in monomer_df.
        k_percentile: Percentile of interface_contrib to use for K
            (default: 100.0 = max).
        ligand_cluster_col: Optional ligand-cluster column (e.g.
            ``q_pn_unit_bmsm_ligand_cluster_id``). When present, forms the
            ligand side of pair_cluster; missing values fall back to the
            individual CCD code via ``ligand_ccd_key``.
    """
    interface_df = interface_df.copy()
    monomer_df = monomer_df.copy()

    if "protein_cluster_multiset" not in interface_df.columns:
        raise ValueError(
            "interface_df is expected to be ligand-centered with a "
            "`protein_cluster_multiset` column; legacy `_1`/`_2` pair layout is "
            "no longer supported."
        )

    def _valid_ligand_cluster_value(value):
        try:
            value_int = int(value) if value is not None and not pd.isna(value) else -1
        except (TypeError, ValueError):
            value_int = -1
        return value_int if value_int >= 0 else None

    def _tagged_seq_cluster(value):
        return ("seq", value)

    def _sort_key(value):
        return repr(value)

    def _protein_clusters(row):
        clusters = row.get("protein_cluster_multiset", ())
        if isinstance(clusters, float) and pd.isna(clusters):
            return []
        return list(clusters)

    # Pre-resolve the alpha for each known interface_type once. Unknown
    # interface_type values fall through to weight 0 with a warning so that
    # dataset/version drift is detected early.
    resolved_alpha_by_type: dict[str, float] = {
        itype: float(alphas_interface.get(alpha_key, 0.0))
        for itype, alpha_key in _INTERFACE_TYPE_TO_ALPHA_KEY.items()
    }
    if "interface_type" in interface_df.columns:
        unknown_types = sorted(
            t for t in interface_df["interface_type"].dropna().unique()
            if t not in _INTERFACE_TYPE_TO_ALPHA_KEY
        )
        if unknown_types:
            logger.warning(
                f"Unknown interface_type values in interface_df (alpha=0): {unknown_types}. "
                f"Known types: {sorted(_INTERFACE_TYPE_TO_ALPHA_KEY.keys())}"
            )

    def _compute_alpha(row):
        itype = row.get("interface_type")
        return resolved_alpha_by_type.get(itype, 0.0)

    use_ligand_cluster_col = ligand_cluster_col is not None and ligand_cluster_col in interface_df.columns
    if ligand_cluster_col is not None and not use_ligand_cluster_col:
        logger.warning(
            f"ligand_cluster_col={ligand_cluster_col!r} was requested but is not present on "
            "ligand-centered interface_df. Falling back to individual CCD keys."
        )

    def _ligand_key(row):
        if use_ligand_cluster_col:
            cluster_id = _valid_ligand_cluster_value(row.get(ligand_cluster_col))
            if cluster_id is not None:
                return ("lig_cluster", cluster_id)
        return row.get("ligand_ccd_key", _normalize_ligand_ccd_key(row.get("q_pn_unit_id")))

    interface_df["pair_cluster"] = interface_df.apply(
        lambda row: (
            _ligand_key(row),
            tuple(sorted((_tagged_seq_cluster(c) for c in _protein_clusters(row)), key=_sort_key)),
        ),
        axis=1,
    )

    pair_cluster_sizes = interface_df["pair_cluster"].value_counts()
    interface_df["pair_cluster_size"] = interface_df["pair_cluster"].map(pair_cluster_sizes)

    if not interface_df.empty:
        logger.info(
            f"pair_cluster stats: n_unique={pair_cluster_sizes.size:,}, "
            f"size_min={int(pair_cluster_sizes.min())}, "
            f"size_median={int(pair_cluster_sizes.median())}, "
            f"size_max={int(pair_cluster_sizes.max())}"
        )

    interface_df["alpha"] = interface_df.apply(_compute_alpha, axis=1)
    interface_df["sampling_weight"] = interface_df["alpha"] / interface_df["pair_cluster_size"]

    # ===== Step 2: Compute each protein cluster's interface contribution =====

    # For each interface, identify which protein clusters are involved
    # protein-protein: both c1 and c2 contribute
    # protein-X (where X is not protein): only the protein side contributes

    # Create a mapping: protein_cluster -> total interface contribution
    interface_contrib = {}

    def _compute_interface_contrib() -> dict:
        contrib = {}
        for _, row in interface_df.iterrows():
            weight = row["sampling_weight"]
            for c in _protein_clusters(row):
                contrib[c] = contrib.get(c, 0.0) + weight
        return contrib

    interface_contrib = _compute_interface_contrib()

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

            factors = [scaling_factors[c] for c in _protein_clusters(row) if c in scaling_factors]
            if factors:
                # Use min factor to ensure neither cluster exceeds K
                return weight * min(factors)
            return weight

        interface_df["sampling_weight"] = interface_df.apply(scale_interface_weight, axis=1)

        # Recompute interface_contrib after scaling
        interface_contrib = _compute_interface_contrib()

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

#### Biologically meaningful small molecule helpers
_NUM_CONTACTING_COL = "num_contacting_protein_chains"
_NUM_CONTACTING_OPS = ("==", "!=", ">=", "<=", ">", "<")

def _build_count_query(value, count_col: str = _NUM_CONTACTING_COL) -> str | None:
    """Normalize a contact-count yaml value into a pandas-eval expression.

    Accepted forms:
      * ``None`` / empty string   — disable the narrowing entirely.
      * ``int`` (e.g. ``1``)       — backward-compat shorthand for ``== <int>``.
      * ``str`` starting with a comparison op (``==``, ``!=``, ``>=``, ``<=``, ``>``, ``<``)
        — auto-prefixed with ``count_col``.
      * any other ``str``         — used as-is as a full pandas-eval expression; caller
        is responsible for referencing the count column in it. Expressions using
        ``num_contacting_protein_chains`` are rewritten to ``count_col`` so the
        same config can filter metadata rows and constructed interface rows.

    Returns the resulting query string (or ``None`` if disabled).
    """
    if value is None:
        return None
    # bool is an int subclass in Python; reject to avoid silent misuse.
    if isinstance(value, bool):
        raise TypeError(
            "num_contacting_protein_chains must be int/str/null, not bool"
        )
    if isinstance(value, int):
        return f"{count_col} == {value}"
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        for op in _NUM_CONTACTING_OPS:
            if s.startswith(op):
                return f"{count_col} {s}"
        return s.replace(_NUM_CONTACTING_COL, count_col)
    raise TypeError(
        f"Unsupported type for num_contacting_protein_chains: {type(value).__name__}"
    )


def _build_num_contacting_query(value) -> str | None:
    return _build_count_query(value, count_col=_NUM_CONTACTING_COL)


def _add_num_contacting_protein_chains(
    metadata_df: pd.DataFrame,
    distance_cutoff: float = 5.0,
    contacts_col: str = "q_pn_unit_contacting_pn_unit_iids",
    protein_col: str = "q_pn_unit_is_protein",
    iid_col: str = "q_pn_unit_iid",
) -> pd.DataFrame:
    """Add ``num_contacting_protein_chains`` to ``metadata_df``.

    For each pn_unit row, counts distinct contacted pn_units that are
    protein AND whose ``min_distance`` is ``<= distance_cutoff`` Å. Used by
    BMSM filtering to keep small molecules with the configured number of
    contacting protein chains. Assumes the contact list is (pdb_id,
    assembly_id)-partitioned, matching the v8/v9 metadata parquet.
    """
    has_assembly_key = (
        "pdb_id" in metadata_df.columns and "assembly_id" in metadata_df.columns
    )
    if has_assembly_key:
        keys = list(
            zip(
                metadata_df["pdb_id"],
                metadata_df["assembly_id"].astype(str),
                metadata_df[iid_col],
            )
        )
        is_protein_lookup = dict(zip(keys, metadata_df[protein_col].astype(bool)))
    else:
        is_protein_lookup = dict(
            zip(metadata_df[iid_col], metadata_df[protein_col].astype(bool))
        )

    def _count(row) -> int:
        contacts = row[contacts_col]
        if contacts is None:
            return 0
        if isinstance(contacts, float) and pd.isna(contacts):
            return 0
        if isinstance(contacts, str):
            try:
                contacts = json.loads(contacts)
            except (json.JSONDecodeError, TypeError):
                return 0

        protein_iids = set()
        for c in contacts:
            if not isinstance(c, dict):
                continue
            cid = c.get("pn_unit_iid")
            md = c.get("min_distance")
            if cid is None or md is None or md > distance_cutoff:
                continue
            key = (row["pdb_id"], str(row["assembly_id"]), cid) if has_assembly_key else cid
            if is_protein_lookup.get(key, False):
                protein_iids.add(cid)
        return len(protein_iids)

    out = metadata_df.copy()
    out["num_contacting_protein_chains"] = out.apply(_count, axis=1).astype("int32")
    return out


def _add_n_contacting_protein_atoms_and_ratio(
    metadata_df: pd.DataFrame,
    contacts_col: str = "q_pn_unit_per_partner_contacts_to_protein_small_molecule",
    expected_atoms_col: str = "q_pn_unit_expected_heavy_atoms_non_polymer",
    n_atoms_out: str = "q_pn_unit_n_contacting_protein_atoms",
    ratio_out: str = "q_pn_unit_contacting_protein_atom_ratio",
) -> pd.DataFrame:
    """Add atom-level protein contact count and ligand-coverage ratio columns.

    Sums per-partner contact atom counts from the precomputed JSON column
    `q_pn_unit_per_partner_contacts_to_protein_small_molecule` (atomworks v9+),
    and divides by `q_pn_unit_expected_heavy_atoms_non_polymer` to get a
    pocket-coverage ratio (high ratio -> ligand atoms are well-buried by
    protein atoms). Used by the BMSM narrowing block to drop crystal-artifact
    ligands that barely contact protein.

    The contacts column may hold a JSON string (object/str dtype on disk),
    a list-of-dicts (already deserialized), or NaN/None (no contacts).

    No-op for missing input columns: returns a copy with no new columns,
    keeping the function safe for v8 parquets that lack the source column.
    """
    out = metadata_df.copy()
    if contacts_col not in out.columns:
        return out

    def _sum_count(v) -> int:
        if v is None:
            return 0
        if isinstance(v, float) and pd.isna(v):
            return 0
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return 0
        if not isinstance(v, list):
            return 0
        total = 0
        for c in v:
            if not isinstance(c, dict):
                continue
            try:
                total += int(c.get("count", 0))
            except (TypeError, ValueError):
                continue
        return total

    out[n_atoms_out] = out[contacts_col].apply(_sum_count).astype("int32")

    if expected_atoms_col in out.columns:
        denom = out[expected_atoms_col].astype(float)
        # Guard against denom <= 0 / NaN; resulting NaN ratios fail any >= threshold.
        denom_safe = denom.where(denom > 0, np.nan)
        out[ratio_out] = out[n_atoms_out].astype(float) / denom_safe
    else:
        out[ratio_out] = np.nan

    return out


# ---------------------------------------------------------------------------
# Helpers for iterative partner-filtered flag narrowing (BMSM_context / BMM / BMH).
#
# Each `q_pn_unit_per_partner_contacts_<modality>` column on the v9 parquet
# is a JSON list of `{pn_unit_iid, chain_iid, count}` dicts keyed by the
# *source* row's modality. The list mixes partners of any modality
# (protein / sm / metal / halide / other), so resolving a partner's modality
# requires looking the partner pn_unit_iid back up in metadata_df.
#
# `_build_partner_edges` flattens these JSONs into a long edge table.
# `_compute_partner_passes` decides per-edge whether the partner passes
# its current modality flag, given the current iteration's flag state.
# ---------------------------------------------------------------------------

_PARTNER_CONTACT_SPECS = (
    # (source_kind, contacts_col, source_flag_col)
    ("sm",     "q_pn_unit_per_partner_contacts_small_molecule", "q_pn_unit_is_small_molecule"),
    ("metal",  "q_pn_unit_per_partner_contacts_metal",          "q_pn_unit_is_metal"),
    ("halide", "q_pn_unit_per_partner_contacts_halide",         "q_pn_unit_is_halide"),
)


def _parse_partner_list(v) -> list:
    """Parse a per_partner_contacts cell into a list of dicts; safe for NaN/str/list."""
    if v is None:
        return []
    if isinstance(v, float) and pd.isna(v):
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(v, list):
        return []
    return v


def _build_partner_edges(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Explode per-partner contact JSON columns into a long edge table.

    For each row in `metadata_df` whose raw `q_pn_unit_is_<modality>` flag is
    True (modality in {sm, metal, halide}), parses the corresponding
    `q_pn_unit_per_partner_contacts_<modality>` JSON list and emits one row
    per (source, partner) pair with columns:
      * `source_idx`    — metadata_df row index of the source pn_unit
      * `source_kind`   — 'sm' / 'metal' / 'halide'
      * `partner_idx`   — metadata_df row index of partner, or NaN if unresolved
      * `partner_kind`  — 'sm' / 'metal' / 'halide' / 'protein' / 'other' / 'unknown'
      * `count`         — atom-level contact count from the JSON dict

    Used by `_iterative_narrow_bm_flags` to recompute effective contact counts
    while excluding partners whose own modality flag has been narrowed out.
    Modalities whose contacts column is missing contribute zero edges (caller
    treats this as raw `is_X` flag passing through unchanged).
    """
    edge_cols = ["source_idx", "source_kind", "partner_idx", "partner_kind", "count"]

    # (pdb_id, assembly_id, q_pn_unit_iid) -> row index lookup for partner resolution.
    keys = list(zip(
        metadata_df["pdb_id"],
        metadata_df["assembly_id"].astype(str),
        metadata_df["q_pn_unit_iid"],
    ))
    iid_to_row = dict(zip(keys, metadata_df.index))

    # Per-row partner_kind series. Last assignment wins where multiple flags
    # could be True; raw flags should be mutually exclusive in practice but
    # we order most-specific-last defensively.
    kind_series = pd.Series("other", index=metadata_df.index, dtype=object)
    if "q_pn_unit_is_protein" in metadata_df.columns:
        kind_series[metadata_df["q_pn_unit_is_protein"].fillna(False).astype(bool)] = "protein"
    if "q_pn_unit_is_halide" in metadata_df.columns:
        kind_series[metadata_df["q_pn_unit_is_halide"].fillna(False).astype(bool)] = "halide"
    if "q_pn_unit_is_metal" in metadata_df.columns:
        kind_series[metadata_df["q_pn_unit_is_metal"].fillna(False).astype(bool)] = "metal"
    if "q_pn_unit_is_small_molecule" in metadata_df.columns:
        kind_series[metadata_df["q_pn_unit_is_small_molecule"].fillna(False).astype(bool)] = "sm"

    edge_frames = []
    for source_kind, contacts_col, source_flag_col in _PARTNER_CONTACT_SPECS:
        if contacts_col not in metadata_df.columns or source_flag_col not in metadata_df.columns:
            continue
        src_mask = metadata_df[source_flag_col].fillna(False).astype(bool)
        if not src_mask.any():
            continue

        sub = metadata_df.loc[src_mask, ["pdb_id", "assembly_id", contacts_col]].copy()
        sub["_partners"] = sub[contacts_col].apply(_parse_partner_list)
        sub = sub.loc[sub["_partners"].apply(len) > 0]
        if sub.empty:
            continue

        sub_exp = sub.explode("_partners", ignore_index=False)
        # Drop non-dict entries (defensive against malformed JSON).
        sub_exp = sub_exp.loc[sub_exp["_partners"].apply(lambda c: isinstance(c, dict))]
        if sub_exp.empty:
            continue

        partner_iids = sub_exp["_partners"].apply(lambda c: c.get("pn_unit_iid"))
        counts = sub_exp["_partners"].apply(
            lambda c: int(c["count"]) if isinstance(c.get("count"), (int, float)) and not pd.isna(c["count"]) else 0
        )

        # Drop entries with missing partner pn_unit_iid (malformed JSON).
        valid = partner_iids.notna()
        sub_exp = sub_exp.loc[valid]
        partner_iids = partner_iids.loc[valid]
        counts = counts.loc[valid]
        if sub_exp.empty:
            continue

        # Resolve partner (pdb_id, assembly_id, partner_iid) -> row index.
        lookup_keys = list(zip(
            sub_exp["pdb_id"],
            sub_exp["assembly_id"].astype(str),
            partner_iids,
        ))
        partner_idx_arr = np.array([iid_to_row.get(k) for k in lookup_keys], dtype=object)

        edge_df = pd.DataFrame({
            "source_idx": sub_exp.index.values,
            "source_kind": source_kind,
            "partner_idx": partner_idx_arr,
            "partner_kind": "unknown",  # default; overwritten below for resolved partners
            "count": counts.values.astype(np.int32),
        })

        resolved = pd.notna(edge_df["partner_idx"])
        if resolved.any():
            edge_df.loc[resolved, "partner_kind"] = (
                kind_series.loc[edge_df.loc[resolved, "partner_idx"].values].values
            )

        edge_frames.append(edge_df)

    if not edge_frames:
        return pd.DataFrame(columns=edge_cols)
    return pd.concat(edge_frames, ignore_index=True)[edge_cols]


def _compute_partner_passes(
    edges: pd.DataFrame,
    sm_pass: pd.Series,
    mt_pass: pd.Series,
    hd_pass: pd.Series,
) -> np.ndarray:
    """Per-edge boolean: does the partner pass its current modality flag?

    * `sm`     -> sm_pass.loc[partner_idx]
    * `metal`  -> mt_pass.loc[partner_idx]
    * `halide` -> hd_pass.loc[partner_idx]
    * `protein` / `other` -> True (always counted)
    * `unknown` -> False (partner row missing from metadata_df, drop)
    """
    out = np.ones(len(edges), dtype=bool)
    kinds = edges["partner_kind"].values
    partner_idx = edges["partner_idx"].values

    out[kinds == "unknown"] = False

    for kind, pass_series in (("sm", sm_pass), ("metal", mt_pass), ("halide", hd_pass)):
        mask = kinds == kind
        if mask.any():
            out[mask] = pass_series.loc[partner_idx[mask]].values

    return out
