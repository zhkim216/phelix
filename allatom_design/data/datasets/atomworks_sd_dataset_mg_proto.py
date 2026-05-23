import ast
import json
import logging
import random
from pathlib import Path
from typing import Literal

import atomworks.enums as aw_enums
import lightning as L
import numpy as np
import pandas as pd
import torch
from atomworks.ml.datasets import MolecularDataset
from atomworks.ml.datasets.parsers import GenericDFParser
from atomworks.ml.example_id import generate_example_id
from atomworks.ml.utils.io import read_parquet_with_metadata
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing_extensions import override

from allatom_design.data.sampler import Sampler
from allatom_design.data.transform import sd_featurizer
from allatom_design.data.transform.pad import pad_to_max
from allatom_design.utils.metadata_utils import split_components

logger = logging.getLogger(__name__)

MG_PROTO_EVIDENCE_COLUMNS = {
    "substring": "q_pn_unit_has_pubmed_evidence_substring",
    "gpt": "q_pn_unit_has_pubmed_evidence_gpt",
}


class AtomworksSDMGProtoDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path
        self._train_set = MGProtoSDDataset(cfg, phase="train")
        self._val_set = MGProtoSDDataset(cfg, phase="val")

    def train_dataloader(self) -> DataLoader:
        num_workers = self.cfg.get("num_workers", 0)
        train_loader = DataLoader(
            dataset=self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=sd_collator,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        num_workers = self.cfg.get("num_workers", 0)
        val_loader = DataLoader(
            dataset=self._val_set,
            batch_size=self.cfg.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=sd_collator,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )
        return val_loader


class MGProtoSDDataset(MolecularDataset):
    def __init__(self, cfg: DictConfig, phase: Literal["train", "val"]):
        super().__init__(name=f"sd_mg_proto::{phase}", transform=None)
        self.cfg = cfg
        self.phase = phase
        self.save_failed_examples_to_dir = cfg.save_failed_examples_to_dir
        self.mg_cfg = cfg.get("mg_proto", {}) or {}
        self.val_cluster_ids: list = []

        self.featurizer = sd_featurizer.sd_featurizer(
            **cfg.featurizer_cfg,
            remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS,
        )

        if self.phase == "train":
            self.metadata_path = self.cfg.train_metadata_path
            self.metadata_df = self._process_train_metadata_df(self.metadata_path)
            dataset_name = self.mg_cfg.get("dataset_name", Path(self.metadata_path).parent.name)
            self.protein_monomer_chain_df = self._process_protein_monomer_chain_df(dataset_name)
            self.interface_df = self._process_interface_df(dataset_name)
            self.protein_monomer_chain_df, self.interface_df = add_mg_proto_sampling_weights(
                monomer_df=self.protein_monomer_chain_df,
                interface_df=self.interface_df,
                alphas_interface=self.cfg.sampling_weights["alphas_interface"],
                cluster_col="q_pn_unit_cluster_id",
                k_percentile=self.cfg.sampling_weights["k_percentile"],
            )
            self._validate_sampling_weights()
            self.parsed_df = self._parse_train_dfs()
            self._sampler = Sampler(self.get_sampling_weights())
            self._rng, self._samples = None, None
        else:
            self.metadata_path = self.cfg.val_metadata_path
            self.metadata_df = self._process_val_metadata_df(self.metadata_path)
            self.parsed_df = self._parse_val_df()

    @override
    def __getitem__(self, idx: int):
        if self.phase == "train":
            self._ensure_worker_rng()
            idx = next(self._samples)

        example_id = self.idx_to_id(idx)
        parsed_row = self.parsed_df.loc[example_id]

        try:
            example = self._load_cached_example(parsed_row["extra_info"]["pdb_id"])
        except FileNotFoundError:
            logger.warning(
                "Cached example for %s not found in %s/cached_examples in %s dataset, skipping...",
                parsed_row["extra_info"]["pdb_id"],
                self.cfg.pdb_path,
                self.phase,
            )
            if len(self.parsed_df) == 0:
                raise
            return self.__getitem__((idx + 1) % len(self.parsed_df))

        example.update(parsed_row)
        example["phase"] = self.phase

        try:
            return self.featurizer(example)
        except Exception as exc:
            logger.error(
                "Error applying train-time transforms to example %s in %s dataset: %s",
                example_id,
                self.phase,
                exc,
            )
            if len(self.parsed_df) == 0:
                raise
            return self.__getitem__((idx + 1) % len(self.parsed_df))

    def _ensure_worker_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._samples = self._sampler.sample(self._rng)

    def _process_train_metadata_df(self, metadata_path: str) -> pd.DataFrame:
        metadata_df = read_parquet_with_metadata(metadata_path)
        metadata_df = _ensure_example_id_column(metadata_df)
        metadata_df = _add_proto_modality_columns(metadata_df)
        metadata_df = attach_mg_external_evidence_flag(metadata_df, self.mg_cfg)

        if self.mg_cfg.get("require_cached_examples", True):
            metadata_df = _filter_to_cached_examples(metadata_df, self.cfg.pdb_path)

        metadata_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        metadata_df = self._add_phase_split(metadata_df)
        metadata_df = metadata_df[metadata_df["phase"] == self.phase]
        metadata_df = self._apply_filters(self.cfg.train_filters.metadata_filter, metadata_df)
        return metadata_df

    def _add_phase_split(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        metadata_df = metadata_df.copy()
        with open(self.cfg.validation_ids_file, "r") as f:
            val_split = {x.lower().split(".")[0] for x in f.read().splitlines()}
        logger.info("Loading validation IDs from %s", self.cfg.validation_ids_file)

        if self.cfg.debug:
            pdb_ids = metadata_df["pdb_id"].unique().tolist()
            n_debug = min(int(self.cfg.debug_num_ids), len(pdb_ids))
            debug_pdb_list = np.random.choice(pdb_ids, size=n_debug, replace=False)
            split_idx = 3 * n_debug // 4
            metadata_df.loc[metadata_df["pdb_id"].isin(debug_pdb_list[:split_idx]), "phase"] = "train"
            metadata_df.loc[metadata_df["pdb_id"].isin(debug_pdb_list[split_idx:]), "phase"] = "val"
        else:
            metadata_df.loc[~metadata_df["pdb_id"].str.lower().isin(val_split), "phase"] = "train"
            metadata_df.loc[metadata_df["pdb_id"].str.lower().isin(val_split), "phase"] = "val"

        if self.cfg.exclude_val_cluster:
            val_proteins = metadata_df[
                metadata_df["q_pn_unit_is_protein"].fillna(False).astype(bool)
                & (metadata_df["phase"] == "val")
            ]
            self.val_cluster_ids = list(set(val_proteins["q_pn_unit_cluster_id"]))
        return metadata_df

    def _process_protein_monomer_chain_df(self, dataset_name: str) -> pd.DataFrame:
        monomer_df = self._apply_filters(
            self.cfg.train_filters.protein_monomer_chain_filter,
            self.metadata_df.copy(),
        )
        if self.cfg.exclude_val_cluster and self.val_cluster_ids:
            before = len(monomer_df)
            monomer_df = monomer_df[~monomer_df["q_pn_unit_cluster_id"].isin(self.val_cluster_ids)]
            logger.info("Excluded %d MG proto monomer rows by val cluster.", before - len(monomer_df))

        monomer_df = add_mg_proto_chain_counts_info(monomer_df)
        monomer_df["example_id"] = monomer_df.apply(
            lambda row: generate_example_id(
                [dataset_name, "protein_monomer_chain"],
                row["pdb_id"],
                row["assembly_id"],
                [row["q_pn_unit_iid"]],
            ),
            axis=1,
        )
        monomer_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        return monomer_df

    def _process_interface_df(self, dataset_name: str) -> pd.DataFrame:
        protein_df = self._apply_filters(
            self.cfg.train_filters.protein_monomer_chain_filter,
            self.metadata_df.copy(),
        )
        interface_df = build_mg_proto_interface_df(
            metadata_df=self.metadata_df.copy(),
            protein_df=protein_df,
            dataset_name=dataset_name,
            mg_cfg=self.mg_cfg,
        )
        if self.cfg.exclude_val_cluster and self.val_cluster_ids and not interface_df.empty:
            val_clusters = set(self.val_cluster_ids)
            before = len(interface_df)
            interface_df = interface_df[
                ~interface_df["protein_cluster_multiset"].apply(
                    lambda clusters: any(c in val_clusters for c in clusters)
                )
            ]
            logger.info("Excluded %d MG proto interface rows by val cluster.", before - len(interface_df))

        interface_df = add_mg_proto_chain_counts_info(interface_df)
        interface_filters = self.cfg.train_filters.get("interface_filter", {}) or {}
        interface_df = self._apply_filters(interface_filters.get("2", []), interface_df)
        return interface_df

    def _process_val_metadata_df(self, metadata_path: str) -> pd.DataFrame:
        metadata_df = pd.read_parquet(metadata_path)
        metadata_df = _ensure_example_id_column(metadata_df)
        if "query_pn_unit_iids" in metadata_df.columns:
            metadata_df["query_pn_unit_iids"] = metadata_df["query_pn_unit_iids"].apply(_parse_pn_unit_iids_value)
        elif "q_pn_unit_iid" in metadata_df.columns:
            metadata_df["query_pn_unit_iids"] = metadata_df["q_pn_unit_iid"].apply(lambda value: [str(value)])
        else:
            raise KeyError(
                "MG proto val metadata must contain either `query_pn_unit_iids` "
                "or row-level `q_pn_unit_iid`."
            )
        metadata_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        logger.info("Final MG proto val dataset contains %d pdbs", metadata_df["pdb_id"].nunique())
        return metadata_df

    def _parse_train_dfs(self):
        chain_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid"])
        interface_parser = GenericDFParser(pn_unit_iid_colnames=[])

        def parse_interface_row(row):
            parsed = interface_parser.parse(row)
            parsed["query_pn_unit_iids"] = list(row["query_pn_unit_iids"])
            parsed["ligand_pn_unit_iids"] = list(row["ligand_pn_unit_iids"])
            parsed["protein_pn_unit_iids"] = list(row["protein_pn_unit_iids"])
            parsed["crop_center_pn_unit_iids"] = list(row["crop_center_pn_unit_iids"])
            parsed["biologically_meaningful_pn_unit_iids"] = list(row["biologically_meaningful_pn_unit_iids"])
            for key in (
                "query_pn_unit_iids",
                "ligand_pn_unit_iids",
                "protein_pn_unit_iids",
                "crop_center_pn_unit_iids",
                "biologically_meaningful_pn_unit_iids",
            ):
                parsed["extra_info"].pop(key, None)
            return parsed

        parsed_df = pd.concat(
            [
                self.protein_monomer_chain_df.apply(chain_parser.parse, axis=1),
                self.interface_df.apply(parse_interface_row, axis=1),
            ],
            axis=0,
        )
        logger.info(
            "Final MG proto train dataset has %d monomer rows and %d interface rows.",
            len(self.protein_monomer_chain_df),
            len(self.interface_df),
        )
        return parsed_df

    def _parse_val_df(self):
        parser = GenericDFParser(pn_unit_iid_colnames=[])

        def parse_val_row(row):
            parsed = parser.parse(row)
            parsed["query_pn_unit_iids"] = list(row["query_pn_unit_iids"])
            parsed["extra_info"].pop("query_pn_unit_iids", None)
            return parsed

        return self.metadata_df.apply(parse_val_row, axis=1)

    def _validate_sampling_weights(self):
        weights = np.concatenate(
            [
                self.protein_monomer_chain_df["sampling_weight"].to_numpy(dtype=float),
                self.interface_df["sampling_weight"].to_numpy(dtype=float),
            ]
        )
        if len(weights) == 0:
            raise ValueError("MG proto train dataset has no rows after filtering.")
        if not np.isfinite(weights).all():
            raise ValueError("MG proto sampling weights contain non-finite values.")
        if (weights < 0).any():
            raise ValueError("MG proto sampling weights contain negative values.")
        if weights.sum() <= 0:
            raise ValueError("MG proto sampling weights have zero total mass.")

        if len(self.interface_df) > 0 and self.interface_df["sampling_weight"].sum() <= 0:
            raise ValueError("MG proto interface rows exist but have zero total sampling mass.")
        if len(self.protein_monomer_chain_df) > 0 and self.protein_monomer_chain_df["sampling_weight"].sum() <= 0:
            raise ValueError("MG proto monomer rows exist but have zero total sampling mass.")

    def get_sampling_weights(self) -> np.ndarray:
        return self.parsed_df.apply(lambda row: row["extra_info"]["sampling_weight"]).to_numpy()

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
        if filters is None:
            return df
        for query in filters:
            df = self._apply_query(query, df)
        return df

    def _apply_query(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.query(query)
        self._validate_filter_impact(query, before, len(df))
        return df

    def _validate_filter_impact(self, query: str, original_num_rows: int, filtered_num_rows: int) -> None:
        if original_num_rows == 0:
            logger.warning("Query %r was applied to an empty MG proto dataset.", query)
            return
        if filtered_num_rows == original_num_rows:
            logger.warning("Query %r on MG proto dataset did not remove any rows.", query)
        elif filtered_num_rows == 0:
            raise ValueError(f"Query {query!r} on MG proto dataset removed all rows.")
        else:
            logger.info(
                "Query %r filtered MG proto dataset from %d to %d rows.",
                query,
                original_num_rows,
                filtered_num_rows,
            )


def attach_mg_external_evidence_flag(metadata_df: pd.DataFrame, mg_cfg: dict | DictConfig) -> pd.DataFrame:
    mg_cfg = mg_cfg or {}
    policy = mg_cfg.get("external_evidence_policy", "no_filter")
    allowed_codes = mg_cfg.get("allowed_ccd_codes", ["MG"])

    out = metadata_df.copy()
    exact_mg = _series_has_any_exact_ccd(
        out.get("q_pn_unit_non_polymer_res_names"),
        allowed_codes,
        index=out.index,
    )

    if policy == "no_filter":
        evidence = pd.Series(True, index=out.index)
    elif policy in MG_PROTO_EVIDENCE_COLUMNS:
        source_col = MG_PROTO_EVIDENCE_COLUMNS[policy]
        if source_col not in out.columns:
            raise KeyError(
                f"MG proto external evidence policy {policy!r} requires missing column {source_col!r}."
            )
        evidence = out[source_col].fillna(False).astype(bool)
    else:
        raise ValueError(
            f"Unknown MG proto external_evidence_policy={policy!r}. "
            "Supported values: 'no_filter', 'substring', 'gpt'."
        )

    out["q_pn_unit_has_external_evidence"] = exact_mg & evidence
    return out


def build_mg_proto_interface_df(
    metadata_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    dataset_name: str,
    mg_cfg: dict | DictConfig,
) -> pd.DataFrame:
    mg_cfg = mg_cfg or {}
    required_cols = [
        "pdb_id",
        "assembly_id",
        "path",
        "q_pn_unit_iid",
        "q_pn_unit_is_metal",
        "q_pn_unit_non_polymer_res_names",
        "q_pn_unit_avg_occupancy_nonpolymer",
        "q_pn_unit_per_partner_contacts_metal",
        "q_pn_unit_has_external_evidence",
        "q_pn_unit_cluster_id",
    ]
    protein_required_cols = ["pdb_id", "assembly_id", "q_pn_unit_iid", "q_pn_unit_cluster_id", "q_pn_unit_is_protein"]
    missing = [c for c in required_cols if c not in metadata_df.columns]
    missing += [c for c in protein_required_cols if c not in protein_df.columns]
    if missing:
        raise KeyError(f"build_mg_proto_interface_df missing required columns: {sorted(set(missing))}")

    allowed_codes = mg_cfg.get("allowed_ccd_codes", ["MG"])
    min_donors = int(mg_cfg.get("min_protein_donor_atoms", 3))
    min_occupancy = float(mg_cfg.get("min_avg_occupancy_nonpolymer", 0.5))

    protein_df = protein_df[protein_df["q_pn_unit_is_protein"].fillna(False).astype(bool)].copy()
    protein_lookup = {
        (row.pdb_id, str(row.assembly_id), row.q_pn_unit_iid): row
        for row in protein_df.itertuples(index=False)
    }

    center_mask = (
        metadata_df["q_pn_unit_is_metal"].fillna(False).astype(bool)
        & _series_has_any_exact_ccd(
            metadata_df["q_pn_unit_non_polymer_res_names"],
            allowed_codes,
            index=metadata_df.index,
        )
        & metadata_df["q_pn_unit_has_external_evidence"].fillna(False).astype(bool)
        & (metadata_df["q_pn_unit_avg_occupancy_nonpolymer"].fillna(-np.inf) >= min_occupancy)
    )
    center_df = metadata_df[center_mask].copy()

    rows = []
    for center in center_df.itertuples(index=False):
        donor_count, protein_rows = _collect_mg_protein_donor_partners(center, protein_lookup)
        if donor_count < min_donors or len(protein_rows) != 1:
            continue

        protein_rows = sorted(protein_rows, key=lambda row: row.q_pn_unit_iid)
        protein_iids = tuple(row.q_pn_unit_iid for row in protein_rows)
        protein_clusters = tuple(row.q_pn_unit_cluster_id for row in protein_rows)
        query_iids = [center.q_pn_unit_iid, *protein_iids]

        row = center._asdict()
        row.update(
            {
                "query_pn_unit_iids": query_iids,
                "ligand_pn_unit_iids": (center.q_pn_unit_iid,),
                "protein_pn_unit_iids": protein_iids,
                "crop_center_pn_unit_iids": [center.q_pn_unit_iid],
                "biologically_meaningful_pn_unit_iids": query_iids,
                "protein_cluster_multiset": protein_clusters,
                "ligand_ccd_key": ("ccd", "MG"),
                "interface_type": "bmm_protein",
                "n_coordinating_protein_donor_atoms": donor_count,
            }
        )
        row["example_id"] = generate_example_id(
            [dataset_name, "interface"],
            row["pdb_id"],
            row["assembly_id"],
            row["query_pn_unit_iids"],
        )
        rows.append(row)

    output_cols = [
        "example_id",
        "pdb_id",
        "assembly_id",
        "path",
        "query_pn_unit_iids",
        "ligand_pn_unit_iids",
        "protein_pn_unit_iids",
        "crop_center_pn_unit_iids",
        "biologically_meaningful_pn_unit_iids",
        "protein_cluster_multiset",
        "ligand_ccd_key",
        "interface_type",
        "n_coordinating_protein_donor_atoms",
        *metadata_df.columns.tolist(),
    ]
    output_cols = list(dict.fromkeys(output_cols))
    interface_df = pd.DataFrame(rows)
    if interface_df.empty:
        interface_df = pd.DataFrame(columns=output_cols)
    else:
        interface_df = interface_df[output_cols]
    interface_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
    logger.info(
        "Built MG proto interface_df with %d rows from %d exact-MG centers.",
        len(interface_df),
        len(center_df),
    )
    return interface_df


def add_mg_proto_sampling_weights(
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    alphas_interface: dict[str, float],
    cluster_col: str = "q_pn_unit_cluster_id",
    k_percentile: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monomer_df = monomer_df.copy()
    interface_df = interface_df.copy()

    if "protein_cluster_multiset" not in interface_df.columns:
        raise ValueError("MG proto interface_df must contain `protein_cluster_multiset`.")

    alpha_metal = float(alphas_interface.get("alpha_protein_metal", 0.0))

    def _protein_clusters(row):
        clusters = row.get("protein_cluster_multiset", ())
        if isinstance(clusters, float) and pd.isna(clusters):
            return []
        return list(clusters)

    def _sort_key(value):
        return repr(value)

    if interface_df.empty:
        interface_df["pair_cluster"] = []
        interface_df["pair_cluster_size"] = []
        interface_df["alpha"] = []
        interface_df["sampling_weight"] = []
        interface_contrib = {}
        k_value = 1.0
    else:
        interface_df["pair_cluster"] = interface_df.apply(
            lambda row: (
                ("ccd", "MG"),
                tuple(sorted((("seq", c) for c in _protein_clusters(row)), key=_sort_key)),
            ),
            axis=1,
        )
        pair_cluster_sizes = interface_df["pair_cluster"].value_counts()
        interface_df["pair_cluster_size"] = interface_df["pair_cluster"].map(pair_cluster_sizes)
        interface_df["alpha"] = alpha_metal
        interface_df["sampling_weight"] = interface_df["alpha"] / interface_df["pair_cluster_size"]
        interface_contrib = _compute_interface_contrib(interface_df, _protein_clusters)

        if interface_contrib and max(interface_contrib.values()) > 0:
            k_value = float(np.percentile(list(interface_contrib.values()), k_percentile))
        else:
            k_value = 1.0

        scaling = {
            cluster_id: k_value / contrib
            for cluster_id, contrib in interface_contrib.items()
            if contrib > k_value and contrib > 0
        }
        if scaling:
            interface_df["sampling_weight"] = interface_df.apply(
                lambda row: row["sampling_weight"]
                * min([scaling[c] for c in _protein_clusters(row) if c in scaling] or [1.0]),
                axis=1,
            )
            interface_contrib = _compute_interface_contrib(interface_df, _protein_clusters)

    monomer_counts = monomer_df[cluster_col].value_counts().to_dict()

    def _monomer_weight(row):
        cluster_id = row[cluster_col]
        target = k_value - interface_contrib.get(cluster_id, 0.0)
        return max(target, 0.0) / monomer_counts.get(cluster_id, 1)

    monomer_df["sampling_weight"] = monomer_df.apply(_monomer_weight, axis=1)
    logger.info(
        "MG proto sampling weights: monomer_rows=%d, interface_rows=%d, K=%.4f, alpha_metal=%.4f",
        len(monomer_df),
        len(interface_df),
        k_value,
        alpha_metal,
    )
    return monomer_df, interface_df


def add_mg_proto_chain_counts_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "protein_cluster_multiset" in df.columns:
        df["n_prot"] = df["protein_cluster_multiset"].apply(lambda clusters: len(clusters))
        df["n_nuc"] = 0
        df["n_peptide"] = 0
        df["n_small_molecule"] = 0
        df["n_metal"] = 1
        df["n_loi"] = 0
        return df

    df["n_prot"] = df["q_pn_unit_is_protein"].fillna(False).astype(bool).astype(int)
    df["n_nuc"] = df.get("q_pn_unit_is_nuc", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["n_peptide"] = df.get("q_pn_unit_is_peptide", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["n_small_molecule"] = df.get("q_pn_unit_is_small_molecule", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["n_metal"] = df["q_pn_unit_is_metal"].fillna(False).astype(bool).astype(int)
    df["n_loi"] = 0
    return df


def sd_collator(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = data[0].keys()
    collated = {}
    for key in keys:
        values = [d[key] for d in data]
        if key not in ["example_id", *sd_featurizer.INFERENCE_ONLY_KEYS]:
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)
        collated[key] = values
    return collated


def worker_init_fn(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _collect_mg_protein_donor_partners(center, protein_lookup: dict) -> tuple[int, list]:
    contacts = _parse_partner_list(getattr(center, "q_pn_unit_per_partner_contacts_metal"))
    total = 0
    protein_rows_by_iid = {}
    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        try:
            count = int(contact.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        resolved = _resolve_partner_rows(
            center.pdb_id,
            center.assembly_id,
            contact.get("pn_unit_iid"),
            contact.get("chain_iid"),
            protein_lookup,
        )
        if not resolved:
            continue
        total += count
        for protein in resolved:
            protein_rows_by_iid[protein.q_pn_unit_iid] = protein
    return total, list(protein_rows_by_iid.values())


def _resolve_partner_rows(pdb_id: str, assembly_id, raw_iid, chain_iid, lookup: dict) -> list:
    candidates = []
    for value in (raw_iid, chain_iid):
        if value is not None:
            candidates.append(str(value))

    seen_candidates = set()
    for candidate in candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        key = (pdb_id, str(assembly_id), candidate)
        if key in lookup:
            return [lookup[key]]

    rows = []
    seen_rows = set()
    for candidate in candidates:
        for iid in split_components(candidate):
            key = (pdb_id, str(assembly_id), iid)
            if iid in seen_rows or key not in lookup:
                continue
            seen_rows.add(iid)
            rows.append(lookup[key])
    return rows


def _compute_interface_contrib(interface_df: pd.DataFrame, protein_clusters_fn) -> dict:
    contrib = {}
    for _, row in interface_df.iterrows():
        weight = float(row["sampling_weight"])
        for cluster_id in protein_clusters_fn(row):
            contrib[cluster_id] = contrib.get(cluster_id, 0.0) + weight
    return contrib


def _ensure_example_id_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "example_id" not in out.columns and out.index.name == "example_id":
        out = out.reset_index()
    if "example_id" not in out.columns:
        raise KeyError("metadata dataframe is missing required column `example_id`.")
    return out


def _filter_to_cached_examples(metadata_df: pd.DataFrame, pdb_path: str) -> pd.DataFrame:
    cached_dir = Path(pdb_path) / "cached_examples"
    if not cached_dir.exists():
        raise FileNotFoundError(f"MG proto require_cached_examples=True but {cached_dir} does not exist.")
    cached_pdb_ids = {path.stem for path in cached_dir.glob("*.pt")}
    if not cached_pdb_ids:
        raise FileNotFoundError(f"MG proto found no cached examples under {cached_dir}.")
    before = len(metadata_df)
    out = metadata_df[metadata_df["pdb_id"].isin(cached_pdb_ids)].copy()
    if out.empty:
        raise ValueError(f"MG proto cached-example filter removed all {before} metadata rows.")
    logger.info(
        "Filtered MG proto metadata to cached examples: %d -> %d rows across %d cached PDBs.",
        before,
        len(out),
        out["pdb_id"].nunique(),
    )
    return out


def _add_proto_modality_columns(metadata_df: pd.DataFrame) -> pd.DataFrame:
    out = metadata_df.copy()
    nuc_chain_types = [chain_type.value for chain_type in aw_enums.ChainType.get_nucleic_acids()]
    if "q_pn_unit_is_nuc" not in out.columns:
        out["q_pn_unit_is_nuc"] = (
            out.get("q_pn_unit_is_polymer", pd.Series(False, index=out.index)).fillna(False).astype(bool)
            & out.get("q_pn_unit_type", pd.Series(np.nan, index=out.index)).isin(nuc_chain_types)
        )
    if "q_pn_unit_is_peptide" not in out.columns:
        out["q_pn_unit_is_peptide"] = False
    if "q_pn_unit_is_halide" not in out.columns:
        out["q_pn_unit_is_halide"] = False
    if "q_pn_unit_is_small_molecule" not in out.columns:
        out["q_pn_unit_is_small_molecule"] = (
            ~out.get("q_pn_unit_is_polymer", pd.Series(False, index=out.index)).fillna(False).astype(bool)
            & ~out.get("q_pn_unit_is_metal", pd.Series(False, index=out.index)).fillna(False).astype(bool)
            & ~out.get("q_pn_unit_is_halide", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        )
    return out


def _parse_partner_list(value) -> list:
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
