import json
import logging
from pathlib import Path
from typing import Any, List, Union, override

import atomworks.enums as aw_enums
import atomworks.ml.preprocessing.constants as aw_const
import lightning as L
import numpy as np
import pandas as pd
import torch
from atomworks.ml.common import generate_example_id
from atomworks.ml.datasets.datasets import BaseDataset
from atomworks.ml.datasets.parsers import GenericDFParser
from atomworks.ml.transforms.base import Compose, Transform
from atomworks.ml.utils.io import read_parquet_with_metadata
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader, WeightedRandomSampler
from allatom_design.data.transform.pad import pad_to_max
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.common import parse_example_id
from allatom_design.data.transform.sd_featurizer import sd_featurizer

logger = logging.getLogger(__name__)

class AtomworksSDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path
        self._train_set = SDDataset(pdb_path=cfg.pdb_path,
                                    parquet_path=cfg.parquet_path,
                                    sampling_weights=cfg.sampling_weights,
                                    featurizer_cfg=cfg.featurizer_cfg)


    def train_dataloader(self) -> DataLoader:
        sampler = WeightedRandomSampler(self._train_set.get_sampling_weights(),
                                        num_samples=self.cfg.samples_per_epoch,
                                        replacement=True)

        train_loader = DataLoader(self._train_set,
                                  sampler=sampler,
                                  batch_size=self.cfg.batch_size,
                                  num_workers=self.cfg.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=sd_collator)

        return train_loader


class SDDataset(BaseDataset):
    def __init__(self,
                 pdb_path: str,
                 parquet_path: str,
                 sampling_weights: dict[str, dict[str, float]],
                 featurizer_cfg: Transform | Compose | None,
                 ):
        super().__init__()

        self.pdb_path = pdb_path
        self.parquet_path = parquet_path
        self.sampling_weights = sampling_weights
        self.featurizer = sd_featurizer(**featurizer_cfg)

        # Read in chain metadata parquet
        self.chain_df = read_parquet_with_metadata(parquet_path)
        self.chain_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        self.chain_df["q_pn_unit_contacting_pn_unit_iids"] = self.chain_df["q_pn_unit_contacting_pn_unit_iids"].apply(json.loads)
        self.chain_df["sampling_weight"] = get_sampling_weights(self.chain_df,
                                                                 alphas=sampling_weights["alphas"],
                                                                 beta=sampling_weights["betas"]["beta_chain"],
                                                                 chain_type_cols=["q_pn_unit_type"],
                                                                 cluster_cols=["q_pn_unit_cluster_id"],
                                                                 seq_length_cols=["q_pn_unit_sequence_length"])

        # Apply filters

        # Build interface df from contacts in chain df
        self.interface_df = build_interface_df(self.chain_df, dataset_name=Path(parquet_path).parent.name)
        self.interface_df["sampling_weight"] = get_sampling_weights(self.interface_df,
                                                                 alphas=sampling_weights["alphas"],
                                                                 beta=sampling_weights["betas"]["beta_interface"],
                                                                 chain_type_cols=["q_pn_unit_type_1", "q_pn_unit_type_2"],
                                                                 cluster_cols=["q_pn_unit_cluster_id_1", "q_pn_unit_cluster_id_2"],
                                                                 seq_length_cols=["q_pn_unit_sequence_length_1", "q_pn_unit_sequence_length_2"])

        # Parse dfs into a common format
        self.chain_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid"])
        self.interface_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid_1", "q_pn_unit_iid_2"])

        self.parsed_df = pd.concat([
            self.chain_df.apply(self.chain_parser.parse, axis=1),
            self.interface_df.apply(self.interface_parser.parse, axis=1)
        ], axis=0)


    def get_sampling_weights(self) -> pd.Series:
        return self.parsed_df.apply(lambda x: x["extra_info"]["sampling_weight"])

    @override
    def __getitem__(self, idx: int):
        example_id = self.idx_to_id(idx)
        parsed_row = self.parsed_df.loc[example_id]
        example = self._load_cached_example(parsed_row["extra_info"]["pdb_id"])
        example.update(parsed_row)  # add in query_pn_unit_iids

        # apply train-time transforms
        feats = self.featurizer(example)

        return feats

    @override
    def __len__(self) -> int:
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
        cached_example_path = f"{self.pdb_path}/cached_examples/{pdb_id}.pt"
        if not Path(cached_example_path).exists():
            raise FileNotFoundError(f"Cached example for {pdb_id} not found in {self.pdb_path}/cached_examples")
        return torch.load(cached_example_path, weights_only=False)


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

        if key not in ["example_id", "atom_array", "crop_info"]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


def build_interface_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    # bring example_id into a column if it's the index
    df = df.reset_index(drop=True)

    # columns we need from the source df
    chain_specific_cols = ["q_pn_unit_iid", "q_pn_unit_type", "q_pn_unit_sequence_length", "q_pn_unit_cluster_id"]
    base_cols = [
        "example_id", "pdb_id", "assembly_id", "path", "q_pn_unit_contacting_pn_unit_iids",
        *chain_specific_cols,
    ]
    interface_df = df[base_cols].copy()

    # explode interface contacts
    interface_df = interface_df.explode("q_pn_unit_contacting_pn_unit_iids", ignore_index=True)
    interface_df = interface_df.dropna(subset=["q_pn_unit_contacting_pn_unit_iids"])  # drop pn_units without interface contacts

    # extract the contacted iid
    interface_df["q_pn_unit_iid_2"] = interface_df["q_pn_unit_contacting_pn_unit_iids"].map(
        lambda d: d.get("pn_unit_iid") if isinstance(d, dict) else None
    )
    interface_df = interface_df.dropna(subset=["q_pn_unit_iid_2"])

    # join back to get chain info for chain_2
    right = df[["pdb_id", "assembly_id"] + chain_specific_cols].rename(
                    columns={f"{c}": f"{c}_2" for c in chain_specific_cols})
    interface_df = interface_df.merge(
        right, on=["pdb_id", "assembly_id", "q_pn_unit_iid_2"], how="left", validate="many_to_one"
    )

    # canonicalize pair ordering to dedupe (A_1, B_1) == (B_1, A_1)
    interface_df = _canonicalize_pair_columns(interface_df, order_by="q_pn_unit_iid", paired_cols=chain_specific_cols)

    # drop exact duplicate interfaces within (pdb_id, assembly_id)
    interface_df = interface_df.drop_duplicates(subset=["pdb_id", "assembly_id", "q_pn_unit_iid_1", "q_pn_unit_iid_2"], keep="first")

    # build example_id for interfaces by appending 'interfaces' to the source dataset_names
    def _get_interface_example_id(row):
        dataset_names = [dataset_name, "interfaces"]
        query_pn_unit_iids = [row["q_pn_unit_iid_1"], row["q_pn_unit_iid_2"]]
        return generate_example_id(dataset_names, row["pdb_id"], row["assembly_id"], query_pn_unit_iids)

    interface_df["example_id"] = interface_df.apply(_get_interface_example_id, axis=1)

    # final selection / order of columns
    interface_df = interface_df[
        ["example_id", "pdb_id", "assembly_id", "path"] + [f"{c}_1" for c in chain_specific_cols] + [f"{c}_2" for c in chain_specific_cols]
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


def get_sampling_weights(df: pd.DataFrame,
                         alphas: dict[str, float],
                         beta: float,
                         chain_type_cols: list[str],
                         cluster_cols: list[str],
                         seq_length_cols: list[str]) -> pd.DataFrame:
    """
    Based on the cluster ID in cluster_col and chain types in chain_type_col, get the sampling weights for each example.
    """
    df = df.copy()

    # Get cluster size
    df["clusters"] = df[cluster_cols].apply(lambda x: tuple(sorted(tuple(x))), axis=1)  # sort cluster ids to dedupe
    cluster_id_to_size = df["clusters"].value_counts()
    df["cluster_size"] = df["clusters"].map(cluster_id_to_size)

    # Compute chain type counts
    chain_count_cols = ["n_prot", "n_nuc", "n_ligand", "n_peptide", "is_loi"]
    df["chain_types"] = df[chain_type_cols].apply(lambda x: tuple(x), axis=1)
    df["seq_lengths"] = df[seq_length_cols].apply(lambda x: tuple(x), axis=1)

    def _get_chain_type_counts(row) -> dict[str, int]:
        chain_types: tuple[str] = row["chain_types"]
        seq_lengths: tuple[int] = row["seq_lengths"]
        chain_type_counts = {c: 0 for c in chain_count_cols}

        for t, l in zip(chain_types, seq_lengths):
            if t in aw_enums.ChainTypeInfo.PROTEINS:
                if l < aw_const.PEPTIDE_MAX_RESIDUES:
                    chain_type_counts["n_peptide"] += 1
                else:
                    chain_type_counts["n_prot"] += 1
            elif t in aw_enums.ChainTypeInfo.NUCLEIC_ACIDS:
                chain_type_counts["n_nuc"] += 1
            else:
                chain_type_counts["n_ligand"] += 1

        return pd.Series(chain_type_counts)

    df[chain_count_cols] = df.apply(_get_chain_type_counts, axis=1)

    # Compute weights
    missing_alphas = set(alphas.keys()) - {"a_prot", "a_peptide", "a_nuc", "a_ligand", "a_loi"}
    missing_counts = {"n_prot", "n_peptide", "n_nuc", "n_ligand"} - set(df.columns)

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
        + alphas.get("a_ligand", 0) * df["n_ligand"]
        + alphas.get("a_loi", 0) * df["is_loi"]  # always 0 for now
    )

    return weights
