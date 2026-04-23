#!/usr/bin/env python3
"""
Pure helpers for augmenting metadata rows with RCSB ligand-validity scores.

Single source of truth for the `q_pn_unit_ligand_validity` shape (a stringified
dict-of-dicts), reused by `augment_shard.py`. The augmentation reproduces the
exact data shape DataPreprocessor would have produced if it had fetched scores
inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LIGAND_SCORES = [
    "RSCC",
    "RSR",
    "completeness",
    "intermolecular_clashes",
    "is_best_instance",
    "ranking_model_fit",
    "ranking_model_geometry",
]


def _build_scores_table(cache_df: pd.DataFrame, ligand_scores: list[str]) -> dict[str, pd.DataFrame]:
    """
    Build per-pdb_id score tables matching DataPreprocessor's internal format:
      index: (asym_id, res_name)
      columns: ligand_scores
    """
    if cache_df.empty:
        return {}
    if "pdb_id" not in cache_df.columns:
        raise ValueError("cache parquet must include a 'pdb_id' column (lowercase entry id).")

    by_pdb: dict[str, pd.DataFrame] = {}
    keep_cols = ["pdb_id", "asym_id", "res_name"] + [c for c in ligand_scores if c in cache_df.columns]
    sub = cache_df[keep_cols].copy()
    for c in ligand_scores:
        if c not in sub.columns:
            sub[c] = np.nan

    for pdb_id, g in sub.groupby("pdb_id", sort=False):
        g = g.dropna(subset=["asym_id", "res_name"])
        if g.empty:
            continue
        g = g.set_index(["asym_id", "res_name"])
        # If duplicates exist, DataPreprocessor effectively keeps the last one
        g = g[~g.index.duplicated(keep="last")]
        g = g.sort_index()
        by_pdb[str(pdb_id)] = g[ligand_scores]
    return by_pdb


def _infer_ligand_ids(chain_ids: list[str], res_names: list[str]) -> list[tuple[str, str]]:
    chain_ids = [c.strip() for c in chain_ids if c.strip()]
    res_names = [r.strip() for r in res_names if r.strip()]
    if not chain_ids or not res_names:
        return []
    if len(chain_ids) == 1 and len(res_names) >= 1:
        return [(chain_ids[0], r) for r in res_names]
    if len(res_names) == 1 and len(chain_ids) >= 1:
        return [(c, res_names[0]) for c in chain_ids]
    if len(chain_ids) == len(res_names):
        return list(zip(chain_ids, res_names, strict=False))
    # Fallback: cartesian product
    return [(c, r) for c in chain_ids for r in res_names]


def _augment_group(meta_g: pd.DataFrame, scores_df: pd.DataFrame | None, ligand_scores: list[str]) -> pd.DataFrame:
    meta_g = meta_g.copy()
    if scores_df is None or scores_df.empty:
        return meta_g

    def build_validity(row) -> str:
        # Only non-polymers have ligand validity in original pipeline
        if bool(row.get("q_pn_unit_is_polymer", True)):
            return str(row.get("q_pn_unit_ligand_validity", "{}"))

        chain_str = str(row.get("q_pn_unit_id", ""))
        res_str = str(row.get("q_pn_unit_non_polymer_res_names", ""))
        chain_ids = chain_str.split(",") if chain_str else []
        res_names = res_str.split(",") if res_str else []

        ligand_ids = _infer_ligand_ids(chain_ids, res_names)
        if not ligand_ids:
            return "{}"

        present = [lid for lid in ligand_ids if lid in scores_df.index]
        if not present:
            # Match DataPreprocessor: when scores exist but no ids match,
            # df.loc[[]].to_dict() yields {col: {} ...} rather than {}.
            return str({k: {} for k in ligand_scores})

        # Mimic DataPreprocessor: df.loc[ids].to_dict()
        selected = scores_df.loc[present]
        return str(selected.to_dict())

    meta_g["q_pn_unit_ligand_validity"] = meta_g.apply(build_validity, axis=1)
    return meta_g
