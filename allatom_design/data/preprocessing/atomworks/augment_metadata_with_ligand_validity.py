#!/usr/bin/env python3
"""
Augment metadata parquet with ligand validity scores (previously fetched via RCSB GraphQL).

This reproduces the exact data-shape used by DataPreprocessor:
`q_pn_unit_ligand_validity` is a dict-of-dicts serialized to string:
  {score_name: {(asym_id, res_name): value, ...}, ...}
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    # keep only relevant columns (if missing, fill with NaN)
    keep_cols = ["pdb_id", "asym_id", "res_name"] + [c for c in ligand_scores if c in cache_df.columns]
    sub = cache_df[keep_cols].copy()
    for c in ligand_scores:
        if c not in sub.columns:
            sub[c] = np.nan

    for pdb_id, g in sub.groupby("pdb_id", sort=False):
        # Drop rows without required identifiers
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
        # no scores for this pdb_id -> leave as-is
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
            # Match DataPreprocessor behavior:
            # when ligand_validity_scores exists but no ids match,
            # df.loc[[]].to_dict() yields {col: {} ...} rather than {}.
            return str({k: {} for k in ligand_scores})

        # Mimic DataPreprocessor: df.loc[ids].to_dict()
        selected = scores_df.loc[present]
        d = selected.to_dict()
        return str(d)

    meta_g["q_pn_unit_ligand_validity"] = meta_g.apply(build_validity, axis=1)
    return meta_g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-in", required=True, help="Input metadata parquet (built without ligand validity)")
    ap.add_argument("--validity-cache", required=True, help="Ligand validity cache parquet from fetch step")
    ap.add_argument("--metadata-out", required=True, help="Output metadata parquet (augmented)")
    ap.add_argument("--ligand-score", action="append", default=None, help="Score field to include (repeatable)")
    args = ap.parse_args()

    ligand_scores = args.ligand_score or DEFAULT_LIGAND_SCORES

    meta_in = Path(args.metadata_in)
    cache_in = Path(args.validity_cache)
    out_path = Path(args.metadata_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_parquet(meta_in)
    cache_df = pd.read_parquet(cache_in)

    by_pdb = _build_scores_table(cache_df, ligand_scores)

    out_groups = []
    for pdb_id, g in meta_df.groupby("pdb_id", sort=False):
        scores_df = by_pdb.get(str(pdb_id), None)
        out_groups.append(_augment_group(g, scores_df, ligand_scores))

    out_df = pd.concat(out_groups, ignore_index=True) if out_groups else meta_df
    out_df.to_parquet(out_path)
    print(f"wrote augmented metadata -> {out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    main()

