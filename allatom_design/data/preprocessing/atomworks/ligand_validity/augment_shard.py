#!/usr/bin/env python3
"""
Per-shard worker: run the slow augment loop on this shard's pdb_ids only.

Splits `sorted(by_pdb.keys())[shard_id::num_shards]` (matches the stride
pattern used by the fetch sbatch) and writes only the augmented rows'
`(row_idx, q_pn_unit_ligand_validity)` so the merge step is cheap.

Reuses `_build_scores_table` and `_augment_group` from .augment_helpers — the
output column values are byte-equivalent to the legacy single-task augment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from allatom_design.data.preprocessing.atomworks.ligand_validity.augment_helpers import (
    DEFAULT_LIGAND_SCORES,
    _augment_group,
    _build_scores_table,
)


# Columns the augment loop reads. Loading only these keeps memory bounded
# (~150 MB instead of ~750 MB on the full v9 metadata).
AUGMENT_INPUT_COLUMNS = [
    "pdb_id",
    "q_pn_unit_is_polymer",
    "q_pn_unit_id",
    "q_pn_unit_non_polymer_res_names",
    "q_pn_unit_ligand_validity",  # polymer rows pass through this value verbatim
]


def _atomic_write_parquet(df: pd.DataFrame, final_path: Path) -> None:
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, final_path)


def run_shard(
    *,
    metadata_in: Path,
    cache_parquet: Path,
    shard_out_dir: Path,
    shard_id: int,
    num_shards: int,
    ligand_scores: list[str],
) -> int:
    if not (0 <= shard_id < num_shards):
        raise SystemExit(f"--shard-id {shard_id} out of range for --num-shards {num_shards}")
    if not cache_parquet.exists():
        raise SystemExit(
            f"cache_parquet not found: {cache_parquet}\n"
            f"Run consolidate_cache.py first to build it from the JSON cache."
        )

    print(f"[shard {shard_id}/{num_shards}] loading metadata columns: {AUGMENT_INPUT_COLUMNS}")
    meta_df = pd.read_parquet(metadata_in, columns=AUGMENT_INPUT_COLUMNS)
    print(f"[shard {shard_id}/{num_shards}] meta_df rows={len(meta_df)}")

    cache_df = pd.read_parquet(cache_parquet)
    print(f"[shard {shard_id}/{num_shards}] cache rows={len(cache_df)}")
    by_pdb = _build_scores_table(cache_df, ligand_scores)
    del cache_df
    print(f"[shard {shard_id}/{num_shards}] by_pdb has scores for {len(by_pdb)} pdb_ids")

    # Stride sharding: matches existing fetch convention (pdb_ids[shard_id::num_shards])
    all_scored_pdb_ids = sorted(by_pdb.keys())
    shard_pdb_ids = set(all_scored_pdb_ids[shard_id::num_shards])
    print(f"[shard {shard_id}/{num_shards}] this shard owns {len(shard_pdb_ids)} pdb_ids")

    scored_rows = meta_df[meta_df["pdb_id"].astype(str).isin(shard_pdb_ids)]
    print(f"[shard {shard_id}/{num_shards}] scored_rows={len(scored_rows)} (groupby + augment)")

    out_indices: list = []
    out_values: list = []
    for pdb_id, g in tqdm(
        scored_rows.groupby("pdb_id", sort=False),
        desc=f"shard {shard_id}",
        total=scored_rows["pdb_id"].nunique(),
    ):
        scores_df = by_pdb[str(pdb_id)]
        augmented_g = _augment_group(g, scores_df, ligand_scores)
        if "q_pn_unit_ligand_validity" in augmented_g.columns:
            out_indices.append(augmented_g.index.to_numpy())
            out_values.append(augmented_g["q_pn_unit_ligand_validity"].to_numpy())

    if out_indices:
        idx = pd.Index(pd.concat([pd.Series(a) for a in out_indices], ignore_index=True))
        vals = pd.concat([pd.Series(v) for v in out_values], ignore_index=True)
    else:
        idx = pd.Index([], dtype="int64")
        vals = pd.Series([], dtype="object")

    out_df = pd.DataFrame({"row_idx": idx, "q_pn_unit_ligand_validity": vals})

    shard_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_out_dir / f"augment_shard_{shard_id:05d}.parquet"
    _atomic_write_parquet(out_df, out_path)
    print(f"[shard {shard_id}/{num_shards}] wrote {len(out_df)} rows -> {out_path}")
    return len(out_df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-in", required=True, type=Path)
    ap.add_argument("--cache-parquet", required=True, type=Path)
    ap.add_argument("--shard-out-dir", required=True, type=Path)
    ap.add_argument("--shard-id", required=True, type=int)
    ap.add_argument("--num-shards", required=True, type=int)
    ap.add_argument(
        "--ligand-score",
        action="append",
        default=None,
        help="Score field to include (repeatable); default = DEFAULT_LIGAND_SCORES",
    )
    args = ap.parse_args()

    ligand_scores = args.ligand_score or list(DEFAULT_LIGAND_SCORES)
    run_shard(
        metadata_in=args.metadata_in,
        cache_parquet=args.cache_parquet,
        shard_out_dir=args.shard_out_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        ligand_scores=ligand_scores,
    )


if __name__ == "__main__":
    main()
