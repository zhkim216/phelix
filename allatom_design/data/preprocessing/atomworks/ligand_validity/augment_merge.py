#!/usr/bin/env python3
"""
Merge per-shard augment outputs into the final metadata_ligval.parquet.

Each shard parquet has columns `[row_idx, q_pn_unit_ligand_validity]` where
`row_idx` is the integer row index in the original metadata. We load the full
metadata once, apply each shard's column update via .loc, and atomically write.

Matches the legacy single-task augment behavior:
- rows with scored pdb_ids are updated (polymers + non-polymers in those groups)
- rows whose pdb_id has no scores keep whatever value the input metadata had
- if `q_pn_unit_ligand_validity` was missing from the input, it is added as NA
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _atomic_write_parquet(df: pd.DataFrame, final_path: Path) -> None:
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, final_path)


def _collect_shard_files(shard_dir: Path, num_shards: int) -> list[Path]:
    expected = [shard_dir / f"augment_shard_{i:05d}.parquet" for i in range(num_shards)]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} shard parquet(s) in {shard_dir}:\n"
            + "\n".join(f"  {p.name}" for p in missing[:10])
            + (f"\n  ... ({len(missing) - 10} more)" if len(missing) > 10 else "")
        )
    return expected


def run_merge(
    *,
    metadata_in: Path,
    shard_dir: Path,
    metadata_out: Path,
    num_shards: int,
) -> int:
    shard_files = _collect_shard_files(shard_dir, num_shards)
    print(f"[merge] found all {num_shards} shard parquets")

    meta_df = pd.read_parquet(metadata_in)
    print(f"[merge] loaded metadata: rows={len(meta_df)}, cols={len(meta_df.columns)}")

    if "q_pn_unit_ligand_validity" not in meta_df.columns:
        meta_df["q_pn_unit_ligand_validity"] = pd.NA
        print("[merge] added q_pn_unit_ligand_validity column (was missing)")

    total_updated = 0
    for p in tqdm(shard_files, desc="merge"):
        sdf = pd.read_parquet(p)
        if sdf.empty:
            continue
        meta_df.loc[sdf["row_idx"].to_numpy(), "q_pn_unit_ligand_validity"] = (
            sdf["q_pn_unit_ligand_validity"].to_numpy()
        )
        total_updated += len(sdf)
    print(f"[merge] applied updates to {total_updated} rows from {num_shards} shards")

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(meta_df, metadata_out)

    n_na = int(meta_df["q_pn_unit_ligand_validity"].isna().sum())
    print(
        f"[merge] wrote {metadata_out} (rows={len(meta_df)}, "
        f"q_pn_unit_ligand_validity NA={n_na} / {len(meta_df)})"
    )
    return len(meta_df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-in", required=True, type=Path)
    ap.add_argument("--shard-dir", required=True, type=Path)
    ap.add_argument("--metadata-out", required=True, type=Path)
    ap.add_argument("--num-shards", required=True, type=int)
    args = ap.parse_args()

    run_merge(
        metadata_in=args.metadata_in,
        shard_dir=args.shard_dir,
        metadata_out=args.metadata_out,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
