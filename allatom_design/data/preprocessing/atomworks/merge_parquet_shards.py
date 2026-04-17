#!/usr/bin/env python3
"""
Merge parquet shards into a single metadata file.

Input layout (produced by `build_metadata_parquet_shards.py`):
    <out_dir>/shards/metadata_shard_{id:05d}.parquet       # per-shard parquet (normal case)
    <out_dir>/shards/shard_{id:05d}_batches/batch_*.parquet # intermediate batches

Output:
    <out_dir>/metadata.parquet               # concatenated, deduplicated on example_id
    <out_dir>/metadata_for_caching.parquet   # one row per pdb_id

Behavior:
- If a shard's final parquet is missing but its batch directory has batch parquets,
  those are merged first (atomic write). If the final parquet already exists, the
  batch directory is left untouched.
- Before concatenating shards we validate that all shard parquets share the same
  column layout (dtype mismatches are warned; structural schema mismatches are
  treated as errors).
- All output files are written atomically (tmp + os.replace).
"""
import argparse
import glob
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _atomic_write_parquet(df: pd.DataFrame, final_path: Path) -> None:
    """Write a parquet atomically via tmp + os.replace."""
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, final_path)


def merge_batch_parquets_for_shard(batch_dir: Path, shard_dir: Path, shard_id: int) -> Path | None:
    """Merge batch parquets for a single shard if the final shard parquet is missing.

    If the final shard parquet already exists we return it unchanged — the
    recovery path is the build script itself (`build_metadata_parquet_shards.py`),
    which writes the final shard parquet atomically at the end of its run.
    This function is only a convenience fallback when we want to merge what is
    already on disk without re-running the build job.
    """
    shard_parquet = shard_dir / f"metadata_shard_{shard_id:05d}.parquet"
    if shard_parquet.exists():
        return shard_parquet

    batch_files = sorted(batch_dir.glob("batch_*.parquet"))
    if not batch_files:
        return None

    tqdm.write(f"  Merging {len(batch_files)} batch files for shard {shard_id}...")
    dfs = [pd.read_parquet(f) for f in batch_files]
    df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        return None

    rows_before = len(df)
    if "example_id" in df.columns:
        df = df.drop_duplicates(subset=["example_id"], keep="first")
    duplicates_removed = rows_before - len(df)
    if duplicates_removed > 0:
        tqdm.write(
            f"  Removed {duplicates_removed} duplicate rows "
            f"(by example_id) in shard {shard_id}"
        )

    _atomic_write_parquet(df, shard_parquet)
    tqdm.write(f"  Created {shard_parquet} with {len(df)} rows")
    return shard_parquet


def _validate_shard_schemas(loaded: list[tuple[Path, pd.DataFrame]]) -> None:
    """Raise SystemExit on column-set mismatch; warn on dtype mismatch.

    Column-set differences are structural and usually mean v7 and v8 shards
    got mixed in the same directory — refuse to concat rather than produce
    a parquet with sparsely-populated columns. Dtype diffs are softer (int
    vs. int64 promotion, nullable vs. non-nullable) — warn only.
    """
    if not loaded:
        return
    ref_path, ref = loaded[0]
    ref_cols = list(ref.columns)
    ref_dtypes = {c: str(ref[c].dtype) for c in ref_cols}
    for p, df in loaded[1:]:
        if list(df.columns) != ref_cols:
            only_ref = sorted(set(ref_cols) - set(df.columns))
            only_new = sorted(set(df.columns) - set(ref_cols))
            raise SystemExit(
                f"Schema mismatch between {ref_path.name} and {p.name}:\n"
                f"  columns only in {ref_path.name}: {only_ref}\n"
                f"  columns only in {p.name}:        {only_new}\n"
                f"Refusing to merge mixed-schema shards."
            )
        dtype_diffs = {
            c: (ref_dtypes[c], str(df[c].dtype))
            for c in ref_cols
            if str(df[c].dtype) != ref_dtypes[c]
        }
        if dtype_diffs:
            tqdm.write(f"WARNING: dtype diffs in {p.name}: {dtype_diffs}")


def main(out_dir: str, merge_batches: bool = True) -> None:
    out = Path(out_dir)
    shard_dir = out / "shards"

    if not shard_dir.exists():
        raise SystemExit(f"Shard directory not found: {shard_dir}")

    # Step 1: fallback-merge any batch directories whose final parquet is missing.
    if merge_batches:
        batch_dirs = sorted(shard_dir.glob("shard_*_batches"))
        if batch_dirs:
            print(f"Found {len(batch_dirs)} batch directories; merging any incomplete shards...")
            for batch_dir in batch_dirs:
                # shard_00001_batches -> 1
                try:
                    shard_id = int(batch_dir.name.split("_")[1])
                except (ValueError, IndexError):
                    tqdm.write(f"  WARNING: could not parse shard_id from {batch_dir.name}, skipping")
                    continue
                merge_batch_parquets_for_shard(batch_dir, shard_dir, shard_id)

    # Step 2: collect per-shard parquets.
    shard_files = sorted(glob.glob(str(shard_dir / "metadata_shard_*.parquet")))
    if not shard_files:
        raise SystemExit(f"No shard files found in {shard_dir}")

    print(f"Found {len(shard_files)} shard files; loading and validating schemas...")
    loaded: list[tuple[Path, pd.DataFrame]] = []
    failed: list[tuple[str, str]] = []
    for p in tqdm(shard_files, desc="Loading shards"):
        try:
            loaded.append((Path(p), pd.read_parquet(p)))
        except Exception as e:
            failed.append((p, repr(e)))

    if failed:
        tqdm.write(f"\nWARNING: {len(failed)} shard parquets failed to load:")
        for p, err in failed:
            tqdm.write(f"  {p}: {err}")

    if not loaded:
        raise SystemExit("No valid parquet files found")

    _validate_shard_schemas(loaded)

    per_shard_rows = [len(df) for _, df in loaded]
    total_rows_pre_dedup = sum(per_shard_rows)
    df = pd.concat([df for _, df in loaded], ignore_index=True)
    assert len(df) == total_rows_pre_dedup, (
        f"concat row mismatch: {len(df)} != {total_rows_pre_dedup}"
    )

    # Step 3: deduplicate by example_id (safety net for cross-shard duplicates).
    rows_before = len(df)
    if "example_id" in df.columns:
        df = df.drop_duplicates(subset=["example_id"], keep="first")
    duplicates_removed = rows_before - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows (by example_id) in final merge")

    # Step 4: write metadata.parquet and metadata_for_caching.parquet atomically.
    meta_path = out / "metadata.parquet"
    _atomic_write_parquet(df, meta_path)
    print(f"Wrote {len(df)} rows to {meta_path}")

    df_cache = df.groupby("pdb_id", as_index=False).first()
    cache_path = out / "metadata_for_caching.parquet"
    _atomic_write_parquet(df_cache, cache_path)
    print(f"Wrote {len(df_cache)} rows to {cache_path}")

    # Summary.
    print("\nSummary:")
    print(f"  Shard parquets merged:   {len(loaded)}")
    if failed:
        print(f"  Shard parquets failed:   {len(failed)}")
    print(f"  Rows before dedup:       {rows_before}")
    print(f"  Duplicates removed:      {duplicates_removed}")
    print(f"  Final rows:              {len(df)}")
    print(f"  Unique PDB IDs:          {len(df_cache)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge parquet shards into a single metadata file")
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory containing shards/ subdirectory (e.g. /scratch/.../atomworks_pdb_full_v8)",
    )
    ap.add_argument(
        "--no-merge-batches",
        action="store_true",
        help="Skip merging batch parquets for shards missing their final parquet",
    )
    args = ap.parse_args()
    main(args.out_dir, merge_batches=not args.no_merge_batches)
