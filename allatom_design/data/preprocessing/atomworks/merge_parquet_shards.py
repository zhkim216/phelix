#!/usr/bin/env python3
"""
Merge parquet shards into a single metadata file.

This script handles two cases:
1. Final shard parquets: shards/metadata_shard_*.parquet
2. Batch parquets (for incomplete shards): shards/shard_*_batches/batch_*.parquet

If a shard's final parquet doesn't exist but batch parquets do, 
they will be merged first.
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def build_file_size_map(cached_examples_dir: Path) -> dict[str, int]:
    """Scan directory once and build pdb_id -> file_size mapping."""
    size_map = {}
    cached_dir = Path(cached_examples_dir)
    
    if not cached_dir.exists():
        return size_map
    
    print("Scanning cached_examples directory...")
    for entry in tqdm(os.scandir(cached_dir), desc="Building file size map"):
        if entry.name.endswith(".pt") and entry.is_file():
            pdb_id = entry.name[:-3]  # Remove .pt extension
            size_map[pdb_id] = entry.stat().st_size
    
    return size_map


def merge_batch_parquets_for_shard(batch_dir: Path, shard_dir: Path, shard_id: int) -> Path | None:
    """Merge batch parquets for a single shard if final parquet doesn't exist."""
    shard_parquet = shard_dir / f"metadata_shard_{shard_id:05d}.parquet"
    
    # If final shard parquet already exists, use it
    if shard_parquet.exists():
        return shard_parquet
    
    # Otherwise, try to merge batch parquets
    batch_files = sorted(batch_dir.glob("batch_*.parquet"))
    if not batch_files:
        return None
    
    print(f"  Merging {len(batch_files)} batch files for shard {shard_id}...")
    dfs = [pd.read_parquet(f) for f in batch_files]
    df = pd.concat(dfs, ignore_index=True)
    
    if df.empty:
        return None
    
    # Deduplicate by example_id within shard (handles resume with different batch_size)
    rows_before = len(df)
    if 'example_id' in df.columns:
        df = df.drop_duplicates(subset=['example_id'], keep='first')
    duplicates_removed = rows_before - len(df)
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate rows (by example_id) in shard {shard_id}")
    
    df.to_parquet(shard_parquet)
    print(f"  Created {shard_parquet} with {len(df)} rows")
    return shard_parquet


def main(out_dir: str, merge_batches: bool = True):        
    
    out = Path(out_dir)
    shard_dir = out / "shards"
    
    if not shard_dir.exists():
        raise SystemExit(f"Shard directory not found: {shard_dir}")
    
    # First, merge any incomplete batch parquets into shard parquets
    if merge_batches:
        batch_dirs = sorted(shard_dir.glob("shard_*_batches"))
        if batch_dirs:
            print(f"Found {len(batch_dirs)} batch directories, checking for incomplete shards...")
            for batch_dir in batch_dirs:
                # Extract shard_id from directory name (shard_00001_batches -> 1)
                try:
                    shard_id = int(batch_dir.name.split("_")[1])
                    merge_batch_parquets_for_shard(batch_dir, shard_dir, shard_id)
                except (ValueError, IndexError):
                    print(f"  Warning: Could not parse shard_id from {batch_dir.name}")
    
    # Now collect all shard parquets
    shard_files = sorted(glob.glob(str(shard_dir / "metadata_shard_*.parquet")))
    if not shard_files:
        raise SystemExit(f"No shard files found in {shard_dir}")
    
    print(f"Found {len(shard_files)} shard files, concatenating...")
    dfs = []
    for p in tqdm(shard_files, desc="Loading shards"):
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            print(f"  Warning: Failed to read {p}: {e}")
    
    if not dfs:
        raise SystemExit("No valid parquet files found")
    
    df = pd.concat(dfs, ignore_index=True)

    # Deduplicate by example_id across all shards (safety net for cross-shard duplicates)
    rows_before_final = len(df)
    if 'example_id' in df.columns:
        df = df.drop_duplicates(subset=['example_id'], keep='first')
    duplicates_removed_final = rows_before_final - len(df)
    if duplicates_removed_final > 0:
        print(f"Removed {duplicates_removed_final} duplicate rows (by example_id) in final merge")
    
    # Add file_size column
    # file_size_map = build_file_size_map(cached_examples_dir)
    # df["cached_file_size"] = df["pdb_id"].map(file_size_map)

    # Write combined metadata
    meta_path = out / "metadata.parquet"
    df.to_parquet(meta_path)
    print(f"Wrote {len(df)} rows to {meta_path}")

    # Unique-by-pdb_id parquet for caching examples in downstream steps
    df_cache = df.groupby("pdb_id", as_index=False).first()
    cache_path = out / "metadata_for_caching.parquet"
    df_cache.to_parquet(cache_path)
    print(f"Wrote {len(df_cache)} rows to {cache_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total shards merged: {len(shard_files)}")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique PDB IDs: {len(df_cache)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge parquet shards into a single metadata file")
    ap.add_argument("--out_dir", default="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8",
                    help="Output directory containing shards/ subdirectory")
    ap.add_argument("--no-merge-batches", action="store_true",
                    help="Skip merging batch parquets for incomplete shards")
    args = ap.parse_args()
    main(args.out_dir, merge_batches=not args.no_merge_batches)
