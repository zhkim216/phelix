#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import pandas as pd


def main(out_dir: str, save_imd: bool | None = None):
    out = Path(out_dir)
    # Detect IMD mode by argument or by existing directory
    if save_imd is None:
        # Prefer explicit detection by directory existence
        shards_with_imd = out / "shards_with_imd"
        shards_plain = out / "shards"
        save_imd = shards_with_imd.exists() and not shards_plain.exists() or (
            shards_with_imd.exists() and not any(shards_plain.iterdir())
        )
    shard_dir = out / ("shards_with_imd" if save_imd else "shards")
    suffix = "_with_imd" if save_imd else ""
    shard_files = sorted(glob.glob(str(shard_dir / f"metadata_shard_*{suffix}.parquet")))
    if not shard_files:
        raise SystemExit(f"No shard files found in {shard_dir}")
    print(f"Found {len(shard_files)} shard files, concatenating...")
    dfs = [pd.read_parquet(p) for p in shard_files]
    df = pd.concat(dfs, ignore_index=True)

    # Write combined metadata
    meta_path = out / ("metadata_with_imd.parquet" if save_imd else "metadata.parquet")
    df.to_parquet(meta_path)
    print(f"Wrote {len(df)} rows to {meta_path}")

    # Unique-by-pdb_id parquet for caching examples in downstream steps
    df_cache = df.groupby("pdb_id", as_index=False).first()
    cache_path = out / ("metadata_for_caching_with_imd.parquet" if save_imd else "metadata_for_caching.parquet")
    df_cache.to_parquet(cache_path)
    print(f"Wrote {len(df_cache)} rows to {cache_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/possu/jinho/datasets/atomworks_lmpnn_valset_filtered")
    ap.add_argument("--save_imd", default=False)
    args = ap.parse_args()
    main(args.out_dir, args.save_imd)
