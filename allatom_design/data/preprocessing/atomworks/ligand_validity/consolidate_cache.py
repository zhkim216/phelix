#!/usr/bin/env python3
"""
Consolidate per-PDB JSON cache into a single ligand_validity_cache.parquet.

Wraps `_consolidate_cache_to_parquet` from .fetch so the augment array tasks
all read one parquet instead of each rescanning the JSON. Run once after the
sharded fetch array completes (or skip if fetch.py wrote the parquet directly,
i.e. without --no-consolidate).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from allatom_design.data.preprocessing.atomworks.ligand_validity.fetch import (
    _consolidate_cache_to_parquet,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True, type=Path,
                    help="Per-PDB JSON cache directory (e.g. ${DATASET_DIR}/ligand_validity_cache_json)")
    ap.add_argument("--cache-parquet", required=True, type=Path,
                    help="Output consolidated parquet path")
    args = ap.parse_args()

    if not args.cache_dir.exists():
        raise SystemExit(f"cache_dir not found: {args.cache_dir}")

    n = _consolidate_cache_to_parquet(args.cache_dir, args.cache_parquet)
    print(f"[consolidate] wrote {n} records -> {args.cache_parquet}")


if __name__ == "__main__":
    main()
