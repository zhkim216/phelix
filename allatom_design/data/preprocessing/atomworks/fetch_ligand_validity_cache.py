#!/usr/bin/env python3
"""
Fetch ligand validity scores from RCSB (GraphQL) and save as a cache parquet.

Why:
- The main metadata build step should avoid network dependencies (cluster hangs/timeouts).
- This cache can be computed/retried independently and later joined into metadata.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from pathlib import Path

import pandas as pd

import atomworks.ml.preprocessing.utils.structure_utils as dp


def _read_pdb_ids_from_metadata(parquet_path: Path) -> list[str]:
    df = pd.read_parquet(parquet_path, columns=["pdb_id"])
    pdb_ids = sorted({str(x).lower() for x in df["pdb_id"].tolist() if pd.notna(x)})
    return pdb_ids


def _fetch_one(pdb_id: str, timeout: tuple[float, float], num_retries: int) -> list[dict]:
    # dp expects the entry id; it is case-insensitive, but we normalize to lower everywhere.
    recs = dp.get_ligand_validity_scores_from_pdb_id(pdb_id.lower(), timeout=timeout, num_retries=num_retries)
    for r in recs:
        r["pdb_id"] = pdb_id.lower()
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-parquet", type=str, default=None, help="Read unique pdb_id from this metadata parquet")
    ap.add_argument("--pdb-id", type=str, action="append", default=None, help="Fetch a single pdb id (repeatable)")
    ap.add_argument("--out", type=str, required=True, help="Output parquet path")
    ap.add_argument("--num-workers", type=int, default=8, help="Thread workers for network I/O")
    ap.add_argument("--connect-timeout", type=float, default=5.0)
    ap.add_argument("--read-timeout", type=float, default=30.0)
    ap.add_argument("--num-retries", type=int, default=2)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_ids: list[str] = []
    if args.metadata_parquet:
        pdb_ids.extend(_read_pdb_ids_from_metadata(Path(args.metadata_parquet)))
    if args.pdb_id:
        pdb_ids.extend([p.lower() for p in args.pdb_id])
    pdb_ids = sorted(set(pdb_ids))
    if not pdb_ids:
        raise SystemExit("No pdb ids provided. Use --metadata-parquet or --pdb-id.")

    timeout = (float(args.connect_timeout), float(args.read_timeout))
    num_retries = int(args.num_retries)

    records: list[dict] = []
    if args.num_workers <= 1:
        for pdb_id in pdb_ids:
            records.extend(_fetch_one(pdb_id, timeout=timeout, num_retries=num_retries))
    else:
        with cf.ThreadPoolExecutor(max_workers=int(args.num_workers)) as ex:
            futs = {ex.submit(_fetch_one, pdb_id, timeout, num_retries): pdb_id for pdb_id in pdb_ids}
            for fut in cf.as_completed(futs):
                records.extend(fut.result())

    df = pd.DataFrame(records)
    df.to_parquet(out_path)
    print(f"wrote {len(df)} records for {len(pdb_ids)} pdb_ids -> {out_path}")


if __name__ == "__main__":
    main()

