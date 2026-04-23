#!/usr/bin/env python3
"""
Fetch RCSB ligand-validity scores per PDB and cache them locally.

For each pdb_id in --metadata-in:
  1. GET via RCSB GraphQL (atomworks.ml.preprocessing.utils.structure_utils)
  2. Save to <cache_dir>/{pdb[:2]}/{pdb}.json (atomic; resume-friendly)

Resume model: --skip-existing (default) checks per-PDB cache and skips already-
fetched entries, so re-runs only hit network for what's missing.

Sharding: pass --shard-id S --num-shards N to fetch only pdb_ids[S::N] —
disjoint across shards, safe in SLURM array. Pass --no-consolidate when running
in a SLURM array so workers don't race on a single cache_parquet write; then
run consolidate_cache.py once after the array completes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

import atomworks.ml.preprocessing.utils.structure_utils as dp


# -----------------------------
# Cache directory (per-PDB JSON)
# -----------------------------

def _cache_path(cache_dir: Path, pdb_id: str) -> Path:
    """Two-letter prefix bucketing avoids a single 200k-file directory."""
    pdb_id = pdb_id.lower()
    return cache_dir / pdb_id[:2] / f"{pdb_id}.json"


def _save_cached(cache_dir: Path, pdb_id: str, records: list[dict]) -> None:
    """Atomic per-PDB JSON write. Empty records are still written so
    `--skip-existing` knows we already tried this pdb."""
    p = _cache_path(cache_dir, pdb_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(records, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _load_cached(cache_dir: Path, pdb_id: str) -> list[dict] | None:
    p = _cache_path(cache_dir, pdb_id)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: cached file {p} unreadable ({e!r}); will re-fetch.")
        return None


def _iter_cached_pdb_ids(cache_dir: Path) -> Iterable[str]:
    """Yield pdb_ids that have a cache file in cache_dir, regardless of content."""
    if not cache_dir.exists():
        return
    for sub in sorted(cache_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.json")):
            yield f.stem


def _atomic_write_parquet(df: pd.DataFrame, final_path: Path) -> None:
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, final_path)


def _consolidate_cache_to_parquet(cache_dir: Path, out_parquet: Path) -> int:
    """Scan the per-PDB JSON cache and write a single consolidated parquet.

    Returns number of records written.
    """
    records: list[dict] = []
    for pdb_id in _iter_cached_pdb_ids(cache_dir):
        recs = _load_cached(cache_dir, pdb_id)
        if not recs:
            continue
        records.extend(recs)
    df = pd.DataFrame(records)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(df, out_parquet)
    return len(df)


# -----------------------------
# Fetch step (network-bound)
# -----------------------------

def _fetch_one(pdb_id: str, timeout: tuple[float, float], num_retries: int) -> list[dict]:
    """Hit RCSB GraphQL for one pdb_id; tag each returned record with pdb_id."""
    recs = dp.get_ligand_validity_scores_from_pdb_id(pdb_id.lower(), timeout=timeout, num_retries=num_retries)
    for r in recs:
        r["pdb_id"] = pdb_id.lower()
    return recs


def _fetch_one_safe(
    pdb_id: str,
    cache_dir: Path,
    timeout: tuple[float, float],
    num_retries: int,
) -> tuple[str, int, str | None]:
    """Fetch one PDB, save to cache, return (pdb_id, n_records, error_or_None).

    Errors are *not* raised; failures land in the returned error slot so one
    bad PDB can't abort the whole run.
    """
    try:
        recs = _fetch_one(pdb_id, timeout=timeout, num_retries=num_retries)
        _save_cached(cache_dir, pdb_id, recs)
        return (pdb_id, len(recs), None)
    except Exception as e:
        return (pdb_id, 0, repr(e))


def fetch_step(
    *,
    metadata_in: Path,
    cache_dir: Path,
    cache_parquet: Path,
    num_workers: int,
    connect_timeout: float,
    read_timeout: float,
    num_retries: int,
    skip_existing: bool,
    checkpoint_every: int,
    shard_id: int | None = None,
    num_shards: int | None = None,
    consolidate: bool = True,
) -> tuple[int, list[tuple[str, str]]]:
    """Populate per-PDB JSON cache, then (optionally) consolidate to a cache parquet.

    Sharding: when ``shard_id`` and ``num_shards`` are both set, the sorted
    global PDB list is sliced as ``all_pdb_ids[shard_id::num_shards]``. Disjoint
    across shards → safe to run in a SLURM array without write contention.

    ``consolidate=False`` skips the per-task cache_parquet rewrite; use this in
    array tasks so parallel workers don't race on a single parquet file.

    Returns (num_records_in_cache_parquet_or_0, errors).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_parquet(metadata_in, columns=["pdb_id"])
    all_pdb_ids = sorted({str(x).lower() for x in meta_df["pdb_id"].tolist() if pd.notna(x)})
    print(f"[fetch] {len(all_pdb_ids)} unique pdb_ids in {metadata_in.name}")

    if (shard_id is None) != (num_shards is None):
        raise SystemExit("--shard-id and --num-shards must be set together")
    if shard_id is not None and num_shards is not None:
        if not (0 <= shard_id < num_shards):
            raise SystemExit(f"--shard-id {shard_id} out of range for --num-shards {num_shards}")
        all_pdb_ids = all_pdb_ids[shard_id::num_shards]
        print(f"[fetch] shard {shard_id}/{num_shards}: {len(all_pdb_ids)} pdb_ids assigned to this task")

    if skip_existing:
        cached = {pid for pid in _iter_cached_pdb_ids(cache_dir)}
        todo = [p for p in all_pdb_ids if p not in cached]
        print(f"[fetch] {len(cached)} already cached, {len(todo)} to fetch")
    else:
        todo = all_pdb_ids
        print(f"[fetch] --skip-existing disabled; will fetch all {len(todo)} pdb_ids")

    errors: list[tuple[str, str]] = []
    if not todo:
        if consolidate:
            print("[fetch] nothing to do; consolidating existing cache...")
            n = _consolidate_cache_to_parquet(cache_dir, cache_parquet)
            print(f"[fetch] consolidated {n} records -> {cache_parquet}")
            return n, errors
        print("[fetch] nothing to do; --no-consolidate requested, skipping parquet write")
        return 0, errors

    timeout = (float(connect_timeout), float(read_timeout))

    def maybe_checkpoint(completed_now: int) -> None:
        if consolidate and checkpoint_every and completed_now % checkpoint_every == 0:
            n = _consolidate_cache_to_parquet(cache_dir, cache_parquet)
            tqdm.write(f"[fetch] checkpoint at {completed_now}/{len(todo)}: cache parquet has {n} records")

    completed = 0
    if num_workers <= 1:
        for pdb_id in tqdm(todo, desc="fetch (serial)"):
            _pid, _n, err = _fetch_one_safe(pdb_id, cache_dir, timeout, num_retries)
            if err:
                errors.append((_pid, err))
            completed += 1
            maybe_checkpoint(completed)
    else:
        with cf.ThreadPoolExecutor(max_workers=int(num_workers)) as ex:
            futures = {
                ex.submit(_fetch_one_safe, pdb_id, cache_dir, timeout, num_retries): pdb_id
                for pdb_id in todo
            }
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="fetch"):
                _pid, _n, err = fut.result()
                if err:
                    errors.append((_pid, err))
                completed += 1
                maybe_checkpoint(completed)

    if consolidate:
        n = _consolidate_cache_to_parquet(cache_dir, cache_parquet)
        print(f"[fetch] done: cache parquet has {n} records ({len(errors)} failed pdbs)")
        return n, errors
    print(f"[fetch] done: {completed} pdbs processed ({len(errors)} failed); --no-consolidate → skipping parquet write")
    return 0, errors


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-in", required=True, help="Input metadata parquet (source of pdb_ids)")
    ap.add_argument("--cache-dir", required=True, help="Directory for per-PDB JSON cache files")
    ap.add_argument("--cache-parquet", required=True, help="Consolidated cache parquet path")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--connect-timeout", type=float, default=5.0)
    ap.add_argument("--read-timeout", type=float, default=30.0)
    ap.add_argument("--num-retries", type=int, default=2)
    ap.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force re-fetch even if a per-PDB cache file exists (default: skip existing)",
    )
    ap.add_argument("--checkpoint-every", type=int, default=1000,
                    help="Consolidate cache-dir to cache-parquet every N fetched PDBs (0=disable)")
    ap.add_argument("--shard-id", type=int, default=None,
                    help="Fetch only pdb_ids[shard_id::num_shards] (SLURM array use). Must be set with --num-shards.")
    ap.add_argument("--num-shards", type=int, default=None,
                    help="Total number of shards in a SLURM array fetch. Must be set with --shard-id.")
    ap.add_argument("--no-consolidate", dest="consolidate", action="store_false",
                    help="Skip the per-task cache-parquet rewrite during fetch (use in array mode to avoid races). "
                         "Run consolidate_cache.py once after the array completes.")
    ap.set_defaults(skip_existing=True, consolidate=True)
    args = ap.parse_args()

    metadata_in = Path(args.metadata_in)
    cache_dir = Path(args.cache_dir)
    cache_parquet = Path(args.cache_parquet)

    _, errors = fetch_step(
        metadata_in=metadata_in,
        cache_dir=cache_dir,
        cache_parquet=cache_parquet,
        num_workers=args.num_workers,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        num_retries=args.num_retries,
        skip_existing=args.skip_existing,
        checkpoint_every=args.checkpoint_every,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        consolidate=args.consolidate,
    )

    if errors:
        print(f"\n[summary] {len(errors)} PDBs failed during fetch (they are omitted from the cache):")
        for pid, err in errors[:20]:
            print(f"  {pid}: {err}")
        if len(errors) > 20:
            print(f"  ... ({len(errors) - 20} more)")
        print(
            "Re-run the command to retry failed PDBs (successful ones will be skipped via --skip-existing)."
        )


if __name__ == "__main__":
    main()
