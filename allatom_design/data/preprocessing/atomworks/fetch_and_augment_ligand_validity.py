#!/usr/bin/env python3
"""
Fetch RCSB ligand validity scores AND augment a metadata parquet, in a single run.

Network failures are isolated per-PDB: each successful fetch is written
immediately to `<cache_dir>/{pdb[:2]}/{pdb}.json`, so re-runs only hit the
network for the still-missing PDBs. On a clean cache-dir this is equivalent
to running `fetch_ligand_validity_cache.py` followed by
`augment_metadata_with_ligand_validity.py`, but with:

- Per-PDB JSON cache (resume-friendly; a bad PDB doesn't poison the parquet).
- `--skip-existing` (default on): do not re-fetch PDBs already cached.
- Periodic consolidation of the JSON cache into a single parquet
  (`--cache-parquet`, `--checkpoint-every`) for fault tolerance.
- `--only-fetch` / `--only-augment` so each step can be run/debugged alone.

Relationship to the legacy scripts:
- `fetch_ligand_validity_cache.py` and `augment_metadata_with_ligand_validity.py`
  are unchanged and still usable standalone. This script imports their
  internal helpers (`_fetch_one`, `_build_scores_table`, `_augment_group`,
  `DEFAULT_LIGAND_SCORES`) so there is exactly one source of truth for
  fetch semantics and the augmentation shape.
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

from allatom_design.data.preprocessing.atomworks.fetch_ligand_validity_cache import (
    _fetch_one,
)
from allatom_design.data.preprocessing.atomworks.augment_metadata_with_ligand_validity import (
    DEFAULT_LIGAND_SCORES,
    _augment_group,
    _build_scores_table,
)


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
) -> tuple[int, list[tuple[str, str]]]:
    """Populate per-PDB JSON cache, then consolidate to a cache parquet.

    Returns (num_records_in_cache_parquet, errors).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_parquet(metadata_in, columns=["pdb_id"])
    all_pdb_ids = sorted({str(x).lower() for x in meta_df["pdb_id"].tolist() if pd.notna(x)})
    print(f"[fetch] {len(all_pdb_ids)} unique pdb_ids in {metadata_in.name}")

    if skip_existing:
        cached = {pid for pid in _iter_cached_pdb_ids(cache_dir)}
        todo = [p for p in all_pdb_ids if p not in cached]
        print(f"[fetch] {len(cached)} already cached, {len(todo)} to fetch")
    else:
        todo = all_pdb_ids
        print(f"[fetch] --skip-existing disabled; will fetch all {len(todo)} pdb_ids")

    errors: list[tuple[str, str]] = []
    if not todo:
        print("[fetch] nothing to do; consolidating existing cache...")
        n = _consolidate_cache_to_parquet(cache_dir, cache_parquet)
        print(f"[fetch] consolidated {n} records -> {cache_parquet}")
        return n, errors

    timeout = (float(connect_timeout), float(read_timeout))

    completed = 0
    if num_workers <= 1:
        for pdb_id in tqdm(todo, desc="fetch (serial)"):
            _pid, _n, err = _fetch_one_safe(pdb_id, cache_dir, timeout, num_retries)
            if err:
                errors.append((_pid, err))
            completed += 1
            if checkpoint_every and completed % checkpoint_every == 0:
                n = _consolidate_cache_to_parquet(cache_dir, cache_parquet)
                tqdm.write(f"[fetch] checkpoint at {completed}/{len(todo)}: cache parquet has {n} records")
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
                if checkpoint_every and completed % checkpoint_every == 0:
                    n = _consolidate_cache_to_parquet(cache_dir, cache_parquet)
                    tqdm.write(f"[fetch] checkpoint at {completed}/{len(todo)}: cache parquet has {n} records")

    n = _consolidate_cache_to_parquet(cache_dir, cache_parquet)
    print(f"[fetch] done: cache parquet has {n} records ({len(errors)} failed pdbs)")
    return n, errors


# -----------------------------
# Augment step (local)
# -----------------------------

def augment_step(
    *,
    metadata_in: Path,
    cache_parquet: Path,
    metadata_out: Path,
    ligand_scores: list[str],
) -> int:
    """Join cache parquet into metadata and write augmented metadata atomically.

    Uses in-place column assignment instead of accumulating augmented
    groups in a list followed by pd.concat. The legacy concat pattern
    allocated roughly 2x meta_df at peak and OOM-killed on multi-million-row
    inputs; the in-place variant keeps memory bounded at ~meta_df size.
    """
    meta_df = pd.read_parquet(metadata_in)
    cache_df = pd.read_parquet(cache_parquet)
    print(
        f"[augment] input metadata rows={len(meta_df)}, cache rows={len(cache_df)}"
    )
    by_pdb = _build_scores_table(cache_df, ligand_scores)
    del cache_df  # no longer needed — score tables are in by_pdb

    # Ensure the target column exists (legacy concat path left it NaN for
    # pdb_ids without scores; replicate that here).
    if "q_pn_unit_ligand_validity" not in meta_df.columns:
        meta_df["q_pn_unit_ligand_validity"] = pd.NA

    # Only touch rows whose pdb_id has ligand-validity scores.
    scored_pdb_ids = set(by_pdb.keys())
    scored_mask = meta_df["pdb_id"].astype(str).isin(scored_pdb_ids)
    scored_rows = meta_df.loc[scored_mask]

    for pdb_id, g in tqdm(
        scored_rows.groupby("pdb_id", sort=False),
        desc="augment",
        total=scored_rows["pdb_id"].nunique(),
    ):
        scores_df = by_pdb[str(pdb_id)]
        augmented_g = _augment_group(g, scores_df, ligand_scores)
        if "q_pn_unit_ligand_validity" in augmented_g.columns:
            meta_df.loc[augmented_g.index, "q_pn_unit_ligand_validity"] = augmented_g["q_pn_unit_ligand_validity"]

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(meta_df, metadata_out)
    print(f"[augment] wrote augmented metadata -> {metadata_out} (rows={len(meta_df)})")
    return len(meta_df)


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-in", required=True, help="Input metadata parquet (pre-validity)")
    ap.add_argument("--metadata-out", required=True, help="Output metadata parquet (post-validity)")
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
    ap.add_argument("--ligand-score", action="append", default=None,
                    help="Score field to include (repeatable); default = DEFAULT_LIGAND_SCORES")
    ap.add_argument("--only-fetch", action="store_true", help="Skip the augment step")
    ap.add_argument("--only-augment", action="store_true", help="Skip the fetch step")
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args()

    if args.only_fetch and args.only_augment:
        raise SystemExit("--only-fetch and --only-augment are mutually exclusive")

    metadata_in = Path(args.metadata_in)
    metadata_out = Path(args.metadata_out)
    cache_dir = Path(args.cache_dir)
    cache_parquet = Path(args.cache_parquet)
    ligand_scores = args.ligand_score or list(DEFAULT_LIGAND_SCORES)

    errors: list[tuple[str, str]] = []
    if not args.only_augment:
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
        )

    if not args.only_fetch:
        augment_step(
            metadata_in=metadata_in,
            cache_parquet=cache_parquet,
            metadata_out=metadata_out,
            ligand_scores=ligand_scores,
        )

    if errors:
        print(f"\n[summary] {len(errors)} PDBs failed during fetch (they are omitted from the cache):")
        # Only show first 20 — full list can be recovered by re-running with --only-fetch
        for pid, err in errors[:20]:
            print(f"  {pid}: {err}")
        if len(errors) > 20:
            print(f"  ... ({len(errors) - 20} more)")
        print(
            "Re-run the command to retry failed PDBs (successful ones will be skipped via --skip-existing)."
        )


if __name__ == "__main__":
    main()
