#!/usr/bin/env python3
"""
Detect shard_ids whose `metadata_shard_{id:05d}.parquet` is missing from the
output directory.

Rationale: `build_metadata_parquet_shards.py` only calls `_merge_batch_parquets`
at the very end of its main loop. If the SLURM job dies mid-way (wall-time,
OOM, etc.) the final per-shard parquet never gets written, while partial
`batch_*.parquet` files may remain in `shard_{id:05d}_batches/`.

Therefore the presence/absence of the final per-shard parquet is a clean
single signal for "did this shard's build finish".

Output:
  - Human-readable summary (counts, shard_ids, SLURM --array string)
  - Optional JSON report (--write-report)

Usage:
    python3 -m allatom_design.data.preprocessing.atomworks.detect_incomplete_shards \\
        --out_dir /scratch/.../atomworks_pdb_full_v8 \\
        --num-shards 100 \\
        [--verbose] \\
        [--write-report path.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ShardDiagnostic:
    shard_id: int
    final_parquet_exists: bool
    progress_completed: list[int] = field(default_factory=list)
    progress_attempted_empty: list[int] = field(default_factory=list)
    progress_batch_size: int | None = None
    progress_total_batches: int | None = None
    batch_files_on_disk: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def find_incomplete_shards(shard_dir: Path, num_shards: int) -> list[int]:
    """Return sorted list of shard_ids that are missing `metadata_shard_*.parquet`."""
    return [
        i for i in range(num_shards)
        if not (shard_dir / f"metadata_shard_{i:05d}.parquet").exists()
    ]


def format_slurm_array(shard_ids: list[int]) -> str:
    """Compress a sorted list of ints into SLURM --array syntax.

    >>> format_slurm_array([])
    ''
    >>> format_slurm_array([3])
    '3'
    >>> format_slurm_array([3, 17, 42, 43, 44])
    '3,17,42-44'
    >>> format_slurm_array([0, 1, 2, 3, 4, 5])
    '0-5'
    >>> format_slurm_array([0, 5])
    '0,5'
    """
    if not shard_ids:
        return ""
    ids = sorted(set(shard_ids))
    runs: list[tuple[int, int]] = []
    start = prev = ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in runs]
    return ",".join(parts)


def diagnose_shard(shard_dir: Path, shard_id: int) -> ShardDiagnostic:
    diag = ShardDiagnostic(
        shard_id=shard_id,
        final_parquet_exists=(shard_dir / f"metadata_shard_{shard_id:05d}.parquet").exists(),
    )

    progress_file = shard_dir / f"metadata_shard_{shard_id:05d}_progress.json"
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                prog = json.load(f)
            diag.progress_completed = sorted(prog.get("completed_batches", []))
            diag.progress_attempted_empty = sorted(prog.get("attempted_empty_batches", []))
            diag.progress_batch_size = prog.get("batch_size")
            diag.progress_total_batches = prog.get("total_batches")
        except Exception as e:
            diag.notes.append(f"progress_json_unreadable: {e!r}")

    batch_dir = shard_dir / f"shard_{shard_id:05d}_batches"
    if batch_dir.is_dir():
        on_disk: list[int] = []
        for p in batch_dir.glob("batch_*.parquet"):
            try:
                on_disk.append(int(p.stem.split("_")[1]))
            except Exception:
                continue
        diag.batch_files_on_disk = sorted(on_disk)

    return diag


def _format_verbose_line(diag: ShardDiagnostic) -> str:
    expected = diag.progress_total_batches
    completed_n = len(diag.progress_completed)
    empty_n = len(diag.progress_attempted_empty)
    on_disk_n = len(diag.batch_files_on_disk)
    expected_str = str(expected) if expected is not None else "?"
    notes_str = f"  notes={diag.notes}" if diag.notes else ""
    return (
        f"  shard {diag.shard_id:>5d}: "
        f"final={'Y' if diag.final_parquet_exists else 'N'}  "
        f"completed={completed_n}/{expected_str}  "
        f"empty={empty_n}  "
        f"batch_files_on_disk={on_disk_n}"
        f"{notes_str}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="Dataset root dir (contains 'shards/' subdir)")
    ap.add_argument("--num-shards", type=int, required=True, help="Total number of shards (SLURM_ARRAY_TASK_MAX + 1)")
    ap.add_argument("--verbose", action="store_true", help="Print per-shard diagnostics (progress JSON, batch file counts)")
    ap.add_argument("--write-report", type=str, default=None, help="Optional path to write a JSON report")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    if not shard_dir.is_dir():
        raise SystemExit(f"Shard directory not found: {shard_dir}")

    incomplete = find_incomplete_shards(shard_dir, args.num_shards)
    complete = [i for i in range(args.num_shards) if i not in set(incomplete)]
    slurm_array = format_slurm_array(incomplete)

    print(f"Scanning {shard_dir} (num_shards={args.num_shards})")
    print(f"Complete:    {len(complete):>5d}  (metadata_shard_*.parquet present)")
    print(f"Incomplete:  {len(incomplete):>5d}")
    print()

    if incomplete:
        print(f"Incomplete shard_ids: {slurm_array}")
        print(f"SLURM --array= {slurm_array}")
        print()
        print(
            "Next step: "
            "sbatch --array="
            f"{slurm_array} "
            "scripts/sherlock_scripts/jinho/preprocessing/atomworks/build_metadata_parquet_shards_v8.sbatch"
        )
    else:
        print("All shards complete.")

    diagnostics: list[ShardDiagnostic] = []
    if args.verbose or args.write_report:
        target_ids = incomplete if args.verbose else range(args.num_shards)
        diagnostics = [diagnose_shard(shard_dir, i) for i in target_ids]

    if args.verbose and diagnostics:
        print()
        print("Per-incomplete-shard diagnostics:")
        for d in diagnostics:
            print(_format_verbose_line(d))

    if args.write_report:
        report = {
            "out_dir": str(out_dir),
            "shard_dir": str(shard_dir),
            "num_shards": args.num_shards,
            "complete_count": len(complete),
            "incomplete_count": len(incomplete),
            "complete_shard_ids": complete,
            "incomplete_shard_ids": incomplete,
            "slurm_array": slurm_array,
            "per_shard_diagnostics": [asdict(d) for d in diagnostics],
        }
        report_path = Path(args.write_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nReport written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
