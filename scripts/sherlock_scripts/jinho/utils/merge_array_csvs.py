"""
Merge CSV files from array jobs into single combined CSVs.

Usage:
    # Single step folder processing
    python merge_array_csvs.py --results_dir /path/to/eval_.../step_62500_epoch_X

    # Specify parent directory to automatically traverse subdirectories starting with "step_"
    python merge_array_csvs.py --results_dir /path/to/eval_exp19_cfg0_lmpnnval_sm_filt_nogaphetero_a.../

This script finds all CSV files with pattern *_array_*.csv in the given directory,
groups them by base name, and concatenates them into merged CSVs.

For example:
    seq_recovery_metrics_array_0.csv  \
    seq_recovery_metrics_array_1.csv   -> seq_recovery_metrics.csv
    ...                               /

Safety behavior (strict by default):
- If the merged output file already exists, it is backed up to
  ``{base_name}.bak.csv`` before being overwritten (use ``--force`` to skip
  the backup).
- If any shard CSV fails to read, the run exits with a non-zero status
  (use ``--allow-broken`` to downgrade to a warning).
- If shard indices have gaps (e.g. ``[0, 1, 3]`` is missing shard 2), the
  run exits with a non-zero status (use ``--allow-gaps`` to downgrade to a
  warning).
"""
import argparse
import glob
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from natsort import natsorted


SHARD_RE = re.compile(r"^(.+)_array_(\d+)\.csv$")


def _format_stat(path: Path) -> str:
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return f"size={st.st_size}B, mtime={mtime}"


def _backup_existing(out_path: Path) -> Path | None:
    """Copy an existing merged CSV to ``{base_name}.bak.csv``.

    Returns the backup path (or ``None`` if no existing file was present).
    An existing ``.bak.csv`` is overwritten — we only keep the most recent
    pre-merge snapshot.
    """
    if not out_path.exists():
        return None
    backup_path = out_path.with_name(out_path.stem + ".bak.csv")
    shutil.copy2(out_path, backup_path)
    return backup_path


def merge_array_csvs(
    results_dir: Path,
    *,
    force: bool = False,
    allow_gaps: bool = False,
    allow_broken: bool = False,
) -> bool:
    """Merge all array-shard CSV files inside a single directory.

    Returns ``True`` on success, ``False`` if any strict check failed.
    """
    array_csvs = natsorted(glob.glob(str(results_dir / "*_array_*.csv")))

    if not array_csvs:
        print(f"  No array CSV files found in {results_dir}")
        return True

    # Group by base name; track (base_name -> [(shard_id, path), ...]).
    groups: dict[str, list[tuple[int, str]]] = {}
    for csv_path in array_csvs:
        match = SHARD_RE.match(Path(csv_path).name)
        if not match:
            continue
        base_name = match.group(1)
        shard_id = int(match.group(2))
        groups.setdefault(base_name, []).append((shard_id, csv_path))

    ok = True
    for base_name, entries in groups.items():
        entries.sort(key=lambda x: x[0])
        shard_ids = [sid for sid, _ in entries]
        csv_paths = [p for _, p in entries]

        print(f"  Merging {len(csv_paths)} files for '{base_name}':")
        for p in csv_paths:
            print(f"    - {Path(p).name}")

        # --- B: shard-gap detection ---
        if len(set(shard_ids)) != len(shard_ids):
            duplicates = sorted({sid for sid in shard_ids if shard_ids.count(sid) > 1})
            msg = f"duplicate shard ids for '{base_name}': {duplicates}"
            if allow_gaps:
                print(f"    WARN: {msg} (allowed via --allow-gaps)")
            else:
                print(f"    ERROR: {msg}")
                ok = False
        expected = set(range(max(shard_ids) + 1))
        missing = sorted(expected - set(shard_ids))
        if missing:
            msg = (
                f"missing shard ids for '{base_name}': {missing} "
                f"(have {len(shard_ids)} of {max(shard_ids) + 1} expected)"
            )
            if allow_gaps:
                print(f"    WARN: {msg} (allowed via --allow-gaps)")
            else:
                print(f"    ERROR: {msg}")
                ok = False

        # --- A2: broken shard detection ---
        dfs = []
        failed: list[str] = []
        for csv_path in csv_paths:
            try:
                dfs.append(pd.read_csv(csv_path))
            except Exception as e:
                failed.append(f"{Path(csv_path).name}: {e}")

        if failed:
            for f in failed:
                print(f"    WARN: failed to read shard {f}")
            if not allow_broken:
                print(
                    f"    ERROR: {len(failed)} shard(s) failed to read for "
                    f"'{base_name}' (use --allow-broken to downgrade)"
                )
                ok = False

        if not dfs:
            print(f"    WARN: no readable shards for '{base_name}', skipping write")
            continue

        merged_df = pd.concat(dfs, ignore_index=True)
        out_path = results_dir / f"{base_name}.csv"

        # --- A1: overwrite backup ---
        if out_path.exists():
            try:
                prior_rows = len(pd.read_csv(out_path))
            except Exception:
                prior_rows = -1  # unreadable, report as -1
            print(
                f"    NOTE: {out_path.name} already exists "
                f"({_format_stat(out_path)}, rows={prior_rows}); "
                f"new merged rows={len(merged_df)}"
            )
            if force:
                print(f"    NOTE: --force set, overwriting without backup")
            else:
                backup = _backup_existing(out_path)
                print(f"    NOTE: backed up existing file to {backup.name}")

        merged_df.to_csv(out_path, index=False)
        print(f"    -> Saved merged CSV: {out_path} ({len(merged_df)} rows)")

    return ok


def merge_all(
    root_dir: str,
    *,
    force: bool = False,
    allow_gaps: bool = False,
    allow_broken: bool = False,
) -> bool:
    """Walk ``step_*`` subdirectories of ``root_dir`` and merge each one.

    If ``root_dir`` has no ``step_*`` subdirectories but contains
    ``*_array_*.csv`` files directly, those are merged in place.

    Returns ``True`` iff every directory merged cleanly.
    """
    root = Path(root_dir)

    step_dirs = natsorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("step_")]
    )

    overall_ok = True
    if step_dirs:
        print(f"Found {len(step_dirs)} step directories in {root}:\n")
        for step_dir in step_dirs:
            print(f"[{step_dir.name}]")
            overall_ok &= merge_array_csvs(
                step_dir,
                force=force,
                allow_gaps=allow_gaps,
                allow_broken=allow_broken,
            )
            print()
    else:
        print(f"No step_* subdirectories found. Processing {root} directly.\n")
        overall_ok &= merge_array_csvs(
            root,
            force=force,
            allow_gaps=allow_gaps,
            allow_broken=allow_broken,
        )

    return overall_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge array job CSV files")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Parent directory (auto-traverses step_* subfolders) or a single step/flat directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing merged CSVs without creating a .bak.csv backup",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Downgrade missing/duplicate shard-id errors to warnings",
    )
    parser.add_argument(
        "--allow-broken",
        action="store_true",
        help="Downgrade shard read failures to warnings",
    )
    args = parser.parse_args()

    ok = merge_all(
        args.results_dir,
        force=args.force,
        allow_gaps=args.allow_gaps,
        allow_broken=args.allow_broken,
    )
    sys.exit(0 if ok else 1)
