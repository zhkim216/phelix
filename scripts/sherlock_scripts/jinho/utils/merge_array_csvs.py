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
"""
import argparse
import glob
import re
from pathlib import Path

import pandas as pd
from natsort import natsorted


def merge_array_csvs(results_dir: Path) -> None:
    """단일 디렉토리 내의 array CSV 파일들을 머지한다."""
    # Find all array CSV files
    array_csvs = natsorted(glob.glob(str(results_dir / "*_array_*.csv")))
    
    if not array_csvs:
        print(f"  No array CSV files found in {results_dir}")
        return
    
    # Group by base name (strip _array_{id} suffix)
    pattern = re.compile(r"^(.+)_array_\d+\.csv$")
    groups: dict[str, list[str]] = {}
    for csv_path in array_csvs:
        match = pattern.match(Path(csv_path).name)
        if match:
            base_name = match.group(1)
            groups.setdefault(base_name, []).append(csv_path)
    
    # Merge each group
    for base_name, csv_paths in groups.items():
        print(f"  Merging {len(csv_paths)} files for '{base_name}':")
        for p in csv_paths:
            print(f"    - {Path(p).name}")
        
        dfs = []
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
                dfs.append(df)
            except Exception as e:
                print(f"    Warning: Failed to read {csv_path}: {e}")
        
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            out_path = results_dir / f"{base_name}.csv"
            merged_df.to_csv(out_path, index=False)
            print(f"    -> Saved merged CSV: {out_path} ({len(merged_df)} rows)")
        else:
            print(f"    Warning: No valid CSVs to merge for '{base_name}'")


def merge_all(root_dir: str) -> None:
    """root_dir 내의 step_* 하위 폴더들을 순회하며 각각 머지한다.
    
    만약 root_dir 자체에 *_array_*.csv 파일이 있으면 그것도 머지한다.
    """
    root_dir = Path(root_dir)
    
    # step_* 패턴의 하위 폴더 탐색
    step_dirs = natsorted(
        [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    )
    
    if step_dirs:
        print(f"Found {len(step_dirs)} step directories in {root_dir}:\n")
        for step_dir in step_dirs:
            print(f"[{step_dir.name}]")
            merge_array_csvs(step_dir)
            print()
    else:
        # step_* 폴더가 없으면 root_dir 자체를 처리
        print(f"No step_* subdirectories found. Processing {root_dir} directly.\n")
        merge_array_csvs(root_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge array job CSV files")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="상위 디렉토리 (step_* 하위 폴더 자동 순회) 또는 단일 step 디렉토리")
    args = parser.parse_args()
    merge_all(args.results_dir)