"""
Gather CSV metric files from evaluation step directories and compress into a tar.gz.

Usage:
    python gather_csvs.py <output_tar_gz> <src_dir1> [src_dir2 ...]
"""
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

CSV_FILES = [
    "all_docking_metrics_per_designed_sample.csv",
    "all_sc_metrics_per_designed_sample.csv",
    "seq_recovery_metrics.csv",
]


def gather(src_dirs: list[Path], output_tar: Path):
    total_copied = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for src_dir in src_dirs:
            src_dir = src_dir.resolve()
            if not src_dir.is_dir():
                print(f"  [SKIP] Not a directory: {src_dir}")
                continue

            exp_name = src_dir.name
            print(f"  Processing: {exp_name}")

            step_dirs = sorted(
                [d for d in src_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
            )
            if not step_dirs:
                print(f"    No step_* directories found")
                continue

            for step_dir in step_dirs:
                dest = tmp / exp_name / step_dir.name
                copied = 0
                for csv_name in CSV_FILES:
                    csv_path = step_dir / csv_name
                    if csv_path.exists():
                        dest.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(csv_path, dest / csv_name)
                        copied += 1
                if copied > 0:
                    total_copied += copied
                else:
                    print(f"    [SKIP] No CSVs in {step_dir.name}")

            print(f"    {len(step_dirs)} step dirs processed")

        # Create tar.gz
        output_tar = output_tar.resolve()
        output_tar.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(output_tar, "w:gz") as tar:
            for item in tmp.iterdir():
                tar.add(item, arcname=item.name)

    size_mb = output_tar.stat().st_size / (1024 * 1024)
    print(f"\nDone: {total_copied} CSV files -> {output_tar} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Gather CSV metrics into a tar.gz")
    parser.add_argument("output_tar", type=Path, help="Output .tar.gz path")
    parser.add_argument("src_dirs", type=Path, nargs="+", help="Source eval directories")
    args = parser.parse_args()
    gather(args.src_dirs, args.output_tar)


if __name__ == "__main__":
    main()
