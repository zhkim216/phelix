"""
Gather CSV metric files from ligand-evaluation batch runs into a tar.gz.

Sibling of :mod:`gather_csvs` for outputs produced by
``allatom_design.eval.glide.run_ligand_eval_batch``. Those filenames carry a
variable cutoff suffix (e.g. ``_lplddt80_lrmsd2``) and an optional SLURM
array suffix (e.g. ``_array_3``), so a simple stem match is not enough —
this script globs by stem prefix and classifies each hit.

Usage:
    python gather_ligand_eval_csvs.py <output_tar_gz> <src_dir1> [src_dir2 ...]
    python gather_ligand_eval_csvs.py --array-jobs <output_tar_gz> <src_dir1> [...]
"""
import argparse
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

CSV_STEM_PREFIXES = [
    "ligand_eval_metrics",
    "pb_af3_metrics",
    "inplace_metrics",
    "redock_metrics",
    "pb_mininplace_metrics",
    "pb_redocking_metrics",
    "failed_samples",
    "selected_samples",
]

EXTRA_FILES = ["config.yaml"]

ARRAY_RE = re.compile(r"_array_\d+\.csv$")


def _find_csvs(step_dir: Path, array_jobs: bool) -> list[Path]:
    """Return ligand-eval CSVs in ``step_dir``.

    Always includes base (non-array) files. Array shards (``*_array_N.csv``)
    are included only when ``array_jobs=True``.
    """
    found: list[Path] = []
    seen: set[str] = set()
    for stem in CSV_STEM_PREFIXES:
        bare = step_dir / f"{stem}.csv"
        if bare.exists() and bare.name not in seen:
            found.append(bare)
            seen.add(bare.name)
        for p in sorted(step_dir.glob(f"{stem}_*.csv")):
            if p.name in seen:
                continue
            is_array = bool(ARRAY_RE.search(p.name))
            if is_array and not array_jobs:
                continue
            found.append(p)
            seen.add(p.name)
    return found


def _find_extras(step_dir: Path) -> list[Path]:
    return [step_dir / name for name in EXTRA_FILES if (step_dir / name).exists()]


def _copy_into(dest: Path, paths: list[Path]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, dest / p.name)


def gather(src_dirs: list[Path], output_tar: Path, array_jobs: bool = False) -> None:
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

            if step_dirs:
                # Case 1: step_* subdirectories exist
                for step_dir in step_dirs:
                    csvs = _find_csvs(step_dir, array_jobs)
                    extras = _find_extras(step_dir)
                    if csvs:
                        dest = tmp / exp_name / step_dir.name
                        _copy_into(dest, csvs + extras)
                        total_copied += len(csvs)
                    else:
                        print(f"    [SKIP] No ligand eval CSVs in {step_dir.name}")
                print(f"    {len(step_dirs)} step dirs processed")
            else:
                # Case 2: CSVs directly in src_dir
                csvs = _find_csvs(src_dir, array_jobs)
                extras = _find_extras(src_dir)
                if csvs:
                    dest = tmp / exp_name
                    _copy_into(dest, csvs + extras)
                    total_copied += len(csvs)
                    print(f"    {len(csvs)} CSVs found directly in directory")
                else:
                    print(f"    No step_* directories or ligand eval CSVs found")

        output_tar = output_tar.resolve()
        output_tar.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(output_tar, "w:gz") as tar:
            for item in tmp.iterdir():
                tar.add(item, arcname=item.name)

    size_mb = output_tar.stat().st_size / (1024 * 1024)
    print(f"\nDone: {total_copied} CSV files -> {output_tar} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather ligand-eval-batch CSVs into a tar.gz"
    )
    parser.add_argument("output_tar", type=Path, help="Output .tar.gz path")
    parser.add_argument(
        "src_dirs", type=Path, nargs="+",
        help="Source ligand-eval directories",
    )
    parser.add_argument(
        "--array-jobs", action="store_true",
        help="Also collect *_array_N.csv files from SLURM array jobs",
    )
    args = parser.parse_args()
    gather(args.src_dirs, args.output_tar, array_jobs=args.array_jobs)


if __name__ == "__main__":
    main()
