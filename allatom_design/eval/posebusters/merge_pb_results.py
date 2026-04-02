"""Merge PoseBusters array job results into a single CSV.

Run after all SLURM array tasks have completed.

Usage:
    python -m allatom_design.eval.posebusters.merge_pb_results \
        --results_dir /path/to/pb_output \
        --output /path/to/pb_output/pb_metrics_merged.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_array_csvs(
    results_dir: str,
    pattern: str = "pb_metrics_array_*.csv",
) -> list[Path]:
    """Find all per-array CSV files, sorted by array index."""
    paths = sorted(
        Path(results_dir).glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return paths


def merge_results(csv_paths: list[Path]) -> pd.DataFrame:
    """Concatenate per-array CSV files into one DataFrame."""
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        logger.info(f"  {path.name}: {len(df)} rows")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge PoseBusters array job results."
    )
    p.add_argument(
        "--results_dir", required=True,
        help="Directory containing pb_metrics_array_*.csv files.",
    )
    p.add_argument(
        "--output", default=None,
        help="Output merged CSV path (default: {results_dir}/pb_metrics_merged.csv).",
    )
    p.add_argument(
        "--pattern", default="pb_metrics_array_*.csv",
        help="Glob pattern for array CSV files.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)

    csv_paths = find_array_csvs(str(results_dir), args.pattern)
    if not csv_paths:
        logger.error(f"No CSV files matching '{args.pattern}' in {results_dir}")
        return

    logger.info(f"Found {len(csv_paths)} array CSV files:")
    merged = merge_results(csv_paths)

    output = Path(args.output) if args.output else results_dir / "pb_metrics_merged.csv"
    merged.to_csv(output, index=False)
    logger.info(f"Merged {len(merged)} rows -> {output}")

    # Summary
    if "pb_valid" in merged.columns:
        valid = merged["pb_valid"].dropna()
        if len(valid) > 0:
            n_valid = int(valid.sum())
            logger.info(
                f"pb_valid: {n_valid}/{len(valid)} "
                f"({100 * n_valid / len(valid):.1f}%)"
            )

    if "error" in merged.columns:
        n_err = merged["error"].notna().sum()
        if n_err > 0:
            logger.warning(f"Entries with errors: {n_err}/{len(merged)}")


if __name__ == "__main__":
    main()
