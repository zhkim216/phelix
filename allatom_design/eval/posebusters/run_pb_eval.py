"""CLI entry point for PoseBusters evaluation of AF3 predictions.

Supports SLURM array jobs and Python multiprocessing.

Usage (local):
    python -m allatom_design.eval.posebusters.run_pb_eval \
        --af3_pred_dir /path/to/af3_ss_preds \
        --out_dir /path/to/pb_output \
        --num_workers 4

Usage (SLURM array job):
    # In sbatch script with --array=0-9:
    python -m allatom_design.eval.posebusters.run_pb_eval \
        --af3_pred_dir /path/to/af3_ss_preds \
        --out_dir /path/to/pb_output \
        --num_workers 4 \
        --num_arrays 10
    # array_id is auto-detected from $SLURM_ARRAY_TASK_ID
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

import pandas as pd

from allatom_design.eval.posebusters.core import (
    discover_af3_cif_paths,
    evaluate_batch,
    split_entries_for_array_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PoseBusters evaluation on AF3 predicted structures."
    )
    p.add_argument(
        "--af3_pred_dir", required=True,
        help="Directory containing AF3 predictions "
             "(e.g. .../step_82500_epoch_52/af3_ss_preds).",
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Output directory for PB results and intermediate files.",
    )
    p.add_argument(
        "--cif_pattern", default="*_model_pocket_aligned.cif",
        help="Glob pattern for CIF files (default: *_model_pocket_aligned.cif).",
    )
    p.add_argument(
        "--config", default="dock", choices=["dock", "redock"],
        help="PoseBusters config mode (default: dock).",
    )
    p.add_argument(
        "--num_workers", type=int, default=1,
        help="Python multiprocessing workers per array task (default: 1).",
    )
    # SLURM array job parameters
    p.add_argument(
        "--array_id", type=int, default=None,
        help="Array task ID (default: auto-detect from $SLURM_ARRAY_TASK_ID).",
    )
    p.add_argument(
        "--num_arrays", type=int, default=None,
        help="Total number of array tasks.",
    )
    # Optional CIF parse config
    p.add_argument(
        "--cif_parse_cfg", default=None,
        help="Path to YAML file with CIF parse config, or inline YAML string.",
    )
    p.add_argument(
        "--full_report", action="store_true", default=False,
        help="Include detailed per-subtest columns (default: False = summary only).",
    )
    return p.parse_args(argv)


def load_cif_parse_cfg(cfg_arg: str | None) -> dict | None:
    """Load CIF parse config from a YAML file path or inline string."""
    if cfg_arg is None:
        return None
    path = Path(cfg_arg)
    if path.is_file():
        with open(path) as f:
            return yaml.safe_load(f)
    return yaml.safe_load(cfg_arg)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover CIF files
    logger.info(f"Discovering CIF files in {args.af3_pred_dir}")
    entries = discover_af3_cif_paths(args.af3_pred_dir, args.cif_pattern)
    logger.info(f"Found {len(entries)} CIF files")

    if not entries:
        logger.warning("No CIF files found. Exiting.")
        sys.exit(0)

    # 2. Split for array job
    entries = split_entries_for_array_job(
        entries, array_id=args.array_id, num_arrays=args.num_arrays,
    )
    logger.info(f"Processing {len(entries)} entries in this task")

    # 3. Load optional CIF parse config
    cif_parse_cfg = load_cif_parse_cfg(args.cif_parse_cfg)

    # 4. Run evaluation
    work_dir = str(out_dir / "work")
    df = evaluate_batch(
        entries=entries,
        out_dir=work_dir,
        config=args.config,
        cif_parse_cfg=cif_parse_cfg,
        num_workers=args.num_workers,
        full_report=args.full_report,
    )

    # 5. Save results
    array_id = args.array_id
    if array_id is None:
        import os
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is not None:
            array_id = int(env_id)

    if array_id is not None:
        csv_name = f"pb_metrics_array_{array_id}.csv"
    else:
        csv_name = "pb_metrics.csv"

    csv_path = out_dir / csv_name
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} results to {csv_path}")

    # 6. Print summary
    if "pb_valid" in df.columns:
        valid = df["pb_valid"].dropna()
        if len(valid) > 0:
            n_valid = valid.sum()
            logger.info(
                f"pb_valid: {n_valid}/{len(valid)} "
                f"({100 * n_valid / len(valid):.1f}%)"
            )

    errors = df.get("error")
    if errors is not None:
        n_err = errors.notna().sum()
        if n_err > 0:
            logger.warning(f"Errors: {n_err}/{len(df)}")


if __name__ == "__main__":
    main()
