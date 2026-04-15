"""Unified ligand evaluation CLI: PoseBusters + Glide on AF3 predictions.

The per-sample and batch logic lives in
:mod:`allatom_design.eval.glide.pipeline`; this module only handles Hydra
config parsing, sample selection, array-job splitting, and CSV output.

Usage:
    python -m allatom_design.eval.glide.run_ligand_eval_batch

Retry failed samples:
    python -m allatom_design.eval.glide.run_ligand_eval_batch retry_csv=path/to/failed_samples.csv
"""

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.glide.pipeline import evaluate_batch
from allatom_design.eval.glide.sample_selection import (
    load_af3_metrics,
    select_best_diffusion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _make_cutoff_suffix(sel_cfg: dict[str, Any]) -> str:
    """Build filename suffix from selection cutoffs."""
    parts = []
    plddt = sel_cfg.get("ligand_plddt_cutoff")
    rmsd = sel_cfg.get("ligand_rmsd_cutoff")

    if plddt is not None:
        v = int(plddt) if plddt == int(plddt) else f"{plddt:.1f}"
        parts.append(f"lplddt{v}")
    if rmsd is not None:
        v = int(rmsd) if rmsd == int(rmsd) else f"{rmsd:.1f}"
        parts.append(f"lrmsd{v}")

    return f"_{'_'.join(parts)}" if parts else ""


def _array_suffix() -> str:
    """Build array suffix from SLURM env, or empty string."""
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_id is not None:
        return f"_array_{array_id}"
    return ""


# ---------------------------------------------------------------------------
# Array job splitting
# ---------------------------------------------------------------------------

def split_for_array_job(
    df: pd.DataFrame,
    array_id: int | None = None,
    num_arrays: int | None = None,
) -> pd.DataFrame:
    """Slice DataFrame for a SLURM array task.

    Falls back to SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT env vars.
    """
    if array_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is not None:
            array_id = int(env_id)
            num_arrays = int(
                os.environ.get("SLURM_ARRAY_TASK_COUNT", num_arrays or 1)
            )

    if array_id is None:
        return df

    num_arrays = num_arrays or 1
    chunk_size = math.ceil(len(df) / num_arrays)
    start = array_id * chunk_size
    end = min(start + chunk_size, len(df))
    chunk = df.iloc[start:end]
    logger.info(
        f"Array {array_id}/{num_arrays}: rows [{start}:{end}] "
        f"({len(chunk)} of {len(df)} total)"
    )
    return chunk


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def _save_phase_csv(
    df: pd.DataFrame,
    output_dir: Path,
    phase_prefix: str,
    suffix: str,
) -> str | None:
    """Extract columns for a phase and save as CSV."""
    meta_cols = [
        "designed_sample_id", "input_sample_id", "diffusion_idx", "cif_path",
    ]
    phase_cols = [c for c in df.columns if c.startswith(phase_prefix)]

    if not phase_cols:
        return None

    keep = [c for c in meta_cols if c in df.columns] + phase_cols
    phase_df = df[keep].copy()

    has_data = phase_df[phase_cols].notna().any(axis=1)
    phase_df = phase_df[has_data]

    if phase_df.empty:
        return None

    csv_name = f"{phase_prefix}_metrics{suffix}{_array_suffix()}.csv"
    csv_path = str(output_dir / csv_name)
    phase_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_name} ({len(phase_df)} rows)")
    return csv_path


def save_results(
    results_df: pd.DataFrame,
    output_dir: str,
    cutoff_suffix: str,
) -> None:
    """Save per-phase CSVs + unified CSV + failed samples CSV."""
    out = Path(output_dir)
    sfx = cutoff_suffix
    arr = _array_suffix()

    unified_path = out / f"ligand_eval_metrics{sfx}{arr}.csv"
    results_df.to_csv(str(unified_path), index=False)
    logger.info(f"Unified: {unified_path.name} ({len(results_df)} rows)")

    _save_phase_csv(results_df, out, "pb_af3", sfx)
    _save_phase_csv(results_df, out, "inplace", sfx)
    _save_phase_csv(results_df, out, "redock", sfx)
    _save_phase_csv(results_df, out, "pb_mininplace", sfx)
    _save_phase_csv(results_df, out, "pb_redocking", sfx)

    error_cols = [c for c in results_df.columns if c.endswith("_error") or c == "error"]
    if error_cols:
        has_error = results_df[error_cols].notna().any(axis=1)
        failed = results_df[has_error].copy()
        if not failed.empty:
            meta = ["designed_sample_id", "input_sample_id", "diffusion_idx", "cif_path"]
            keep = [c for c in meta if c in failed.columns] + error_cols
            failed_df = failed[keep]
            failed_path = out / f"failed_samples{sfx}{arr}.csv"
            failed_df.to_csv(str(failed_path), index=False)
            logger.info(
                f"Failed: {failed_path.name} ({len(failed_df)} samples)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../../configs_local/eval/glide",
    config_name="run_ligand_eval_batch",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    """Run unified ligand evaluation on AF3 predicted structures."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    af3_eval_dir = Path(cfg.af3_eval_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sel_cfg = cfg_dict.get("selection", {})
    cutoff_suffix = _make_cutoff_suffix(sel_cfg)

    # ================================================================
    # Phase 0: Load metrics and select samples (or retry)
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 0: Sample selection")
    print("=" * 60 + "\n")

    retry_csv = cfg_dict.get("retry_csv")
    if retry_csv:
        print(f"Retry mode: loading {retry_csv}")
        selected = pd.read_csv(retry_csv)
        for col in ["designed_sample_id", "diffusion_idx"]:
            if col not in selected.columns:
                print(f"ERROR: retry CSV missing column '{col}'")
                return
        print(f"Retrying {len(selected)} samples")
    else:
        docking_csv = str(af3_eval_dir / "all_docking_metrics_per_designed_sample.csv")
        sc_csv = str(af3_eval_dir / "all_sc_metrics_per_designed_sample.csv")

        flat_df = load_af3_metrics(docking_csv, sc_csv)
        print(f"Loaded {len(flat_df)} total (designed_sample, diffusion) pairs")

        selected = select_best_diffusion(
            flat_df,
            ligand_rmsd_cutoff=sel_cfg.get("ligand_rmsd_cutoff"),
            ligand_plddt_cutoff=sel_cfg.get("ligand_plddt_cutoff"),
        )
        print(f"Selected {len(selected)} samples after cutoff filtering")

    if selected.empty:
        print("No samples selected. Exiting.")
        return

    debug = cfg_dict.get("debug", False)
    num_debug_samples = cfg_dict.get("num_debug_samples", 10)
    if debug and len(selected) > num_debug_samples:
        print(f"Debug mode: limiting to {num_debug_samples} samples")
        selected = selected.head(num_debug_samples)

    selected = split_for_array_job(selected)
    if selected.empty:
        print("No samples for this array task. Exiting.")
        return

    sel_path = output_dir / f"selected_samples{cutoff_suffix}{_array_suffix()}.csv"
    selected.to_csv(str(sel_path), index=False)
    print(f"Processing {len(selected)} samples")

    # ================================================================
    # Run evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("Running ligand evaluation (PB → Glide → PB)")
    print("=" * 60 + "\n")

    af3_preds_dir = str(af3_eval_dir / "af3_ss_preds")
    num_workers = cfg_dict.get("num_workers", 1)

    results_df = evaluate_batch(
        selected_df=selected,
        af3_preds_dir=af3_preds_dir,
        sample_dir=cfg.sample_dir,
        output_dir=str(output_dir),
        schrodinger_cfg=cfg_dict.get("schrodinger", {}),
        glide_cfg=cfg_dict.get("glide", {}),
        pb_cfg=cfg_dict.get("posebusters", {}),
        cif_parse_cfg=cfg_dict.get("cif_parse_cfg"),
        seed=sel_cfg.get("seed", 42),
        num_workers=num_workers,
        ref_sample_is_designed=cfg_dict.get("ref_sample_is_designed", False),
    )

    # ================================================================
    # Save results
    # ================================================================
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60 + "\n")

    save_results(results_df, str(output_dir), cutoff_suffix)

    # Summary
    print(f"\nTotal samples: {len(results_df)}")
    for col_name, label in [
        ("pb_af3_pb_valid", "PB AF3 valid"),
        ("inplace_best_docking_score", "Inplace DockingScore (median)"),
        ("redock_best_docking_score", "Redock DockingScore (median)"),
        ("redock_vs_ref_ligand_rmsd", "Redock vs ref RMSD (median)"),
        ("pb_mininplace_pb_valid", "PB Mininplace valid"),
        ("pb_redocking_pb_valid", "PB Redocking valid"),
    ]:
        if col_name in results_df.columns:
            valid = results_df[col_name].dropna()
            if len(valid) > 0:
                if valid.dtype == bool or set(valid.unique()) <= {True, False}:
                    rate = valid.mean() * 100
                    print(f"  {label}: {rate:.1f}% ({valid.sum()}/{len(valid)})")
                else:
                    print(f"  {label}: {valid.median():.3f}")

    error_cols = [c for c in results_df.columns if "error" in c]
    n_errors = results_df[error_cols].notna().any(axis=1).sum() if error_cols else 0
    if n_errors > 0:
        print(f"\n  Samples with errors: {n_errors}")
        for ec in error_cols:
            n = results_df[ec].notna().sum()
            if n > 0:
                print(f"    {ec}: {n}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
