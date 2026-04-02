"""Unified ligand evaluation: PoseBusters + Glide on AF3 predictions.

Pipeline per sample:
    1. Shared preprocessing  (CIF → protein PDB + ligand SDF)
    2. Phase 1: PoseBusters on AF3 raw ligand pose
    3. Phase 2: Glide prep + mininplace + redocking
    4. Phase 3: PoseBusters on Glide output SDF files

Each phase is independent — failure in one does not block others.
Per-sample failures are tracked and can be retried.

Usage:
    python -m allatom_design.eval.glide.run_ligand_eval_batch

Retry failed samples:
    python -m allatom_design.eval.glide.run_ligand_eval_batch retry_csv=path/to/failed_samples.csv
"""

import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.glide.preprocessing import (
    compute_dynamic_outerbox,
    preprocess_structure,
)
from allatom_design.eval.glide.result_parser import (
    extract_best_scores,
    parse_glide_csv,
)
from allatom_design.eval.glide.sample_selection import (
    find_af3_prediction_path,
    load_af3_metrics,
)
from allatom_design.eval.glide.schrodinger_runner import (
    find_schrodinger,
    run_glide,
    run_grid_generation,
    run_ligprep,
    run_prepwizard,
    write_docking_input,
    write_gridgen_input,
)
from allatom_design.eval.eval_utils.eval_posebusters import (
    add_pb_valid,
    run_pb_single,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample selection (flexible cutoffs)
# ---------------------------------------------------------------------------

def select_best_diffusion(
    flat_df: pd.DataFrame,
    ligand_rmsd_cutoff: float | None = 2.0,
    ligand_plddt_cutoff: float | None = 70.0,
) -> pd.DataFrame:
    """Select best diffusion per sample with optional cutoffs.

    Unlike sample_selection.select_best_diffusion, this handles:
    - None cutoff values (skip that filter)
    - Missing columns (skip that filter)
    """
    mask = pd.Series(True, index=flat_df.index)

    if ligand_rmsd_cutoff is not None and "ligand_rmsd" in flat_df.columns:
        mask &= flat_df["ligand_rmsd"] <= ligand_rmsd_cutoff

    if ligand_plddt_cutoff is not None and "ligand_plddt" in flat_df.columns:
        mask &= flat_df["ligand_plddt"] >= ligand_plddt_cutoff

    filtered = flat_df[mask]

    if filtered.empty:
        logger.warning("No samples pass cutoffs")
        return pd.DataFrame()

    n_before = flat_df["designed_sample_id"].nunique()
    n_after = filtered["designed_sample_id"].nunique()
    logger.info(f"Selection: {n_after}/{n_before} designed samples pass cutoffs")

    # Pick best diffusion per sample
    if "ligand_plddt" in filtered.columns:
        idx = filtered.groupby("designed_sample_id")["ligand_plddt"].idxmax()
    elif "ligand_rmsd" in filtered.columns:
        idx = filtered.groupby("designed_sample_id")["ligand_rmsd"].idxmin()
    else:
        idx = filtered.groupby("designed_sample_id").head(1).index

    return filtered.loc[idx].reset_index(drop=True)


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
# PB helper
# ---------------------------------------------------------------------------

def _run_pb_and_summarize(
    mol_pred: str,
    mol_cond: str,
    pb_cfg: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Run PB bust() and return a flat dict with prefixed column names."""
    result: dict[str, Any] = {}
    config = pb_cfg.get("config", "dock")
    full_report = pb_cfg.get("full_report", False)

    df = run_pb_single(
        mol_pred=mol_pred,
        mol_cond=mol_cond,
        config=config,
        full_report=full_report,
    )
    if len(df) == 0:
        result[f"{prefix}_error"] = "empty_result"
        return result

    df = add_pb_valid(df)
    for col in df.columns:
        result[f"{prefix}_{col}"] = df.iloc[0][col]

    return result


# ---------------------------------------------------------------------------
# Glide docking (reused from run_glide_eval_batch)
# ---------------------------------------------------------------------------

def _run_docking_mode(
    grid_file: str,
    ligand_file: str,
    work_dir: str,
    schrodinger_path: str,
    glide_cfg: dict[str, Any],
    mode: str,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run a single docking mode and return metrics dict."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if mode == "mininplace":
        cfg = glide_cfg.get("inplace", {})
        jobname = "dock_inplace"
        docking_method = cfg.get("docking_method", "mininplace")
        num_poses = 1
    elif mode == "redocking":
        cfg = glide_cfg.get("redocking", {})
        jobname = "dock_redock"
        docking_method = cfg.get("docking_method", "confgen")
        num_poses = cfg.get("num_poses", 1)
    else:
        raise ValueError(f"Unknown docking mode: {mode}")

    dock_input = write_docking_input(
        gridfile=grid_file,
        ligandfile=ligand_file,
        out_dir=work_dir,
        jobname=jobname,
        docking_method=docking_method,
        precision=cfg.get("precision", "SP"),
        num_poses=num_poses,
        write_csv=True,
        pose_outtype="ligandlib_sd",
        compress_poses=False,
        forcefield=cfg.get("forcefield", "OPLS4"),
    )

    outputs = run_glide(
        input_file=dock_input,
        schrodinger_path=schrodinger_path,
        timeout=timeout,
    )

    metrics: dict[str, Any] = {}
    if outputs["csv_path"]:
        df = parse_glide_csv(outputs["csv_path"])
        if not df.empty:
            metrics.update(extract_best_scores(df))
            metrics["num_poses"] = len(df)
        else:
            metrics["error"] = "empty_csv"
    else:
        metrics["error"] = "no_csv_output"

    if outputs.get("sdf_path"):
        metrics["sdf_path"] = outputs["sdf_path"]

    return metrics


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------

def process_single_sample(
    row: dict[str, Any],
    af3_preds_dir: str,
    sample_dir: str,
    output_dir: str,
    schrodinger_path: str | None,
    schrodinger_cfg: dict[str, Any],
    glide_cfg: dict[str, Any],
    pb_cfg: dict[str, Any],
    cif_parse_cfg: dict | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Process one sample: preprocess → PB → Glide → PB.

    Designed to be called from ProcessPoolExecutor (stateless).
    """
    designed_id = row["designed_sample_id"]
    input_id = row.get("input_sample_id", "")
    diff_idx = int(row.get("diffusion_idx", 0))

    out_path = Path(output_dir)
    timeout = schrodinger_cfg.get("timeout", 3600)

    # Find AF3 prediction CIF
    cif_path = find_af3_prediction_path(
        af3_preds_dir, designed_id, diff_idx, seed=seed,
    )
    if not cif_path:
        return {
            "designed_sample_id": designed_id,
            "input_sample_id": input_id,
            "diffusion_idx": diff_idx,
            "error": "af3_prediction_not_found",
        }

    sample_name = Path(cif_path).stem

    # Base result with metadata
    result: dict[str, Any] = {
        "designed_sample_id": designed_id,
        "input_sample_id": input_id,
        "diffusion_idx": diff_idx,
        "cif_path": cif_path,
    }
    # Carry forward AF3 metrics
    for key in [
        "sc_ca_rmsd", "avg_ca_plddt", "ligand_rmsd", "ligand_plddt",
        "binding_site_rmsd", "binding_site_plddt", "iptm",
    ]:
        if key in row:
            result[key] = row[key]

    # ========== Shared preprocessing ==========
    prep_dir = str(out_path / "prep" / sample_name)
    try:
        Path(prep_dir).mkdir(parents=True, exist_ok=True)
        sample_info = preprocess_structure(
            cif_path=cif_path,
            out_dir=prep_dir,
            sample_id=sample_name,
            cif_parse_cfg=cif_parse_cfg,
        )
        protein_pdb = sample_info["protein_pdb_path"]
        ligand_sdf = sample_info["ligand_sdf_path"]
    except Exception as e:
        logger.error(f"[{designed_id}] Preprocessing failed: {e}")
        result["error"] = f"preprocessing_failed: {e}"
        return result

    # ========== Phase 1: PB on AF3 raw ==========
    try:
        logger.info(f"[{designed_id}] Phase 1: PB on AF3 raw")
        pb_af3 = _run_pb_and_summarize(
            mol_pred=ligand_sdf,
            mol_cond=protein_pdb,
            pb_cfg=pb_cfg,
            prefix="pb_af3",
        )
        result.update(pb_af3)
    except Exception as e:
        logger.error(f"[{designed_id}] Phase 1 PB failed: {e}")
        result["pb_af3_error"] = str(e)

    # ========== Phase 2: Glide ==========
    inplace_sdf = None
    redock_sdf = None
    modes = glide_cfg.get("modes", {})

    if schrodinger_path and glide_cfg.get("enabled", True):
        try:
            # PrepWizard
            logger.info(f"[{designed_id}] Phase 2: PrepWizard")
            receptor_mae = str(Path(prep_dir) / f"{sample_name}_protein_prepared.mae")
            run_prepwizard(
                input_file=protein_pdb,
                output_file=receptor_mae,
                schrodinger_path=schrodinger_path,
                options=schrodinger_cfg.get("prepwizard", {}),
                timeout=timeout,
                log_dir=schrodinger_cfg.get("log_dir"),
            )

            # LigPrep
            ligand_file = ligand_sdf
            if glide_cfg.get("use_ligprep", True):
                try:
                    logger.info(f"[{designed_id}] Phase 2: LigPrep")
                    prepared_ligand = str(
                        Path(prep_dir) / f"{sample_name}_ligand_prepared.maegz"
                    )
                    ligand_file = run_ligprep(
                        input_sdf=ligand_sdf,
                        output_file=prepared_ligand,
                        schrodinger_path=schrodinger_path,
                        options=schrodinger_cfg.get("ligprep", {}),
                        timeout=timeout,
                        log_dir=schrodinger_cfg.get("log_dir"),
                    )
                except Exception as e:
                    logger.warning(
                        f"[{designed_id}] LigPrep failed, using raw SDF: {e}"
                    )

            # Grid generation
            logger.info(f"[{designed_id}] Phase 2: Grid generation")
            grid_cfg = glide_cfg.get("grid", {})
            outer_box = grid_cfg.get("outer_box")
            if outer_box is None:
                outer_box = compute_dynamic_outerbox(
                    sample_info["ligand_atom_array"]
                )

            gridgen_input = write_gridgen_input(
                receptor_mae=receptor_mae,
                grid_center=sample_info["ligand_centroid"].tolist(),
                out_dir=prep_dir,
                jobname="gridgen",
                inner_box=grid_cfg.get("inner_box", [10, 10, 10]),
                outer_box=outer_box,
                forcefield=grid_cfg.get("forcefield", "OPLS4"),
            )
            grid_file = run_grid_generation(
                input_file=gridgen_input,
                schrodinger_path=schrodinger_path,
                timeout=timeout,
            )

            # Mininplace
            if modes.get("inplace_scoring", True):
                try:
                    logger.info(f"[{designed_id}] Phase 2: Mininplace")
                    ip_metrics = _run_docking_mode(
                        grid_file=grid_file,
                        ligand_file=ligand_file,
                        work_dir=str(out_path / "mininplace" / sample_name),
                        schrodinger_path=schrodinger_path,
                        glide_cfg=glide_cfg,
                        mode="mininplace",
                        timeout=timeout,
                    )
                    for k, v in ip_metrics.items():
                        result[f"inplace_{k}"] = v
                    inplace_sdf = ip_metrics.get("sdf_path")
                except Exception as e:
                    logger.error(f"[{designed_id}] Mininplace failed: {e}")
                    result["inplace_error"] = str(e)

            # Redocking
            if modes.get("redocking", True):
                try:
                    logger.info(f"[{designed_id}] Phase 2: Redocking")
                    rd_metrics = _run_docking_mode(
                        grid_file=grid_file,
                        ligand_file=ligand_file,
                        work_dir=str(out_path / "redocking" / sample_name),
                        schrodinger_path=schrodinger_path,
                        glide_cfg=glide_cfg,
                        mode="redocking",
                        timeout=timeout,
                    )
                    for k, v in rd_metrics.items():
                        result[f"redock_{k}"] = v
                    redock_sdf = rd_metrics.get("sdf_path")
                except Exception as e:
                    logger.error(f"[{designed_id}] Redocking failed: {e}")
                    result["redock_error"] = str(e)

        except Exception as e:
            logger.error(f"[{designed_id}] Phase 2 Glide prep failed: {e}")
            result["glide_error"] = str(e)

    # ========== Phase 3: PB on Glide outputs ==========
    if inplace_sdf:
        try:
            logger.info(f"[{designed_id}] Phase 3: PB on mininplace")
            pb_ip = _run_pb_and_summarize(
                mol_pred=inplace_sdf,
                mol_cond=protein_pdb,
                pb_cfg=pb_cfg,
                prefix="pb_mininplace",
            )
            result.update(pb_ip)
        except Exception as e:
            logger.error(f"[{designed_id}] Phase 3 PB mininplace failed: {e}")
            result["pb_mininplace_error"] = str(e)

    if redock_sdf:
        try:
            logger.info(f"[{designed_id}] Phase 3: PB on redocking")
            pb_rd = _run_pb_and_summarize(
                mol_pred=redock_sdf,
                mol_cond=protein_pdb,
                pb_cfg=pb_cfg,
                prefix="pb_redocking",
            )
            result.update(pb_rd)
        except Exception as e:
            logger.error(f"[{designed_id}] Phase 3 PB redocking failed: {e}")
            result["pb_redocking_error"] = str(e)

    logger.info(f"[{designed_id}] Done")
    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    selected_df: pd.DataFrame,
    af3_preds_dir: str,
    sample_dir: str,
    output_dir: str,
    schrodinger_cfg: dict[str, Any],
    glide_cfg: dict[str, Any],
    pb_cfg: dict[str, Any],
    cif_parse_cfg: dict | None = None,
    seed: int = 42,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Run ligand evaluation for all selected samples."""
    output_path = Path(output_dir)
    for d in ["prep", "mininplace", "redocking"]:
        (output_path / d).mkdir(parents=True, exist_ok=True)

    # Schrodinger path (None if not available → Glide phases skipped)
    schrodinger_path = None
    raw_path = schrodinger_cfg.get("schrodinger_path")
    if raw_path:
        try:
            schrodinger_path = find_schrodinger(raw_path)
        except Exception as e:
            logger.warning(f"Schrodinger not found ({e}), Glide phases disabled")

    total = len(selected_df)
    rows = selected_df.to_dict("records")

    common_kwargs = dict(
        af3_preds_dir=af3_preds_dir,
        sample_dir=sample_dir,
        output_dir=str(output_dir),
        schrodinger_path=schrodinger_path,
        schrodinger_cfg=schrodinger_cfg,
        glide_cfg=glide_cfg,
        pb_cfg=pb_cfg,
        cif_parse_cfg=cif_parse_cfg,
        seed=seed,
    )

    results: list[dict[str, Any]] = []

    if num_workers <= 1:
        for i, row in enumerate(rows):
            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"[{i + 1}/{total}] {row['designed_sample_id']} "
                f"(diffusion_{int(row.get('diffusion_idx', 0))})"
            )
            logger.info(f"{'=' * 60}")
            result = process_single_sample(row=row, **common_kwargs)
            results.append(result)
    else:
        logger.info(f"Running with {num_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for i, row in enumerate(rows):
                future = executor.submit(
                    process_single_sample, row=row, **common_kwargs,
                )
                futures[future] = (i, row["designed_sample_id"])

            for future in as_completed(futures):
                idx, designed_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    n_done = len(results)
                    logger.info(f"[{n_done}/{total}] {designed_id} complete")
                except Exception as e:
                    logger.error(f"[{designed_id}] Worker failed: {e}")
                    results.append({
                        "designed_sample_id": designed_id,
                        "error": f"worker_failed: {e}",
                    })

    return pd.DataFrame(results)


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

    # Only keep rows that have at least one non-null phase column
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

    # 1. Unified CSV (all columns)
    unified_path = out / f"ligand_eval_metrics{sfx}{arr}.csv"
    results_df.to_csv(str(unified_path), index=False)
    logger.info(f"Unified: {unified_path.name} ({len(results_df)} rows)")

    # 2. Per-phase CSVs
    _save_phase_csv(results_df, out, "pb_af3", sfx)
    _save_phase_csv(results_df, out, "inplace", sfx)
    _save_phase_csv(results_df, out, "redock", sfx)
    _save_phase_csv(results_df, out, "pb_mininplace", sfx)
    _save_phase_csv(results_df, out, "pb_redocking", sfx)

    # 3. Failed samples CSV
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

    # Save config
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
        # Retry mode: load failed samples CSV
        print(f"Retry mode: loading {retry_csv}")
        selected = pd.read_csv(retry_csv)
        # Ensure required columns
        for col in ["designed_sample_id", "diffusion_idx"]:
            if col not in selected.columns:
                print(f"ERROR: retry CSV missing column '{col}'")
                return
        print(f"Retrying {len(selected)} samples")
    else:
        # Normal mode: load AF3 metrics and select
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

    # Debug mode
    debug = cfg_dict.get("debug", False)
    num_debug_samples = cfg_dict.get("num_debug_samples", 10)
    if debug and len(selected) > num_debug_samples:
        print(f"Debug mode: limiting to {num_debug_samples} samples")
        selected = selected.head(num_debug_samples)

    # Array job splitting
    selected = split_for_array_job(selected)
    if selected.empty:
        print("No samples for this array task. Exiting.")
        return

    # Save selection
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
        ("inplace_best_docking_score", "Inplace DockingScore (mean)"),
        ("redock_best_docking_score", "Redock DockingScore (mean)"),
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
                    print(f"  {label}: {valid.mean():.3f}")

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
