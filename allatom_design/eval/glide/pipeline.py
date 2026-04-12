"""Per-sample ligand evaluation pipeline (PoseBusters + Glide).

This module hosts the stateless per-sample and batch entrypoints that the
Hydra CLI in :mod:`allatom_design.eval.glide.run_ligand_eval_batch` drives.

Per-sample flow
---------------
1. Shared preprocessing (CIF → protein PDB + ligand SDF).
2. Phase 1: PoseBusters on the AF3 raw ligand pose.
3. Phase 2: Glide prep (PrepWizard + LigPrep + grid) → mininplace + redocking.
4. Phase 3: PoseBusters on the Glide output SDFs.
5. Reference RMSD: symmetry-corrected RMSD between the redocked pose and the
   original sample ligand (``redock_vs_ref_ligand_rmsd``).

Each phase is independent; a failure in one phase is recorded in the result
dict but does not block later phases.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.glide.preprocessing import (
    compute_dynamic_outerbox,
    get_ligand_pn_unit_iids,
    preprocess_structure,
)
from allatom_design.eval.glide.result_parser import (
    compute_redock_vs_reference_rmsd,
    extract_best_scores,
    parse_glide_csv,
)
from allatom_design.eval.glide.sample_selection import find_af3_prediction_path
from allatom_design.eval.glide.schrodinger_runner import (
    find_schrodinger,
    run_glide,
    run_grid_generation,
    run_ligprep,
    run_prepwizard,
    write_docking_input,
    write_gridgen_input,
)
from allatom_design.eval.posebusters.core import add_pb_valid, run_pb_single
from allatom_design.utils.sample_io_utils import load_example_with_parse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _run_pb_and_summarize(
    mol_pred: str,
    mol_cond: str,
    pb_cfg: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Run PoseBusters ``bust()`` and return a flat dict with prefixed columns."""
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


def _run_docking_mode(
    grid_file: str,
    ligand_file: str,
    work_dir: str,
    schrodinger_path: str,
    glide_cfg: dict[str, Any],
    mode: str,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run one Glide docking mode (mininplace or redocking) and return metrics."""
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


def _compute_redock_vs_ref_rmsd(
    sample_cif: Path,
    cif_parse_cfg: dict | DictConfig | None,
    redock_sdf_path: str,
    designed_id: str,
) -> float | None:
    """Symmetry-corrected RMSD between the redocked pose and the sample ligand.

    The sample CIF is parsed with the same ``cif_parse_cfg`` used elsewhere in
    the pipeline; its ligand subset is handed to
    :func:`compute_redock_vs_reference_rmsd`, which maps atoms via RDKit
    substructure matching and falls back to :func:`AllChem.GetBestRMS`.

    Returns ``None`` on any failure (missing CIF, parse error, RDKit conversion
    error, etc.) — callers should treat a missing column as "not computable"
    rather than an error.
    """
    if not sample_cif.exists():
        logger.warning(
            f"[{designed_id}] Redock vs ref RMSD skipped: sample CIF not found "
            f"({sample_cif})"
        )
        return None

    try:
        ref_cfg = cif_parse_cfg
        if ref_cfg is not None and not isinstance(ref_cfg, DictConfig):
            ref_cfg = OmegaConf.create(ref_cfg)
        ref_example = load_example_with_parse(
            str(sample_cif), cif_parse_cfg=ref_cfg,
        )
        ref_array = ref_example["atom_array"]
        ref_lig_ids = get_ligand_pn_unit_iids(ref_array)
        if not ref_lig_ids:
            logger.warning(
                f"[{designed_id}] Redock vs ref RMSD skipped: no ligand in "
                f"sample CIF"
            )
            return None

        import numpy as np

        ref_lig = ref_array[np.isin(ref_array.pn_unit_iid, ref_lig_ids)]
        rmsd_result = compute_redock_vs_reference_rmsd(
            redock_sdf_path=redock_sdf_path,
            ref_ligand_array=ref_lig,
        )
        return rmsd_result.get("redock_vs_ref_ligand_rmsd")
    except Exception as e:
        logger.warning(f"[{designed_id}] Redock vs ref RMSD failed: {e}")
        return None


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
    """Run the full per-sample pipeline; stateless for ``ProcessPoolExecutor``."""
    designed_id = row["designed_sample_id"]
    input_id = row.get("input_sample_id", "")
    diff_idx = int(row.get("diffusion_idx", 0))

    out_path = Path(output_dir)
    timeout = schrodinger_cfg.get("timeout", 3600)

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

    result: dict[str, Any] = {
        "designed_sample_id": designed_id,
        "input_sample_id": input_id,
        "diffusion_idx": diff_idx,
        "cif_path": cif_path,
    }
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
    inplace_sdf: str | None = None
    redock_sdf: str | None = None
    modes = glide_cfg.get("modes", {})

    if schrodinger_path and glide_cfg.get("enabled", True):
        try:
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

    # ========== Redock vs reference RMSD ==========
    if redock_sdf:
        rmsd_val = _compute_redock_vs_ref_rmsd(
            sample_cif=Path(sample_dir) / f"{input_id}.cif",
            cif_parse_cfg=cif_parse_cfg,
            redock_sdf_path=redock_sdf,
            designed_id=designed_id,
        )
        if rmsd_val is not None:
            result["redock_vs_ref_ligand_rmsd"] = rmsd_val

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
    """Evaluate all selected samples, optionally in parallel."""
    output_path = Path(output_dir)
    for d in ("prep", "mininplace", "redocking"):
        (output_path / d).mkdir(parents=True, exist_ok=True)

    schrodinger_path: str | None = None
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
            logger.info("\n" + "=" * 60)
            logger.info(
                f"[{i + 1}/{total}] {row['designed_sample_id']} "
                f"(diffusion_{int(row.get('diffusion_idx', 0))})"
            )
            logger.info("=" * 60)
            results.append(process_single_sample(row=row, **common_kwargs))
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
                    logger.info(f"[{len(results)}/{total}] {designed_id} complete")
                except Exception as e:
                    logger.error(f"[{designed_id}] Worker failed: {e}")
                    results.append({
                        "designed_sample_id": designed_id,
                        "error": f"worker_failed: {e}",
                    })

    return pd.DataFrame(results)
