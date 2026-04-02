"""Glide evaluation pipeline orchestration.

Coordinates preprocessing, Schrodinger tools, and result collection
for single samples and batches.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.glide.preprocessing import (
    compute_dynamic_outerbox,
    preprocess_structure,
)
from allatom_design.eval.glide.schrodinger_runner import (
    find_schrodinger,
    run_prepwizard,
    run_grid_generation,
    run_ligprep,
    run_glide,
    write_gridgen_input,
    write_docking_input,
)
from allatom_design.eval.glide.result_parser import (
    parse_glide_csv,
    extract_best_scores,
)

logger = logging.getLogger(__name__)


def evaluate_single_sample(
    cif_path: str,
    work_dir: str,
    schrodinger_cfg: dict[str, Any],
    glide_cfg: dict[str, Any],
    cif_parse_cfg: dict | None = None,
    reference_cif_path: str | None = None,
    receptor_pn_unit_iids: list[str] | None = None,
    ligand_pn_unit_iids: list[str] | None = None,
) -> dict[str, Any]:
    """Run Glide evaluation for a single AF3 predicted structure.

    Steps:
        1. Preprocess: CIF -> protein PDB + ligand SDF
        2. PrepWizard: PDB -> prepared MAE
        3. Grid generation: MAE + ligand centroid -> grid .zip
        4. LigPrep (optional): SDF -> prepared SDF
        5. In-place scoring (if enabled)
        6. Re-docking (if enabled)
        7. RMSD vs reference (if reference provided)

    Args:
        cif_path: Path to AF3 predicted CIF file.
        work_dir: Working directory for this sample.
        schrodinger_cfg: Schrodinger tool configuration.
        glide_cfg: Glide evaluation configuration.
        cif_parse_cfg: Config for CIF parser.
        reference_cif_path: Path to reference structure for RMSD comparison.
        receptor_pn_unit_iids: Protein chain IDs.
        ligand_pn_unit_iids: Ligand chain IDs.

    Returns:
        Dict with sample_id and all computed metrics.
    """
    sample_id = Path(cif_path).stem
    work_dir = str(Path(work_dir) / sample_id)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {"sample_id": sample_id}
    schrodinger_path = find_schrodinger(schrodinger_cfg.get("schrodinger_path"))

    # ------------------------------------------------------------------
    # Step 1: Preprocess - separate protein and ligand
    # ------------------------------------------------------------------
    logger.info(f"[{sample_id}] Step 1: Preprocessing")
    sample_info = preprocess_structure(
        cif_path=cif_path,
        out_dir=work_dir,
        sample_id=sample_id,
        cif_parse_cfg=cif_parse_cfg,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
    )
    metrics["receptor_pn_unit_iids"] = sample_info["receptor_pn_unit_iids"]
    metrics["ligand_pn_unit_iids"] = sample_info["ligand_pn_unit_iids"]
    metrics["ligand_centroid"] = sample_info["ligand_centroid"].tolist()

    # ------------------------------------------------------------------
    # Step 2: PrepWizard - prepare protein
    # ------------------------------------------------------------------
    logger.info(f"[{sample_id}] Step 2: PrepWizard")
    receptor_mae = str(Path(work_dir) / f"{sample_id}_protein_prepared.mae")
    prepwizard_opts = schrodinger_cfg.get("prepwizard", {})
    run_prepwizard(
        input_file=sample_info["protein_pdb_path"],
        output_file=receptor_mae,
        schrodinger_path=schrodinger_path,
        options=prepwizard_opts,
        timeout=schrodinger_cfg.get("timeout", 3600),
        log_dir=schrodinger_cfg.get("log_dir"),
    )

    # ------------------------------------------------------------------
    # Step 3: Grid generation
    # ------------------------------------------------------------------
    logger.info(f"[{sample_id}] Step 3: Grid generation")
    grid_cfg = glide_cfg.get("grid", {})
    outer_box = grid_cfg.get("outer_box")
    if outer_box is None:
        outer_box = compute_dynamic_outerbox(sample_info["ligand_atom_array"])
    gridgen_input = write_gridgen_input(
        receptor_mae=receptor_mae,
        grid_center=sample_info["ligand_centroid"].tolist(),
        out_dir=work_dir,
        jobname="gridgen",
        inner_box=grid_cfg.get("inner_box", [10, 10, 10]),
        outer_box=outer_box,
        forcefield=grid_cfg.get("forcefield", "OPLS4"),
    )
    grid_file = run_grid_generation(
        input_file=gridgen_input,
        schrodinger_path=schrodinger_path,
        timeout=schrodinger_cfg.get("timeout", 3600),
    )

    # ------------------------------------------------------------------
    # Step 4: LigPrep (optional)
    # ------------------------------------------------------------------
    ligand_file = sample_info["ligand_sdf_path"]
    if glide_cfg.get("use_ligprep", False):
        logger.info(f"[{sample_id}] Step 4: LigPrep")
        prepared_ligand = str(Path(work_dir) / f"{sample_id}_ligand_prepared.maegz")
        ligprep_opts = schrodinger_cfg.get("ligprep", {})
        try:
            ligand_file = run_ligprep(
                input_sdf=sample_info["ligand_sdf_path"],
                output_file=prepared_ligand,
                schrodinger_path=schrodinger_path,
                options=ligprep_opts,
                timeout=schrodinger_cfg.get("timeout", 3600),
                log_dir=schrodinger_cfg.get("log_dir"),
            )
        except Exception as e:
            logger.warning(f"[{sample_id}] LigPrep failed, using raw SDF: {e}")

    # ------------------------------------------------------------------
    # Step 5: In-place scoring
    # ------------------------------------------------------------------
    modes = glide_cfg.get("modes", {})
    if modes.get("inplace_scoring", True):
        logger.info(f"[{sample_id}] Step 5: In-place scoring")
        try:
            inplace_metrics = _run_inplace_scoring(
                grid_file=grid_file,
                ligand_file=ligand_file,
                work_dir=work_dir,
                schrodinger_path=schrodinger_path,
                glide_cfg=glide_cfg,
                timeout=schrodinger_cfg.get("timeout", 3600),
            )
            for k, v in inplace_metrics.items():
                metrics[f"inplace_{k}"] = v
        except Exception as e:
            logger.error(f"[{sample_id}] In-place scoring failed: {e}")
            metrics["inplace_error"] = str(e)

    # ------------------------------------------------------------------
    # Step 6: Re-docking
    # ------------------------------------------------------------------
    if modes.get("redocking", False):
        logger.info(f"[{sample_id}] Step 6: Re-docking")
        try:
            redock_metrics = _run_redocking(
                grid_file=grid_file,
                ligand_file=ligand_file,
                work_dir=work_dir,
                schrodinger_path=schrodinger_path,
                glide_cfg=glide_cfg,
                timeout=schrodinger_cfg.get("timeout", 3600),
            )
            for k, v in redock_metrics.items():
                metrics[f"redock_{k}"] = v
        except Exception as e:
            logger.error(f"[{sample_id}] Re-docking failed: {e}")
            metrics["redock_error"] = str(e)

    # ------------------------------------------------------------------
    # Step 7: RMSD vs reference
    # ------------------------------------------------------------------
    if modes.get("rmsd_comparison", False) and reference_cif_path:
        logger.info(f"[{sample_id}] Step 7: RMSD comparison")
        try:
            rmsd_metrics = _compute_rmsd_vs_reference(
                pred_cif_path=cif_path,
                ref_cif_path=reference_cif_path,
                receptor_pn_unit_iids=sample_info["receptor_pn_unit_iids"],
                ligand_pn_unit_iids=sample_info["ligand_pn_unit_iids"],
                pocket_distance=glide_cfg.get("pocket_distance", 8.0),
                cif_parse_cfg=cif_parse_cfg,
            )
            for k, v in rmsd_metrics.items():
                metrics[f"rmsd_{k}"] = v
        except Exception as e:
            logger.error(f"[{sample_id}] RMSD comparison failed: {e}")
            metrics["rmsd_error"] = str(e)

    return metrics


def _run_inplace_scoring(
    grid_file: str,
    ligand_file: str,
    work_dir: str,
    schrodinger_path: str,
    glide_cfg: dict[str, Any],
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run Glide in-place scoring (no docking, score AF3 pose as-is)."""
    inplace_cfg = glide_cfg.get("inplace", {})

    dock_input = write_docking_input(
        gridfile=grid_file,
        ligandfile=ligand_file,
        out_dir=work_dir,
        jobname="dock_inplace",
        docking_method=inplace_cfg.get("docking_method", "mininplace"),
        precision=inplace_cfg.get("precision", "SP"),
        num_poses=1,
        write_csv=True,
        pose_outtype="ligandlib_sd",
        compress_poses=False,
        forcefield=inplace_cfg.get("forcefield", "OPLS4"),
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
            scores = extract_best_scores(df)
            metrics.update(scores)
        else:
            metrics["error"] = "empty_csv"
    else:
        metrics["error"] = "no_csv_output"

    return metrics


def _run_redocking(
    grid_file: str,
    ligand_file: str,
    work_dir: str,
    schrodinger_path: str,
    glide_cfg: dict[str, Any],
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run Glide re-docking (flexible docking to find best pose)."""
    redock_cfg = glide_cfg.get("redocking", {})

    dock_input = write_docking_input(
        gridfile=grid_file,
        ligandfile=ligand_file,
        out_dir=work_dir,
        jobname="dock_redock",
        docking_method=redock_cfg.get("docking_method", "confgen"),
        precision=redock_cfg.get("precision", "SP"),
        num_poses=redock_cfg.get("num_poses", 1),
        write_csv=True,
        pose_outtype="ligandlib_sd",
        compress_poses=False,
        forcefield=redock_cfg.get("forcefield", "OPLS4"),
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
            scores = extract_best_scores(df)
            metrics.update(scores)
            metrics["num_poses"] = len(df)
        else:
            metrics["error"] = "empty_csv"
    else:
        metrics["error"] = "no_csv_output"

    if outputs["sdf_path"]:
        metrics["sdf_path"] = outputs["sdf_path"]

    return metrics


def _compute_rmsd_vs_reference(
    pred_cif_path: str,
    ref_cif_path: str,
    receptor_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
    pocket_distance: float = 8.0,
    cif_parse_cfg: dict | None = None,
) -> dict[str, Any]:
    """Compute ligand RMSD between AF3 prediction and reference structure.

    Uses the existing binding-site-superposition RMSD from eval_metrics.
    """
    from allatom_design.eval.eval_utils.eval_metrics import (
        calculate_ligand_rmsd_with_binding_site_superposition,
    )
    from allatom_design.utils.sample_io_utils import load_example_with_parse

    pred_example = load_example_with_parse(pred_cif_path, cif_parse_cfg=cif_parse_cfg)
    ref_example = load_example_with_parse(ref_cif_path, cif_parse_cfg=cif_parse_cfg)

    result = calculate_ligand_rmsd_with_binding_site_superposition(
        pred_example=pred_example,
        sample_example=ref_example,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
        pocket_distance=pocket_distance,
        save_aligned=False,
    )

    metrics: dict[str, Any] = {}
    if result and "error" not in result:
        metrics["ligand_rmsd"] = result.get("ligand_rmsd")
        metrics["bs_rmsd"] = result.get("bs_rmsd")
    elif result:
        metrics["error"] = result.get("error", "unknown")

    return metrics


def run_glide_evaluation(
    sample_paths: list[str],
    cfg: DictConfig,
    log_dir: str,
    reference_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Run Glide evaluation for a batch of samples.

    Args:
        sample_paths: List of CIF file paths to evaluate.
        cfg: Full Hydra config.
        log_dir: Directory for output files.
        reference_map: Optional mapping from sample_id to reference CIF path.

    Returns:
        DataFrame with per-sample metrics.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg

    schrodinger_cfg = cfg_dict.get("schrodinger", {})
    glide_cfg = cfg_dict.get("glide", {})
    cif_parse_cfg = cfg_dict.get("cif_parse_cfg")
    receptor_pn_unit_iids = cfg_dict.get("receptor_pn_unit_iids")
    ligand_pn_unit_iids = cfg_dict.get("ligand_pn_unit_iids")

    work_dir = str(Path(log_dir) / "glide_work")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    results = []
    failed_samples = []

    for sample_path in tqdm(sample_paths, desc="Glide evaluation"):
        sample_id = Path(sample_path).stem
        ref_path = reference_map.get(sample_id) if reference_map else None

        try:
            metrics = evaluate_single_sample(
                cif_path=sample_path,
                work_dir=work_dir,
                schrodinger_cfg=schrodinger_cfg,
                glide_cfg=glide_cfg,
                cif_parse_cfg=cif_parse_cfg,
                reference_cif_path=ref_path,
                receptor_pn_unit_iids=receptor_pn_unit_iids,
                ligand_pn_unit_iids=ligand_pn_unit_iids,
            )
            results.append(metrics)
        except Exception as e:
            logger.error(f"Failed to evaluate {sample_id}: {e}")
            results.append({"sample_id": sample_id, "error": str(e)})
            failed_samples.append(sample_id)

    df = pd.DataFrame(results)

    # Save results
    results_csv = str(Path(log_dir) / "glide_results.csv")
    df.to_csv(results_csv, index=False)
    logger.info(f"Results saved: {results_csv} ({len(df)} samples, {len(failed_samples)} failed)")

    if failed_samples:
        failed_path = str(Path(log_dir) / "glide_failed_samples.txt")
        with open(failed_path, "w") as f:
            f.write("\n".join(failed_samples))
        logger.info(f"Failed sample list: {failed_path}")

    return df
