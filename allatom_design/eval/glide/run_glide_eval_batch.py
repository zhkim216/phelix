"""Batch Glide evaluation on AF3 predicted structures.

Workflow:
    1. Load AF3 metrics (docking + SC) and select best diffusion per design
    2. Copy AF3 predictions and sample CIFs to output structure
    3. For each selected sample: prep → mininplace + redocking
    4. Write combined metrics CSV

Usage:
    python -m allatom_design.eval.glide.run_glide_eval_batch
"""

import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.glide.preprocessing import (
    compute_dynamic_outerbox,
    get_ligand_pn_unit_iids,
    preprocess_structure,
)
from allatom_design.utils.sample_io_utils import load_example_with_parse
from allatom_design.eval.glide.result_parser import (
    compute_redock_vs_reference_rmsd,
    extract_best_scores,
    parse_glide_csv,
)
from allatom_design.eval.glide.sample_selection import (
    find_af3_prediction_path,
    load_af3_metrics,
    select_best_diffusion,
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

logger = logging.getLogger(__name__)


def _make_cutoff_suffix(sel_cfg: dict[str, Any]) -> str:
    """Build filename suffix from selection cutoffs, e.g. '_lplddt70_lrmsd2'."""
    plddt = sel_cfg.get("ligand_plddt_cutoff", 70.0)
    rmsd = sel_cfg.get("ligand_rmsd_cutoff", 2.0)

    def _fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else f"{v:.1f}"

    return f"_lplddt{_fmt(plddt)}_lrmsd{_fmt(rmsd)}"


def run_prep(
    cif_path: str,
    prep_dir: str,
    schrodinger_path: str,
    schrodinger_cfg: dict[str, Any],
    glide_cfg: dict[str, Any],
    cif_parse_cfg: dict | None = None,
    receptor_pn_unit_iids: list[str] | None = None,
    ligand_pn_unit_iids: list[str] | None = None,
) -> dict[str, Any]:
    """Run preparation steps: preprocess, PrepWizard, LigPrep, grid gen."""
    sample_id = Path(cif_path).stem
    Path(prep_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocess
    logger.info(f"[{sample_id}] Preprocessing")
    sample_info = preprocess_structure(
        cif_path=cif_path,
        out_dir=prep_dir,
        sample_id=sample_id,
        cif_parse_cfg=cif_parse_cfg,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
    )

    # Step 2: PrepWizard
    logger.info(f"[{sample_id}] PrepWizard")
    receptor_mae = str(Path(prep_dir) / f"{sample_id}_protein_prepared.mae")
    log_dir = schrodinger_cfg.get("log_dir")
    run_prepwizard(
        input_file=sample_info["protein_pdb_path"],
        output_file=receptor_mae,
        schrodinger_path=schrodinger_path,
        options=schrodinger_cfg.get("prepwizard", {}),
        timeout=schrodinger_cfg.get("timeout", 3600),
        log_dir=log_dir,
    )

    # Step 3: LigPrep
    logger.info(f"[{sample_id}] LigPrep")
    prepared_ligand = str(Path(prep_dir) / f"{sample_id}_ligand_prepared.maegz")
    ligand_file = sample_info["ligand_sdf_path"]
    if glide_cfg.get("use_ligprep", True):
        try:
            ligand_file = run_ligprep(
                input_sdf=sample_info["ligand_sdf_path"],
                output_file=prepared_ligand,
                schrodinger_path=schrodinger_path,
                options=schrodinger_cfg.get("ligprep", {}),
                timeout=schrodinger_cfg.get("timeout", 3600),
                log_dir=log_dir,
            )
        except Exception as e:
            logger.warning(f"[{sample_id}] LigPrep failed, using raw SDF: {e}")

    # Step 4: Grid generation
    logger.info(f"[{sample_id}] Grid generation")
    grid_cfg = glide_cfg.get("grid", {})
    outer_box = grid_cfg.get("outer_box")
    if outer_box is None:
        outer_box = compute_dynamic_outerbox(sample_info["ligand_atom_array"])

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
        timeout=schrodinger_cfg.get("timeout", 3600),
    )

    return {
        "sample_info": sample_info,
        "receptor_mae": receptor_mae,
        "ligand_file": ligand_file,
        "grid_file": grid_file,
    }


def run_docking(
    grid_file: str,
    ligand_file: str,
    work_dir: str,
    schrodinger_path: str,
    glide_cfg: dict[str, Any],
    mode: str,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run a single docking mode (mininplace or redocking)."""
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


def process_single_sample(
    row: dict[str, Any],
    af3_preds_dir: str,
    sample_dir: str,
    output_dir: str,
    schrodinger_path: str,
    schrodinger_cfg: dict[str, Any],
    glide_cfg: dict[str, Any],
    cif_parse_cfg: dict | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Process a single sample: prep → mininplace + redocking.

    Designed to be called from ProcessPoolExecutor.
    """
    designed_id = row["designed_sample_id"]
    input_id = row["input_sample_id"]
    diff_idx = int(row["diffusion_idx"])

    output_path = Path(output_dir)
    af3_pred_out = output_path / "af3_predictions"
    samples_out = output_path / "samples"
    prep_out = output_path / "prep"
    mininplace_out = output_path / "mininplace"
    redocking_out = output_path / "redocking"

    modes = glide_cfg.get("modes", {})
    timeout = schrodinger_cfg.get("timeout", 3600)

    # Find AF3 prediction CIF
    cif_path = find_af3_prediction_path(
        af3_preds_dir, designed_id, diff_idx, seed=seed,
    )
    if not cif_path:
        return {
            "input_sample_id": input_id,
            "designed_sample_id": designed_id,
            "chosen_diffusion_idx": diff_idx,
            "error": "af3_prediction_not_found",
        }

    sample_name = Path(cif_path).stem

    # Copy AF3 prediction
    dst = af3_pred_out / Path(cif_path).name
    if not dst.exists():
        shutil.copy2(cif_path, dst)

    # Copy sample CIF (original input sample)
    sample_cif = Path(sample_dir) / f"{input_id}.cif"
    if sample_cif.exists():
        dst = samples_out / sample_cif.name
        if not dst.exists():
            shutil.copy2(sample_cif, dst)

    # Run prep (shared)
    try:
        prep_result = run_prep(
            cif_path=cif_path,
            prep_dir=str(prep_out / sample_name),
            schrodinger_path=schrodinger_path,
            schrodinger_cfg=schrodinger_cfg,
            glide_cfg=glide_cfg,
            cif_parse_cfg=cif_parse_cfg,
        )
    except Exception as e:
        logger.error(f"[{designed_id}] Prep failed: {e}")
        return {
            "input_sample_id": input_id,
            "designed_sample_id": designed_id,
            "chosen_diffusion_idx": diff_idx,
            "error": f"prep_failed: {e}",
        }

    # Build result row with AF3 metrics
    result: dict[str, Any] = {
        "input_sample_id": input_id,
        "designed_sample_id": designed_id,
        "chosen_diffusion_idx": diff_idx,
        "sc_rmsd": row.get("sc_ca_rmsd"),
        "ca_plddt": row.get("avg_ca_plddt"),
        "ligand_rmsd": row.get("ligand_rmsd"),
        "ligand_plddt": row.get("ligand_plddt"),
        "binding_site_rmsd": row.get("binding_site_rmsd"),
        "binding_site_plddt": row.get("binding_site_plddt"),
        "iptm": row.get("iptm"),
    }

    # Mininplace
    if modes.get("inplace_scoring", True):
        try:
            logger.info(f"[{designed_id}] Mininplace scoring")
            inplace_metrics = run_docking(
                grid_file=prep_result["grid_file"],
                ligand_file=prep_result["ligand_file"],
                work_dir=str(mininplace_out / sample_name),
                schrodinger_path=schrodinger_path,
                glide_cfg=glide_cfg,
                mode="mininplace",
                timeout=timeout,
            )
            for k, v in inplace_metrics.items():
                result[f"inplace_{k}"] = v
        except Exception as e:
            logger.error(f"[{designed_id}] Mininplace failed: {e}")
            result["inplace_error"] = str(e)

    # Redocking
    if modes.get("redocking", True):
        try:
            logger.info(f"[{designed_id}] Redocking")
            redock_metrics = run_docking(
                grid_file=prep_result["grid_file"],
                ligand_file=prep_result["ligand_file"],
                work_dir=str(redocking_out / sample_name),
                schrodinger_path=schrodinger_path,
                glide_cfg=glide_cfg,
                mode="redocking",
                timeout=timeout,
            )
            for k, v in redock_metrics.items():
                result[f"redock_{k}"] = v

            # Compute RMSD between redocked pose and original sample ligand
            redock_sdf = redock_metrics.get("sdf_path")
            sample_cif = Path(sample_dir) / f"{input_id}.cif"
            if redock_sdf and sample_cif.exists():
                try:
                    logger.info(f"[{designed_id}] Redock vs reference RMSD")
                    ref_cfg = cif_parse_cfg
                    if ref_cfg is not None and not isinstance(ref_cfg, DictConfig):
                        ref_cfg = OmegaConf.create(ref_cfg)
                    ref_example = load_example_with_parse(
                        str(sample_cif), cif_parse_cfg=ref_cfg,
                    )
                    ref_array = ref_example["atom_array"]
                    ref_lig_ids = get_ligand_pn_unit_iids(ref_array)
                    if ref_lig_ids:
                        ref_lig = ref_array[
                            np.isin(ref_array.pn_unit_iid, ref_lig_ids)
                        ]
                        rmsd_result = compute_redock_vs_reference_rmsd(
                            redock_sdf_path=redock_sdf,
                            ref_ligand_array=ref_lig,
                        )
                        result["redock_vs_ref_ligand_rmsd"] = rmsd_result.get(
                            "redock_vs_ref_ligand_rmsd",
                        )
                except Exception as e:
                    logger.warning(
                        f"[{designed_id}] Redock vs ref RMSD failed: {e}"
                    )
        except Exception as e:
            logger.error(f"[{designed_id}] Redocking failed: {e}")
            result["redock_error"] = str(e)

    logger.info(f"[{designed_id}] Done")
    return result


def evaluate_batch(
    selected_df: pd.DataFrame,
    af3_preds_dir: str,
    sample_dir: str,
    output_dir: str,
    schrodinger_cfg: dict[str, Any],
    glide_cfg: dict[str, Any],
    cif_parse_cfg: dict | None = None,
    seed: int = 42,
    num_workers: int = 1,
    cutoff_suffix: str = "",
) -> pd.DataFrame:
    """Run Glide evaluation for all selected samples."""
    output_path = Path(output_dir)
    for d in ["af3_predictions", "samples", "prep", "mininplace", "redocking"]:
        (output_path / d).mkdir(parents=True, exist_ok=True)

    schrodinger_path = find_schrodinger(schrodinger_cfg.get("schrodinger_path"))
    total = len(selected_df)
    rows = selected_df.to_dict("records")

    common_kwargs = dict(
        af3_preds_dir=af3_preds_dir,
        sample_dir=sample_dir,
        output_dir=str(output_dir),
        schrodinger_path=schrodinger_path,
        schrodinger_cfg=schrodinger_cfg,
        glide_cfg=glide_cfg,
        cif_parse_cfg=cif_parse_cfg,
        seed=seed,
    )

    results: list[dict[str, Any]] = []

    if num_workers <= 1:
        # Sequential
        for i, row in enumerate(rows):
            logger.info(f"\n{'='*60}")
            logger.info(
                f"[{i+1}/{total}] {row['designed_sample_id']} "
                f"(diffusion_{int(row['diffusion_idx'])})"
            )
            logger.info(f"{'='*60}")
            result = process_single_sample(row=row, **common_kwargs)
            results.append(result)
    else:
        # Parallel
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
                    logger.info(
                        f"[{n_done}/{total}] {designed_id} complete"
                    )
                except Exception as e:
                    logger.error(f"[{designed_id}] Worker failed: {e}")
                    results.append({
                        "designed_sample_id": designed_id,
                        "error": f"worker_failed: {e}",
                    })

    df = pd.DataFrame(results)
    csv_name = f"all_glide_metrics_per_designed_sample{cutoff_suffix}.csv"
    csv_path = str(output_path / csv_name)
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved: {csv_path} ({len(df)} samples)")

    return df


@hydra.main(
    config_path="../../configs_local/eval/glide",
    config_name="run_glide_eval_batch",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    """Run batch Glide evaluation on AF3 predicted structures."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    af3_eval_dir = Path(cfg.af3_eval_dir)
    output_dir = Path(cfg.output_dir)

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sel_cfg = cfg_dict.get("selection", {})
    cutoff_suffix = _make_cutoff_suffix(sel_cfg)

    # ================================================================
    # Phase 1: Load metrics and select samples
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Loading AF3 metrics and selecting samples")
    print("=" * 60 + "\n")

    docking_csv = str(af3_eval_dir / "all_docking_metrics_per_designed_sample.csv")
    sc_csv = str(af3_eval_dir / "all_sc_metrics_per_designed_sample.csv")

    flat_df = load_af3_metrics(docking_csv, sc_csv)
    print(f"Loaded {len(flat_df)} total (designed_sample, diffusion) pairs")

    selected = select_best_diffusion(
        flat_df,
        ligand_rmsd_cutoff=sel_cfg.get("ligand_rmsd_cutoff", 2.0),
        ligand_plddt_cutoff=sel_cfg.get("ligand_plddt_cutoff", 70.0),
    )
    print(f"Selected {len(selected)} samples after cutoff filtering")

    if selected.empty:
        print("No samples selected. Exiting.")
        return

    # Debug mode: limit to num_debug_samples
    debug = cfg_dict.get("debug", False)
    num_debug_samples = cfg_dict.get("num_debug_samples", 10)
    if debug and len(selected) > num_debug_samples:
        print(f"Debug mode: limiting to {num_debug_samples} samples")
        selected = selected.head(num_debug_samples)

    # Save selection
    selected.to_csv(
        str(output_dir / f"selected_samples{cutoff_suffix}.csv"), index=False,
    )

    # ================================================================
    # Phase 2: Run Glide evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Running Glide evaluation")
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
        cif_parse_cfg=cfg_dict.get("cif_parse_cfg"),
        seed=sel_cfg.get("seed", 42),
        num_workers=num_workers,
        cutoff_suffix=cutoff_suffix,
    )

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results: {output_dir}/all_glide_metrics_per_designed_sample{cutoff_suffix}.csv")
    print(f"Total samples: {len(results_df)}")

    for col in ["inplace_best_docking_score", "redock_best_docking_score"]:
        if col in results_df.columns:
            valid = results_df[col].dropna()
            if len(valid) > 0:
                print(f"{col}: mean={valid.mean():.3f}, median={valid.median():.3f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
