import gc
import glob
import logging
import os
import shutil
import subprocess
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_cached_example_files, get_pdb_files, get_training_checkpoints, wandb_setup)
from allatom_design.eval.eval_utils.seq_des_utils import (
    get_seq_des_model, run_lc_seq_des)
from allatom_design.eval.eval_utils.eval_metrics import (
    compute_self_consistency_metrics_atomworks,
    compute_template_conditioned_docking_metrics,
)
from allatom_design.eval.eval_utils.folding_utils import (
    _chain_letters, make_af3_json, run_af3_single_sequence, 
    run_af3_template_conditioned, find_pred_sample_path_af3
)
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def load_outputs_from_samples_dir(samples_dir: str, metadata: pd.DataFrame = None) -> dict:
    """
    Load outputs from an existing samples directory (from a previous sampling run).
    Uses sample_metadata.pt if available. Loads atom_arrays from .cif files using load_any
    (which preserves annotations like pn_unit_iid needed for make_af3_json).
    
    Args:
        samples_dir: Path to the samples/ directory containing .cif files
        metadata: Optional metadata DataFrame
        
    Returns:
        outputs dict with keys: example_id, out_pdb, out_pdb_for_af3_tc, atom_array, U, sample_seq_recovery, sample_sp_seq_recovery
    """
    outputs = defaultdict(list)
    samples_dir = Path(samples_dir)
    
    # Load sample_metadata.pt if it exists
    metadata_path = samples_dir / "sample_metadata.pt"
    sample_metadata = None
    if metadata_path.exists():
        sample_metadata = torch.load(metadata_path, weights_only=False)
        print(f"Loaded sample_metadata.pt with {len(sample_metadata)} samples")
    
    # Find all .cif files in the samples directory (exclude _for_af3_tc.cif files)
    all_cif_files = sorted(glob.glob(os.path.join(samples_dir, "*.cif")))
    cif_files = [f for f in all_cif_files if not f.endswith("_for_af3_tc.cif")]
    
    if len(cif_files) == 0:
        raise ValueError(f"No .cif files found in {samples_dir}")
    
    print(f"Loading {len(cif_files)} samples from {samples_dir}")
    
    for cif_file in cif_files:
        sample_stem = Path(cif_file).stem
        
        # Get metadata from sample_metadata.pt if available
        if sample_metadata and sample_stem in sample_metadata:
            meta = sample_metadata[sample_stem]
            example_id = meta["example_id"]
            outputs["U"].append(meta.get("U"))
            outputs["sample_seq_recovery"].append(meta.get("sample_seq_recovery"))
            outputs["sample_sp_seq_recovery"].append(meta.get("sample_sp_seq_recovery"))
        else:
            # Fallback: parse example_id from sample stem
            parts = sample_stem.rsplit("_sample", 1)
            example_id = parts[0] if len(parts) == 2 else sample_stem
            outputs["U"].append(None)
            outputs["sample_seq_recovery"].append(None)
            outputs["sample_sp_seq_recovery"].append(None)
        
        # Load atom_array from .cif file using load_any (preserves pn_unit_iid annotation)
        try:
            atom_array = load_any(cif_file, extra_fields="all")
            # load_any may return AtomArrayStack, extract first array if so
            if isinstance(atom_array, AtomArrayStack):
                atom_array = atom_array[0]
            outputs["atom_array"].append(atom_array)
        except Exception as e:
            print(f"Warning: Failed to load atom_array from {cif_file}: {e}")
            outputs["atom_array"].append(None)
        
        outputs["example_id"].append(example_id)
        outputs["out_pdb"].append(str(cif_file))
        
        # Check for corresponding _for_af3_tc.cif file
        af3_tc_cif_file = cif_file.replace(".cif", "_for_af3_tc.cif")
        if os.path.exists(af3_tc_cif_file):
            outputs["out_pdb_for_af3_tc"].append(af3_tc_cif_file)
        else:
            outputs["out_pdb_for_af3_tc"].append(None)
    
    # Set placeholder values for total seq recovery metrics
    outputs["total_avg_seq_recovery"] = None
    outputs["total_avg_sp_seq_recovery"] = None
    
    print(f"Loaded {len(outputs['out_pdb'])} samples")
    
    return outputs


@hydra.main(config_path="../configs_local/eval", config_name="eval_lc_seq_des_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Sequence denoiser training eval with AF3 self-consistency.

    Two-phase approach to avoid PyTorch/JAX memory conflicts:
    - Phase 1: Sample sequences on processed input structures (PyTorch)
    - Phase 2: Run AF3 predictions and compute metrics (JAX)
    
    This separation ensures PyTorch models are fully unloaded before JAX allocates GPU memory.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Seeds and determinism
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.protein_only:
        cfg.exp_name = f"{cfg.exp_name}_protein_only"

    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"
        path_base_out_dir = Path(cfg.base_out_dir)
        cfg.base_out_dir = str(path_base_out_dir.parent) + "/debug_" + str(path_base_out_dir.stem)

    # Logging / output root
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)

    # Load in metadata
    metadata = pd.read_parquet(cfg.metadata_path)    
    pdb_keys = metadata['pdb_id'].tolist()
    
    if cfg.debug:
        pdb_keys = pdb_keys[:cfg.num_sample_debug]

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)
    
    # Load in PDB file to eval on
    if cfg.data_cfg_for_design.load_from_cache:
        pdb_files = get_cached_example_files(cached_example_path=cfg.data_cfg_for_design.load_cache_cfg.cached_example_path, 
                                             pdb_name_list=pdb_keys, 
                                             pdb_name_ext=cfg.data_cfg_for_design.load_cache_cfg.pdb_name_ext, 
                                             n_subsample=cfg.data_cfg_for_design.load_cache_cfg.n_subsample)
    else:
        pdb_files = get_pdb_files(pdb_dir=cfg.data_cfg_for_design.pdb_cfg.pdb_dir, pdb_name_list=pdb_keys, 
                              pdb_name_ext=cfg.data_cfg_for_design.pdb_cfg.pdb_name_ext, 
                              n_subsample=cfg.data_cfg_for_design.pdb_cfg.n_subsample)

    # Device
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Denoiser checkpoints
    sd_ckpts, pattern = get_training_checkpoints(cfg.denoiser_train_dir, "seq_denoiser",
                                                 cfg.eval_every_n_ckpts, cfg.start_step, cfg.end_step, cfg.use_ema,
                                                 cfg.get("eval_last_ckpt", True))

    # AF3 config defaults
    af3_runner_path = cfg.struct_pred_cfg.af3.get("runner_path", None)
    if af3_runner_path is None:
        raise ValueError("af3_runner_path is not set")
    af3_model_seeds = list(cfg.struct_pred_cfg.af3.json_config.get("model_seeds", [42]))

    # Check if we should skip sampling
    skip_sampling = cfg.get("skip_sampling", False)
    skip_sampling_dir = cfg.get("skip_sampling_dir", None)

    # Build checkpoint info list for both phases
    ckpt_info_list = []
    for sd_ckpt in sd_ckpts:
        match = pattern.search(Path(sd_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))
        log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}"
        ckpt_info_list.append({
            "sd_ckpt": sd_ckpt,
            "global_step": global_step,
            "epoch": epoch,
            "log_dir_i": log_dir_i,
        })

    # =========================================================================
    # PHASE 1: Sampling (PyTorch only)
    # =========================================================================
    if not skip_sampling:
        print("\n" + "="*80)
        print("PHASE 1: Sequence Sampling (PyTorch)")
        print("="*80 + "\n")
        
        pbar = tqdm(ckpt_info_list, desc="Phase 1: Sampling sequences...")
        for ckpt_info in pbar:
            sd_ckpt = ckpt_info["sd_ckpt"]
            global_step = ckpt_info["global_step"]
            epoch = ckpt_info["epoch"]
            log_dir_i = ckpt_info["log_dir_i"]
            
            pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")
            
            # Create output directory
            Path(log_dir_i).mkdir(parents=True, exist_ok=True)
            
            # Check if sampling already done for this checkpoint
            samples_dir = Path(log_dir_i) / "samples"
            if samples_dir.exists() and len(list(samples_dir.glob("*.cif"))) > 0:
                print(f"Sampling already done for step {global_step}, skipping...")
                continue
            
            # Load sequence design model
            cfg.seq_des_cfg.atom_mpnn.ckpt_path = sd_ckpt
            seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
            seq_des_model["sampling_cfg"].num_workers = cfg.num_workers  # To avoid memory issues

            # Reset seed per checkpoint
            L.seed_everything(cfg.seed)

            # Run sequence design
            outputs = run_lc_seq_des(model=seq_des_model["model"], 
                                    data_cfg=cfg.data_cfg_for_design,
                                    transform_cfg=cfg.transform_cfg_for_design,                                     
                                    sampling_cfg=seq_des_model["sampling_cfg"],                          
                                    metadata=metadata,
                                    pdb_paths=pdb_files, device=device, 
                                    pos_constraint_df=None,
                                    out_dir=log_dir_i,
                                    protein_only=cfg.get("protein_only", False))
            
            # === Save sequence recovery metrics ===
            sample_len = len(outputs["example_id"])
            seq_recovery_rows = []
            for idx in range(sample_len):
                row = {
                    "example_id": outputs["example_id"][idx],
                    "sample_id": Path(outputs["out_pdb"][idx]).stem,
                }
                if "sample_seq_recovery" in outputs and len(outputs["sample_seq_recovery"]) == sample_len:
                    row["sample_seq_recovery"] = outputs["sample_seq_recovery"][idx]
                if "sample_sp_seq_recovery" in outputs and len(outputs["sample_sp_seq_recovery"]) == sample_len:
                    row["sample_sp_seq_recovery"] = outputs["sample_sp_seq_recovery"][idx]
                seq_recovery_rows.append(row)
            
            seq_recovery_df = pd.DataFrame(seq_recovery_rows)
            seq_recovery_df.to_csv(f"{log_dir_i}/seq_recovery_metrics.csv", index=False)
            
            # Log sequence recovery metrics to wandb
            seq_recovery_wandb_metrics = {
                "trainer/global_step": global_step,
                "trainer/epoch": epoch,
            }
            if "total_avg_seq_recovery" in outputs and outputs["total_avg_seq_recovery"] is not None:
                seq_recovery_wandb_metrics["eval/total_avg_seq_recovery"] = outputs["total_avg_seq_recovery"]
            if "total_avg_sp_seq_recovery" in outputs and outputs["total_avg_sp_seq_recovery"] is not None:
                seq_recovery_wandb_metrics["eval/total_avg_sp_seq_recovery"] = outputs["total_avg_sp_seq_recovery"]
            
            # Also log per-sample mean seq recovery from the DataFrame
            if len(seq_recovery_df) > 0:
                if "sample_seq_recovery" in seq_recovery_df.columns:
                    mean_sr = seq_recovery_df["sample_seq_recovery"].mean()
                    if pd.notna(mean_sr):
                        seq_recovery_wandb_metrics["eval/mean_sample_seq_recovery"] = mean_sr
                if "sample_sp_seq_recovery" in seq_recovery_df.columns:
                    mean_sp_sr = seq_recovery_df["sample_sp_seq_recovery"].mean()
                    if pd.notna(mean_sp_sr):
                        seq_recovery_wandb_metrics["eval/mean_sample_sp_seq_recovery"] = mean_sp_sr
            
            if not cfg.wandb.no_wandb:
                wandb.log(seq_recovery_wandb_metrics, commit=True)
                print(f"Logged Phase 1 metrics to wandb for step {global_step}")
            
            # === Free PyTorch memory before next checkpoint ===
            del seq_des_model
            del outputs
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            
            print(f"Completed sampling for step {global_step}, freed PyTorch memory")
        
        print("\n" + "="*80)
        print("PHASE 1 COMPLETE: All sampling done, PyTorch memory freed")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("PHASE 1 SKIPPED: skip_sampling=True")
        print("="*80 + "\n")

    # =========================================================================
    # PHASE 2: AF3 Predictions (JAX only)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: AF3 Predictions (JAX)")
    print("="*80 + "\n")
    
    pbar = tqdm(ckpt_info_list, desc="Phase 2: AF3 predictions...")
    for ckpt_info in pbar:
        global_step = ckpt_info["global_step"]
        epoch = ckpt_info["epoch"]
        log_dir_i = ckpt_info["log_dir_i"]
        
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")
        
        # Determine samples directory
        if skip_sampling and skip_sampling_dir is not None:
            samples_dir = Path(skip_sampling_dir) / "samples"
        else:
            samples_dir = Path(log_dir_i) / "samples"
        
        if not samples_dir.exists():
            print(f"Samples directory does not exist: {samples_dir}, skipping...")
            continue
        
        # Load outputs from samples directory
        print(f"Loading samples from {samples_dir}")
        outputs = load_outputs_from_samples_dir(str(samples_dir), metadata=metadata)
        
        # AF3 input/output dirs
        af3_ss_input_dir = Path(log_dir_i, "af3_ss_inputs")
        af3_ss_pred_dir = Path(log_dir_i, "af3_ss_preds")
        af3_tc_input_dir = Path(log_dir_i, "af3_tc_inputs")
        af3_tc_pred_dir = Path(log_dir_i, "af3_tc_preds")
        af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
        af3_ss_pred_dir.mkdir(parents=True, exist_ok=True)
        af3_tc_input_dir.mkdir(parents=True, exist_ok=True)
        af3_tc_pred_dir.mkdir(parents=True, exist_ok=True)

        # Make AF3 JSON for self-consistency evaluation
        af3_ss_json_paths, af3_tc_json_paths, pdb_chain_info = make_af3_json(
            af3_ss_input_dir=af3_ss_input_dir,
            af3_tc_input_dir=af3_tc_input_dir,                                                            
            outputs=outputs, 
            metadata=metadata,
            pdb_chain_info=None,                
            json_config=cfg.struct_pred_cfg.af3.json_config,
        )   
    
        # AF3 self-consistency and docking metrics per sample
        id_to_per_pred_metrics = {}                

        af3_ss_pred_dir = Path(af3_ss_pred_dir)                               
        
        for i in tqdm(range(len(outputs["out_pdb"])), desc="AF3 predictions", leave=False):
            sample_id = Path(outputs["out_pdb"][i]).stem
            sample_path = outputs["out_pdb"][i] 
            job_name = sample_id

            json_path_ss = af3_ss_json_paths[i]
            json_path_tc = af3_tc_json_paths[i]            

            # Self-consistency evaluation
            per_pred_sc_metrics = {}                           
            if cfg.evaluate_self_consistency:
                try:
                    run_af3_single_sequence(str(json_path_ss), str(af3_ss_pred_dir), 
                                          runner_path=af3_runner_path, 
                                          inference_config=cfg.struct_pred_cfg.af3.inference_config)
                except Exception as e:
                    print(f"AF3 single sequence prediction failed for {job_name}: {e}")
                    traceback.print_exc()
                    if not cfg.evaluate_template_conditioned_docking:
                        continue
                else:
                    try:
                        _, pred_ss_sample_paths = find_pred_sample_path_af3(
                            out_dir=str(af3_ss_pred_dir), job_name=job_name)        
                    except Exception as e:
                        print(f"Failed to find AF3 predicted structure for {job_name}: {e}")
                        pred_ss_sample_paths = []
                                                                                                        
                    if len(pred_ss_sample_paths) == 0:
                        print(f"No AF3 predicted structure found for {job_name}")
                    else:
                        try:
                            per_pred_sc_metrics = compute_self_consistency_metrics_atomworks(
                                sample_path=sample_path, 
                                pred_sample_paths=pred_ss_sample_paths,
                                data_cfg_for_af3_prediction=cfg.data_cfg_for_af3_prediction,
                                transform_cfg_for_af3_prediction=cfg.transform_cfg_for_af3_prediction,
                                num_diffusion_samples=cfg.struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples,
                                struct_pred_cfg=cfg.struct_pred_cfg,
                                metadata=metadata,
                                pdb_chain_info=pdb_chain_info)
                        except Exception as e:
                            print(f"Self-consistency metrics computation failed for {job_name}: {e}")
                            traceback.print_exc()
            
            # AF3 docking evaluation
            per_pred_docking_metrics = {}
            if cfg.evaluate_template_conditioned_docking:
                try:
                    run_af3_template_conditioned(str(json_path_tc), str(af3_tc_pred_dir), 
                                                runner_path=af3_runner_path, 
                                                inference_config=cfg.struct_pred_cfg.af3.inference_config)
                except Exception as e:
                    print(f"AF3 template-conditioned prediction failed for {job_name}: {e}")
                    traceback.print_exc()
                else:
                    try:
                        _, pred_tc_sample_paths = find_pred_sample_path_af3(
                            out_dir=str(af3_tc_pred_dir), job_name=job_name)        
                    except Exception as e:
                        print(f"Failed to find AF3 predicted structure for {job_name}: {e}")
                        pred_tc_sample_paths = []
                    
                    if len(pred_tc_sample_paths) == 0:
                        print(f"No AF3 predicted structure found for {job_name}")
                    else:
                        try:                                                
                            per_pred_docking_metrics = compute_template_conditioned_docking_metrics(
                                sample_path=sample_path, 
                                pred_sample_paths=pred_tc_sample_paths,
                                pdb_chain_info=pdb_chain_info,
                                binding_site_radius=cfg.docking_metrics_cfg.binding_site_radius,
                                save_aligned=cfg.docking_metrics_cfg.get("save_aligned", True),                                
                                data_cfg_for_af3_prediction=cfg.data_cfg_for_af3_prediction,
                                transform_cfg_for_af3_prediction=cfg.transform_cfg_for_af3_prediction,
                                metadata=metadata,
                            )
                        except Exception as e:
                            print(f"AF3 docking metrics computation failed for {job_name}: {e}")
                            traceback.print_exc()
                
            # Store metrics
            combined_metrics = {}
            if per_pred_sc_metrics:
                combined_metrics.update(per_pred_sc_metrics)
            if per_pred_docking_metrics:
                combined_metrics.update(per_pred_docking_metrics)
            
            if combined_metrics:
                id_to_per_pred_metrics[sample_id] = combined_metrics
            else:
                print(f"No metrics computed for {job_name} (skipping this sample)")

        # Save metrics CSV and log to wandb
        out_metrics = {
            "trainer/global_step": global_step,
            "trainer/epoch": epoch,
        }
        
        if id_to_per_pred_metrics:
            rows = []
            for sample_id, metrics_dict in id_to_per_pred_metrics.items():
                if not metrics_dict:
                    continue
                    
                record_id = sample_id.split("_")[0]
                num_diffusion = len(next(iter(metrics_dict.values())))
                
                for diffusion_id in range(num_diffusion):
                    row = {
                        "record_id": record_id,
                        "sample_id": sample_id,
                        "diffusion_id": diffusion_id,
                    }
                    for metric_name, values in metrics_dict.items():
                        row[metric_name] = values[diffusion_id]
                    rows.append(row)
            
            metrics_df = pd.DataFrame(rows)
            metrics_csv_name = f"lc_seq_des_metrics_{cfg.struct_pred_cfg.model_name}.csv"
            metrics_df.to_csv(f"{log_dir_i}/{metrics_csv_name}", index=False)
            print(f"Saved metrics to {log_dir_i}/{metrics_csv_name}")

            # Aggregate metrics for wandb logging
            skip_metrics = {"aligned_pred_array", "num_bs_residues"}
            
            all_metrics = defaultdict(list)
            for sample_id, metrics_dict in id_to_per_pred_metrics.items():
                for metric_name, values in metrics_dict.items():
                    if metric_name not in skip_metrics:
                        all_metrics[metric_name].extend(values)
            
            def filter_none(values):
                return [v for v in values if v is not None]
            
            out_metrics.update({f"eval/mean/{k}": np.nanmean(filter_none(v)) for k, v in all_metrics.items() if filter_none(v)})
            out_metrics.update({f"eval/median/{k}": np.nanmedian(filter_none(v)) for k, v in all_metrics.items() if k != "num_bs_residues" and filter_none(v)})
            
            # Ranked metrics: best pLDDT sample per sample_id
            ranked_metrics = defaultdict(list)
            for sample_id, metrics_dict in id_to_per_pred_metrics.items():
                if "avg_ca_plddt" not in metrics_dict:
                    continue
                
                plddt_values = metrics_dict["avg_ca_plddt"]
                best_diffusion_id = int(np.argmax(plddt_values))
                
                for metric_name, values in metrics_dict.items():
                    if metric_name not in skip_metrics:
                        ranked_metrics[metric_name].append(values[best_diffusion_id])
            
            out_metrics.update({f"eval/ranked/mean_{k}": np.nanmean(v) for k, v in ranked_metrics.items()})
            out_metrics.update({f"eval/ranked/median_{k}": np.nanmedian(v) for k, v in ranked_metrics.items()})
        else:
            print(f"No metrics computed for step {global_step} (id_to_per_pred_metrics is empty)")

        # Always log to wandb at end of each checkpoint (even if metrics are empty)
        if not cfg.wandb.no_wandb:
            wandb.log(out_metrics, commit=True)
            print(f"Logged Phase 2 metrics to wandb for step {global_step}")

        # Cleanup temp dirs
        for d in [Path(log_dir_i, "tmp")]:
            if Path(d).exists():
                shutil.rmtree(d, ignore_errors=True)
    
    print("\n" + "="*80)
    print("PHASE 2 COMPLETE: All AF3 predictions done")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
