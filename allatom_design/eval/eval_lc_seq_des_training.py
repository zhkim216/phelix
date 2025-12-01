import glob
import os
import shutil
import subprocess
import traceback
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


def load_outputs_from_samples_dir(samples_dir: str, metadata: pd.DataFrame = None) -> dict:
    """
    Load outputs from an existing samples directory (from a previous sampling run).
    Uses sample_metadata.pt if available. Loads atom_arrays from .pt files (which preserve
    annotations like pn_unit_iid needed for make_af3_json).
    
    Args:
        samples_dir: Path to the samples/ directory containing .cif and .pt files
        metadata: Optional metadata DataFrame
        
    Returns:
        outputs dict with keys: example_id, out_pdb, atom_array, U, sample_seq_recovery, sample_sp_seq_recovery
    """
    outputs = defaultdict(list)
    samples_dir = Path(samples_dir)
    
    # Load sample_metadata.pt if it exists
    metadata_path = samples_dir / "sample_metadata.pt"
    sample_metadata = None
    if metadata_path.exists():
        sample_metadata = torch.load(metadata_path, weights_only=False)
        print(f"Loaded sample_metadata.pt with {len(sample_metadata)} samples")
    
    # Find all .cif files in the samples directory
    cif_files = sorted(glob.glob(os.path.join(samples_dir, "*.cif")))
    
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
        
        # Load atom_array from .pt file (preserves pn_unit_iid annotation)
        pt_file = samples_dir / f"{sample_stem}.pt"
        if pt_file.exists():
            atom_array = torch.load(pt_file, weights_only=False)
            outputs["atom_array"].append(atom_array)
        else:
            print(f"Warning: .pt file not found for {sample_stem}, atom_array will be None")
            outputs["atom_array"].append(None)
        
        outputs["example_id"].append(example_id)
        outputs["out_pdb"].append(str(cif_file))
    
    # Set placeholder values for total seq recovery metrics
    outputs["total_avg_seq_recovery"] = None
    outputs["total_avg_sp_seq_recovery"] = None
    
    print(f"Loaded {len(outputs['out_pdb'])} samples")
    
    return outputs


@hydra.main(config_path="../configs_local/eval", config_name="eval_lc_seq_des_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Sequence denoiser training eval with AF3 self-consistency.

    Flow per checkpoint:
    - sample sequences on processed input structures
    - for each sample, build AF3 JSON (MSA-free) and run AF3
    - compute structure self-consistency metrics vs sampled structure
    - save CSV and optionally log to wandb
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Seeds and determinism
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.debug:
        cfg.exp_name = f"debug_{cfg.exp_name}"

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
    if cfg.input_cfg.load_from_cache:
        pdb_files = get_cached_example_files(cached_example_path=cfg.input_cfg.load_cache_cfg.cached_example_path, pdb_name_list=pdb_keys, \
                                             pdb_name_ext=cfg.input_cfg.load_cache_cfg.pdb_name_ext, n_subsample=cfg.input_cfg.load_cache_cfg.n_subsample)
    else:
        pdb_files = get_pdb_files(pdb_dir=cfg.input_cfg.pdb_cfg.pdb_dir, pdb_name_list=pdb_keys, \
                              pdb_name_ext=cfg.input_cfg.pdb_cfg.pdb_name_ext, n_subsample=cfg.input_cfg.pdb_cfg.n_subsample)

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

    pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints (seq_recovery & AF3 metrics)...")
    for sd_ckpt in pbar:
        match = pattern.search(Path(sd_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Per-ckpt out dir
        log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}"
        Path(log_dir_i).mkdir(parents=True, exist_ok=True)

        # Check if we should skip sampling and load from existing directory
        skip_sampling = cfg.get("skip_sampling", False)
        skip_sampling_dir = cfg.get("skip_sampling_dir", None)
        
        if skip_sampling:
            # Determine the samples directory
            if skip_sampling_dir is not None:
                samples_dir = Path(skip_sampling_dir) / "samples"
            else:
                # Use the default log_dir_i/samples
                samples_dir = Path(log_dir_i) / "samples"
            
            if not samples_dir.exists():
                raise ValueError(f"skip_sampling is True but samples directory does not exist: {samples_dir}")
            
            print(f"Skipping sampling, loading from {samples_dir}")
            outputs = load_outputs_from_samples_dir(str(samples_dir), metadata=metadata)
        else:
            # Load sequence design model
            cfg.seq_des_cfg.atom_mpnn.ckpt_path = sd_ckpt
            seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
            seq_des_model["sampling_cfg"].num_workers = cfg.num_workers #! To avoid memory issues

            # Reset seed per checkpoint
            L.seed_everything(cfg.seed)

            # Run sequence design
            outputs = run_lc_seq_des(model = seq_des_model["model"], 
                                         load_from_cache = cfg.input_cfg.load_from_cache,
                                         featurizer_cfg = cfg.featurizer_cfg, 
                                         sampling_cfg = seq_des_model["sampling_cfg"],                          
                                         metadata = metadata,
                                         pdb_paths = pdb_files, device=device, 
                                         pos_constraint_df = None,
                                         out_dir = log_dir_i)
        
        # === Save sequence recovery metrics and log to wandb (before AF3 prediction) ===
        # Only save/log seq recovery metrics if not skipping sampling
        if not skip_sampling:
            # Build per-sample dataframe
            sample_len = len(outputs["example_id"])
            seq_recovery_rows = []
            for idx in range(sample_len):
                row = {
                    "example_id": outputs["example_id"][idx],
                    "sample_id": Path(outputs["out_pdb"][idx]).stem,
                }
                # Add per-sample seq recovery if available
                if "sample_seq_recovery" in outputs and len(outputs["sample_seq_recovery"]) == sample_len:
                    row["sample_seq_recovery"] = outputs["sample_seq_recovery"][idx]
                if "sample_sp_seq_recovery" in outputs and len(outputs["sample_sp_seq_recovery"]) == sample_len:
                    row["sample_sp_seq_recovery"] = outputs["sample_sp_seq_recovery"][idx]
                seq_recovery_rows.append(row)
            
            seq_recovery_df = pd.DataFrame(seq_recovery_rows)
            seq_recovery_df.to_csv(f"{log_dir_i}/seq_recovery_metrics.csv", index=False)
            
            # Log sequence recovery metrics to wandb (before AF3 prediction)
            seq_recovery_wandb_metrics = {}
            if "total_avg_seq_recovery" in outputs and outputs["total_avg_seq_recovery"] is not None:
                seq_recovery_wandb_metrics["eval/total_avg_seq_recovery"] = outputs["total_avg_seq_recovery"]
            if "total_avg_sp_seq_recovery" in outputs and outputs["total_avg_sp_seq_recovery"] is not None:
                seq_recovery_wandb_metrics["eval/total_avg_sp_seq_recovery"] = outputs["total_avg_sp_seq_recovery"]
            
            if not cfg.wandb.no_wandb and seq_recovery_wandb_metrics:
                seq_recovery_wandb_metrics["trainer/global_step"] = global_step
                seq_recovery_wandb_metrics["trainer/epoch"] = epoch
                wandb.log(seq_recovery_wandb_metrics, step=global_step)
        else:
            print("Skipping sequence recovery metrics logging (skip_sampling=True)")
        
        # AF3 input/output dirs
        # When skip_sampling is True, use temporary directories to avoid conflicts with existing predictions
        suffix = "_skip_test" if skip_sampling else ""
        af3_ss_input_dir = Path(log_dir_i, f"af3_ss_inputs{suffix}")
        af3_ss_pred_dir = Path(log_dir_i, f"af3_ss_preds{suffix}")
        af3_tc_input_dir = Path(log_dir_i, f"af3_tc_inputs{suffix}")
        af3_tc_pred_dir = Path(log_dir_i, f"af3_tc_preds{suffix}")
        af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
        af3_ss_pred_dir.mkdir(parents=True, exist_ok=True)
        af3_tc_input_dir.mkdir(parents=True, exist_ok=True)
        af3_tc_pred_dir.mkdir(parents=True, exist_ok=True)

        # Make AF3 JSON for self-consistency evaluation
        af3_ss_json_paths, af3_tc_json_paths, pdb_chain_info = make_af3_json(af3_ss_input_dir=af3_ss_input_dir,
                                                            af3_tc_input_dir=af3_tc_input_dir,                                                            
                                                            outputs=outputs, metadata=metadata,\
                                                            pdb_chain_info = None,                
                                                            json_config=cfg.struct_pred_cfg.af3.json_config
                                                            )   
    
        # AF3 self-consistency and docking metrics per sample
        # Structure: {sample_id: {metric_name: [values per diffusion_id]}}
        id_to_per_pred_metrics = {}                

        # Output directory for AF3 predictions
        af3_ss_pred_dir = Path(af3_ss_pred_dir)                               
        
        for i in tqdm(range(len(outputs["out_pdb"])), desc="AF3 single sequence self-consistency and docking scoring", leave=False):
            sample_id = Path(outputs["out_pdb"][i]).stem
            sample_path = outputs["out_pdb"][i] 
            job_name = sample_id

            # Get AF3 JSON paths
            json_path_ss = af3_ss_json_paths[i]
            json_path_tc = af3_tc_json_paths[i]            

            ## Self-consistency evaluation ###         
            per_pred_sc_metrics = {}                           
            if cfg.evaluate_self_consistency:
                try:
                    run_af3_single_sequence(str(json_path_ss), str(af3_ss_pred_dir), runner_path=af3_runner_path, inference_config=cfg.struct_pred_cfg.af3.inference_config)
                except Exception as e:
                    print(f"AF3 single sequence prediction failed for {job_name}: {e}")
                    traceback.print_exc()
                    # Continue to try template-conditioned docking if enabled
                    if not cfg.evaluate_template_conditioned_docking:
                        continue
                else:
                    # Find predicted structure file (only if AF3 succeeded)
                    try:
                        _, pred_ss_sample_paths = find_pred_sample_path_af3(out_dir = str(af3_ss_pred_dir), job_name = job_name)        
                    except Exception as e:
                        print(f"Failed to find AF3 predicted structure for {job_name}: {e}")
                        pred_ss_sample_paths = []
                                                                                                        
                    if len(pred_ss_sample_paths) == 0:
                        print(f"No AF3 predicted structure found for {job_name}")
                    else:
                        # Compute self-consistency metrics
                        try:
                            per_pred_sc_metrics = compute_self_consistency_metrics_atomworks(sample_path = sample_path, 
                                                                    pred_sample_paths = pred_ss_sample_paths,
                                                                    num_diffusion_samples = cfg.struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples,                                                        
                                                                    data_cfg = cfg.data_cfg_for_metrics,
                                                                    preprocess_transform_cfg = cfg.preprocess_transform_cfg,                                                                                                                
                                                                    featurizer_cfg = cfg.featurizer_cfg,
                                                                    struct_pred_cfg = cfg.struct_pred_cfg,
                                                                    metadata = metadata,
                                                                    pdb_chain_info = pdb_chain_info)
                        except Exception as e:
                            print(f"Self-consistency metrics computation failed for {job_name}: {e}")
                            traceback.print_exc()
            
            ### AF3 docking evaluation ###
            per_pred_docking_metrics = {}
            if cfg.evaluate_template_conditioned_docking:
                # Run template-conditioned AF3
                try:
                    run_af3_template_conditioned(str(json_path_tc), str(af3_tc_pred_dir), runner_path=af3_runner_path, inference_config=cfg.struct_pred_cfg.af3.inference_config)
                except Exception as e:
                    print(f"AF3 template-conditioned prediction failed for {job_name}: {e}")
                    traceback.print_exc()
                    # Continue to aggregate whatever metrics we have
                else:
                    # Find predicted structure file (only if AF3 succeeded)
                    try:
                        _, pred_tc_sample_paths = find_pred_sample_path_af3(out_dir = str(af3_tc_pred_dir), job_name = job_name)        
                    except Exception as e:
                        print(f"Failed to find AF3 predicted structure for {job_name}: {e}")
                        pred_tc_sample_paths = []
                    
                    if len(pred_tc_sample_paths) == 0:
                        print(f"No AF3 predicted structure found for {job_name}")
                    else:
                        try:
                            #! (JH) 251129 added: pass cif_parser_args to docking metrics
                            parser_kwargs = dict(cfg.data_cfg_for_metrics.cif_parser_args)
                            per_pred_docking_metrics = compute_template_conditioned_docking_metrics(
                                sample_path=sample_path, 
                                pred_sample_paths=pred_tc_sample_paths,
                                pdb_chain_info=pdb_chain_info,
                                binding_site_radius=cfg.docking_metrics_cfg.binding_site_radius,
                                save_aligned=cfg.docking_metrics_cfg.get("save_aligned", True),
                                parser_kwargs=parser_kwargs,  #! (JH) 251129 added
                            )
                        except Exception as e:
                            print(f"AF3 docking metrics computation failed for {job_name}: {e}")
                            traceback.print_exc()
                
            # Store per-prediction metrics for this sample
            # per_pred_sc_metrics is {metric_name: [val_0, val_1, ...]} where each val corresponds to a diffusion_id
            combined_metrics = {}
            if per_pred_sc_metrics:
                combined_metrics.update(per_pred_sc_metrics)
            if per_pred_docking_metrics:
                combined_metrics.update(per_pred_docking_metrics)
            
            if combined_metrics:
                id_to_per_pred_metrics[sample_id] = combined_metrics
            else:
                print(f"No metrics computed for {job_name} (skipping this sample)")

        # Save metrics CSV with record_id / sample_id / diffusion_id structure
        if id_to_per_pred_metrics:
            rows = []
            for sample_id, metrics_dict in id_to_per_pred_metrics.items():
                # Skip empty metrics_dict
                if not metrics_dict:
                    continue
                    
                # Extract record_id from sample_id (e.g., "1a28_A1_C1_sample0" -> "1a28")
                record_id = sample_id.split("_")[0]
                
                # Get number of diffusion samples from any metric
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
            metrics_csv_name = f"lc_seq_des_metrics_{cfg.struct_pred_cfg.model_name}{suffix}.csv"
            metrics_df.to_csv(f"{log_dir_i}/{metrics_csv_name}", index=False)
            print(f"Saved metrics to {log_dir_i}/{metrics_csv_name}")

            # Aggregate metrics for wandb logging
            # Skip ligand_rmsd and sym_ligand_rmsd, only log best_ligand_rmsd
            skip_metrics = {"ligand_rmsd", "sym_ligand_rmsd", "num_matched_atoms", "aligned_path"}
            
            # 1. All diffusion samples (flatten all predictions)
            all_metrics = defaultdict(list)
            for sample_id, metrics_dict in id_to_per_pred_metrics.items():
                for metric_name, values in metrics_dict.items():
                    if metric_name not in skip_metrics:
                        all_metrics[metric_name].extend(values)
            
            out_metrics = {f"eval/mean/{k}": np.nanmean(v) for k, v in all_metrics.items()}
            out_metrics.update({f"eval/median/{k}": np.nanmedian(v) for k, v in all_metrics.items() if k != "num_bs_residues"})
            
            # 2. Ranked metrics: best pLDDT sample per sample_id
            # For each sample_id, find the diffusion_id with highest avg_ca_plddt, then use those metrics
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

            if not cfg.wandb.no_wandb:
                out_metrics["trainer/global_step"] = global_step
                out_metrics["trainer/epoch"] = epoch
                wandb.log(out_metrics, step=global_step)

        # Cleanup temp dirs to save space
        for d in [Path(log_dir_i, "tmp")]:
            if Path(d).exists():
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()



