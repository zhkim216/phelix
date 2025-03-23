import os
import pickle
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

from allatom_design.data.data import load_feats_from_pdb
from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.bb_gen_utils import (
    get_bb_gen_model, run_bb_uncond_sampling)
from allatom_design.eval.eval_utils.eval_setup_utils import wandb_setup
from allatom_design.eval.eval_utils.fampnn_utils import get_seq_des_model
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model


@hydra.main(config_path="../../configs/eval/sampling", config_name="bb_unconditional", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging / output directory
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in atom denoiser
    bb_gen_model = get_bb_gen_model(cfg.bb_gen_cfg, device=device)
    sampling_cfg = bb_gen_model["sampling_cfg"]

    # Define the range of lengths to sample
    start, end = cfg.length_range
    lengths_to_sample = np.arange(start, end + 1, cfg.length_step_size).repeat(cfg.n_samples_per_length)  # get the length of each protein we'll sample

    if cfg.save_traj.enabled:
        save_traj_inputs = {
            "save_traj_mask": np.tile(np.arange(cfg.n_samples_per_length) < cfg.save_traj.n_traj_per_length, len(lengths_to_sample)),  # for each protein, True if we should save the trajectory
            "save_traj_steps": np.linspace(0, sampling_cfg.num_steps - 1, cfg.save_traj.limit_traj_steps, dtype=int),  # get the which diffusion timesteps we'll save along the trajectory
            "traj_conect": cfg.save_traj.traj_conect,
            "align_traj_to_last_step": cfg.save_traj.align_traj_to_last_step
        }
    else:
        save_traj_inputs = None

    # === Sample structures === #
    print(f"Drawing {cfg.n_samples_per_length} samples each of lengths {start} to {end} with step size {cfg.length_step_size}")

    # Run unconditional sampling
    sampled_pdb_paths = run_bb_uncond_sampling(model=bb_gen_model["model"],
                                               cfg=sampling_cfg,
                                               device=device,
                                               lengths=lengths_to_sample,
                                               out_dir=log_dir,
                                               save_traj_inputs=save_traj_inputs)

    ### CALCULATE STRUCTURE METRICS ###
    if cfg.compute_self_consistency:
        # Load in MPNN + structure prediction model for self-consistency evals
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

        # First, get bin sampled structures into length bins
        bins = [int(length / cfg.length_bin_size) * cfg.length_bin_size for length in lengths_to_sample]
        bin_to_pdbs = defaultdict(list)
        for pdb, b in zip(sampled_pdb_paths, bins):
            bin_to_pdbs[b].append(pdb)

        # Compute metrics for each length bin
        for b in sorted(bin_to_pdbs.keys()):
            pdbs_b = bin_to_pdbs[b]
            log_dir_b = Path(log_dir, f"bin{b}")
            Path(log_dir_b).mkdir(parents=True, exist_ok=True)

            ## Get per-pdb info
            per_pdb_info_b, sample_metrics_b = eval_metrics.compute_per_pdb_info(pdbs_b, seq_des_model, struct_pred_model, device,
                                                                                 out_dir=log_dir_b, temp_dir=f"{log_dir_b}/tmp", nntm_dataset=cfg.nntm_dataset)

            # Save per-pdb info
            torch.save(per_pdb_info_b, f"{log_dir_b}/per_pdb_info.pt")

            # Calculate a scalar for each metric to log
            metrics_b = {}
            metrics_b.update({f"mean/{k}": np.mean(v) for k, v in sample_metrics_b.items()})
            metrics_b.update({f"median/{k}": np.median(v) for k, v in sample_metrics_b.items()})

            # Compute diversity metrics across all PDBs in bin
            diversity_metrics = eval_metrics.run_diversity_eval(pdbs_b, per_pdb_info_b, cfg.diversity_eval, log_dir_b)
            metrics_b.update(diversity_metrics)

            # Log aggregated metrics to wandb
            metrics_b = {f"bb_gen/{k}": v for k, v in metrics_b.items()}
            torch.save(metrics_b, f"{log_dir_b}/metrics.pt")
            if not cfg.wandb.no_wandb:
                metrics_b["length_bin"] = b  # log the starting length of the bin
                wandb.log(metrics_b, step=b)

    # Finish wandb logging
    if not cfg.wandb.no_wandb:
        wandb.finish()



if __name__ == "__main__":
    main()
