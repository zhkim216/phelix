import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import pad_to_max_len, trim_to_max_len
from allatom_design.data.datasets.sd_dataset import SDDataset
from allatom_design.eval import eval_metrics
from allatom_design.eval.eval_setup_utils import (get_pdb_files,
                                                 get_training_checkpoints,
                                                 wandb_setup)
from allatom_design.eval.fampnn_utils import (get_seq_des_model, run_fampnn_packing)
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_scn_pack_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating the sidechain packing capabilities of a denoiser model during its training run.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging
    log_dir = wandb_setup(no_wandb=cfg.no_wandb, out_dir=cfg.out_dir,
                          project=cfg.project, wandb_id=cfg.wandb_id, exp_name=cfg.exp_name, group=cfg.group,
                          cfg_dict=cfg_dict)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load in PDB files to eval on ###
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Get checkpoints from denoiser training run
    sd_ckpts, pattern = get_training_checkpoints(cfg.denoiser_train_dir, "seq_denoiser",
                                                 cfg.eval_every_n_ckpts,
                                                 cfg.start_step, cfg.end_step)

    pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints for sidechain packing...")
    for sd_ckpt in pbar:
        match = pattern.search(Path(sd_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Create output directory for this epoch
        log_dir_i = f"{log_dir}/step_{global_step}_epoch_{epoch}"
        Path(log_dir_i).mkdir(parents=True, exist_ok=True)

        # Load in sequence design model
        cfg.seq_des_cfg.fampnn.fampnn_ckpt = sd_ckpt

        metrics = {}

        for S_scd in cfg.num_steps_list:
            # Set up sidechain diffusion inputs
            cfg.sampling_cfg_overrides.scn_diffusion.num_steps = S_scd
            seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

            # Sidechain pack
            _, aux = run_fampnn_packing(seq_des_model["fampnn_model"], seq_des_model["fampnn_cfg"],
                                        pdb_paths=pdb_files, device=device)
            sample_info = aux["sample_info"]

            # Compute metrics
            core_mask, surface_mask = eval_metrics.get_core_surface_mask(sample_info["x_in"], sample_info["atom_mask"])
            sample_info["core_mask"] = core_mask
            sample_info["surface_mask"] = surface_mask
            scn_info, _ = eval_metrics.compute_structure_metrics(sample_info["x_in"], sample_info["x_denoised"],
                                                                 sample_info["atom_mask"], aatype=sample_info["aatype"],
                                                                 metrics_to_compute=["scn_rmsd_per_pos", "chi_metrics_per_pos"])

            for k, v in scn_info.items():
                sample_info[k] = v

            ### Compute sidechain metrics ###
            # Get average RMSD over all residues
            seq_mask = sample_info["seq_mask"]
            metrics[f"S_scd{S_scd}/scn_rmsd_avg"] = (sample_info["scn_rmsd_per_pos"] * seq_mask).sum() / seq_mask.sum()  # average over all residues in the dataset
            metrics[f"S_scd{S_scd}/scn_rmsd_avg"] = metrics[f"S_scd{S_scd}/scn_rmsd_avg"].item()

            # Get average RMSD per residue
            for aa_idx, aa in enumerate(rc.restypes_with_x):
                aatype_mask = sample_info["aatype"] == aa_idx
                rmsd_i = sample_info["scn_rmsd_per_pos"][aatype_mask]
                rmsd_avg_i = (rmsd_i * seq_mask[aatype_mask]).sum() / seq_mask[aatype_mask].sum()

                metrics[f"S_scd{S_scd}/scn_rmsd_{aa}"] = rmsd_avg_i.item()

            # Get average RMSD over all core and surface residues
            for key in ["core", "surface"]:
                mask = sample_info[f"{key}_mask"]
                scn_rmsd_avg = (sample_info["scn_rmsd_per_pos"][mask] * seq_mask[mask]).sum() / seq_mask[mask].sum()
                metrics[f"S_scd{S_scd}/scn_rmsd_avg_{key}"] = scn_rmsd_avg.item()

            # Get average chi metrics per chi angle
            chi_mask = sample_info["chi_mask"]  # [B, N, 4]
            chi_mae_avg = (sample_info["chi_mae_per_pos"] * chi_mask).sum(dim=(0, 1)) / chi_mask.sum(dim=(0, 1))
            chi_acc_avg = (sample_info["chi_acc_per_pos"] * chi_mask).sum(dim=(0, 1)) / chi_mask.sum(dim=(0, 1))
            for ci in range(4):
                metrics[f"S_scd{S_scd}/chi{ci+1}_mae_avg"] = chi_mae_avg[ci].item()
                metrics[f"S_scd{S_scd}/chi{ci+1}_acc_avg"] = chi_acc_avg[ci].item()

        # Save metrics
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(f"{log_dir_i}/scn_pack_metrics.csv", index=False)

        # Log metrics to wandb
        metrics = {f"scn_pack/{k}": v for k, v in metrics.items()}
        if not cfg.no_wandb:
            metrics["trainer/global_step"] = global_step
            metrics["trainer/epoch"] = epoch

            wandb.log(metrics, step=global_step)


if __name__ == "__main__":
    main()
