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
from allatom_design.data.datasets.ad_dataset import ADDataset
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_scn_pack_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating the sidechain packing capabilities of a denoiser model during its training run.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create wandb dir
    wandb_dir = str(Path(cfg.out_dir))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

    # Set wandb cache directory
    wandb_cache_dir = str(Path(cfg.out_dir, "cache", "wandb"))
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up logging
    if cfg.no_wandb:
        log_dir = Path(cfg.out_dir, "debug")
    else:
        wandb.init(
            project=cfg.project,
            entity=cfg.wandb_id,
            name=cfg.exp_name,
            group=cfg.group,
            config=cfg_dict,
            dir=wandb_dir,
        )
        log_dir = Path(cfg.out_dir, wandb.run.name)  # base log dir

    # Set up out directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get checkpoints from denoiser training run
    pattern = re.compile(r"sd-step(\d+)-epoch(\d+)\.ckpt$")  # Only match checkpoints of the form sd-step{step}-epoch{epoch}.ckpt
    sd_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
    sd_ckpts = natsorted([ckpt for ckpt in sd_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    dataset = None  # we will load the dataset based on the model config

    pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints")
    for sd_ckpt in pbar:
        match = pattern.search(Path(sd_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))
        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Skip if global_step is before start_step
        if (cfg.start_step is not None) and (global_step < cfg.start_step):
            continue

        # Load denoiser model and dataset
        lit_ad_model = LitSeqDenoiser.load_from_checkpoint(sd_ckpt).eval()
        with open_dict(lit_ad_model.cfg.data):
            lit_ad_model.cfg.data.update({k: v for k, v in cfg.data.items() if v is not None})  # override data config where specified

        if dataset is None:
            # Load dataset based on model config
            dataset = ADDataset(phase="eval", **lit_ad_model.cfg.data)
            val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False, drop_last=False)
            dataset.subset_to_length_range(cfg.subset_length_range[0], cfg.subset_length_range[1])  # only eval on proteins within this length range

        # Create sidechain diffusion inputs
        scd_inputs = {"num_steps": None,  # filled in based on S_scd
                     "timesteps": None,  # filled in based on batch size
                     "noise_schedule": None,
                     "churn_cfg": None,
                     "autoguidance_cfg": None,
                     "return_scn_diffusion_aux": False
                     }

        ### BEGIN EVAL ###
        metrics = {}

        for S_scd in cfg.scn_diffusion.num_steps_list:
            # Set up sidechain diffusion inputs
            scd_inputs["num_steps"] = S_scd
            cfg.scn_diffusion.timestep_schedule.num_steps = S_scd
            t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

            # Sidechain pack
            sample_info = defaultdict(list)

            for batch in tqdm(val_dataloader, desc=f"Evaluating sidechain packing on validation set using {S_scd} denoising steps", leave=False):
                x, aatype = batch["x"].to(device), batch["aatype"].to(device)
                scd_inputs["timesteps"] = t_scd.expand(x.shape[0], -1).to(device)
                seq_mask = batch["seq_mask"].to(device)
                residue_index, chain_index = batch["residue_index"].to(device), batch["chain_index"].to(device)
                cond_labels_in = {"crop_aug": batch["cond_labels_in"]["crop_aug"].to(device)}  # we only provide whether cropping was applied

                x_denoised, _, _ = lit_ad_model.model.sidechain_pack(
                    x,
                    aatype,
                    seq_mask=seq_mask,
                    residue_index=residue_index,
                    chain_index=chain_index,
                    cond_labels=cond_labels_in,
                    scd_inputs=scd_inputs,
                )

                # Store sample info
                seq_mask, aatype = seq_mask.cpu(), aatype.cpu()
                sample_info["pdb"] += batch["pdb_key"]
                sample_info["seq_mask"].append(seq_mask)
                sample_info["aatype"].append(aatype)
                core_mask, surface_mask = eval_metrics.get_core_surface_mask(x.cpu(), batch["atom_mask"].cpu())
                sample_info["core_mask"].append(core_mask)
                sample_info["surface_mask"].append(surface_mask)

                # Compute sidechain RMSD per residue
                atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK)[aatype] * seq_mask[..., None]
                atom_mask = atom_mask * (1 - batch["missing_atom_mask"])  # handle atoms missing from the ground truth PDB

                scn_info, _ = eval_metrics.compute_structure_metrics(x.cpu(), x_denoised.cpu(),
                                                                     atom_mask, aatype=aatype,
                                                                     metrics_to_compute=["scn_rmsd_per_pos", "chi_metrics_per_pos"])
                for k, v in scn_info.items():
                    sample_info[k].append(v)

            sample_info = {k: torch.cat(v, dim=0) if k != "pdb" else v for k, v in sample_info.items()}

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
        metrics_dir = f"{log_dir}/scn_pack_metrics"
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)

        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(f"{metrics_dir}/scn_pack_metrics_epoch{epoch}.csv", index=False)

        # Log metrics to wandb
        metrics = {f"scn_pack/{k}": v for k, v in metrics.items()}
        if not cfg.no_wandb:
            metrics["trainer/global_step"] = global_step
            metrics["trainer/epoch"] = epoch

            wandb.log(metrics, step=global_step)


if __name__ == "__main__":
    main()
