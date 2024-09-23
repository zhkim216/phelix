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
    pattern = re.compile(r"sd-epoch\d+\.ckpt$")  # only consider ckpts of form sd-epochXXXX.ckpt
    sd_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
    sd_ckpts = natsorted([ckpt for ckpt in sd_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]
    sd_ckpts = sd_ckpts

    dataset = None  # we will load the dataset based on the model config

    pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints")
    for sd_ckpt in pbar:
        # Skip if epoch is before start_epoch
        epoch = int(Path(sd_ckpt).stem.replace("sd-epoch", ""))
        pbar.set_postfix_str(f"Epoch: {epoch}")
        if (cfg.start_epoch is not None) and (epoch < cfg.start_epoch):
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
        sd_inputs = {"num_steps": None,  # filled in based on S_sd
                     "timesteps": None,  # filled in based on batch size
                     "noise_schedule": None,
                     "churn_cfg": None,
                     "return_scn_diffusion_aux": False
                     }

        ### BEGIN EVAL ###
        metrics = {}

        for S_sd in cfg.scn_diffusion.num_steps_list:
            # Set up sidechain diffusion inputs
            sd_inputs["num_steps"] = S_sd
            cfg.scn_diffusion.timestep_schedule.num_steps = S_sd
            t_sd = sampling_utils.get_timestep_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

            # Sidechain pack
            sample_info = defaultdict(list)

            for batch in tqdm(val_dataloader, desc=f"Evaluating sidechain packing on validation set using {S_sd} denoising steps", leave=False):
                x, aatype = batch["x"].to(device), batch["aatype"].to(device)
                sd_inputs["timesteps"] = t_sd.expand(x.shape[0], -1).to(device)
                seq_mask, residue_index = batch["seq_mask"].to(device), batch["residue_index"].to(device)
                cond_labels_in = {"crop_aug": batch["cond_labels_in"]["crop_aug"].to(device)}  # we only provide whether cropping was applied


                x_denoised, _, _ = lit_ad_model.model.sidechain_pack(
                    x,
                    aatype,
                    seq_mask=seq_mask,
                    residue_index=residue_index,
                    cond_labels=cond_labels_in,
                    sd_inputs=sd_inputs,
                )

                # Store sample info
                seq_mask, aatype = seq_mask.cpu(), aatype.cpu()
                sample_info["pdb"] += batch["pdb_key"]
                sample_info["seq_mask"].append(seq_mask)
                sample_info["aatype"].append(aatype)

                # Compute sidechain RMSD per residue
                atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK)[aatype] * seq_mask[..., None]
                atom_mask = atom_mask * (1 - batch["missing_atom_mask"])  # handle atoms missing from the ground truth PDB

                scn_info, ca_aligned_coords1 = eval_metrics.compute_structure_metrics(x.cpu(), x_denoised.cpu(), atom_mask, metrics_to_compute=["scn_rmsd_per_pos"])
                sample_info["scn_rmsd_per_pos"].append(scn_info["scn_rmsd_per_pos"])

            sample_info = {k: torch.cat(v, dim=0) if k != "pdb" else v for k, v in sample_info.items()}

            ### Compute sidechain metrics ###
            metrics[f"S_sd{S_sd}/scn_rmsd_avg"] = (sample_info["scn_rmsd_per_pos"] * sample_info["seq_mask"]).sum() / sample_info["seq_mask"].sum()  # average over all residues in the dataset
            metrics[f"S_sd{S_sd}/scn_rmsd_avg"] = metrics[f"S_sd{S_sd}/scn_rmsd_avg"].item()

            # Get average RMSD per residue
            for aa_idx, aa in enumerate(rc.restypes_with_x):
                aatype_mask = sample_info["aatype"] == aa_idx
                rmsd_i = sample_info["scn_rmsd_per_pos"][aatype_mask]
                rmsd_avg_i = (rmsd_i * sample_info["seq_mask"][aatype_mask]).sum() / sample_info["seq_mask"][aatype_mask].sum()

                metrics[f"S_sd{S_sd}/scn_rmsd_{aa}"] = rmsd_avg_i.item()

        # Save metrics
        metrics_dir = f"{log_dir}/scn_pack_metrics"
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)

        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(f"{metrics_dir}/scn_pack_metrics_epoch{epoch}.csv", index=False)

        # Log metrics to wandb
        metrics = {f"scn_pack/{k}": v for k, v in metrics.items()}
        if not cfg.no_wandb:
            # Get global step
            global_step = torch.load(sd_ckpt, map_location="cpu")["global_step"]
            metrics["trainer/global_step"] = global_step

            wandb.log(metrics, step=global_step)


if __name__ == "__main__":
    main()
