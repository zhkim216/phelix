import glob
import os
import shutil
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import pad_to_max_len, trim_to_max_len
from allatom_design.data.datasets.ad_dataset import ADDataset
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_validation", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        model_run_dir = Path(cfg.sd_ckpt).parent.parent
        model_name = Path(cfg.sd_ckpt).stem
        cfg.out_dir = f"{model_run_dir}/eval_validation/{model_name}/{cfg.exp_name}"

    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # Delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

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
        results_dir = Path(log_dir, "results")
        results_dir.mkdir(parents=True, exist_ok=True)
        logger = False  # disables logging
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

        # path for run outputs
        results_dir = Path(log_dir, "results")
        results_dir.mkdir(parents=True, exist_ok=True)

        logger = WandbLogger(
            name=cfg.exp_name,
            project=cfg.project,
            entity=cfg.wandb_id,
            experiment=wandb.run,
            save_dir=results_dir,
        )

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.sd_ckpt).eval()
    device = lit_sd_model.device

    # Load dataset
    with open_dict(lit_sd_model.cfg.data):
        lit_sd_model.cfg.data.update({k: v for k, v in cfg.data.items() if v is not None})  # override data config where specified

        # Load dataset based on model config
        dataset = ADDataset(phase="eval", evaluation_mode = True, **lit_sd_model.cfg.data)

        val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False, drop_last=False)

    ### Run validation loop ###
    # if hasattr(lit_sd_model.model, "_orig_mod"):
    #     # decompile model since it's too much overhead for 1 validation epoch
    #     lit_sd_model.model = lit_sd_model.model._orig_mod

    # Set up trainer
    trainer = L.Trainer(logger=logger,
                        default_root_dir=cfg.out_dir,
                        **cfg.trainer
                        )

    # Evaluate
    trainer.validate(lit_sd_model, dataloaders=val_dataloader)

    # Send model back to device since PTL moves it to CPU after validation
    lit_sd_model.model.to(device)

    ### Run sidechain packing ###
    # Create sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

    # create noise schedule
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)

    # create churn config
    S_scd = cfg.scn_diffusion.num_steps
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs = {"num_steps": S_scd,
                 "timesteps": None,  # filled in based on batch size
                 "noise_schedule": noise_schedule,
                 "churn_cfg": churn_cfg,
                 "return_scn_diffusion_aux": False
                 }

    # Sidechain pack
    sample_info = defaultdict(list)
    for batch in tqdm(val_dataloader, desc=f"Evaluating sidechain packing on validation set using {S_scd} denoising steps", leave=False):
        batch = trim_to_max_len(batch)
        x, aatype = batch["x"].to(device), batch["aatype"].to(device)
        scd_inputs["timesteps"] = t_scd.expand(x.shape[0], -1).to(device)
        seq_mask, missing_atom_mask = batch["seq_mask"].to(device), batch["missing_atom_mask"].to(device)
        residue_index, chain_index = batch["residue_index"].to(device), batch["chain_index"].to(device)
        cond_labels_in = {"crop_aug": batch["cond_labels_in"]["crop_aug"].to(device)}  # we only provide whether cropping was applied

        x_denoised, _, _ = lit_sd_model.model.sidechain_pack(
            x,
            aatype,
            seq_mask=seq_mask,
            missing_atom_mask=missing_atom_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            cond_labels=cond_labels_in,
            scd_inputs=scd_inputs,
        )

        # Store sample info
        seq_mask, aatype = seq_mask.cpu(), aatype.cpu()
        core_mask, surface_mask = eval_metrics.get_core_surface_mask(x.cpu(), batch["atom_mask"].cpu())
        sample_info_i = {"pdb_key": batch["pdb_key"], "seq_mask": seq_mask, "aatype": aatype, "core_mask": core_mask, "surface_mask": surface_mask}

        # Compute sidechain RMSD per residue
        atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK)[aatype] * seq_mask[..., None]
        atom_mask = atom_mask * (1 - batch["missing_atom_mask"])  # handle atoms missing from the ground truth PDB

        scn_info, _ = eval_metrics.compute_structure_metrics(x.cpu(), x_denoised.cpu(),
                                                                atom_mask, aatype=aatype,
                                                                metrics_to_compute=["scn_rmsd_per_pos", "chi_metrics_per_pos"])
        for k, v in scn_info.items():
            sample_info_i[k] = v

        # Pad sample_info for this batch back to max length
        sample_info_i = pad_to_max_len(sample_info_i, max_len=dataset.fixed_size)

        # Append sample info for this batch
        for k, v in sample_info_i.items():
            sample_info[k].append(v)

    sample_info = {k: torch.cat(v, dim=0) if k != "pdb_key" else v for k, v in sample_info.items()}

    ### Compute sidechain metrics ###
    metrics = {}
    metric_prefix = f"S_scd{S_scd}_ss{cfg.scn_diffusion.noise_schedule.c}"

    # Get average RMSD over all residues
    seq_mask = sample_info["seq_mask"]
    metrics[f"{metric_prefix}/scn_rmsd_avg"] = (sample_info["scn_rmsd_per_pos"] * seq_mask).sum() / seq_mask.sum()  # average over all residues in the dataset
    metrics[f"{metric_prefix}/scn_rmsd_avg"] = metrics[f"{metric_prefix}/scn_rmsd_avg"].item()

    # Get average RMSD per residue
    for aa_idx, aa in enumerate(rc.restypes_with_x):
        aatype_mask = sample_info["aatype"] == aa_idx
        rmsd_i = sample_info["scn_rmsd_per_pos"][aatype_mask]
        rmsd_avg_i = (rmsd_i * seq_mask[aatype_mask]).sum() / seq_mask[aatype_mask].sum()

        metrics[f"{metric_prefix}/scn_rmsd_{aa}"] = rmsd_avg_i.item()

    # Get average RMSD over all core and surface residues
    for key in ["core", "surface"]:
        mask = sample_info[f"{key}_mask"]
        scn_rmsd_avg = (sample_info["scn_rmsd_per_pos"][mask] * seq_mask[mask]).sum() / seq_mask[mask].sum()
        metrics[f"{metric_prefix}/scn_rmsd_avg_{key}"] = scn_rmsd_avg.item()

    # Get average chi metrics per chi angle
    chi_mask = sample_info["chi_mask"]  # [B, N, 4]
    chi_mae_avg = (sample_info["chi_mae_per_pos"] * chi_mask).sum(dim=(0, 1)) / chi_mask.sum(dim=(0, 1))
    chi_acc_avg = (sample_info["chi_acc_per_pos"] * chi_mask).sum(dim=(0, 1)) / chi_mask.sum(dim=(0, 1))
    for ci in range(4):
        metrics[f"{metric_prefix}/chi{ci+1}_mae_avg"] = chi_mae_avg[ci].item()
        metrics[f"{metric_prefix}/chi{ci+1}_acc_avg"] = chi_acc_avg[ci].item()

    # Save metrics
    metrics_dir = f"{log_dir}/scn_pack_metrics"
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(f"{metrics_dir}/scn_pack_metrics.csv", index=False)

    # Log metrics to wandb
    metrics = {f"scn_pack/{k}": v for k, v in metrics.items()}
    if not cfg.no_wandb:
        wandb.log(metrics, step=1)

    wandb.finish()


if __name__ == "__main__":
    main()
