import glob
import os
import pickle
import re
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from allatom_design.data import protein
from allatom_design.data import residue_constants as rc
from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.datasets.ad_dataset import ADDataset
from allatom_design.data.pdb_utils import write_to_pdb_frames
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_inv_fold_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating the inverse folding capabilities of a denoiser model during its training run.

    We refer to "sequence recovery" as opposed to "sequence accuracy" for evaluating median across sequences rather than mean across residues.
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

    # Load in ESMFold for co-design self-consistency evals
    if cfg.run_codes_sc:
        esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
        esmfold.esm = esmfold.esm.half()
        esmfold = esmfold.to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # Get checkpoints from denoiser training run
    pattern = re.compile(r"sd-epoch\d+\.ckpt$")  # only consider ckpts of form sd-epochXXXX.ckpt
    sd_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
    sd_ckpts = natsorted([ckpt for ckpt in sd_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    dataset = None  # we will load the dataset based on the model config

    pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints")
    for sd_ckpt in pbar:
        # Skip if epoch is before start_epoch
        epoch = int(Path(sd_ckpt).stem.replace("sd-epoch", ""))
        pbar.set_postfix_str(f"Epoch: {epoch}")
        if (cfg.start_epoch is not None) and (epoch < cfg.start_epoch):
            continue

        # Load denoiser model and dataset
        lit_sd_model = LitSeqDenoiser.load_from_checkpoint(sd_ckpt).eval()
        with open_dict(lit_sd_model.cfg.data):
            lit_sd_model.cfg.data.update({k: v for k, v in cfg.data.items() if v is not None})  # override data config where specified

        if dataset is None:
            # Load dataset based on model config
            dataset = ADDataset(phase="eval", **lit_sd_model.cfg.data)
            val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False, drop_last=False)
            dataset.subset_to_length_range(cfg.subset_length_range[0], cfg.subset_length_range[1])  # only eval on proteins within this length range

        # Set up sidechain diffusion inputs
        t_sd = sampling_utils.get_timestep_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

        # create sidechain diffusion noise schedule
        noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)

        # create sidechain diffusion churn config
        churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
        sd_inputs = {"num_steps": cfg.scn_diffusion.num_steps,
                    "timesteps": None,  # filled in based on batch size
                    "noise_schedule": noise_schedule,
                    "churn_cfg": churn_cfg,
                    "return_scn_diffusion_aux": False
                    }

        ### BEGIN EVAL ###
        metrics = {}

        ### Sequence recovery eval on validation set ###
        if cfg.run_seq_recovery:
            for S in cfg.num_steps_list:
                # Evaluate sequence recovery with different numbers of steps
                print(f"Evaluating with num denoising steps S={S}")
                cfg.timestep_schedule.num_steps = S
                t_seq = sampling_utils.get_timestep_schedule(**cfg.timestep_schedule)

                # Inverse fold
                seq_recovery_dir = f"{log_dir}/seq_recovery"
                Path(seq_recovery_dir).mkdir(parents=True, exist_ok=True)
                seq_rec_df_S = defaultdict(list)
                for batch in tqdm(val_dataloader, desc="Evaluating sequence recovery on validation set", leave=False):
                    x, seq_mask, residue_index = batch["x"].to(device), batch["seq_mask"].to(device), batch["residue_index"].to(device)
                    timesteps = t_seq[None].expand(x.shape[0], -1).to(device)

                    # Define sidechain diffusion timesteps
                    sd_inputs["timesteps"] = t_sd[None].expand(x.shape[0], -1).to(device)

                    # Define conditioning labels when we inverse fold
                    cond_labels_in = {"crop_aug": batch["cond_labels_in"]["crop_aug"].to(device)}  # we only provide whether cropping was applied

                    x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
                        x,
                        seq_mask=seq_mask,
                        residue_index=residue_index,
                        timesteps=timesteps,
                        aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
                        cond_labels=cond_labels_in,
                        sd_inputs=sd_inputs,
                    )
                    samples = {"x_denoised": x_denoised,
                            "seq_mask": seq_mask,
                            "residue_index": residue_index,
                            "pred_aatype": aatype_denoised,
                            "aatype_pred_traj": aux["aatype_pred_traj"],
                            "aatype_t_traj": aux["aatype_t_traj"],
                    }

                    # Update info for sequence recovery eval
                    seq_rec_df_S["pdb"] += batch["pdb_key"]

                    seq_mask = seq_mask.cpu()
                    for i in range(batch["x"].shape[0]):
                        # Ground truth seqs
                        gt_aatype = batch["aatype"][i][seq_mask[i].bool()]
                        gt_seq = "".join([rc.restypes_with_x[i] for i in gt_aatype])
                        seq_rec_df_S["gt_seq"].append(gt_seq)

                        # Predicted seqs
                        pred_aatype = samples["pred_aatype"][i][seq_mask[i].bool()]
                        pred_seq = "".join([rc.restypes_with_x[i] for i in pred_aatype])
                        seq_rec_df_S["pred_seq"].append(pred_seq)

                        for t in np.linspace(0, 0.9, 10):
                            ti = int(t * S)

                            # Denoised seqs along trajectory
                            pred_aatype_traj = samples["aatype_pred_traj"][i, ti][seq_mask[i].bool()]
                            pred_seq_traj = "".join([rc.restypes_with_x[i] for i in pred_aatype_traj])
                            seq_rec_df_S[f"pred_seq_t{t:.1f}"].append(pred_seq_traj)

                            # Noisy seqs along trajectory
                            aatype_traj = samples["aatype_t_traj"][i, ti][seq_mask[i].bool()]
                            seq_traj = "".join([rc.restypes_with_x[i] for i in aatype_traj])
                            seq_rec_df_S[f"noisy_seq_t{t:.1f}"].append(seq_traj)


                # Save inverse folding results
                seq_rec_df_S = pd.DataFrame(seq_rec_df_S)
                seq_rec_df_S["seq_rec"] = seq_rec_df_S.apply(lambda x: np.mean(np.array(list(x["gt_seq"])) == np.array(list(x["pred_seq"]))), axis=1)
                for t in np.linspace(0, 0.9, 10):
                    seq_rec_df_S[f"pred_seq_rec_t{t:.1f}"] = seq_rec_df_S.apply(lambda x: np.mean(np.array(list(x["gt_seq"])) == np.array(list(x[f"pred_seq_t{t:.1f}"]))), axis=1)
                    seq_rec_df_S[f"noisy_seq_rec_t{t:.1f}"] = seq_rec_df_S.apply(partial(get_unmasked_rec, t=t), axis=1)  # get accuracy among unmasked residues

                seq_rec_df_S.to_csv(f"{seq_recovery_dir}/seq_rec_epoch{epoch}_S{S}.csv", index=False)

                med_seq_rec = seq_rec_df_S["seq_rec"].median()
                print(f"Sequence recovery accuracy: {med_seq_rec:.4f}")
                metrics[f"inv_fold/S{S}/median_seq_recovery"] = med_seq_rec

                # # Save seq accuracy across trajectory
                # for t in np.linspace(0, 0.9, 10):
                #     # Denoised seqs
                #     pred_seq_rec_t_med = seq_rec_df_S[f"pred_seq_rec_t{t:.1f}"].median()
                #     # print(f"Sequence recovery accuracy at t={t:.1f}: {pred_seq_rec_t_med:.4f}")
                #     metrics[f"inv_fold/S{S}/traj/med_seq_rec_t{t:.1f}"] = pred_seq_rec_t_med

                #     # Noisy seqs
                #     noisy_seq_rec_t_med = seq_rec_df_S[f"noisy_seq_rec_t{t:.1f}"].median()
                #     # print(f"Sequence recovery among unmasked tokens at t={t:.1f}: {noisy_seq_rec_t_med:.4f}")
                #     metrics[f"inv_fold/S{S}/traj/med_noisy_seq_rec_t{t:.1f}"] = noisy_seq_rec_t_med


        ### Co-design self-consistency eval ###
        if cfg.run_codes_sc:
            print("Running co-design self-consistency evaluation...")
            codes_sc_dir = f"{log_dir}/codesign_sc"
            sampled_pdbs_dir = f"{codes_sc_dir}/sampled_pdbs"
            Path(sampled_pdbs_dir).mkdir(parents=True, exist_ok=True)

            # Grab a batch for inverse folding
            B = cfg.num_codes_sc_samples
            val_dataloader = DataLoader(dataset, batch_size=B, num_workers=cfg.num_workers, pin_memory=True, shuffle=True, drop_last=False)
            batch = next(iter(val_dataloader))  # get a random batch for inverse folding
            x, seq_mask, residue_index = batch["x"].to(device), batch["seq_mask"].to(device), batch["residue_index"].to(device)
            cond_labels_in = create_cond_labels_input(B, {"designability": "DESIGNABLE"}, device)  # for now we always use "DESIGNABLE" for eval
            cond_labels_in["crop_aug"] = batch["cond_labels_in"]["crop_aug"].to(device)

            for S in cfg.num_steps_list:
                # Define multi-time timesteps
                timesteps = t_seq[None].expand(x.shape[0], -1).to(device)
                sd_inputs["timesteps"] = t_sd[None].expand(x.shape[0], -1).to(device)

                x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
                    x,
                    seq_mask=seq_mask,
                    residue_index=residue_index,
                    timesteps=timesteps,
                    aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
                    cond_labels=cond_labels_in,
                    sd_inputs=sd_inputs,
                )

                samples = {"x_denoised": x_denoised,
                           "seq_mask": seq_mask.cpu(),
                           "residue_index": residue_index,
                           "pred_aatype": aatype_denoised.cpu()}

                # Save samples
                filenames = [f"{sampled_pdbs_dir}/epoch_{epoch}_S{S}_sample_{i}.pdb" for i in range(B)]
                SeqDenoiser.save_samples_to_pdb(samples, filenames)

                # Run self-consistency eval
                pdbs = natsorted(filenames)
                codes_sc_info = eval_metrics.run_self_consistency_eval(pdbs,
                                                                    None, None,  # no MPNN model for co-design eval
                                                                    esmfold, tokenizer,
                                                                    device,
                                                                    out_dir=codes_sc_dir,
                                                                    eval_codesign=True)

                # Aggregate results
                codes_metrics = defaultdict(list)
                for pdb in pdbs:
                    for k, v in codes_sc_info[pdb]["sc_metrics"].items():
                        codes_metrics[f"codes_{k}"].append(v.item())

                metrics.update({f"inv_fold/S{S}/{k}": np.mean(v) for k, v in codes_metrics.items()})

        # Log metrics to wandb
        if not cfg.no_wandb:
            # Get global step
            global_step = torch.load(sd_ckpt, map_location="cpu")["global_step"]
            metrics["trainer/global_step"] = global_step
            metrics["trainer/epoch"] = epoch

            wandb.log(metrics, step=global_step)


def get_unmasked_rec(row: pd.Series, t: float):
    """Get number of correct predictions out of total number of non-masked residues"""
    correct = np.array(list(row["gt_seq"])) == np.array(list(row[f"noisy_seq_t{t:.1f}"]))
    masked = np.array(list(row[f"noisy_seq_t{t:.1f}"])) == "X"  # we use "X" to denote masked residues
    return np.sum(correct * ~masked) / np.sum(~masked)


if __name__ == "__main__":
    main()
