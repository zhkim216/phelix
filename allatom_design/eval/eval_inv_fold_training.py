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

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import trim_to_max_len
from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.data.datasets.ad_dataset import ADDataset
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.folding_utils import get_struct_pred_model
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

    # Load in structure prediction model for co-design self-consistency evals
    if cfg.run_codes_sc:
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Get checkpoints from denoiser training run
    pattern = re.compile(r"sd-step(\d+)-epoch(\d+)\.ckpt$")  # Only match checkpoints of the form sd-step{step}-epoch{epoch}.ckpt
    sd_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
    sd_ckpts = natsorted([ckpt for ckpt in sd_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    pbar = tqdm(sd_ckpts, desc="Evaluating checkpoints")
    for sd_ckpt in pbar:
        match = pattern.search(Path(sd_ckpt).name)
        epoch = int(match.group(1))
        global_step = torch.load(sd_ckpt).get('global_step')

        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Skip if global_step is before start_step
        if (cfg.start_step is not None) and (global_step < cfg.start_step):
            continue

        # Load denoiser model and dataset
        lit_sd_model = LitSeqDenoiser.load_from_checkpoint(sd_ckpt).eval()
        with open_dict(lit_sd_model.cfg.data):
            lit_sd_model.cfg.data.update({k: v for k, v in cfg.data.items() if v is not None})  # override data config where specified

        # Load dataset based on model config
        dataset = ADDataset(phase="eval", evaluation_mode= True, **lit_sd_model.cfg.data)
        val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False, drop_last=False)
        dataset.subset_to_length_range(cfg.subset_length_range[0], cfg.subset_length_range[1])  # only eval on proteins within this length range

        # Set up sidechain diffusion inputs
        t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)  # sidechain diffusion time

        # create sidechain diffusion noise schedule
        noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)

        # create sidechain diffusion churn config
        churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
        scd_inputs = {"num_steps": cfg.scn_diffusion.num_steps,
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
                t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

                # Inverse fold
                seq_recovery_dir = f"{log_dir}/seq_recovery"
                Path(seq_recovery_dir).mkdir(parents=True, exist_ok=True)
                seq_rec_df_S = defaultdict(list)
                for batch in tqdm(val_dataloader, desc="Evaluating sequence recovery on validation set", leave=False):
                    batch = trim_to_max_len(batch)
                    x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = batch["x"].to(device), batch['aatype'].to(device), batch["seq_mask"].to(device), batch["missing_atom_mask"].to(device), batch["residue_index"].to(device), batch["chain_index"].to(device)
                    timesteps = t_seq[None].expand(x.shape[0], -1).to(device)

                    # Define sidechain diffusion timesteps
                    scd_inputs["timesteps"] = t_scd[None].expand(x.shape[0], -1).to(device)

                    # Define conditioning labels when we inverse fold
                    cond_labels_in = {"crop_aug": batch["cond_labels_in"]["crop_aug"].to(device)}  # we only provide whether cropping was applied

                    x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
                        x,
                        aatype=aatype,
                        seq_mask=seq_mask,
                        missing_atom_mask=missing_atom_mask,
                        residue_index=residue_index,
                        chain_index=chain_index,
                        cond_labels=cond_labels_in,
                        timesteps=timesteps,
                        temperature=cfg.temperature,
                        aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
                        num_corrector_steps=cfg.num_corrector_steps,
                        corrector_step_ratio=cfg.corrector_step_ratio,
                        scd_inputs=scd_inputs,
                    )


                    samples = {"x_denoised": x_denoised,
                            "seq_mask": seq_mask,
                            "residue_index": residue_index,
                            "pred_aatype": aatype_denoised,
                            "aatype_pred_traj": aux["aatype_pred_traj"],
                            "aatype_t_traj": aux["aatype_t_traj"],
                            "psce": torch.zeros((x.shape[0], x.shape[1], 33))
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


        ### Co-design self-consistency eval ###
        if cfg.run_codes_sc:
            print("Running co-design self-consistency evaluation...")
            codes_sc_dir = f"{log_dir}/codesign_sc"
            sampled_pdbs_dir = f"{codes_sc_dir}/sampled_pdbs"
            Path(sampled_pdbs_dir).mkdir(parents=True, exist_ok=True)

            for S in cfg.num_steps_list:
                print(f"Evaluating with num denoising steps S={S}")
                cfg.timestep_schedule.num_steps = S
                t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

                val_dataloader = DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    shuffle=True,
                    drop_last=False,
                )

                pdbs = []
                for bi, batch in enumerate(val_dataloader):
                    if bi >= cfg.num_codes_sc_batches:
                        break

                    batch = trim_to_max_len(batch)
                    x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = batch["x"].to(device), batch['aatype'].to(device), batch["seq_mask"].to(device), batch["missing_atom_mask"].to(device), batch["residue_index"].to(device), batch["chain_index"].to(device)

                    B = x.shape[0]
                    cond_labels_in = create_cond_labels_input(B, {"designability": "DESIGNABLE"}, device)
                    cond_labels_in["crop_aug"] = batch["cond_labels_in"]["crop_aug"].to(device)  # we provide whether this example was cropped

                    # Define timesteps for sequence and sidechain diffusion
                    timesteps = t_seq[None].expand(B, -1).to(device)
                    scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(device)

                    # Generate samples
                    x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
                        x,
                        aatype=torch.zeros_like(aatype),
                        seq_mask=seq_mask,
                        missing_atom_mask=missing_atom_mask,
                        residue_index=residue_index,
                        chain_index=chain_index,
                        cond_labels=cond_labels_in,
                        timesteps=timesteps,
                        temperature=cfg.temperature,
                        aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
                        num_corrector_steps=cfg.num_corrector_steps,
                        corrector_step_ratio=cfg.corrector_step_ratio,
                        scd_inputs=scd_inputs,
                    )

                    samples = {
                        "x_denoised": x_denoised,
                        "seq_mask": seq_mask.cpu(),
                        "missing_atom_mask": missing_atom_mask.cpu(),
                        "residue_index": residue_index,
                        "chain_index": chain_index,
                        "pred_aatype": aatype_denoised.cpu(),
                        "psce": aux["psce"]
                    }

                    # Save samples
                    filenames = [f"{sampled_pdbs_dir}/epoch_{epoch}_S{S}_batch_{bi}_sample_{idx}.pdb" for i in range(B)]
                    SeqDenoiser.save_samples_to_pdb(samples, filenames)
                    pdbs.extend(filenames)

                # After processing the specified number of batches, run self-consistency eval
                pdbs = natsorted(pdbs)

                codes_sc_info = eval_metrics.run_self_consistency_eval(
                    pdbs,
                    None, None,  # no MPNN model for co-design eval
                    struct_pred_model,
                    device,
                    out_dir=codes_sc_dir,
                    eval_codesign=True,
                    temp_dir=f"{cfg.out_dir}/tmp")
            

                # Aggregate results
                codes_metrics = defaultdict(list)
                for pdb in pdbs:
                    for k, v in codes_sc_info[pdb]["sc_metrics"].items():
                        codes_metrics[f"codes_{k}"].append(v.item())

                # Update metrics
                metrics.update({f"inv_fold/S{S}/{k}": np.mean(v) for k, v in codes_metrics.items()})

        # Log metrics to wandb
        if not cfg.no_wandb:
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