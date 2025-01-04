import os
import shutil
from functools import partial
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

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.eval import sampling_utils
from allatom_design.eval.eval_metrics import fpd
from allatom_design.eval.proteinmpnn_utils import (create_mpnn_embeddings,
                                                   load_mpnn,
                                                   load_mpnn_embeddings)
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


@hydra.main(config_path="../configs/eval", config_name="eval_fpd", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Load atom denoiser
    lit_ad_model = LitAtomDenoiser.load_from_checkpoint(cfg.model_ckpt).eval()
    device = lit_ad_model.device

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        model_run_dir = Path(cfg.model_ckpt).parent.parent
        model_name = Path(cfg.model_ckpt).stem
        cfg.out_dir = f"{model_run_dir}/eval_fpd/{model_name}/{cfg.exp_name}"

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sample_out_dir = Path(cfg.out_dir, "samples")
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    # Subsample embeddings
    train_subsample_df = subsample_embeddings(cfg.train_lengths_csv, cfg.pct_subsample_train, phase="train")
    eval_subsample_df = subsample_embeddings(cfg.eval_lengths_csv, cfg.pct_subsample_eval, phase="eval")
    eval2_subsample_df = subsample_embeddings(cfg.eval2_lengths_csv, cfg.pct_subsample_eval, phase="eval2")

    print(f"Sampling {len(train_subsample_df)} train, {len(eval_subsample_df)} eval, {len(eval2_subsample_df)} eval2 structures for FPD calculation.")

    # Concatenate into one dataframe
    df = pd.concat([train_subsample_df, eval_subsample_df, eval2_subsample_df], ignore_index=True)
    df = df.sort_values("length").reset_index(drop=True)  # sort by length for sampling efficiency

    # Override s_max
    if cfg.s_max_override is not None:
        lit_ad_model.model.denoiser.interpolant.set_s_max(cfg.s_max_override)

    # === Sample structures === #
    pbar = tqdm(total=len(df), desc="Sampling backbones")
    all_lengths = df["length"].values
    pdb_keys = df["pdb_key"].values
    sampled_pdbs = []

    for i in range(0, len(all_lengths), cfg.batch_size):
        lengths = torch.tensor(all_lengths[i : i + cfg.batch_size], dtype=torch.long).to(device)
        B = lengths.shape[0]

        residue_index = torch.arange(lengths.max(), dtype=torch.long).to(device)
        residue_index = residue_index[None].expand(B, -1)

        # Create timesteps for backbone
        t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)
        t_bb = t_bb[None].expand(B, -1).to(device)
        timesteps = t_bb

        # Create noise schedule for backbone
        noise_schedule = NoiseSchedule(cfg.noise_schedule)

        # Create churn config for backbone
        churn_cfg = dict(cfg.churn_cfg)

        cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device)
        x_bb_denoised, aux = lit_ad_model.model.sample(
            lengths,
            residue_index=residue_index,
            timesteps=timesteps,
            cond_labels=cond_labels_in,
            noise_schedule=noise_schedule,
            churn_cfg=churn_cfg,
            autoguidance_cfg=dict(cfg.autoguidance_cfg),
        )

        # Save samples
        samples = {
            "x_bb_denoised": x_bb_denoised.cpu(),
            "seq_mask": aux["seq_mask"].cpu(),
            "residue_index": residue_index.cpu(),
        }

        filenames = [f"{sample_out_dir}/{pdb_keys[i + j]}_L{lengths[j]}.pdb" for j in range(B)]
        sampled_pdbs.extend(filenames)
        AtomDenoiser.save_samples_to_pdb(samples, filenames)

        pbar.update(B)
    pbar.close()

    # Add sampled pdb names to df
    df["sample_name"] = [Path(pdb).stem for pdb in sampled_pdbs]

    # Create output directories
    samp_embeddings_dir = Path(cfg.out_dir, "mpnn_embeddings")
    samp_embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load MPNN model
    device = torch.device("cuda")
    mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
    mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
    mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

    # Run MPNN to create embeddings
    create_mpnn_embeddings(mpnn_model, pdb_paths=sampled_pdbs, out_dir=str(samp_embeddings_dir), device=device, cfg=mpnn_cfg)

    # Load sampled embeddings
    train_samp_embeddings = load_mpnn_embeddings([f"{str(samp_embeddings_dir)}/{Path(pdb).stem}.npy" for pdb in df[df["phase"] == "train"]["sample_name"].values])
    eval_samp_embeddings = load_mpnn_embeddings([f"{str(samp_embeddings_dir)}/{Path(pdb).stem}.npy" for pdb in df[df["phase"] == "eval"]["sample_name"].values])
    eval2_samp_embeddings = load_mpnn_embeddings([f"{str(samp_embeddings_dir)}/{Path(pdb).stem}.npy" for pdb in df[df["phase"] == "eval2"]["sample_name"].values])

    # Load pre-computed embeddings
    train_embeddings = load_mpnn_embeddings([f"{cfg.embeddings_dir}/{pdb}.npy" for pdb in train_subsample_df["pdb_key"].values])
    eval_embeddings = load_mpnn_embeddings([f"{cfg.embeddings_dir}/{pdb}.npy" for pdb in eval_subsample_df["pdb_key"].values])
    eval2_embeddings = load_mpnn_embeddings([f"{cfg.embeddings_dir}/{pdb}.npy" for pdb in eval2_subsample_df["pdb_key"].values])

    # Calculate FPD scores
    train_fpd_scores = []
    eval_fpd_scores = []
    eval2_fpd_scores = []

    for i in range(3): # 3 layers
        train_fpd_score = fpd_safe(train_samp_embeddings[:, i], train_embeddings[:, i])
        train_fpd_scores.append(train_fpd_score)

        eval_fpd_score = fpd_safe(eval_samp_embeddings[:, i], eval_embeddings[:, i])
        eval_fpd_scores.append(eval_fpd_score)

        eval2_fpd_score = fpd_safe(eval2_samp_embeddings[:, i], eval2_embeddings[:, i])
        eval2_fpd_scores.append(eval2_fpd_score)

   # TODO: Add Wandb logging here -- y-axis can be the FPD score, x-axis can be the layer number
    # Set up wandb logging
    if not cfg.no_wandb:
        # Create wandb dir
        wandb_dir = str(Path(cfg.out_dir))
        Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

        # Set wandb cache directory
        wandb_cache_dir = str(Path(cfg.out_dir, "cache", "wandb"))
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

        wandb.init(
            project=cfg.project,
            entity=cfg.wandb_id,
            name=cfg.exp_name,
            group=cfg.group,
            config=cfg_dict,
            dir=wandb_dir,
        )

        # Log FPD scores
        for phase, fpd_scores in zip(["train", "eval", "eval2"], [train_fpd_scores, eval_fpd_scores, eval2_fpd_scores]):
            for i, fpd_score in enumerate(fpd_scores):
                wandb.log({f"{phase}/mpnn_layer_{i}_fpd": fpd_score})

        wandb.finish()


def subsample_embeddings(lengths_csv_path: str, frac: float, phase: str) -> pd.DataFrame:
    """
    Returns a subsampled dataframe (pdb_key, length, phase).
    """
    lengths_df = pd.read_csv(lengths_csv_path)
    subsampled_df = lengths_df.sample(frac=frac)
    subsampled_df["phase"] = phase
    return subsampled_df


def fpd_safe(*args, **kwargs):
    """
    fpd() wrapper defaulting to nan if ValueError is raised.
    """
    try:
        return fpd(*args, **kwargs)
    except ValueError as e:
        print(f"Error calculating FPD: {e}, defaulting to nan.")
        return np.nan


if __name__ == "__main__":
    main()
