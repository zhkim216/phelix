import glob
import os
import pickle
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
from huggingface_hub import login
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.esm3_utils import (create_esm3_embeddings,
                                            load_esm3_embeddings)
from allatom_design.eval.eval_fpd import fpd_safe, subsample_embeddings
from allatom_design.eval.eval_metrics import fpd
from allatom_design.eval.proteinmpnn_utils import (create_mpnn_embeddings,
                                                   load_mpnn,
                                                   load_mpnn_embeddings)
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from esm3.esm.models.esm3 import ESM3


@hydra.main(config_path="../configs/eval", config_name="eval_fpd_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating FPD for backbone generation.
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

    # Subsample embeddings
    train_subsample_df = subsample_embeddings(cfg.train_lengths_csv, cfg.pct_subsample_train, phase="train")
    eval_subsample_df = subsample_embeddings(cfg.eval_lengths_csv, cfg.pct_subsample_eval, phase="eval")
    eval2_subsample_df = subsample_embeddings(cfg.eval2_lengths_csv, cfg.pct_subsample_eval, phase="eval2")

    print(f"Sampling {len(train_subsample_df)} train, {len(eval_subsample_df)} eval, {len(eval2_subsample_df)} eval2 structures for FPD calculation.")

    # Concatenate into one dataframe
    df = pd.concat([train_subsample_df, eval_subsample_df, eval2_subsample_df], ignore_index=True)
    df = df.sort_values("length").reset_index(drop=True)  # sort by length for sampling efficiency

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get checkpoints from denoiser training run
    pattern = re.compile(r"ad-epoch\d+\.ckpt$")  # only consider ckpts of form ad-epochXXXX.ckpt
    ad_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
    ad_ckpts = natsorted([ckpt for ckpt in ad_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    pbar = tqdm(ad_ckpts, desc="Evaluating checkpoints")
    for ad_ckpt in pbar:
        L.seed_everything(cfg.seed)  # Reset the random seed for each checkpoint

        # Skip if epoch is before start_epoch
        epoch = int(Path(ad_ckpt).stem.replace("ad-epoch", ""))
        pbar.set_postfix_str(f"Epoch: {epoch}")
        if (cfg.start_epoch is not None) and (epoch < cfg.start_epoch):
            continue

        # Create output directory for this epoch
        log_dir_i = f"{log_dir}/epoch_{epoch}"
        Path(log_dir_i).mkdir(parents=True, exist_ok=True)
        sampled_pdbs_dir_i = f"{log_dir_i}/sampled_pdbs"
        Path(sampled_pdbs_dir_i).mkdir(parents=True, exist_ok=True)

        # Load denoiser model and dataset
        lit_ad_model = LitAtomDenoiser.load_from_checkpoint(ad_ckpt).eval()

        ### BEGIN EVAL ###
        pbar = tqdm(total=len(df), desc="Sampling backbones")
        all_lengths = df["length"].values
        pdb_keys = df["pdb_key"].values
        sampled_pdbs = []

        for i in range(0, len(all_lengths), cfg.batch_size):
            lengths = torch.tensor(all_lengths[i : i + cfg.batch_size], dtype=torch.long).to(device)
            B = lengths.shape[0]

            residue_index = torch.arange(lengths.max(), dtype=torch.long).to(device)
            # TEMP: pad residue index to the next largest multiple of 8
            residue_index = torch.cat([residue_index, torch.zeros(8 - (residue_index.shape[0] % 8), dtype=torch.long).to(device)])
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

            filenames = [f"{sampled_pdbs_dir_i}/epoch_{epoch}_{pdb_keys[i + j]}_L{lengths[j]}.pdb" for j in range(B)]
            sampled_pdbs.extend(filenames)
            AtomDenoiser.save_samples_to_pdb(samples, filenames)

            pbar.update(B)
        pbar.close()

        # Add sampled pdb names to df
        df["sample_name"] = [Path(pdb).stem for pdb in sampled_pdbs]

        # Create output directories
        samp_embeddings_dir = Path(log_dir_i, "mpnn_embeddings")
        samp_embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Load MPNN model
        mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
        mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

        # Load ESM3 model
        login(token=os.environ["HUGGINGFACE_TOKEN"])
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        vqvae_encoder = model.get_structure_encoder()

        # Run MPNN to create embeddings
        create_mpnn_embeddings(mpnn_model, pdb_paths=sampled_pdbs, out_dir=str(samp_embeddings_dir), device=device, cfg=mpnn_cfg)
        create_esm3_embeddings(vqvae_encoder, pdb_paths=sampled_pdbs, out_dir=log_dir_i, device=device)

        esm3_samp_embed_path = Path(log_dir_i) / "esm3_embed.pkl"
        esm3_embed_path = Path(cfg.esm3_embed_dir) / "esm3_embed.pkl"

        # Load sampled embeddings
        mpnn_train_samp_embeddings = load_mpnn_embeddings([f"{str(samp_embeddings_dir)}/{Path(pdb).stem}.npy" for pdb in df[df["phase"] == "train"]["sample_name"].values])
        mpnn_eval_samp_embeddings = load_mpnn_embeddings([f"{str(samp_embeddings_dir)}/{Path(pdb).stem}.npy" for pdb in df[df["phase"] == "eval"]["sample_name"].values])
        mpnn_eval2_samp_embeddings = load_mpnn_embeddings([f"{str(samp_embeddings_dir)}/{Path(pdb).stem}.npy" for pdb in df[df["phase"] == "eval2"]["sample_name"].values])

        esm3_train_samp_embeddings = load_esm3_embeddings([Path(pdb).stem for pdb in df[df["phase"] == "train"]["sample_name"].values], str(esm3_samp_embed_path))
        esm3_eval_samp_embeddings = load_esm3_embeddings([Path(pdb).stem for pdb in df[df["phase"] == "eval"]["sample_name"].values], str(esm3_samp_embed_path))
        esm3_eval2_samp_embeddings = load_esm3_embeddings([Path(pdb).stem for pdb in df[df["phase"] == "eval2"]["sample_name"].values], str(esm3_samp_embed_path))

        # Load pre-computed embeddings
        mpnn_train_embeddings = load_mpnn_embeddings([f"{cfg.mpnn_embeddings_dir}/{pdb}.npy" for pdb in train_subsample_df["pdb_key"].values])
        mpnn_eval_embeddings = load_mpnn_embeddings([f"{cfg.mpnn_embeddings_dir}/{pdb}.npy" for pdb in eval_subsample_df["pdb_key"].values])
        mpnn_eval2_embeddings = load_mpnn_embeddings([f"{cfg.mpnn_embeddings_dir}/{pdb}.npy" for pdb in eval2_subsample_df["pdb_key"].values])

        esm3_train_embeddings = load_esm3_embeddings(train_subsample_df["pdb_key"].values, str(esm3_embed_path))
        esm3_eval_embeddings = load_esm3_embeddings(eval_subsample_df["pdb_key"].values, str(esm3_embed_path))
        esm3_eval2_embeddings = load_esm3_embeddings(eval2_subsample_df["pdb_key"].values, str(esm3_embed_path))


        # Calculate FPD scores
        mpnn_train_fpd_scores = []
        mpnn_eval_fpd_scores = []
        mpnn_eval2_fpd_scores = []

        for i in range(3): # 3 layers
            train_fpd_score = fpd_safe(mpnn_train_samp_embeddings[:, i], mpnn_train_embeddings[:, i])
            mpnn_train_fpd_scores.append(train_fpd_score)

            eval_fpd_score = fpd_safe(mpnn_eval_samp_embeddings[:, i], mpnn_eval_embeddings[:, i])
            mpnn_eval_fpd_scores.append(eval_fpd_score)

            eval2_fpd_score = fpd_safe(mpnn_eval2_samp_embeddings[:, i], mpnn_eval2_embeddings[:, i])
            mpnn_eval2_fpd_scores.append(eval2_fpd_score)

        esm3_train_fpd_score = fpd_safe(esm3_train_samp_embeddings, esm3_train_embeddings)
        esm3_eval_fpd_score = fpd_safe(esm3_eval_samp_embeddings, esm3_eval_embeddings)
        esm3_eval2_fpd_score = fpd_safe(esm3_eval2_samp_embeddings, esm3_eval2_embeddings)

        # Aggregate metrics
        metrics = {}
        for phase, fpd_scores in zip(["train", "eval", "eval2"], [mpnn_train_fpd_scores, mpnn_eval_fpd_scores, mpnn_eval2_fpd_scores]):
            for i, fpd_score in enumerate(fpd_scores):
                metrics[f"fpd/{phase}/mpnn_layer_{i}"] = fpd_score

        metrics[f"fpd/train/esm3"] = esm3_train_fpd_score
        metrics[f"fpd/eval/esm3"] = esm3_eval_fpd_score
        metrics[f"fpd/eval2/esm3"] = esm3_eval2_fpd_score

        # Dump metrics to pickle
        with open(f"{log_dir_i}/metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)

        # Log metrics to wandb
        if not cfg.no_wandb:
            # Get global step
            global_step = torch.load(ad_ckpt, map_location="cpu")["global_step"]
            metrics["trainer/global_step"] = global_step
            metrics["trainer/epoch"] = epoch

            wandb.log(metrics, step=global_step)

    wandb.finish()



if __name__ == "__main__":
    main()
