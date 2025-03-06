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
from allatom_design.eval.eval_fpd import fpd_safe
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

    # Get precomputed embedding keys for each phase
    phases = ["train", "eval", "eval2"]
    precomputed_key_files = [f"{cfg.fpd_embeddings_dir}/precomputed_{phase}_pdb_keys.csv" for phase in phases]
    phase_to_keys = {phase: pd.read_csv(file_path, header=None).values.flatten() for phase, file_path in zip(phases, precomputed_key_files)}

    if cfg.annotation_csv is not None:
        # Get sequence length, designability info, and other useful info directly from annotation csv
        df = pd.read_csv(cfg.annotation_csv)

        # subset to only precomputed pdb keys
        all_pdb_keys = set().union(*phase_to_keys.values())
        df = df[df["pdb_key"].isin(all_pdb_keys)]
    else:
        # Get sequence lengths for all pdb keys and store as dataframe
        df_dict = {"phase": [], "pdb_key": [], "seq_length": []}
        for phase in phases:
            cached_lengths_file = f"{cfg.fpd_embeddings_dir}/precomputed_{phase}_pdb_keys_lengths.csv"
            if Path(cached_lengths_file).exists():
                # Check if cached lengths file exists
                print(f"Loading cached lengths from {cached_lengths_file}")
                lengths_df = pd.read_csv(cached_lengths_file)
                lengths = lengths_df["seq_length"].values
                assert (phase_to_keys[phase] == lengths_df["pdb_key"].values).all(), "Keys in cached lengths csv do not match keys in precomputed pdb keys csv."
            else:
                # Otherwise, compute lengths and cache them
                if cfg.use_esm3:
                    _, lengths = load_esm3_embeddings([f"{cfg.fpd_embeddings_dir}/esm3/{pdb}.pkl" for pdb in phase_to_keys[phase]])
                elif cfg.use_mpnn:
                    _, lengths = load_mpnn_embeddings([f"{cfg.fpd_embeddings_dir}/mpnn/{pdb}.npy" for pdb in phase_to_keys[phase]])
                else:
                    raise ValueError("Must set at least one of use_esm3 or use_mpnn to True.")

                # Cache the lengths
                lengths_df = pd.DataFrame({
                    "pdb_key": phase_to_keys[phase],
                    "seq_length": lengths
                })
                lengths_df.to_csv(cached_lengths_file, index=False)
                print(f"Cached lengths to {cached_lengths_file}")

            df_dict["phase"].extend([phase] * len(phase_to_keys[phase]))
            df_dict["pdb_key"].extend(phase_to_keys[phase])
            df_dict["seq_length"].extend(lengths)
        df = pd.DataFrame(df_dict)

    # Filter by designability / radius of gyration
    if cfg.max_scrmsd is not None or cfg.max_rel_rog is not None:
        assert cfg.annotation_csv is not None, "Must provide annotation csv to filter by designability / radius of gyration."
        if cfg.max_scrmsd is not None:
            print(f"Filtering by max scrmsd: {cfg.max_scrmsd}")
            df = df[df["sc_ca_rmsd"] <= cfg.max_scrmsd]
        if cfg.max_rel_rog is not None:
            print(f"Filtering by max rel_rog: {cfg.max_rel_rog}")
            df = df[df["rel_rog"] <= cfg.max_rel_rog]

    # Subset by length
    if cfg.subset_length_range is not None:
        min_len, max_len = cfg.subset_length_range
        df = df[(df["seq_length"] >= min_len) & (df["seq_length"] <= max_len)]

    # Randomly subsample
    subsample_fracs = {"train": cfg.pct_subsample_train, "eval": cfg.pct_subsample_eval, "eval2": cfg.pct_subsample_eval}
    df = df.groupby("phase", group_keys=False)[["phase", "pdb_key", "seq_length"]].apply(lambda x: x.sample(frac=subsample_fracs[x.name], replace=False, random_state=cfg.subsample_seed))
    df = df.sort_values("seq_length").reset_index(drop=True)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load precomputed embeddings (for FPD reference)
    phase_to_p_embeds = defaultdict(dict)
    for phase in phases:
        phase_pdb_keys = df[df["phase"] == phase]["pdb_key"].values
        if cfg.use_esm3:
            phase_to_p_embeds[phase]["esm3"], _ = load_esm3_embeddings([f"{cfg.fpd_embeddings_dir}/esm3/{pdb}.pkl" for pdb in phase_pdb_keys])
        if cfg.use_mpnn:
            phase_to_p_embeds[phase]["mpnn"], _ = load_mpnn_embeddings([f"{cfg.fpd_embeddings_dir}/mpnn/{pdb}.npy" for pdb in phase_pdb_keys])

    # Get checkpoints from denoiser training run
    ema_ckpt_dir = f"{cfg.denoiser_train_dir}/checkpoints/ema"
    if Path(ema_ckpt_dir).exists():
        # Use EMA checkpoints if they exist
        print(f"Using EMA checkpoints from {ema_ckpt_dir}")
        pattern = re.compile(r"ad-step(\d+)-epoch(\d+)-ema(\d+\.\d+)\.ckpt$")  # match checkpoints of the form ad-step{step}-epoch{epoch}-ema{decay_rate}.ckpt
        ad_ckpts = glob.glob(f"{ema_ckpt_dir}/*.ckpt")
        ad_ckpts = natsorted([ckpt for ckpt in ad_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]
    else:
        print(f"Using non-EMA checkpoints from {cfg.denoiser_train_dir}/checkpoints")
        pattern = re.compile(r"ad-step(\d+)-epoch(\d+)\.ckpt$")  # Only match checkpoints of the form ad-step{step}-epoch{epoch}.ckpt
        ad_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
        ad_ckpts = natsorted([ckpt for ckpt in ad_ckpts if pattern.search(Path(ckpt).name)])[::cfg.eval_every_n_ckpts]

    # Sample structures for FPD calculation
    print(f"Sampling {len(df[df['phase'] == 'train'])} train, {len(df[df['phase'] == 'eval'])} eval, {len(df[df['phase'] == 'eval2'])} eval2 structures for FPD calculation.")
    df = df.sort_values("seq_length").reset_index(drop=True)  # sort by length for sampling efficiency

    pbar = tqdm(ad_ckpts, desc="Evaluating checkpoints")
    for ad_ckpt in pbar:
        match = pattern.search(Path(ad_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))

        pbar.set_postfix_str(f"Step: {global_step}, Epoch: {epoch}")

        # Skip if global_step is before start_step
        if (cfg.start_step is not None) and (global_step < cfg.start_step):
            continue

        # Create output directory for this epoch
        log_dir_i = f"{log_dir}/step_{global_step}"
        Path(log_dir_i).mkdir(parents=True, exist_ok=True)
        sampled_pdbs_dir_i = f"{log_dir_i}/sampled_pdbs"
        Path(sampled_pdbs_dir_i).mkdir(parents=True, exist_ok=True)

        # Load denoiser model and dataset
        lit_ad_model = LitAtomDenoiser.load_from_checkpoint(ad_ckpt).eval()

        # === Sample new backbones ===
        pbar_i = tqdm(total=len(df), desc="Sampling backbones")
        all_lengths = df["seq_length"].values
        pdb_keys = df["pdb_key"].values
        sampled_pdbs = []

        L.seed_everything(cfg.seed)  # Reset the random seed for each checkpoint
        for i in range(0, len(all_lengths), cfg.batch_size):
            lengths = torch.tensor(all_lengths[i : i + cfg.batch_size], dtype=torch.long).to(device)
            B = lengths.shape[0]

            residue_index = torch.arange(lengths.max(), dtype=torch.long).to(device)
            residue_index = torch.cat([residue_index, torch.zeros(8 - (residue_index.shape[0] % 8), dtype=torch.long).to(device)])
            residue_index = residue_index[None].expand(B, -1)

            # Create timesteps
            t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)
            t_bb = t_bb[None].expand(B, -1).to(device)
            timesteps = t_bb

            # Create noise schedule
            noise_schedule = NoiseSchedule(cfg.noise_schedule)

            # Create churn config
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

            filenames = [f"{sampled_pdbs_dir_i}/step_{global_step}_{pdb_keys[i + j]}_L{lengths[j]}.pdb" for j in range(B)]
            sampled_pdbs.extend(filenames)
            AtomDenoiser.save_samples_to_pdb(samples, filenames)

            pbar_i.update(B)
        pbar_i.close()

        df["sample_name"] = [Path(pdb).stem for pdb in sampled_pdbs]

        # Prepare to compute FPD metrics
        fpd_metrics = {}
        phase_to_s_embeds = defaultdict(dict)

        # === MPNN ===
        if cfg.use_mpnn:
            mpnn_samp_dir = f"{log_dir_i}/mpnn"
            Path(mpnn_samp_dir).mkdir(parents=True, exist_ok=True)

            mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
            mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
            mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

            create_mpnn_embeddings(
                mpnn_model,
                pdb_paths=sampled_pdbs,
                out_dir=str(mpnn_samp_dir),
                device=device,
                cfg=mpnn_cfg
            )

            for phase in phases:
                phase_pdb_keys = df[df["phase"] == phase]["sample_name"].values
                phase_to_s_embeds[phase]["mpnn"], _ = load_mpnn_embeddings([f"{mpnn_samp_dir}/{pdb}.npy" for pdb in phase_pdb_keys])

            # Compute FPD for MPNN layers
            for phase in phases:
                for i in range(3):
                    fpd = fpd_safe(phase_to_s_embeds[phase]["mpnn"][:, i], phase_to_p_embeds[phase]["mpnn"][:, i])
                    fpd_metrics[f"{phase}/mpnn_layer_{i}_fpd"] = fpd

        # === ESM3 ===
        if cfg.use_esm3:
            esm3_samp_dir = f"{log_dir_i}/esm3"
            Path(esm3_samp_dir).mkdir(parents=True, exist_ok=True)

            login(token=os.environ["HUGGINGFACE_TOKEN"])
            model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
            vqvae_encoder = model.get_structure_encoder()

            create_esm3_embeddings(
                vqvae_encoder,
                pdb_paths=sampled_pdbs,
                out_dir=esm3_samp_dir,
                device=device
            )

            for phase in phases:
                phase_pdb_keys = df[df["phase"] == phase]["sample_name"].values
                phase_to_s_embeds[phase]["esm3"], _ = load_esm3_embeddings([f"{esm3_samp_dir}/{pdb}.pkl" for pdb in phase_pdb_keys])

            # Compute FPD for ESM3
            for phase in phases:
                fpd = fpd_safe(phase_to_s_embeds[phase]["esm3"],phase_to_p_embeds[phase]["esm3"])
                fpd_metrics[f"{phase}/esm3_fpd"] = fpd

        # Dump metrics to pickle
        with open(f"{log_dir_i}/fpd_scores.pkl", "wb") as f:
            pickle.dump(fpd_metrics, f)

        # Log metrics to wandb
        if not cfg.no_wandb:
            fpd_metrics["trainer/global_step"] = global_step
            fpd_metrics["trainer/epoch"] = epoch
            wandb.log(fpd_metrics, step=global_step)

    wandb.finish()


if __name__ == "__main__":
    main()
