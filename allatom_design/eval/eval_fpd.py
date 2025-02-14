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
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.eval import sampling_utils
from allatom_design.eval.esm3_utils import (create_esm3_embeddings,
                                            load_esm3_embeddings)
from allatom_design.eval.eval_metrics import fpd
from allatom_design.eval.proteinmpnn_utils import (create_mpnn_embeddings,
                                                   load_mpnn,
                                                   load_mpnn_embeddings)
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from esm3.esm.models.esm3 import ESM3


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

    # Get precomputed embedding keys for each phase
    phases = ["train", "eval", "eval2"]
    precomputed_key_files = [f"{cfg.fpd_embeddings_dir}/precomputed_{phase}_pdb_keys.csv" for phase in phases]
    phase_to_keys = {phase: pd.read_csv(file_path, header=None).values.flatten() for phase, file_path in zip(phases, precomputed_key_files)}

    # Get dataframes with length info
    df = {"phase": [], "pdb_key": [], "seq_length": []}
    for phase in phases:
        if cfg.use_esm3:
            _, lengths = load_esm3_embeddings([f"{cfg.fpd_embeddings_dir}/esm3/{pdb}.pkl" for pdb in phase_to_keys[phase]])
        elif cfg.use_mpnn:
            _, lengths = load_mpnn_embeddings([f"{cfg.fpd_embeddings_dir}/mpnn/{pdb}.npy" for pdb in phase_to_keys[phase]])
        else:
            raise ValueError("Must use at least one embedding type.")
        df["phase"].extend([phase] * len(phase_to_keys[phase]))
        df["pdb_key"].extend(phase_to_keys[phase])
        df["seq_length"].extend(lengths)
    df = pd.DataFrame(df)

    # Subset by length
    if cfg.subset_length_range is not None:
        min_len, max_len = cfg.subset_length_range
        df = df[(df["seq_length"] >= min_len) & (df["seq_length"] <= max_len)]

    # Randomly subsample
    subsample_fracs = {"train": cfg.pct_subsample_train, "eval": cfg.pct_subsample_eval, "eval2": cfg.pct_subsample_eval}
    df = df.groupby("phase", group_keys=False).apply(lambda x: x.sample(frac=subsample_fracs[x.name], replace=False))  # subsample

    # Load in precomputed embeddings
    phase_to_p_embeds = defaultdict(dict)  # phase -> model -> precomputed embeddings
    for phase in phases:
        phase_pdb_keys = df[df["phase"] == phase]["pdb_key"].values
        if cfg.use_esm3:
            phase_to_p_embeds[phase]["esm3"], _ = load_esm3_embeddings([f"{cfg.fpd_embeddings_dir}/esm3/{pdb}.pkl" for pdb in phase_pdb_keys])
        if cfg.use_mpnn:
            phase_to_p_embeds[phase]["mpnn"], _ = load_mpnn_embeddings([f"{cfg.fpd_embeddings_dir}/mpnn/{pdb}.npy" for pdb in phase_pdb_keys])

    # Sample structures for FPD calculation
    print(f"Sampling {len(df[df['phase'] == 'train'])} train, {len(df[df['phase'] == 'eval'])} eval, {len(df[df['phase'] == 'eval2'])} eval2 structures for FPD calculation.")
    df = df.sort_values("seq_length").reset_index(drop=True)  # sort by length for sampling efficiency

    # === Sample structures === #
    pbar = tqdm(total=len(df), desc="Sampling backbones")
    all_lengths = df["seq_length"].values
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

        filenames = [f"{sample_out_dir}/{pdb_keys[i + j]}_L{lengths[j]}.pdb" for j in range(B)]
        sampled_pdbs.extend(filenames)
        AtomDenoiser.save_samples_to_pdb(samples, filenames)

        pbar.update(B)
    pbar.close()

    # Add sampled pdb names to df
    df["sample_name"] = [Path(pdb).stem for pdb in sampled_pdbs]

    ### Compute FPD ###
    fpd_metrics = {}  # f"{phase}/{model}" -> FPD
    phase_to_s_embeds = defaultdict(dict)  # phase -> model -> sampled embeddings
    if cfg.use_mpnn:
        mpnn_sampled_embeddings_dir = f"{cfg.out_dir}/mpnn"
        Path(mpnn_sampled_embeddings_dir).mkdir(parents=True, exist_ok=True)

        # Load MPNN
        device = torch.device("cuda")
        mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
        mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

        # Create MPNN embeddings
        create_mpnn_embeddings(mpnn_model, pdb_paths=sampled_pdbs, out_dir=str(mpnn_sampled_embeddings_dir), device=device, cfg=mpnn_cfg)

        # Load sampled embeddings
        for phase in phases:
            phase_pdb_keys = df[df["phase"] == phase]["sample_name"].values
            phase_to_s_embeds[phase]["mpnn"], _ = load_mpnn_embeddings([f"{mpnn_sampled_embeddings_dir}/{pdb}.npy" for pdb in phase_pdb_keys])

        # Calculate FPD
        for phase in phases:
            for i in range(3):
                fpd = fpd_safe(phase_to_s_embeds[phase]["mpnn"][:, i], phase_to_p_embeds[phase]["mpnn"][:, i])
                fpd_metrics[f"{phase}/mpnn_layer_{i}_fpd"] = fpd

    if cfg.use_esm3:
        esm3_sampled_embeddings_dir = f"{cfg.out_dir}/esm3"
        Path(esm3_sampled_embeddings_dir).mkdir(parents=True, exist_ok=True)

        # Load ESM3
        login(token=os.environ["HUGGINGFACE_TOKEN"])
        device = "cuda"
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        vqvae_encoder = model.get_structure_encoder()

        # Create ESM3 embeddings
        create_esm3_embeddings(vqvae_encoder, pdb_paths=sampled_pdbs, out_dir=esm3_sampled_embeddings_dir, device=device)

        # Load sampled embeddings
        for phase in phases:
            phase_pdb_keys = df[df["phase"] == phase]["sample_name"].values
            phase_to_s_embeds[phase]["esm3"], _ = load_esm3_embeddings([f"{esm3_sampled_embeddings_dir}/{pdb}.pkl" for pdb in phase_pdb_keys])

        # Calculate FPD
        for phase in phases:
            fpd = fpd_safe(phase_to_s_embeds[phase]["esm3"], phase_to_p_embeds[phase]["esm3"])
            fpd_metrics[f"{phase}/esm3_fpd"] = fpd

    # Dump FPD scores to pickle
    with open(f"{cfg.out_dir}/fpd_scores.pkl", "wb") as f:
        pickle.dump(fpd_metrics, f)

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
        wandb.log(fpd_metrics)
        wandb.finish()


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
