"""
Utils for sampling from backbone generation models.
"""
from functools import partial
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.eval.eval_utils import sampling_utils
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.atom_denoiser.ad_model import AtomDenoiser
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser


def get_bb_gen_model(cfg: DictConfig, device: str) -> dict[str, Any]:
    """
    Load in a backbone generation model.
    """
    lit_ad_model = LitAtomDenoiser.load_from_checkpoint(cfg.ckpt_path).eval()
    sampling_cfg = OmegaConf.load(cfg.sampling_cfg)
    sampling_cfg = OmegaConf.merge(sampling_cfg, cfg.overrides)
    bb_gen_model = {"model": lit_ad_model.model,
                    "sampling_cfg": sampling_cfg,
                    "device": device}

    return bb_gen_model


def run_bb_uncond_sampling(model: AtomDenoiser,
                           cfg: DictConfig,
                           lengths: list[int],
                           device: str,
                           out_dir: str,
                           save_traj_inputs: dict[str, Any] | None = None) -> list[str]:
    """
    Run unconditional sampling from a backbone generation model.
    """
    # Set up output directories
    sample_out_dir = Path(out_dir, "samples")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    if save_traj_inputs is not None:
        # Save diffusion trajectories
        traj_out_dir = Path(out_dir, "traj")
        Path(traj_out_dir).mkdir(parents=True, exist_ok=True)

    sampled_pdb_paths = []
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    for i in tqdm(range(0, len(lengths), cfg.batch_size)):
        lengths_batch = lengths[i:i + cfg.batch_size]
        B = lengths_batch.shape[0]
        residue_index = torch.arange(lengths.max(), dtype=torch.long, device=device)  # assume residue index is 0 to max length
        residue_index = residue_index[None].expand(B, -1)

        # Set up backbone diffusion inputs
        diffusion_inputs = {}
        diffusion_inputs["num_steps"] = cfg.num_steps
        t_bb = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)  # timesteps for backbone diffusion
        diffusion_inputs["timesteps"] = t_bb[None].expand(B, -1).to(device)
        diffusion_inputs["noise_schedule"] = NoiseSchedule(cfg.noise_schedule)  # noise schedule, used for step_scale
        diffusion_inputs["churn_cfg"] = dict(cfg.churn_cfg)  # churn config for stochastic sampling
        diffusion_inputs["autoguidance_cfg"] = dict(cfg.autoguidance_cfg)  # autoguidance config

        # Create conditioning labels
        cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device)

        # Sample backbones
        x_bb_denoised, aux = model.sample(lengths=lengths_batch,
                                          residue_index=residue_index,
                                          diffusion_inputs=diffusion_inputs,
                                          cond_labels=cond_labels_in)
        samples = {"x_bb_denoised": x_bb_denoised,
                   "seq_mask": aux["seq_mask"],
                   "residue_index": residue_index}
        samples = {k: v.cpu() if v is not None else v for k, v in samples.items()}

        # Save samples
        filenames = [f"{sample_out_dir}/sample_len{lengths_batch[j]}_{i + j}.pdb" for j in range(B)]
        AtomDenoiser.save_samples_to_pdb(samples, filenames)
        sampled_pdb_paths.extend(filenames)

        if save_traj_inputs is not None:
            # Save trajectories
            save_trajs_fn = partial(AtomDenoiser.save_trajs_to_pdb, aux, residue_index=residue_index, chain_index=torch.zeros_like(residue_index),
                                    save_traj_mask=save_traj_inputs["save_traj_mask"], save_traj_steps=save_traj_inputs["save_traj_steps"],
                                    traj_conect=save_traj_inputs["traj_conect"], align_models_to_idx=save_traj_inputs["align_traj_to_last_step"])
            save_trajs_fn(x_traj_key="x1_bb_traj", filenames=[f"{traj_out_dir}/x1_traj_sample_len{lengths_batch[j]}_{i + j}.pdb" for j in range(B)])
            save_trajs_fn(x_traj_key="xt_bb_traj", filenames=[f"{traj_out_dir}/xt_traj_sample_len{lengths_batch[j]}_{i + j}.pdb" for j in range(B)])


    return sampled_pdb_paths
