import glob
import pickle
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
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from allatom_design.data.conditioning_labels import create_cond_labels_input
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.proteinmpnn_utils import load_mpnn
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.allatom_denoiser.allatom_model import AllAtomModel
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


@hydra.main(config_path="../../configs/eval/sampling", config_name="draw_allatom_samples", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Construct allatom model
    lit_ad_model = LitAtomDenoiser.load_from_checkpoint(cfg.ad_ckpt).eval()
    device = lit_ad_model.device
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.sd_ckpt).eval()
    allatom_model = AllAtomModel(lit_ad_model, lit_sd_model)

    # Create out dirs in atom denoiser directory and preserve config
    if cfg.out_dir is None:
        model_run_dir = Path(cfg.ad_ckpt).parent.parent
        model_name = Path(cfg.ad_ckpt).stem
        cfg.out_dir = f"{model_run_dir}/draw_samples/{model_name}/{cfg.exp_name}"

    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sample_out_dir = Path(cfg.out_dir, "samples")
    traj_out_dir = Path(cfg.out_dir, "traj")
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(traj_out_dir).mkdir(parents=True, exist_ok=True)

    # Define the range of lengths to sample
    start, end = cfg.length_range
    lengths_to_sample = np.arange(start, end + 1, cfg.length_step_size)
    all_lengths = lengths_to_sample.repeat(cfg.n_samples_per_length)  # get the length of each protein we'll sample
    print(f"Drawing {cfg.n_samples_per_length} samples each of lengths {start} to {end} with step size {cfg.length_step_size}")

    ### SAMPLE ###
    pbar = tqdm(total=len(all_lengths))

    for i in range(0, len(all_lengths), cfg.batch_size):
        # Choose lengths and residue index
        lengths = torch.tensor(all_lengths[i:i + cfg.batch_size], dtype=torch.long).to(device)
        B = lengths.shape[0]
        residue_index = torch.arange(lengths.max(), dtype=torch.long).to(device)
        residue_index = residue_index[None].expand(B, -1)
        cond_labels_in = create_cond_labels_input(B, cfg.cond_labels, device)

        # === Handle atom denoiser inputs === #
        ad_sampling_inputs = {}

        # Create timesteps, separating timesteps for CA and NCO
        t_ca = sampling_utils.get_timesteps_from_schedule(**cfg.ad.ca.timestep_schedule)
        t_nco = sampling_utils.get_timesteps_from_schedule(**cfg.ad.nco.timestep_schedule)
        t_ca = t_ca[None].expand(B, -1).to(device)
        t_nco = t_nco[None].expand(B, -1).to(device)
        ad_sampling_inputs["timesteps"] = (t_ca, t_nco)

        # Create noise schedules for CA and NCO
        ad_sampling_inputs["noise_schedule"] = (NoiseSchedule(cfg.ad.ca.noise_schedule),
                                                NoiseSchedule(cfg.ad.nco.noise_schedule))

        # Create churn configs for CA and NCO
        ad_sampling_inputs["churn_cfg"] = (dict(cfg.ad.ca.churn_cfg), dict(cfg.ad.nco.churn_cfg))

        # === Handle sequence denoiser inputs === #
        sd_sampling_inputs = {}

        # Define sequence denoising timesteps
        t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.sd.seq.timestep_schedule)
        t_seq = t_seq[None].expand(B, -1).to(device)
        sd_sampling_inputs["timesteps"] = t_seq
        sd_sampling_inputs["aatype_decoding_order_mode"] = cfg.sd.seq.aatype_decoding_order_mode

        # Set up sidechain diffusion inputs
        t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.sd.scd.timestep_schedule)
        t_scd = t_scd[None].expand(B, -1).to(device)
        noise_schedule = NoiseSchedule(cfg.sd.scd.noise_schedule)
        churn_cfg = dict(cfg.sd.scd.churn_cfg)
        scd_inputs = {"num_steps": cfg.sd.scd.num_steps,
                      "timesteps": t_scd,
                      "noise_schedule": noise_schedule,
                      "churn_cfg": churn_cfg,
                      "return_scn_diffusion_aux": False}
        sd_sampling_inputs["scd_inputs"] = scd_inputs

        # === Sample from allatom model === #
        x_denoised, aatype_denoised, aux = allatom_model.sample(lengths=lengths,
                                                                residue_index=residue_index,
                                                                ad_sampling_inputs=ad_sampling_inputs,
                                                                sd_sampling_inputs=sd_sampling_inputs,
                                                                cond_labels=cond_labels_in)


        samples = {"x_denoised": x_denoised,
                   "aatype_denoised": aatype_denoised,
                   "seq_mask": aux["seq_mask"],
                   "residue_index": residue_index}
        samples = {k: v.cpu() if v is not None else v for k, v  in samples.items()}

        # Save samples
        filenames = [f"{sample_out_dir}/sample_len{lengths[j]}_{i + j}.pdb" for j in range(B)]
        AllAtomModel.save_samples_to_pdb(samples, filenames)

        pbar.update(B)

    pbar.close()



if __name__ == "__main__":
    main()
