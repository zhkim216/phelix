import copy
import glob
import re
import shutil
from pathlib import Path

import hydra
import lightning as L
import torch
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

import allatom_design.model.ema.phema as phema
from tqdm import tqdm
from allatom_design.model.atom_denoiser.lit_ad_model import LitAtomDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
import numpy as np


@hydra.main(config_path="../configs/eval", config_name="reconstruct_phema", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for running EDM2's post hoc EMA.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        cfg.out_dir = f"{cfg.denoiser_train_dir}/post_hoc_ema_ckpts"

    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Gather EMA checkpoints
    ema_ckpts = natsorted(glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/ema_tracker/*.ckpt"))

    # Compute posthoc EMA coefficients based on input steps/stds and desired output step/stds
    out_step = cfg.out_step
    if cfg.out_stds is None:
        out_stds = np.arange(*cfg.out_stds_arange)
    else:
        out_stds = cfg.out_stds
    print(f"Computing posthoc EMA coefficients for: step {out_step} \nStds: {out_stds}...")

    in_steps = []
    in_stds = []
    in_ckpts = []
    for ckpt in tqdm(ema_ckpts, desc="Loading checkpoints for computing phema coefficients"):
        ckpt_dict = torch.load(ckpt, map_location="cpu")
        if ckpt_dict["global_step"] > out_step:
            # only consider ckpts up to the desired output step
            break

        stds = ckpt_dict["ema_state"]["stds"]
        in_stds += stds
        in_steps += [ckpt_dict["global_step"]] * len(stds)
        in_ckpts.append(ckpt)

    coefs = phema.solve_posthoc_coefficients(in_steps, in_stds, out_step, out_stds)

    # Initialize output state dicts
    out_state_dicts = []

    for _ in out_stds:
        out_state_dict = copy.deepcopy(ckpt_dict["ema_state"]["emas"][0])
        for p in out_state_dict.values():
            p.data.zero_()
        out_state_dicts.append(out_state_dict)

    # Compute EMAs
    ei = 0
    for ckpt in tqdm(in_ckpts, desc="Computing EMA weights"):
        ckpt_dict = torch.load(ckpt, map_location="cpu")
        for i in range(len(ckpt_dict["ema_state"]["emas"])):
            ema_weights_in = ckpt_dict["ema_state"]["emas"][i]

            # Update output state dicts
            for j, out_state_dict in enumerate(out_state_dicts):
                for n, p in out_state_dict.items():
                    p += coefs[ei, j] * ema_weights_in[n]
            ei += 1

    # Gather denoiser checkpoints
    pattern = re.compile(r"(?:ad|sd)-step(\d+)-epoch(\d+)\.ckpt$")  # only consider checkpoints of the form ad-step{step}-epoch{epoch}.ckpt or sd-step{step}-epoch{epoch}.ckpt
    denoiser_ckpts = glob.glob(f"{cfg.denoiser_train_dir}/checkpoints/*.ckpt")
    denoiser_ckpts = natsorted([ckpt for ckpt in denoiser_ckpts if pattern.search(Path(ckpt).name)])

    for out_std, out_state_dict in zip(out_stds, out_state_dicts):
        # Create new checkpoints containing EMA weights
        out_ckpt = Path(cfg.out_dir, f"ema-step{out_step}-std{out_std:.3f}.ckpt")
        base_ckpt = denoiser_ckpts[-1]  # besides the updated EMA weights, the rest of the model will be the same as the last checkpoint
        ckpt_dict = torch.load(base_ckpt, map_location="cpu")

        # Load the base model
        if cfg.model_type == "seq_denoiser":
            lit_model = LitSeqDenoiser.load_from_checkpoint(base_ckpt)
        elif cfg.model_type == "atom_denoiser":
            lit_model = LitAtomDenoiser.load_from_checkpoint(base_ckpt)
        else:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")

        # Make sure every parameter that required grad in the base model is overridden with the new state dict
        for n, p in lit_model.model.named_parameters():
            if p.requires_grad:
                assert n in out_state_dict

        # Update the model state dict with the new EMA weights
        model_state_dict = lit_model.model.state_dict()
        model_state_dict.update(out_state_dict)
        lit_model.model.load_state_dict(model_state_dict, strict=True)

        # Save the new checkpoint
        ckpt_dict["state_dict"] = lit_model.state_dict()
        torch.save(ckpt_dict, out_ckpt)


if __name__ == "__main__":
    main()
