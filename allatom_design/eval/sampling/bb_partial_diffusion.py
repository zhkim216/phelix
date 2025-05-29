import pickle
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.bb_gen_utils import get_bb_gen_model, run_bb_partial_diffusion
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             process_pdb_files,
                                                             wandb_setup)
import shutil


@hydra.main(config_path="../../configs/eval/sampling", config_name="bb_partial_diffusion", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for running partial diffusion on an input structure.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging / output directory
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    ### Load in PDB files to eval on ###
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Process PDB files into .npz structure format
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg, keep_order=True)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in atom denoiser
    bb_gen_model = get_bb_gen_model(cfg.bb_gen_cfg, device=device)
    sampled_pdb_paths = run_bb_partial_diffusion(bb_gen_model["model"], bb_gen_model["data_cfg"], bb_gen_model["sampling_cfg"], device, processed_struct_files, n_samples_per_pdb=cfg.n_samples_per_pdb, out_dir=log_dir)

    # Rename files to expected format for multistate seq des
    for pdb_file in pdb_files:
        record_id = Path(pdb_file).stem
        pdb_out_dir = f"{cfg.base_out_dir}/{record_id}"
        Path(pdb_out_dir).mkdir(parents=True, exist_ok=True)

        # Copy over original pdb file
        shutil.copy(pdb_file, f"{pdb_out_dir}/{record_id}.cif")

        # Copy over sampled pdb files
        for sampled_pdb_path in sampled_pdb_paths:
            if record_id in sampled_pdb_path:
                shutil.copy(sampled_pdb_path, f"{pdb_out_dir}/{Path(sampled_pdb_path).stem}.pdb")

    # Delete log dir
    shutil.rmtree(log_dir)

    if not cfg.wandb.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
