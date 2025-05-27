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

    # Process PDB files into .npz structure format
    processed_struct_files = process_pdb_files([cfg.pdb_path], processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg)
    processed_struct_files = np.repeat(processed_struct_files, cfg.n_samples_per_pdb).tolist()

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in atom denoiser
    bb_gen_model = get_bb_gen_model(cfg.bb_gen_cfg, device=device)

    sampled_pdb_paths = run_bb_partial_diffusion(bb_gen_model["model"], bb_gen_model["data_cfg"], bb_gen_model["sampling_cfg"], device, processed_struct_files, log_dir)

    if not cfg.wandb.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
