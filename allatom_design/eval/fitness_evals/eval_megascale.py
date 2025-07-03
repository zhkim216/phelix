import glob
import itertools
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import (wandb_setup, get_conformer_dirs, process_conformer_dirs)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (
    get_seq_des_model, run_seq_des_ensemble)


@hydra.main(config_path="../../configs/eval/fitness_evals", config_name="eval_megascale", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for ranking mutational effects on protein stability using the Megascale dataset (Tsuboyama et al., 2023).
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

    # Load in conformer directories to eval on
    conformer_dirs = get_conformer_dirs(**cfg.input_cfg)

    # Process conformer directories
    pdb_to_processed_conformers = process_conformer_dirs(conformer_dirs, cfg.max_num_conformers, cfg.include_primary_conformer, f"{log_dir}/processed_structures", cfg.pdb_processing_cfg)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Run sequence design model
    outputs = run_seq_des_ensemble(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                                    pdb_to_processed_conformers=pdb_to_processed_conformers, device=device, pos_constraint_df=None,
                                    out_dir=log_dir)

    del seq_des_model



if __name__ == "__main__":
    main()
