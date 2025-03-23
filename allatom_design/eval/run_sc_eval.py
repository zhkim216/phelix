import os
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

from allatom_design.data.data import load_feats_from_pdb
from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             wandb_setup)
from allatom_design.eval.eval_utils.fampnn_utils import get_seq_des_model
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model


@hydra.main(config_path="../configs/eval", config_name="run_sc_eval", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Simple script for running self-consistency eval on a set of PDBs and logging to wandb.
    If use_seq_design is False, we will evaluate using the sequence in the PDB.
    If use_seq_design is True, we will evaluate using a sequence designed by seq_des_cfg.model_name ("proteinmpnn" or "fampnn")
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load in PDB files to eval on ###
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Load in sequence design model
    if cfg.use_seq_design:
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
    else:
        seq_des_model = None  # use sequence found in PDB

    # Load in structure prediction model
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    ### CALCULATE STRUCTURE METRICS ###
    per_pdb_info, sample_metrics = eval_metrics.compute_per_pdb_info(pdb_files, seq_des_model, struct_pred_model, device,
                                                                     out_dir=log_dir, temp_dir=f"{log_dir}/tmp")

    # Save per-pdb info
    torch.save(per_pdb_info, f"{log_dir}/per_pdb_info.pt")

    # === Calculate a scalar for each metric to log === #
    metrics = {}
    metrics.update({f"mean/{k}": np.mean(v) for k, v in sample_metrics.items()})
    metrics.update({f"median/{k}": np.median(v) for k, v in sample_metrics.items()})

    # Optionally compute diversity metrics across entire set of PDBs
    if cfg.compute_diversity_metrics:
        diversity_metrics = eval_metrics.run_diversity_eval(pdb_files, per_pdb_info, cfg.diversity_eval, log_dir)
        metrics.update(diversity_metrics)

    # Log aggregated metrics to wandb
    metrics = {f"sc_eval/{k}": v for k, v in metrics.items()}
    torch.save(metrics, f"{log_dir}/metrics.pt")
    if not cfg.wandb.no_wandb:
        wandb.log(metrics)


if __name__ == "__main__":
    main()
