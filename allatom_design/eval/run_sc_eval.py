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

from allatom_design.eval import eval_metrics
from allatom_design.eval.eval_setup_utils import get_pdb_files, wandb_setup
from allatom_design.eval.folding_utils import get_struct_pred_model


@hydra.main(config_path="../configs/eval", config_name="eval_seq_des_training", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Simple script for running self-consistency eval on a set of PDBs using the sequence in the PDBs.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging
    log_dir = wandb_setup(no_wandb=cfg.no_wandb, out_dir=cfg.out_dir,
                          project=cfg.project, wandb_id=cfg.wandb_id, exp_name=cfg.exp_name, group=cfg.group,
                          cfg_dict=cfg_dict)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load in PDB files to eval on ###
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Load in structure prediction model for co-design self-consistency evals
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    sc_info = eval_metrics.run_self_consistency_eval(
        pdb_files,
        None,
        struct_pred_model,
        device,
        out_dir=log_dir,
        temp_dir=f"{log_dir}/tmp"
    )

    # Aggregate results
    sc_metrics = defaultdict(list)
    for pdb in pdb_files:
        for k, v in sc_info[pdb]["sc_metrics"].items():
            sc_metrics[f"{k}"].append(v.item())

    # Update metrics
    out_metrics = {f"seq_des/mean/{k}": np.mean(v) for k, v in sc_metrics.items()}
    out_metrics.update({f"seq_des/median/{k}": np.median(v) for k, v in sc_metrics.items()})

    # Dump to output directory
    with open(os.path.join(log_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(out_metrics, f)

    # Log metrics to wandb
    if not cfg.no_wandb:
        wandb.log(out_metrics, step=0)


if __name__ == "__main__":
    main()
