import shutil
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             wandb_setup)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model


@hydra.main(config_path="../configs/eval", config_name="predict_structures", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for predicting structures for a set of PDBs.
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

    # Load in PDB file to eval on
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Save original PDBs
    input_pdb_dir = f"{log_dir}/input_pdbs"
    Path(input_pdb_dir).mkdir(parents=True, exist_ok=True)
    for pdb_file in pdb_files:
        shutil.copy(pdb_file, f"{input_pdb_dir}/{Path(pdb_file).name}")

    # Load structure prediction model for self-consistency evaluation
    pred_out_dir = f"{log_dir}/preds"  # directory for structure predictions
    Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
        pdb_files,
        struct_pred_model,
        cfg.pdb_processing_cfg,
        out_dir=pred_out_dir)

    # Save metrics as CSV
    metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
    metrics_df.to_csv(f"{log_dir}/self_consistency_metrics.csv", index=False)

    if not cfg.wandb.no_wandb:
        # Aggregate results
        sc_metrics = defaultdict(list)
        for record_id, metrics in id_to_metrics.items():
            for k, v in metrics.items():
                sc_metrics[f"{k}"].append(v)

        # Update metrics
        out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k != "record_id"}
        out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k != "record_id"})

        # Log metrics to wandb
        wandb.log(out_metrics, step=0)


if __name__ == "__main__":
    main()
