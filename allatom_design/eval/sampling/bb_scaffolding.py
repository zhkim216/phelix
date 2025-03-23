
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
from allatom_design.eval.eval_utils.bb_gen_utils import (get_bb_gen_model,
                                                         run_backbone_scaffolding)
from allatom_design.eval.eval_utils.eval_setup_utils import wandb_setup
from allatom_design.eval.eval_utils.fampnn_utils import get_seq_des_model
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model


@hydra.main(config_path="../../configs/eval/sampling", config_name="bb_scaffolding", version_base="1.3.2")
def main(cfg: DictConfig):
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

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in atom denoiser
    bb_gen_model = get_bb_gen_model(cfg.bb_gen_cfg, device=device)

    ### Load in motif info CSV ###
    motif_info_df = pd.read_csv(cfg.motif_info_csv)
    motif_info_df["pdb_path"] = motif_info_df["pdb_name"].apply(lambda x: f"{cfg.pdb_dir}/{x}")

    ### Run scaffold sampling ###
    sampled_pdb_paths = run_backbone_scaffolding(bb_gen_model["model"], bb_gen_model["sampling_cfg"], device, motif_info_df, log_dir)

    ### CALCULATE STRUCTURE METRICS ###
    # Load in MPNN + structure prediction model for self-consistency evals
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    per_pdb_info, sample_metrics = eval_metrics.compute_per_pdb_info(sampled_pdb_paths, seq_des_model, struct_pred_model, device,
                                                                     out_dir=log_dir, temp_dir=f"{log_dir}/tmp",
                                                                     nntm_dataset=cfg.nntm_dataset)

    # Save per-pdb info to pt file
    torch.save(per_pdb_info, f"{log_dir}/per_pdb_info.pt")

    # === Calculate a scalar for each metric to log === #
    metrics = {}
    metrics.update({f"mean/{k}": np.mean(v) for k, v in sample_metrics.items()})
    metrics.update({f"median/{k}": np.median(v) for k, v in sample_metrics.items()})

    # Log metrics to wandb
    metrics = {f"scaffold/{k}": v for k, v in metrics.items()}
    torch.save(metrics, f"{log_dir}/metrics.pt")
    if not cfg.wandb.no_wandb:
        wandb.log(metrics)


if __name__ == "__main__":
    main()
