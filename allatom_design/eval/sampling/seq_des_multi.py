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
                                                             process_pdb_files,
                                                             wandb_setup)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (get_seq_des_model,
                                                          run_seq_des)


@hydra.main(config_path="../../configs/eval/sampling", config_name="seq_des_multi", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for multiple PDBs.
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
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # # Load structure prediction model for self-consistency evaluation
    if cfg.run_self_consistency_eval:
        pred_out_dir = f"{log_dir}/preds"  # directory for structure predictions
        Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Read in fixed positions
    if cfg.pos_constraint_csv is not None:
        pos_constraint_df = pd.read_csv(cfg.pos_constraint_csv)
    else:
        pos_constraint_df = None

    # Run sequence design model
    outputs = run_seq_des(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                         struct_file_paths=processed_struct_files, device=device, pos_constraint_df=pos_constraint_df,
                         out_dir=log_dir)

    if cfg.run_self_consistency_eval:
        id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
            outputs["out_pdbs"],
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
