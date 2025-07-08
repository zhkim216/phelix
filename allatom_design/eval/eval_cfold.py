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
from allatom_design.eval.eval_utils.eval_setup_utils import (process_pdb_files,
                                                             wandb_setup, get_conformer_dirs, process_conformer_dirs)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (
    get_seq_des_model, run_seq_des_ensemble)


@hydra.main(config_path="../configs/eval", config_name="eval_cfold", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating sequence design performance on the Cfold dataset of fold-switchers.
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
    # DEBUG
    conformer_dirs = conformer_dirs[:1]

    # Process conformer directories
    pdb_to_processed_conformers = process_conformer_dirs(conformer_dirs,
                                                         max_num_conformers=None,
                                                         include_primary_conformer=True,
                                                         processed_struct_dir=f"{log_dir}/processed_structures",
                                                         pdb_processing_cfg=cfg.pdb_processing_cfg)

    # Filter out pdbs where either conformer did not pass processing
    pdb_to_processed_conformers = {k: v for k, v in pdb_to_processed_conformers.items() if len(v) == 2}

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Run sequence design model
    out_dfs = []
    modes = ["ensemble", "state0", "state1"]
    for mode in modes:
        log_dir_mode = f"{log_dir}/{mode}"
        Path(log_dir_mode).mkdir(parents=True, exist_ok=True
                                 )
        if mode == "ensemble":
            # run on both Cfold conformers
            pdb_to_processed_conformers_mode = pdb_to_processed_conformers
        elif mode == "state0":
            # run on state 0 conformer
            pdb_to_processed_conformers_mode = {k: [v[0]] for k, v in pdb_to_processed_conformers.items()}
        elif mode == "state1":
            # run on state 1 conformer
            pdb_to_processed_conformers_mode = {k: [v[1]] for k, v in pdb_to_processed_conformers.items()}

        outputs = run_seq_des_ensemble(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                                       pdb_to_processed_conformers=pdb_to_processed_conformers_mode, device=device, pos_constraint_df=None,
                                       out_dir=log_dir_mode)
        out_df = pd.DataFrame({"record_id": [Path(x).stem.lower() for x in outputs["out_pdbs"]],
                               **{k: v for k, v in outputs.items() if k in ["out_pdbs", "seqs", "input_seqs", "n_conformers", "U"]}})

        # Compute seq recovery
        out_df[f"seq_recovery"] = out_df.apply(lambda x: eval_metrics.compute_seq_recovery(x[f"input_seqs"], x[f"seqs"]), axis=1)

        # Save to csv
        out_df.to_csv(f"{log_dir_mode}/seq_des_outputs.csv", index=False)
        out_dfs.append(out_df)

    del seq_des_model

    # Compute self-consistency metrics for each mode
    if cfg.run_self_consistency_eval:
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

        for mi, mode in enumerate(modes):
            log_dir_mode = f"{log_dir}/{mode}"
            preds_dir_mode = f"{log_dir_mode}/preds"
            Path(preds_dir_mode).mkdir(parents=True, exist_ok=True)

            out_df_mi = out_dfs[mi]
            id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
                out_df_mi["out_pdbs"],
                struct_pred_model,
                cfg.pdb_processing_cfg,
                out_dir=preds_dir_mode)

            # Save metrics as CSV
            metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
            metrics_df = pd.merge(metrics_df, out_df_mi, on="record_id", how="left")
            metrics_df.to_csv(f"{log_dir_mode}/self_consistency_metrics.csv", index=False)

            if not cfg.wandb.no_wandb:
                # Aggregate results to log to wandb
                sc_metrics = defaultdict(list)
                for record_id, metrics in id_to_metrics.items():
                    for k, v in metrics.items():
                        sc_metrics[f"{k}"].append(v)

                # Update metrics
                out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k not in ["record_id", "n_conformers"]}
                out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k not in ["record_id", "n_conformers"]})

                # Log metrics to wandb
                wandb.log(out_metrics, step=0)


if __name__ == "__main__":
    main()
