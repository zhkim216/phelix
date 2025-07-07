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


@hydra.main(config_path="../../configs/eval/sampling", config_name="seq_des_multi_ensemble", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for multiple conformers of multiple PDBs.
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

    # Load structure prediction model for self-consistency evaluation
    if cfg.run_self_consistency_eval:
        pred_out_dir = f"{log_dir}/preds"  # directory for structure predictions
        Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Load in positional constraints
    if cfg.pos_constraint_csv is not None:
        pos_constraint_df = pd.read_csv(cfg.pos_constraint_csv)

        # expand pdb_key to all conformers
        pos_constraint_df["pdb_key"] = pos_constraint_df["pdb_key"].str.lower()
        conformer_dfs = []
        for pdb_key in pos_constraint_df["pdb_key"].unique():
            if pdb_key not in pdb_to_processed_conformers:
                continue
            conformer_df = pos_constraint_df[pos_constraint_df["pdb_key"] == pdb_key]
            conformer_df = pd.concat([conformer_df] * len(pdb_to_processed_conformers[pdb_key]), ignore_index=True)
            conformer_df["pdb_key"] = [Path(x).stem for x in pdb_to_processed_conformers[pdb_key]]
            conformer_dfs.append(conformer_df)
        pos_constraint_df = pd.concat(conformer_dfs)
    else:
        pos_constraint_df = None

    # Run sequence design model
    outputs = run_seq_des_ensemble(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                                   pdb_to_processed_conformers=pdb_to_processed_conformers, device=device, pos_constraint_df=pos_constraint_df,
                                   out_dir=log_dir)

    del seq_des_model

    record_ids = [Path(x).stem.lower() for x in outputs["out_pdbs"]]
    output_df = pd.DataFrame({"record_id": record_ids, "seq": outputs["seqs"], "out_pdb": outputs["out_pdbs"], "n_conformers": outputs["n_conformers"], "U": outputs["U"]})
    output_df.to_csv(f"{log_dir}/seq_des_outputs.csv", index=False)

    if cfg.run_self_consistency_eval:
        id_to_metrics = eval_metrics.run_self_consistency_eval_boltz(
            outputs["out_pdbs"],
            struct_pred_model,
            cfg.pdb_processing_cfg,
            out_dir=pred_out_dir)

        # Save metrics as CSV
        metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
        metrics_df = pd.merge(metrics_df, output_df, on="record_id", how="left")
        metrics_df.to_csv(f"{log_dir}/self_consistency_metrics.csv", index=False)

        if not cfg.wandb.no_wandb:
            # Aggregate results
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
