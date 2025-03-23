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
    per_pdb_metrics = defaultdict(dict)

    # Run self-consistency evaluation
    sc_info = eval_metrics.run_self_consistency_eval(
        pdb_files,
        seq_des_model,
        struct_pred_model,
        device,
        out_dir=log_dir,
        temp_dir=f"{log_dir}/tmp"
    )

    for pdb, v in sc_info.items():
        per_pdb_metrics[pdb]["sc_info"] = v

    # Get secondary structure info
    ss_info = eval_metrics.compute_secondary_structure_content(pdb_files)
    for pdb, v in ss_info.items():
        per_pdb_metrics[pdb]["ss_info"] = v

    # Aggregate per-pdb metrics to map from {metric key: list of values}
    sample_metrics = defaultdict(list)
    for pdb in per_pdb_metrics:
        # secondary structure metrics
        for k, v in per_pdb_metrics[pdb]["ss_info"].items():
            sample_metrics[k].append(v)

        # self-consistency metrics
        for k, v in per_pdb_metrics[pdb]["sc_info"]["sc_metrics"].items():
            best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
            sample_metrics[f"{cfg.seq_des_cfg.model_name}_{k}_best"].append(best_sc_metric.item())

            if len(v) > 1:
                # only report mean if we run multiple sequences per sample
                sample_metrics[f"{cfg.seq_des_cfg.model_name}_{k}_mean"].append(mean_sc_metric.item())
                mean_sc_metric = torch.mean(v)

    # Compute optional diversity metrics across entire set of PDBs
    if cfg.compute_diversity_metrics:
        # === Calculate mean pairwise TM score ===
        coords = [load_feats_from_pdb(pdb)["all_atom_positions"] for pdb in pdb_files]
        sample_metrics["pairwise_tm"] = eval_metrics.compute_pairwise_tm_score(
            coords,
            temp_dir=f"{log_dir}/tmp",
            subsample_pairs=cfg.pairwise_tm_subsample,
        )

        # === Foldseek clustering analysis ===
        for sctm_cutoff in cfg.clustering.sctm_cutoffs:
            # Cluster only on designable samples (scTM > sctm_cutoff)
            designable_pdbs = [pdb for pdb in pdb_files if (per_pdb_metrics[pdb]["sc_info"]["sc_metrics"]["sc_ca_tm"] > sctm_cutoff).any()]
            sample_metrics[f"sctm{sctm_cutoff}_nsamples"] = len(designable_pdbs)

            cluster_out_dir = Path(f"{log_dir}/clustering/sctm{sctm_cutoff}")
            sample_metrics[f"sctm{sctm_cutoff}_ncluster"] = eval_metrics.foldseek_cluster(designable_pdbs, cluster_out_dir, f"{log_dir}/tmp",
                                                                                       **cfg.clustering.foldseek_opts)

    # === Calculate metrics to log === #
    metrics = {}
    metrics.update({f"sc/mean/{k}": np.mean(v) for k, v in sample_metrics.items()})
    metrics.update({f"sc/median/{k}": np.median(v) for k, v in sample_metrics.items()})

    # Save per-sample metrics
    with open(f"{log_dir}/all_metrics.pkl", "wb") as f:
        pickle.dump(per_pdb_metrics, f)

    # Log aggregated metrics to wandb
    if not cfg.wandb.no_wandb:
        wandb.log(metrics)


if __name__ == "__main__":
    main()
