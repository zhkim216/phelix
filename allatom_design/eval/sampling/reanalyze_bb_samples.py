import glob
import os
import pickle
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval import eval_metrics


@hydra.main(config_path="../../configs/eval/sampling", config_name="reanalyze_bb_samples", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    base_dir = Path(cfg.orig_out_dir)

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        cfg.out_dir = f"{cfg.orig_out_dir}/reanalyze_bb_samples/{cfg.exp_name}"

    # Load metrics
    all_metrics_path = f"{cfg.orig_out_dir}/all_metrics.pkl"
    bin_to_metrics_path = f"{cfg.orig_out_dir}/L_to_metrics.pkl"

    with open(all_metrics_path, "rb") as f:
        all_metrics_orig = pickle.load(f)

    with open(bin_to_metrics_path, "rb") as f:
        bin_to_metrics = pickle.load(f)

    # Fix all samples to have current base dir
    all_metrics = {f"{base_dir}/samples/{Path(k).name}": v for k, v in all_metrics_orig.items()}

    # === Run clustering analysis === #
    pdbs = natsorted(all_metrics.keys())
    bins = [all_metrics[pdb]["bin"] for pdb in pdbs]

    for sctm_cutoff in cfg.clustering.sctm_cutoffs:
        # Cluster by length bin
        for bin in set(bins):
            pdbs_b = [pdb for pdb, b in zip(pdbs, bins) if (b == bin) and ("mpnn_sc_info" in all_metrics[pdb])]
            if len(pdbs_b) == 0:
                # skip if we don't have self-consistency info for any samples in this bin
                continue

            # Cluster only on designable samples (scTM > sctm_cutoff)
            designable_pdbs = [pdb for pdb in pdbs_b if all_metrics[pdb]["mpnn_sc_info"]["sc_metrics"]["sc_ca_tm"] > sctm_cutoff]
            bin_to_metrics[bin][f"sctm{sctm_cutoff}_nsamples"] = len(designable_pdbs)

            cluster_out_dir = Path(f"{cfg.out_dir}/clustering/bin{bin}_sctm{sctm_cutoff}")
            bin_to_metrics[bin][f"sctm{sctm_cutoff}_ncluster"] = eval_metrics.foldseek_cluster(designable_pdbs, cluster_out_dir, f"{cfg.out_dir}/tmp",
                                                                                                **cfg.clustering.foldseek_opts)


    # Set up wandb logging
    if not cfg.no_wandb:
        wandb.init(
            project=cfg.project,
            entity=cfg.wandb_id,
            name=cfg.exp_name,
            group=cfg.group,
            config=cfg_dict,
            dir=cfg.out_dir,
        )

        # Log metrics
        for bin in sorted(bin_to_metrics.keys()):
            metrics_b = bin_to_metrics[bin]
            metrics_b["length_bin"] = bin
            wandb.log(metrics_b, step=bin)

        wandb.finish()


if __name__ == "__main__":
    main()
