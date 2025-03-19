# eval_sample_sc.py

import glob
import os
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import wandb
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from allatom_design.data.data import load_feats_from_pdb
from allatom_design.eval.eval_utils import eval_metrics
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.proteinmpnn_utils import load_mpnn


@hydra.main(config_path="../configs/eval", config_name="eval_sample_sc", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating self-consistency on a directory of pre-sampled CA-only PDBs.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create wandb dir
    wandb_dir = str(Path(cfg.out_dir))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

    # Set wandb cache directory
    wandb_cache_dir = str(Path(cfg.out_dir, "cache", "wandb"))
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up logging
    if cfg.no_wandb:
        log_dir = Path(cfg.out_dir, "debug")
    else:
        wandb.init(
            project=cfg.project,
            entity=cfg.wandb_id,
            name=cfg.exp_name,
            group=cfg.group,
            config=cfg_dict,
            dir=wandb_dir,
        )
        log_dir = Path(cfg.out_dir, wandb.run.name)

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MPNN + structure prediction model for self-consistency evals
    mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
    mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)
    mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)

    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    # Gather PDBs from directory
    pdb_dir = Path(cfg.sampled_pdb_dir)
    pdbs = natsorted(glob.glob(f"{pdb_dir}/*.pdb"))
    if len(pdbs) == 0:
        assert False, f"No PDBs found in {pdb_dir}"

    # Create log directories
    sampled_pdbs_dir = Path(log_dir, "sampled_pdbs")
    sampled_pdbs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(log_dir, "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(log_dir, "tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Dump PDBs to sampled_pdbs to log directory
    for pdb in pdbs:
        shutil.copy(pdb, sampled_pdbs_dir)

    # Calculate structure metrics
    all_metrics = defaultdict(dict)

    # Secondary structure info
    ss_info = eval_metrics.compute_secondary_structure_content(pdbs)
    for pdb, v in ss_info.items():
        all_metrics[pdb]["ss_info"] = v

    # MPNN + structure prediction self-consistency
    mpnn_sc_info = eval_metrics.run_self_consistency_eval(
        pdbs,
        mpnn_model,
        mpnn_cfg,
        struct_pred_model,
        device,
        out_dir=log_dir,
        temp_dir=str(tmp_dir),
    )
    for pdb, v in mpnn_sc_info.items():
        all_metrics[pdb]["mpnn_sc_info"] = v

    # Optional nnTM evaluation
    if cfg.nntm_dataset is not None:
        nntm_info = eval_metrics.run_nntm_eval(pdbs, dataset=cfg.nntm_dataset, out_dir=log_dir)
        for pdb, v in nntm_info.items():
            all_metrics[pdb]["nntm_info"] = v

    # Aggregate per-pdb metrics
    sample_metrics = defaultdict(list)
    for pdb in pdbs:
        # Secondary structure metrics
        for k, v in all_metrics[pdb]["ss_info"].items():
            sample_metrics[f"{k}"].append(v)

        # MPNN self-consistency metrics
        for k, v in all_metrics[pdb]["mpnn_sc_info"]["sc_metrics"].items():
            mean_sc_metric = torch.mean(v)
            best_sc_metric = max(v, key=eval_metrics.get_sort_key_fn(k))
            sample_metrics[f"mpnn_{k}_mean"].append(mean_sc_metric.item())
            sample_metrics[f"mpnn_{k}_best"].append(best_sc_metric.item())

        # nnTM metrics
        if cfg.nntm_dataset is not None:
            sample_metrics["nntm"].append(all_metrics[pdb]["nntm_info"])

    # Compute optional diversity metrics across entire set of PDBs
    if cfg.compute_diversity_metrics:
        # === Calculate mean pairwise TM score ===
        coords = [load_feats_from_pdb(pdb)["all_atom_positions"] for pdb in pdbs]
        sample_metrics["pairwise_tm"] = eval_metrics.compute_pairwise_tm_score(
            coords,
            temp_dir=str(tmp_dir),
            subsample_pairs=cfg.pairwise_tm_subsample,
        )

        # === Foldseek clustering analysis ===
        for sctm_cutoff in cfg.clustering.sctm_cutoffs:
            designable_pdbs = [
                pdb
                for pdb in pdbs
                if all_metrics[pdb]["mpnn_sc_info"]["sc_metrics"]["sc_ca_tm"] > sctm_cutoff
            ]
            sample_metrics[f"sctm{sctm_cutoff}_nsamples"] = len(designable_pdbs)

            cluster_out_dir = Path(log_dir, "clustering", f"sctm{sctm_cutoff}")
            cluster_out_dir.mkdir(parents=True, exist_ok=True)
            sample_metrics[f"sctm{sctm_cutoff}_ncluster"] = eval_metrics.foldseek_cluster(
                designable_pdbs,
                cluster_out_dir,
                str(tmp_dir),
                **cfg.clustering.foldseek_opts,
            )

    # Calculate mean metrics
    metrics = {f"eval_sc/{k}": np.mean(v) for k, v in sample_metrics.items()}

    # Save per-sample metrics
    with open(f"{metrics_dir}/all_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)

    # Log aggregated metrics to wandb
    if not cfg.no_wandb:
        wandb.log(metrics)

    print("Done evaluating PDBs.")


if __name__ == "__main__":
    main()
