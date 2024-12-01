import glob
import pickle
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.data.data import load_feats_from_pdb, process_single_pdb
from allatom_design.eval import eval_metrics


@hydra.main(config_path="../configs/eval", config_name="compute_scn_pack_structure_metrics", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        cfg.out_dir = f"{cfg.sample_pdb_dir}/compute_scn_pack_structure_metrics"

    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # Delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Compute metrics for each protein
    metrics_per_prot = defaultdict(list)

    gt_pdbs = natsorted(glob.glob(f"{cfg.gt_pdb_dir}/*.pdb"))
    for gt_pdb in tqdm(gt_pdbs, desc="Computing metrics for each pdb..."):
        # Load in ground truth data
        gt_data = load_feats_from_pdb(gt_pdb)
        gt_batch = process_single_pdb(gt_data)
        x, seq_mask, atom_mask, aatype = gt_batch["x"], gt_batch["seq_mask"], gt_batch["atom_mask"], gt_batch["aatype"]

        # Load in sample pdbs, assuming they are in the same format as the ground truth pdbs
        pdb_key = Path(gt_pdb).stem
        sample_pdb_file = find_sample_pdb(cfg.sample_pdb_dir, pdb_key)  # try to find sample pdb file
        sample_data = load_feats_from_pdb(sample_pdb_file)
        sample_batch = process_single_pdb(sample_data)
        x_sample = sample_batch["x"]

        # Compute metrics
        try:
            scn_info, _ = eval_metrics.compute_structure_metrics(x[None], x_sample[None], # add batch dim
                                                                 atom_mask[None], aatype=aatype[None],
                                                                 metrics_to_compute=["scn_rmsd_per_pos", "chi_metrics_per_pos", "sce"])
        except Exception as e:
            print(f"Error computing metrics for {Path(gt_pdb).stem}: {e}, skipping...")
            continue

        # Compute RMSD per protein
        rmsd_i = (scn_info["scn_rmsd_per_pos"].squeeze() * seq_mask).sum() / seq_mask.sum()
        metrics_per_prot["rmsd"].append(rmsd_i.item())

        # Compute chi metrics per protein
        chi_mask = scn_info["chi_mask"].squeeze()
        chi_mae_i = (scn_info["chi_mae_per_pos"].squeeze() * chi_mask).sum(dim=0) / chi_mask.sum(dim=0)
        chi_acc_i = (scn_info["chi_acc_per_pos"].squeeze() * chi_mask).sum(dim=0) / chi_mask.sum(dim=0)
        for ci in range(4):
            metrics_per_prot[f"chi{ci+1}_mae"].append(chi_mae_i[ci].item())
            metrics_per_prot[f"chi{ci+1}_acc"].append(chi_acc_i[ci].item())

    # Compute average metrics
    avg_metrics = {}
    avg_metrics["rmsd"] = np.mean(metrics_per_prot["rmsd"])
    for ci in range(4):
        avg_metrics[f"chi{ci+1}_mae"] = np.mean(metrics_per_prot[f"chi{ci+1}_mae"])
        avg_metrics[f"chi{ci+1}_acc"] = np.mean(metrics_per_prot[f"chi{ci+1}_acc"])

    for k, v in avg_metrics.items():
        print(f"{k}: {v}")

    # Save metrics as pickle
    with open(Path(cfg.out_dir, "avg_metrics.pkl"), "wb") as f:
        pickle.dump(avg_metrics, f)

    print("DONE")


def find_sample_pdb(sample_dir: str, pdb_key: str) -> str:
    # Look for an exact match first
    exact_match = f"{sample_dir}/{pdb_key}.pdb"
    if Path(exact_match).exists():
        return exact_match

    # Otherwise, search for a file matching {pdb_key}*.pdb
    pattern = f"{sample_dir}/{pdb_key}*.pdb"
    matched_pdbs = glob.glob(pattern)

    if len(matched_pdbs) == 1:
        return matched_pdbs[0]
    elif len(matched_pdbs) == 0:
        raise FileNotFoundError(f"No sample PDB file found for key '{pdb_key}' in '{sample_dir}'.")
    else:
        raise ValueError(
            f"Multiple sample PDB files found for key '{pdb_key}' in '{sample_dir}': {matched_pdbs}"
        )


if __name__ == "__main__":
    main()
