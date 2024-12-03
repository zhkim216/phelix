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

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import load_feats_from_pdb, process_single_pdb, get_rc_tensor
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
        pdb_key = Path(gt_pdb).stem

        # Load in ground truth data
        gt_data = load_feats_from_pdb(gt_pdb)
        gt_batch = process_single_pdb(gt_data)
        x, seq_mask, atom_mask, aatype = gt_batch["x"], gt_batch["seq_mask"], gt_batch["atom_mask"], gt_batch["aatype"]
        atom_mask[:, rc.atom_order["OXT"]] = 0  # remove OXT atoms from atom_mask
        core_mask, surface_mask = eval_metrics.get_core_surface_mask(x.cpu(), atom_mask.cpu())

        # Load in sample pdbs, assuming they are in the same format as the ground truth pdbs
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
            try:
                print(f"Failed for {Path(gt_pdb).stem}, trying again by deleting residues where any backbone atoms are missing...")
                # LigandMPNN doesn't save residues where any of the 4 backbone atoms are missing
                all_bb_exists = atom_mask[:, rc.bb_idxs].all(dim=1)
                x = x[all_bb_exists]
                seq_mask = seq_mask[all_bb_exists]
                atom_mask = atom_mask[all_bb_exists]
                aatype = aatype[all_bb_exists]
                core_mask, surface_mask = core_mask[all_bb_exists], surface_mask[all_bb_exists]

                scn_info, _ = eval_metrics.compute_structure_metrics(x[None], x_sample[None], # add batch dim
                                                                    atom_mask[None], aatype=aatype[None],
                                                                    metrics_to_compute=["scn_rmsd_per_pos", "chi_metrics_per_pos", "sce"])
                print("Success!")
            except Exception as e:
                print(f"Error computing metrics for {Path(gt_pdb).stem}: {e}, skipping...")
                continue

        # Compute RMSD per protein
        rmsd_i = (scn_info["scn_rmsd_per_pos"].squeeze(0) * seq_mask).sum() / seq_mask.sum()
        metrics_per_prot["rmsd"].append(rmsd_i.item())

        # Compute chi metrics per protein
        chi_mask = scn_info["chi_mask"].squeeze(0)
        chi_mae_i = (scn_info["chi_mae_per_pos"].squeeze(0) * chi_mask).sum(dim=0) / chi_mask.sum(dim=0)
        chi_acc_i = (scn_info["chi_acc_per_pos"].squeeze(0) * chi_mask).sum(dim=0) / chi_mask.sum(dim=0)
        for ci in range(4):
            metrics_per_prot[f"chi{ci+1}_mae"].append(chi_mae_i[ci].item())
            metrics_per_prot[f"chi{ci+1}_acc"].append(chi_acc_i[ci].item())

        # Compute RMSD per protein for core and surface residues
        for key in ["core", "surface"]:
            mask = core_mask if key == "core" else surface_mask
            rmsd_key_i = (scn_info["scn_rmsd_per_pos"].squeeze(0)[mask] * seq_mask[mask]).sum() / seq_mask[mask].sum()
            metrics_per_prot[f"rmsd_{key}"].append(rmsd_key_i.item())

        # Compute chi metrics per protein for core and surface residues
        for key in ["core", "surface"]:
            mask = core_mask if key == "core" else surface_mask
            chi_mask = scn_info["chi_mask"].squeeze(0)[mask]
            chi_mae_key_i = (scn_info["chi_mae_per_pos"].squeeze(0)[mask] * chi_mask).sum(dim=0) / chi_mask.sum(dim=0)
            chi_acc_key_i = (scn_info["chi_acc_per_pos"].squeeze(0)[mask] * chi_mask).sum(dim=0) / chi_mask.sum(dim=0)
            for ci in range(4):
                metrics_per_prot[f"chi{ci+1}_mae_{key}"].append(chi_mae_key_i[ci].item())
                metrics_per_prot[f"chi{ci+1}_acc_{key}"].append(chi_acc_key_i[ci].item())

    # Compute average metrics
    avg_metrics = {}
    avg_metrics["rmsd"] = np.mean(metrics_per_prot["rmsd"])

    for ci in range(4):
        avg_metrics[f"chi{ci+1}_mae"] = np.mean(metrics_per_prot[f"chi{ci+1}_mae"])
        avg_metrics[f"chi{ci+1}_acc"] = np.mean(metrics_per_prot[f"chi{ci+1}_acc"])

    for key in ["core", "surface"]:
        avg_metrics[f"rmsd_{key}"] = np.nanmean(metrics_per_prot[f"rmsd_{key}"])
        for ci in range(4):
            avg_metrics[f"chi{ci+1}_mae_{key}"] = np.nanmean(metrics_per_prot[f"chi{ci+1}_mae_{key}"])
            avg_metrics[f"chi{ci+1}_acc_{key}"] = np.nanmean(metrics_per_prot[f"chi{ci+1}_acc_{key}"])

    for k, v in avg_metrics.items():
        print(f"{k}: {v}")

    # Save metrics as pickle
    with open(Path(cfg.out_dir, "avg_metrics.pkl"), "wb") as f:
        pickle.dump(avg_metrics, f)


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
