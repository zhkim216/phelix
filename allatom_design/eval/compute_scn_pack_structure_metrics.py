import glob
import pickle
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import get_rc_tensor, load_feats_from_pdb
from allatom_design.data.datasets.sd_dataset import process_single_pdb
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
    all_sample_info = defaultdict(list)  # stores sample info for all proteins
    metrics_per_prot = defaultdict(list)

    gt_pdbs = natsorted(glob.glob(f"{cfg.gt_pdb_dir}/*.pdb"))
    for gt_pdb in tqdm(gt_pdbs, desc="Computing metrics for each pdb..."):
        pdb_key = Path(gt_pdb).stem

        if cfg.skip_pdb_keys and pdb_key in cfg.skip_pdb_keys:
            print(f"Skipping {pdb_key}...")
            continue

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

        # For FAMPNN, extract b-factors as pSCE
        all_sample_info["psce"].append(sample_data["b_factors"][:, rc.non_bb_idxs])
        all_sample_info["sce"].append(scn_info["sce"][0])
        all_sample_info["atom_mask"].append(sample_batch["atom_mask"])
        all_sample_info["aatype"].append(sample_batch["aatype"])
        all_sample_info["scn_rmsd_per_pos"].append(scn_info["scn_rmsd_per_pos"][0])

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

    ### Plots ###
    for k, v in all_sample_info.items():
        all_sample_info[k] = torch.cat(v, dim=0)

    # Plot RMSD per residue type
    rmsd_per_restype = {}

    scn_atom_mask = all_sample_info["atom_mask"][:, rc.non_bb_idxs]
    for aa_idx, aa in enumerate(rc.restypes_with_x):
        if aa == "X" or aa == "G" or aa == "A":
            continue
        aatype_mask = all_sample_info["aatype"] == aa_idx
        rmsds_i = all_sample_info["scn_rmsd_per_pos"][aatype_mask]

        print(f"Average RMSD for {aa}: {rmsds_i.mean().item():.3f} Å")
        rmsd_per_restype[aa] = rmsds_i

    # Create a box plot with each residue type on the x-axis, sorted by median RMSD
    residues = list(rmsd_per_restype.keys())
    medians = [rmsd_per_restype[res].median().item() for res in residues]
    sorted_residues = [res for _, res in sorted(zip(medians, residues), key=lambda x: x[0])]
    sorted_data = [rmsd_per_restype[res].cpu().numpy() for res in sorted_residues]

    plt.figure(figsize=(6, 3.5))

    # Compute overall min/max for setting y-ticks at 0.5 increments
    all_rmsd_values = np.concatenate(sorted_data)
    y_min = np.floor(all_rmsd_values.min() * 2) / 2
    y_max = np.ceil(all_rmsd_values.max() * 2) / 2

    plt.boxplot(
        sorted_data,
        patch_artist=True,
        showfliers=False,  # get rid of outliers
        boxprops=dict(color='black', facecolor='white'),
        medianprops=dict(color='goldenrod', linewidth=2),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(color='black', markeredgecolor='black', markersize=3)
    )

    plt.xticks(range(1, len(sorted_residues) + 1), sorted_residues, ha='center', fontsize=10)
    plt.xlabel("Residue type", fontsize=12)
    plt.ylabel("RMSD (Å)", fontsize=12)

    # Set y-ticks in increments of 0.5
    plt.yticks(np.arange(y_min, y_max + 0.5, 0.5), fontsize=10)

    # Add grid lines with alpha=0.5
    plt.grid(True, alpha=0.5)

    plt.ylim(-0.1, 5.5)

    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/rmsd_per_residue_type_boxplot.png", dpi=300)
    plt.savefig(f"{cfg.out_dir}/rmsd_per_residue_type_boxplot.pdf", dpi=300)
    plt.close()


    # Get pSCE correlations
    if cfg.plot_psce:
        # Get pSCE correlations

        ### Get correlation per residue ###
        scn_atom_mask = all_sample_info["atom_mask"][:, rc.non_bb_idxs]
        sce_per_res = (all_sample_info["sce"] * scn_atom_mask).nansum(dim=1) / scn_atom_mask.sum(dim=1)
        psce_per_res = (all_sample_info["psce"] * scn_atom_mask).sum(dim=1) / scn_atom_mask.sum(dim=1)

        # get rid of glycines
        nan_mask = torch.isnan(sce_per_res)
        sce_per_res = sce_per_res[~nan_mask]
        psce_per_res = psce_per_res[~nan_mask]

        # subsample to 5k points
        idxs = np.random.choice(len(sce_per_res), 5000, replace=False)
        sce_per_res_sub = sce_per_res[idxs]
        psce_per_res_sub = psce_per_res[idxs]

        plt.figure(figsize=(5, 5))
        plt.scatter(sce_per_res_sub, psce_per_res_sub, s=0.4, color='#1f77b4', alpha=1.0)  # same blue

        # Dark dashed line for y=x
        max_val_res = float(max(sce_per_res.max().item(), psce_per_res.max().item()))
        plt.plot([0, max_val_res], [0, max_val_res], 'k--', linewidth=1.5)

        # Horizontal gold dashed line at y=4.0625
        # plt.axhline(4.0625, color='goldenrod', linestyle='-', linewidth=1)

        # Spearman correlation
        spearman_corr_res, _ = spearmanr(sce_per_res.cpu().numpy(), psce_per_res.cpu().numpy())
        plt.text(
            0.05, 0.95,
            r"Spearman $\rho$: {0:.3f}".format(spearman_corr_res),
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            color="black"
        )

        # Change labels and title
        plt.xlabel("Sidechain error ($\\mathrm{\\AA}$)", fontsize=12)
        plt.ylabel("Predicted sidechain error ($\\mathrm{\\AA}$)", fontsize=12)
        plt.title("Confidence per residue")  # Reflecting per-residue

        plt.xlim(0, 4.5)
        plt.ylim(0, 4.5)
        plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

        # Save as both PNG and PDF
        plt.savefig(f"{cfg.out_dir}/sce_vs_psce_per_res.png", dpi=300)
        plt.savefig(f"{cfg.out_dir}/sce_vs_psce_per_res.pdf", dpi=300, transparent=True)
        plt.close()

        # Plot the same but between 0 and 1
        plt.figure(figsize=(5, 5))
        plt.scatter(sce_per_res_sub, psce_per_res_sub, s=0.4, color='#1f77b4', alpha=1.0)  # same blue
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

        plt.xlabel("Sidechain error", fontsize=12)
        plt.ylabel("Predicted sidechain error", fontsize=12)
        plt.title("Confidence per residue")  # Reflecting per-residue
        plt.savefig(f"{cfg.out_dir}/sce_vs_psce_per_res_01.png", dpi=300)
        plt.savefig(f"{cfg.out_dir}/sce_vs_psce_per_res_01.pdf", dpi=300, transparent=True)
        plt.close("all")


        ### Get correlation per atom ###
        sce = all_sample_info["sce"][scn_atom_mask.bool()].flatten()
        psce = all_sample_info["psce"][scn_atom_mask.bool()].flatten()

        # subsample to 5k points
        idxs = np.random.choice(len(sce), 5000, replace=False)
        sce_sub = sce[idxs]
        psce_sub = psce[idxs]

        plt.figure(figsize=(5, 5))
        plt.scatter(sce_sub, psce_sub, s=0.4, color='#1f77b4')  # same blue

        # Dark dashed line for y=x
        max_val_atom = float(max(sce.max().item(), psce.max().item()))
        plt.plot([0, max_val_atom], [0, max_val_atom], 'k--', linewidth=1.5)

        # Horizontal gold dashed line at y=4.0625
        # plt.axhline(4.0625, color='goldenrod', linestyle='-', linewidth=1)

        # Spearman correlation
        spearman_corr_atom, _ = spearmanr(sce.cpu().numpy(), psce.cpu().numpy())
        plt.text(
            0.05, 0.95,
            r"Spearman $\rho$: {0:.3f}".format(spearman_corr_atom),
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            color="black"
        )

        # Change labels and title
        plt.xlabel("Sidechain error ($\\mathrm{\\AA}$)", fontsize=12)
        plt.ylabel("Predicted sidechain error ($\\mathrm{\\AA}$)", fontsize=12)
        plt.title("Confidence per atom", fontsize=12)  # Reflecting per-atom

        plt.xlim(0, 4.5)
        plt.ylim(0, 4.5)
        plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

        # Save as both PNG and PDF
        plt.savefig(f"{cfg.out_dir}/sce_vs_psce_per_atom.png", dpi=300)
        plt.savefig(f"{cfg.out_dir}/sce_vs_psce_per_atom.pdf", dpi=300, transparent=True)
        plt.close()

        # Plot the same but between 0 and 1
        plt.figure(figsize=(5, 5))
        plt.scatter(sce_sub, psce_sub, s=0.4, color='#1f77b4')  # same blue
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

        plt.xlabel("Sidechain error", fontsize=12)
        plt.ylabel("Predicted sidechain error", fontsize=12)
        plt.title("Confidence per atom")  # Reflecting per-atom
        plt.savefig(f"{cfg.out_dir}/sce_vs_psce_per_atom_01.png", dpi=300)
        plt.savefig(f"{cfg.out_dir}/sce_vs_psce_per_atom_01.pdf", dpi=300, transparent=True)
        plt.close("all")

    print("TEST")


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
