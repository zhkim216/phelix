import pickle
from pathlib import Path

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.data import residue_constants as rc
from allatom_design.eval import eval_metrics
from allatom_design.eval.eval_setup_utils import get_pdb_files
from allatom_design.eval.fampnn_utils import (get_seq_des_model,
                                              run_fampnn_packing)


@hydra.main(config_path="../../configs/eval/sampling", config_name="sidechain_pack", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)

    # Create out dir and preserve config
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in sequence design model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    ### Load in PDB files ###
    pdb_files = get_pdb_files(**cfg.input_cfg)

    ### Sampling ###
    # Run FAMPNN packing
    _, aux = run_fampnn_packing(seq_des_model["fampnn_model"], seq_des_model["fampnn_cfg"],
                                pdb_paths=pdb_files, device=device, out_dir=f"{cfg.out_dir}/packed_pdbs")

    sample_info = aux["sample_info"]

    # Save metrics
    with open(f"{cfg.out_dir}/sample_info.pkl", "wb") as f:
        pickle.dump(sample_info, f)

    ### Compute sidechain metrics ###
    scn_metrics = {}
    seq_mask = sample_info["seq_mask"]

    # Compute metrics against input PDBs
    core_mask, surface_mask = eval_metrics.get_core_surface_mask(sample_info["x_in"], sample_info["atom_mask"])
    sample_info["core_mask"] = core_mask
    sample_info["surface_mask"] = surface_mask
    scn_info, _ = eval_metrics.compute_structure_metrics(sample_info["x_in"], sample_info["x_denoised"],
                                                         sample_info["atom_mask"], aatype=sample_info["aatype"],
                                                         metrics_to_compute=["scn_rmsd_per_pos", "chi_metrics_per_pos"])

    for k, v in scn_info.items():
        sample_info[k] = v

    # Average RMSD per protein over proteins
    scn_rmsd_avg = (sample_info["scn_rmsd_per_pos"] * seq_mask).sum(dim=-1) / seq_mask.sum(dim=-1)
    scn_metrics["scn_rmsd_avg"] = scn_rmsd_avg.mean().item()
    print(f"Average RMSD per protein: {scn_metrics['scn_rmsd_avg']:.3f}")

    # Average RMSD over all residues
    scn_rmsd_avg_all = (sample_info["scn_rmsd_per_pos"] * seq_mask).sum() / seq_mask.sum()
    scn_metrics["scn_rmsd_avg_all"] = scn_rmsd_avg_all.item()

    # Average RMSD over all core and surface residues
    for key in ["core", "surface"]:
        mask = sample_info[f"{key}_mask"]
        scn_rmsd_avg = (sample_info["scn_rmsd_per_pos"][mask] * seq_mask[mask]).sum() / seq_mask[mask].sum()
        scn_metrics[f"scn_rmsd_avg_{key}"] = scn_rmsd_avg.item()

    # Get average RMSD per residue
    for aa_idx, aa in enumerate(rc.restypes_with_x):
        aatype_mask = sample_info["aatype"] == aa_idx
        rmsd_i = sample_info["scn_rmsd_per_pos"][aatype_mask]
        rmsd_avg_i = (rmsd_i * seq_mask[aatype_mask]).sum() / seq_mask[aatype_mask].sum()

        print(f"Average RMSD for {aa}: {rmsd_avg_i:.3f} Å")
        scn_metrics[f"scn_rmsd_avg_{aa}"] = rmsd_avg_i.item()

    print(f"Average RMSD for all residues: {scn_metrics['scn_rmsd_avg_all']:.3f} Å")
    print(f"Average RMSD for core residues: {scn_metrics['scn_rmsd_avg_core']:.3f} Å")
    print(f"Average RMSD for surface residues: {scn_metrics['scn_rmsd_avg_surface']:.3f} Å")

    # Plot average sidechain RMSD per residue
    rmsd_avg_aas = [(aa, scn_metrics[f"scn_rmsd_avg_{aa}"]) for aa in rc.restypes_with_x]
    rmsd_avg_aas = sorted(rmsd_avg_aas, key=lambda x: x[1])

    plt.figure(figsize=(12, 6))
    plt.plot([aa for aa, _ in rmsd_avg_aas], [rmsd for _, rmsd in rmsd_avg_aas], marker="o", linestyle="--")
    plt.xlabel("Residue")
    plt.ylabel("Average sidechain RMSD (Å)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/scn_rmsd_per_res.png")
    plt.close()

    # Get average chi metrics per chi angle
    chi_mask = sample_info["chi_mask"]  # [B, N, 4]
    chi_mae_avg = (sample_info["chi_mae_per_pos"] * chi_mask).sum(dim=(0, 1)) / chi_mask.sum(dim=(0, 1))
    chi_acc_avg = (sample_info["chi_acc_per_pos"] * chi_mask).sum(dim=(0, 1)) / chi_mask.sum(dim=(0, 1))
    for ci in range(4):
        scn_metrics[f"chi{ci+1}_mae_avg"] = chi_mae_avg[ci].item()
        scn_metrics[f"chi{ci+1}_acc_avg"] = chi_acc_avg[ci].item()

    # Save metrics as csv with pandas
    metrics_df = pd.DataFrame(scn_metrics, index=[0])
    metrics_df.to_csv(f"{cfg.out_dir}/scn_metrics.csv", index=False)


if __name__ == "__main__":
    main()
