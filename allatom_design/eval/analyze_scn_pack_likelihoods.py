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
import pandas as pd
import seaborn as sns
import torch
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm


@hydra.main(config_path="../configs/eval", config_name="analyze_scn_pack_likelihoods", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Read in output pickle from sidechain_pack.py and analyze likelihoods from sidechain packing
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)

    # Create out dirs and preserve config
    if cfg.out_dir is None:
        cfg.out_dir = f"{Path(cfg.in_pkl).parent}/analyze_scn_pack_likelihoods"

    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # Delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in pickle file from sidechain_pack.py
    with open(cfg.in_pkl, "rb") as f:
        sample_info = pickle.load(f)

    num_pdbs = len(sample_info["pdb"])
    seq_length = sample_info["seq_mask"].shape[1]
    seq_probs = torch.tensor(sample_info["seq_logits"]).softmax(dim=-1)

    # Create a DataFrame in long format
    df = pd.DataFrame({
        'pdb': np.repeat(sample_info["pdb"], seq_length),
        'scn_rmsd_per_pos': sample_info["scn_rmsd_per_pos"].flatten(),
        'position': np.tile(np.arange(seq_length), num_pdbs),
        'sample_number': np.repeat(np.arange(num_pdbs), seq_length),
        'seq_mask': sample_info["seq_mask"].flatten(),
        'aatype': sample_info["aatype"].flatten(),
        'npa': sample_info["npa"].flatten(),
        'res_num_atoms': sample_info["res_num_atoms"].flatten(),
        'seq_logits_aatype': sample_info["seq_logits"][torch.arange(num_pdbs).unsqueeze(1), torch.arange(seq_length), sample_info["aatype"]].flatten(),
        'seq_probs_aatype': seq_probs[torch.arange(num_pdbs).unsqueeze(1), torch.arange(seq_length), sample_info["aatype"]].flatten(),
    })

    # drop out padding, glycines, and alanines
    df = df[df["seq_mask"] == 1]
    df = df[df["res_num_atoms"] > 1]

    # plot per-protein average npa vs rmsd
    rmsd_per_pdb = df.groupby('pdb').apply(lambda x: x["scn_rmsd_per_pos"].sum() / x["seq_mask"].sum()).reset_index(name='average_rmsd')
    npa_per_pdb = df.groupby('pdb').apply(lambda x: x["npa"].sum() / x["seq_mask"].sum()).reset_index(name='average_npa')
    metrics_per_pdb = pd.merge(rmsd_per_pdb, npa_per_pdb, on='pdb')
    rho, _ = spearmanr(metrics_per_pdb["average_rmsd"], metrics_per_pdb["average_npa"])

    plt.figure()
    plt.scatter(metrics_per_pdb["average_npa"], metrics_per_pdb["average_rmsd"])
    plt.title(f"rmsd vs npa per protein (Spearman's rho = {rho:.4f})")
    plt.ylabel("rmsd averaged over protein")
    plt.xlabel("npa averaged over protein")
    plt.savefig(f"{cfg.out_dir}/rmsd_vs_npa.png")
    plt.close()

    # # plot histogram of sces
    # plt.figure()
    # plt.hist(sample_info["sce"].flatten(), bins=50)
    # plt.title(f"Histogram of sce values\nMedian: {sample_info['sce'].median():.4f},mean: {sample_info['sce'].mean():.4f}")
    # plt.xlabel("sce")
    # plt.ylabel("Frequency")
    # plt.savefig(f"{cfg.out_dir}/sce_hist.png")
    # plt.close()

    # # plot histogram of sces, binning from 0 to 4 in increments
    # plt.figure()
    # plt.hist(sample_info["sce"].flatten(), bins=np.arange(0, 4.1, 0.125))
    # plt.title(f"Histogram of sce values\nMedian: {sample_info['sce'].median():.4f},mean: {sample_info['sce'].mean():.4f}")
    # plt.xlabel("sce")
    # plt.ylabel("Frequency")
    # plt.savefig(f"{cfg.out_dir}/sce_hist_binned.png")
    # plt.close()

    # for each protein, plot spearmanr between npa and rmsd within the protein
    spearmanr_per_pdb = df.groupby('pdb').apply(lambda x: spearmanr(x["npa"], x["scn_rmsd_per_pos"])[0]).reset_index(name='spearmanr')
    plt.figure()
    plt.hist(spearmanr_per_pdb["spearmanr"], bins=20)
    plt.title(f"Spearman's rho between npa and rmsd for each protein\nMedian: {spearmanr_per_pdb['spearmanr'].median():.4f},mean: {spearmanr_per_pdb['spearmanr'].mean():.4f}")
    plt.xlabel("Spearman's rho")
    plt.ylabel("Frequency")
    plt.savefig(f"{cfg.out_dir}/spearmanr_hist.png")
    plt.close()

    # # for each protein, plot spearmanr across different samples of average rmsd vs npa
    # rmsd_per_pdb_per_sample = df.groupby(['pdb', 'sample_number']).apply(lambda x: x["scn_rmsd_per_pos"].sum() / x["seq_mask"].sum()).reset_index(name='average_rmsd')
    # npa_per_pdb_per_sample = df.groupby(['pdb', 'sample_number']).apply(lambda x: x["npa"].sum() / x["seq_mask"].sum()).reset_index(name='average_npa')
    # metrics_per_pdb_per_sample = pd.merge(rmsd_per_pdb_per_sample, npa_per_pdb_per_sample, on=['pdb', 'sample_number'])
    # metrics_per_pdb_per_sample = metrics_per_pdb_per_sample.groupby('pdb').apply(lambda x: spearmanr(x["average_rmsd"], x["average_npa"])[0]).reset_index(name='spearmanr')
    # plt.figure()
    # plt.hist(metrics_per_pdb_per_sample["spearmanr"], bins=20)
    # plt.title(f"Spearman's rho between npa and rmsd for each protein across samples\n average={metrics_per_pdb_per_sample['spearmanr'].mean():.4f}")
    # plt.xlabel("Spearman's rho")
    # plt.ylabel("Frequency")
    # plt.savefig(f"{cfg.out_dir}/spearmanr_hist_across_samples.png")
    # plt.close()

    # # for each protein, plot spearmanr between npa and rmsd across the position
    # metrics_per_pdb_per_pos = df.groupby(['pdb', 'position']).apply(lambda x: spearmanr(x["npa"], x["scn_rmsd_per_pos"])[0]).reset_index(name='spearmanr')
    # plt.figure()
    # plt.hist(metrics_per_pdb_per_pos["spearmanr"], bins=20)
    # plt.title(f"Spearman's rho between npa and rmsd for each protein and sample across positions\n average={metrics_per_pdb_per_pos['spearmanr'].mean():.4f}")
    # plt.xlabel("Spearman's rho")
    # plt.ylabel("Frequency")
    # plt.savefig(f"{cfg.out_dir}/spearmanr_hist_across_positions.png")
    # plt.close()

    # for each protein, plot spearmanr between aatype seq logit and rmsd within the protein
    spearmanr_per_pdb = df.groupby('pdb').apply(lambda x: spearmanr(x["seq_logits_aatype"], x["scn_rmsd_per_pos"])[0]).reset_index(name='spearmanr')
    plt.figure()
    plt.hist(spearmanr_per_pdb["spearmanr"], bins=20)
    plt.title(f"Spearman's rho between seq logits and rmsd for each protein\nMedian={spearmanr_per_pdb['spearmanr'].median():.4f},mean={spearmanr_per_pdb['spearmanr'].mean():.4f}")
    plt.xlabel("Spearman's rho")
    plt.ylabel("Frequency")
    plt.savefig(f"{cfg.out_dir}/spearmanr_hist_seq_logits_vs_rmsd.png")
    plt.close()

    # for each protein, plot spearmanr between aatype seq logit and npa within the protein
    spearmanr_per_pdb = df.groupby('pdb').apply(lambda x: spearmanr(x["seq_logits_aatype"], x["npa"])[0]).reset_index(name='spearmanr')
    plt.figure()
    plt.hist(spearmanr_per_pdb["spearmanr"], bins=20)
    plt.title(f"Spearman's rho between seq logits and npa for each protein\nMedian={spearmanr_per_pdb['spearmanr'].median():.4f},mean={spearmanr_per_pdb['spearmanr'].mean():.4f}")
    plt.xlabel("Spearman's rho")
    plt.ylabel("Frequency")
    plt.savefig(f"{cfg.out_dir}/spearmanr_hist_seq_logits_vs_npa.png")
    plt.close()

    # for each protein, plot spearmanr between aatype seq logit * npa
    spearmanr_per_pdb = df.groupby('pdb').apply(lambda x: spearmanr(x["seq_logits_aatype"] * x["npa"], x["scn_rmsd_per_pos"])[0]).reset_index(name='spearmanr')
    plt.figure()
    plt.hist(spearmanr_per_pdb["spearmanr"], bins=20)
    plt.title(f"Spearman's rho between seq logits * npa and rmsd for each protein\nMedian={spearmanr_per_pdb['spearmanr'].median():.4f},mean={spearmanr_per_pdb['spearmanr'].mean():.4f}")
    plt.xlabel("Spearman's rho")
    plt.ylabel("Frequency")
    plt.savefig(f"{cfg.out_dir}/spearmanr_hist_seq_logits_times_npa_vs_rmsd.png")
    plt.close()

    # for each protein, plot spearmanr between aatype seq prob and rmsd within the protein
    spearmanr_per_pdb = df.groupby('pdb').apply(lambda x: spearmanr(x["seq_probs_aatype"], x["scn_rmsd_per_pos"])[0]).reset_index(name='spearmanr')
    plt.figure()
    plt.hist(spearmanr_per_pdb["spearmanr"], bins=20)
    plt.title(f"Spearman's rho between seq probs and rmsd for each protein\nMedian={spearmanr_per_pdb['spearmanr'].median():.4f},mean={spearmanr_per_pdb['spearmanr'].mean():.4f}")
    plt.xlabel("Spearman's rho")
    plt.ylabel("Frequency")
    plt.savefig(f"{cfg.out_dir}/spearmanr_hist_seq_probs_vs_rmsd.png")
    plt.close()

    # for each protein, plot spearmanr between aatype seq prob and npa within the protein





if __name__ == "__main__":
    main()
