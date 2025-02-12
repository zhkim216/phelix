import glob
import os
import pickle
from pathlib import Path
from typing import Dict, List

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

import allatom_design.data.residue_constants as rc
from tqdm import tqdm


@hydra.main(version_base="1.3.2", config_path="../configs/eval", config_name="plot_monomer_eval")
def main(cfg: DictConfig):
    """
    Loads:
     - sample_pkls from multiple directories to compute sequence recovery
       (ignoring positions where either aatype_override_mask=1 or scn_override_mask=1,
        and ignoring any unknown (X) tokens in predicted or true sequences).
     - self_consistency_metrics.csv from each directory to retrieve sc_ca_rmsd and sc_ca_tm.
    Aggregates results (mean and median sequence recovery, median sc_ca_rmsd, median sc_ca_tm)
    as a function of t for both seq and scn, then plots them.
    """
    torch.set_grad_enabled(False)
    L.seed_everything(cfg.seed)

    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for eval_dir in cfg.eval_dirs:
        # Infer method and fraction (t) from directory name (e.g. "scn_0.1" -> method=scn, fraction=0.1)
        # If the directory name doesn't match this pattern, adjust as needed.
        method, fraction_str = Path(eval_dir).name.split("_")
        try:
            fraction = float(fraction_str)
        except ValueError:
            # If there's any mismatch, skip or handle differently
            print(f"Directory {eval_dir} name does not conform to 'method_fraction'. Skipping.")
            continue

        sample_pkl_dir = Path(eval_dir) / "sample_pkls"
        sc_metrics_csv = Path(eval_dir) / "self_consistency_metrics.csv"
        if (not sample_pkl_dir.is_dir()) or (not sc_metrics_csv.is_file()):
            print(f"Warning: Missing PKLs or CSV in {eval_dir}. Skipping.")
            continue

        # Load the self-consistency CSV
        sc_df = pd.read_csv(sc_metrics_csv)
        # 'pdb_name' typically matches the sample PKL stem, e.g. "7qf8_A_267_sample0"
        # contains columns: [pdb_name, pdb_key, sc_ca_rmsd, sc_aa_rmsd, sc_ca_tm, pred_seq]
        sc_df = sc_df.rename(columns={"pdb_name": "pkl_stem"})

        # Build a dict from pkl_stem -> {sc_ca_rmsd, sc_ca_tm} so we can merge after computing seq recovery
        sc_map = {}
        for _, row in sc_df.iterrows():
            sc_map[row["pkl_stem"]] = {
                "sc_ca_rmsd": row["sc_ca_rmsd"],
                "sc_ca_tm": row["sc_ca_tm"]
            }

        # Read PKLs
        pkl_files = list(sample_pkl_dir.glob("*.pkl"))
        for pkl_file in tqdm(pkl_files, desc=f"Processing PKLs in {eval_dir}"):
            pkl_stem = pkl_file.stem
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            pred_aatype = data["pred_aatype"]
            true_aatype = data["original_aatype"]
            override_seq_mask = data["aatype_override_mask"].astype(bool)
            override_scn_mask = data["scn_override_mask"].astype(bool)

            valid_mask = ~(override_seq_mask | override_scn_mask)

            # Ignore unknown (X) tokens in predicted or true. X is index 20 in restypes_with_x
            x_idx = rc.restype_order_with_x["X"]
            valid_mask = valid_mask & (true_aatype != x_idx) & (pred_aatype != x_idx)

            if np.sum(valid_mask) == 0:
                seq_acc = np.nan
            else:
                correct = (pred_aatype[valid_mask] == true_aatype[valid_mask])
                seq_acc = correct.mean()

            sc_ca_rmsd = np.nan
            sc_ca_tm = np.nan
            if pkl_stem in sc_map:
                sc_ca_rmsd = sc_map[pkl_stem]["sc_ca_rmsd"]
                sc_ca_tm = sc_map[pkl_stem]["sc_ca_tm"]

            results.append({
                "pkl_stem": pkl_stem,
                "method": method,
                "t": fraction,
                "seq_acc": seq_acc,
                "sc_ca_rmsd": sc_ca_rmsd,
                "sc_ca_tm": sc_ca_tm
            })

    df = pd.DataFrame(results)
    df_out_path = Path(out_dir) / "monomer_eval_raw.csv"
    df.to_csv(df_out_path, index=False)

    # Group by method, t
    grouped = df.groupby(["method", "t"], dropna=False)
    agg_df = grouped.agg(
        median_seq_acc=("seq_acc", "median"),
        median_sc_ca_rmsd=("sc_ca_rmsd", "median"),
        median_sc_ca_tm=("sc_ca_tm", "median")
    ).reset_index()

    agg_csv_path = Path(out_dir) / "monomer_eval_summary.csv"
    agg_df.to_csv(agg_csv_path, index=False)

    # Plot
    # We'll create one figure with three subplots:
    #   (1) mean & median seq_acc vs t
    #   (2) median sc_ca_rmsd vs t
    #   (3) median sc_ca_tm vs t

    methods = agg_df["method"].unique()
    ts = sorted(agg_df["t"].unique())
    plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(311)
    for m in methods:
        sub = agg_df[agg_df["method"] == m].sort_values("t")
        ax1.plot(sub["t"], sub["median_seq_acc"], marker="s", label=f"{m} median")
    ax1.set_ylabel("Sequence Recovery")
    ax1.set_title("Sequence Accuracy vs t")
    ax1.legend()

    ax2 = plt.subplot(312)
    for m in methods:
        sub = agg_df[agg_df["method"] == m].sort_values("t")
        ax2.plot(sub["t"], sub["median_sc_ca_rmsd"], marker="o", label=m)
    ax2.set_ylabel("sc_ca_rmsd (median)")
    ax2.set_title("sc_ca_rmsd vs t")
    ax2.legend()

    ax3 = plt.subplot(313)
    for m in methods:
        sub = agg_df[agg_df["method"] == m].sort_values("t")
        ax3.plot(sub["t"], sub["median_sc_ca_tm"], marker="o", label=m)
    ax3.set_xlabel("t")
    ax3.set_ylabel("sc_ca_tm (median)")
    ax3.set_title("sc_ca_tm vs t")
    ax3.legend()

    plt.tight_layout()
    out_plot_path = Path(out_dir) / "monomer_eval_plots.png"
    plt.savefig(out_plot_path, dpi=150)
    plt.close()

    print(f"Done! Raw results in {df_out_path}, summary in {agg_csv_path}, figure in {out_plot_path}")


def load_pkl_data(pkl_file: Path) -> Dict:
    return {}  # Helper function if needed. Currently unused.


if __name__ == "__main__":
    main()
