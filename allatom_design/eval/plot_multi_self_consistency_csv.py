import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
import re

@hydra.main(config_path="../configs/eval", config_name="plot_multi_self_consistency_csv", version_base="1.3.2")
def main(cfg: DictConfig):
    # Convert to a dict for easy usage
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create output directory if it doesn't exist
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the metrics CSV
    df = pd.read_csv(cfg.csv_path)

    # We expect columns like: 'pdb_key', 'sc_ca_rmsd', 'sc_aa_rmsd'.
    required_columns = ["pdb_key", "sc_ca_rmsd", "sc_aa_rmsd"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {cfg.csv_path}")

    # Extract the base PDB key by removing the "_sampleX" part
    df["base_pdb_key"] = df["pdb_key"].str.replace(r"_sample\d+$", "", regex=True)

    # Subsample to N unique base PDB keys if there are more than N
    unique_bases = df["base_pdb_key"].unique()
    if len(unique_bases) > cfg.subsample_n:
        unique_bases = unique_bases[:cfg.subsample_n]
    df = df[df["base_pdb_key"].isin(unique_bases)]

    # Sort by base_pdb_key
    df = df.sort_values("base_pdb_key")

    # Compute range and min values before clipping
    df_range = df.groupby("base_pdb_key").agg(
        sc_ca_rmsd_range=("sc_ca_rmsd", lambda x: x.max() - x.min()),
        sc_aa_rmsd_range=("sc_aa_rmsd", lambda x: x.max() - x.min())
    ).reset_index()

    df_min = df.groupby("base_pdb_key").agg(
        min_sc_ca_rmsd=("sc_ca_rmsd", "min"),
        min_sc_aa_rmsd=("sc_aa_rmsd", "min")
    ).reset_index()

    # Melt range and min dataframes
    df_range_melt = df_range.melt(id_vars=["base_pdb_key"],
                                  value_vars=["sc_ca_rmsd_range", "sc_aa_rmsd_range"],
                                  var_name="metric", value_name="range_value")

    df_min_melt = df_min.melt(id_vars=["base_pdb_key"],
                              value_vars=["min_sc_ca_rmsd", "min_sc_aa_rmsd"],
                              var_name="metric", value_name="min_value")

    # Clip values for plotting
    # For original dataframe
    df["sc_ca_rmsd_clipped"] = df["sc_ca_rmsd"].clip(upper=15)
    df["sc_aa_rmsd_clipped"] = df["sc_aa_rmsd"].clip(upper=15)

    # For range and min data
    df_range_melt["range_value_clipped"] = df_range_melt["range_value"].clip(upper=15)
    df_min_melt["min_value_clipped"] = df_min_melt["min_value"].clip(upper=15)

    sns.set_theme(style="whitegrid")

    # Plot sc_ca_rmsd clipped
    fig_ca, ax_ca = plt.subplots(figsize=(12, 6))
    sns.stripplot(data=df, x="base_pdb_key", y="sc_ca_rmsd_clipped", ax=ax_ca, size=5)
    plt.xticks(rotation=90)
    plt.xlabel("Base PDB Key")
    plt.ylabel("sc_ca_rmsd (Å)")
    plt.title("Self-Consistency sc_ca_rmsd by Base PDB (Subsampled)")
    ax_ca.set_ylim(0, 15)
    plt.tight_layout()
    out_fig_ca = out_dir / "self_consistency_sc_ca_rmsd.png"
    plt.savefig(out_fig_ca, dpi=300)
    plt.close(fig_ca)
    print(f"Plot saved to {out_fig_ca}")

    # Plot sc_aa_rmsd clipped
    fig_aa, ax_aa = plt.subplots(figsize=(12, 6))
    sns.stripplot(data=df, x="base_pdb_key", y="sc_aa_rmsd_clipped", ax=ax_aa, size=5)
    plt.xticks(rotation=90)
    plt.xlabel("Base PDB Key")
    plt.ylabel("sc_aa_rmsd (Å)")
    plt.title("Self-Consistency sc_aa_rmsd by Base PDB (Subsampled)")
    ax_aa.set_ylim(0, 15)
    plt.tight_layout()
    out_fig_aa = out_dir / "self_consistency_sc_aa_rmsd.png"
    plt.savefig(out_fig_aa, dpi=300)
    plt.close(fig_aa)
    print(f"Plot saved to {out_fig_aa}")

    # Plot the ranges clipped
    fig_range, ax_range = plt.subplots(figsize=(12, 6))
    sns.stripplot(data=df_range_melt, x="base_pdb_key", y="range_value_clipped", hue="metric", ax=ax_range, size=5)
    plt.xticks(rotation=90)
    plt.xlabel("Base PDB Key")
    plt.ylabel("Range (max - min) (Å)")
    plt.title("Self-Consistency RMSD Range by Base PDB (Subsampled)")
    ax_range.set_ylim(0, 15)
    plt.legend(title="Metric")
    plt.tight_layout()
    out_fig_range = out_dir / "self_consistency_rmsd_range.png"
    plt.savefig(out_fig_range, dpi=300)
    plt.close(fig_range)
    print(f"Plot saved to {out_fig_range}")

    # Plot the min values clipped
    fig_min, ax_min = plt.subplots(figsize=(12, 6))
    sns.stripplot(data=df_min_melt, x="base_pdb_key", y="min_value_clipped", hue="metric", ax=ax_min, size=5)
    plt.xticks(rotation=90)
    plt.xlabel("Base PDB Key")
    plt.ylabel("Min RMSD (Å)")
    plt.title("Minimum Self-Consistency RMSD by Base PDB (Subsampled)")
    ax_min.set_ylim(0, 15)
    plt.legend(title="Metric")
    plt.tight_layout()
    out_fig_min = out_dir / "self_consistency_min_rmsd.png"
    plt.savefig(out_fig_min, dpi=300)
    plt.close(fig_min)
    print(f"Plot saved to {out_fig_min}")


if __name__ == "__main__":
    main()
