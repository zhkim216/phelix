import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
import re

@hydra.main(config_path="../configs/eval", config_name="plot_sc_dataset", version_base="1.3.2")
def main(cfg: DictConfig):
    # Convert to a dict for easy usage
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create output directory if it doesn't exist
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the metrics CSV
    df = pd.read_csv(cfg.csv_path)

    # We expect columns like: 'pdb_key', 'sc_ca_rmsd'
    required_columns = ["pdb_key", "sc_ca_rmsd"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {cfg.csv_path}")

    # Subsample to N unique PDB keys if there are more than N
    unique_bases = df["pdb_key"].unique()
    if len(unique_bases) > cfg.subsample_n:
        unique_bases = unique_bases[:cfg.subsample_n]
    df = df[df["pdb_key"].isin(unique_bases)]

    # Sort by pdb_key
    df = df.sort_values("pdb_key")

    # Compute range and min values before clipping
    df_range = df.groupby("pdb_key").agg(
        sc_ca_rmsd_range=("sc_ca_rmsd", lambda x: x.max() - x.min()),
    ).reset_index()

    df_min = df.groupby("pdb_key").agg(
        min_sc_ca_rmsd=("sc_ca_rmsd", "min"),
    ).reset_index()

    # Melt range and min dataframes
    df_range_melt = df_range.melt(id_vars=["pdb_key"],
                                  value_vars=["sc_ca_rmsd_range"],
                                  var_name="metric", value_name="range_value")

    df_min_melt = df_min.melt(id_vars=["pdb_key"],
                              value_vars=["min_sc_ca_rmsd"],
                              var_name="metric", value_name="min_value")

    # Clip values for plotting
    df["sc_ca_rmsd_clipped"] = df["sc_ca_rmsd"].clip(upper=15)
    df_range_melt["range_value_clipped"] = df_range_melt["range_value"].clip(upper=15)
    df_min_melt["min_value_clipped"] = df_min_melt["min_value"].clip(upper=15)

    sns.set_theme(style="whitegrid")

    # Plot sc_ca_rmsd clipped
    fig_ca, ax_ca = plt.subplots(figsize=(12, 6))
    sns.stripplot(data=df, x="pdb_key", y="sc_ca_rmsd_clipped", ax=ax_ca, size=5)
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

    # Plot the ranges clipped
    fig_range, ax_range = plt.subplots(figsize=(12, 6))
    sns.stripplot(data=df_range_melt, x="pdb_key", y="range_value_clipped", hue="metric", ax=ax_range, size=5)
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
    sns.stripplot(data=df_min_melt, x="pdb_key", y="min_value_clipped", hue="metric", ax=ax_min, size=5)
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

    # Now repeat the above plotting for each temperature present in the DataFrame
    if "temperature" in df.columns:
        unique_temps = df["temperature"].unique()
        for temp in unique_temps:
            df_temp = df[df["temperature"] == temp].copy()

            # Subsample to N unique PDB keys if there are more than N
            unique_bases_temp = df_temp["pdb_key"].unique()
            if len(unique_bases_temp) > cfg.subsample_n:
                unique_bases_temp = unique_bases_temp[:cfg.subsample_n]
            df_temp = df_temp[df_temp["pdb_key"].isin(unique_bases_temp)]

            df_temp = df_temp.sort_values("pdb_key")

            # Compute range and min values for this temperature
            df_range_temp = df_temp.groupby("pdb_key").agg(
                sc_ca_rmsd_range=("sc_ca_rmsd", lambda x: x.max() - x.min()),
            ).reset_index()
            df_min_temp = df_temp.groupby("pdb_key").agg(
                min_sc_ca_rmsd=("sc_ca_rmsd", "min"),
            ).reset_index()

            df_range_temp_melt = df_range_temp.melt(
                id_vars=["pdb_key"],
                value_vars=["sc_ca_rmsd_range"],
                var_name="metric", value_name="range_value"
            )
            df_min_temp_melt = df_min_temp.melt(
                id_vars=["pdb_key"],
                value_vars=["min_sc_ca_rmsd"],
                var_name="metric", value_name="min_value"
            )

            # Clip
            df_temp["sc_ca_rmsd_clipped"] = df_temp["sc_ca_rmsd"].clip(upper=15)
            df_range_temp_melt["range_value_clipped"] = df_range_temp_melt["range_value"].clip(upper=15)
            df_min_temp_melt["min_value_clipped"] = df_min_temp_melt["min_value"].clip(upper=15)

            # Make a subdirectory or suffix for each temperature's plots
            temp_dir = out_dir / f"temperature_{temp}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # sc_ca_rmsd clipped plot
            fig_ca_temp, ax_ca_temp = plt.subplots(figsize=(12, 6))
            sns.stripplot(data=df_temp, x="pdb_key", y="sc_ca_rmsd_clipped", ax=ax_ca_temp, size=5)
            plt.xticks(rotation=90)
            plt.xlabel("Base PDB Key")
            plt.ylabel("sc_ca_rmsd (Å)")
            plt.title(f"Self-Consistency sc_ca_rmsd (Temp={temp})")
            ax_ca_temp.set_ylim(0, 15)
            plt.tight_layout()
            out_fig_ca_temp = temp_dir / f"self_consistency_sc_ca_rmsd_temp_{temp}.png"
            plt.savefig(out_fig_ca_temp, dpi=300)
            plt.close(fig_ca_temp)
            print(f"Plot saved to {out_fig_ca_temp}")

            # RMSD range plot
            fig_range_temp, ax_range_temp = plt.subplots(figsize=(12, 6))
            sns.stripplot(data=df_range_temp_melt, x="pdb_key", y="range_value_clipped",
                          hue="metric", ax=ax_range_temp, size=5)
            plt.xticks(rotation=90)
            plt.xlabel("Base PDB Key")
            plt.ylabel("Range (max - min) (Å)")
            plt.title(f"Self-Consistency RMSD Range (Temp={temp})")
            ax_range_temp.set_ylim(0, 15)
            plt.legend(title="Metric")
            plt.tight_layout()
            out_fig_range_temp = temp_dir / f"self_consistency_rmsd_range_temp_{temp}.png"
            plt.savefig(out_fig_range_temp, dpi=300)
            plt.close(fig_range_temp)
            print(f"Plot saved to {out_fig_range_temp}")

            # Min RMSD plot
            fig_min_temp, ax_min_temp = plt.subplots(figsize=(12, 6))
            sns.stripplot(data=df_min_temp_melt, x="pdb_key", y="min_value_clipped",
                          hue="metric", ax=ax_min_temp, size=5)
            plt.xticks(rotation=90)
            plt.xlabel("Base PDB Key")
            plt.ylabel("Min RMSD (Å)")
            plt.title(f"Minimum Self-Consistency RMSD (Temp={temp})")
            ax_min_temp.set_ylim(0, 15)
            plt.legend(title="Metric")
            plt.tight_layout()
            out_fig_min_temp = temp_dir / f"self_consistency_min_rmsd_temp_{temp}.png"
            plt.savefig(out_fig_min_temp, dpi=300)
            plt.close(fig_min_temp)
            print(f"Plot saved to {out_fig_min_temp}")


if __name__ == "__main__":
    main()
