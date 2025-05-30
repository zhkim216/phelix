from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="compare_self_consistency")
def main(cfg: DictConfig) -> None:
    """
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Dump config to out_dir
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load CSVs
    model1_boltz_df = pd.read_csv(cfg.model1_boltz_csv)
    model2_boltz_df = pd.read_csv(cfg.model2_boltz_csv)

    # Handle plddt scaling
    model1_boltz_df["avg_ca_plddt"] = model1_boltz_df["avg_ca_plddt"] * 100
    model2_boltz_df["avg_ca_plddt"] = model2_boltz_df["avg_ca_plddt"] * 100

    # Load subset pdb names
    if cfg.subset_pdb_names is not None:
        with open(cfg.subset_pdb_names, "r") as f:
            subset_pdb_names = [line.strip() for line in f.readlines()]
    else:
        subset_pdb_names = None

    # extract pdb_name from record id
    model1_boltz_df["pdb_name"] = model1_boltz_df["record_id"].apply(lambda x: f'{x.split("_sample")[0]}.cif')
    model2_boltz_df["pdb_name"] = model2_boltz_df["record_id"].apply(lambda x: f'{x.split("_sample")[0]}.cif')
    if subset_pdb_names is not None:
        model1_boltz_df = model1_boltz_df[model1_boltz_df["pdb_name"].isin(subset_pdb_names)]
        model2_boltz_df = model2_boltz_df[model2_boltz_df["pdb_name"].isin(subset_pdb_names)]

    # Create a function to generate and save scatter plots
    create_scatter_plots(
        model1_boltz_df,
        model2_boltz_df,
        model1_name=cfg.model1_name,
        model2_name=cfg.model2_name,
        out_dir=cfg.out_dir,
        prefix="boltz",
        metric="sc_ca_rmsd",
        metric_title="Boltz-1x scRMSD",
        best_is_min=True
    )
    create_scatter_plots(
        model1_boltz_df,
        model2_boltz_df,
        model1_name=cfg.model1_name,
        model2_name=cfg.model2_name,
        out_dir=cfg.out_dir,
        prefix="boltz",
        metric="avg_ca_plddt",
        metric_title="Boltz-1x average pLDDT",
        best_is_min=False
    )


def create_scatter_plots(
    model1_df: pd.DataFrame,
    model2_df: pd.DataFrame,
    model1_name: str,
    model2_name: str,
    out_dir: str,
    prefix: str,
    metric: str,
    metric_title: str,
    best_is_min: bool
) -> None:
    # Extract pdb code
    model1_df["pdb_id"] = model1_df["record_id"].apply(lambda x: x.split("_sample")[0])
    model2_df["pdb_id"] = model2_df["record_id"].apply(lambda x: x.split("_sample")[0])

    # Aggregate best or worst
    if best_is_min:
        model1_best = model1_df.groupby("pdb_id")[metric].min().rename(f"{model1_name}_best")
        model2_best = model2_df.groupby("pdb_id")[metric].min().rename(f"{model2_name}_best")
    else:
        model1_best = model1_df.groupby("pdb_id")[metric].max().rename(f"{model1_name}_best")
        model2_best = model2_df.groupby("pdb_id")[metric].max().rename(f"{model2_name}_best")
    model1_median = model1_df.groupby("pdb_id")[metric].median().rename(f"{model1_name}_median")
    model2_median = model2_df.groupby("pdb_id")[metric].median().rename(f"{model2_name}_median")

    best_df = pd.merge(model1_best, model2_best, left_index=True, right_index=True)
    median_df = pd.merge(model1_median, model2_median, left_index=True, right_index=True)

    def plot_and_annotate(df: pd.DataFrame, kind: str):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        ax.scatter(x, y, s=4)

        # Compute unified limits
        lower, upper = 0.0, max(np.max(x), np.max(y))
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

        # Diagonal from (0,0) to (upper,upper)
        ax.plot([lower, upper], [lower, upper], 'k--')

        # Annotations
        for pdb_id, xi, yi in zip(df.index, x, y):
            ax.annotate(
                pdb_id,
                (xi, yi),
                textcoords="offset points",
                xytext=(0, -8),
                ha='center',
                fontsize=6
            )

        # Labels, title, grid
        ax.set_xlabel(model1_name)
        ax.set_ylabel(model2_name)
        n = len(df)
        ax.set_title(f"{metric_title} ({kind} of 8, n={n})")
        plt.grid(True, alpha=0.5)

        # Ticks
        if metric == "sc_ca_rmsd":
            ticks = np.arange(0, upper + 1e-6, 2.0)
        else:
            ticks = np.arange(0, upper + 1e-6, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        plt.savefig(f"{out_dir}/{prefix}_{metric}_{kind}.png", dpi=300)
        plt.close()

    plot_and_annotate(best_df, 'best')
    plot_and_annotate(median_df, 'median')


    # # Plot best
    # plt.figure(figsize=(10, 10))
    # plt.scatter(best_df["chroma_best"], best_df["atommpnn_best"])
    # ax = plt.gca()
    # ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    # plt.xlabel("Chroma")
    # plt.ylabel("ProteinMPNN")
    # plt.title(f"{metric_title} (best of 8, n={len(best_df)})")
    # plt.savefig(f"{out_dir}/{prefix}_{metric}_best.png")
    # plt.close()

    # # Plot median
    # plt.figure(figsize=(10, 10))
    # plt.scatter(median_df["chroma_median"], median_df["atommpnn_median"])
    # ax = plt.gca()
    # ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    # plt.xlabel("Chroma")
    # plt.ylabel("ProteinMPNN")
    # plt.title(f"{metric_title} (median of 8, n={len(median_df)})")
    # plt.savefig(f"{out_dir}/{prefix}_{metric}_median.png")
    # plt.close()


if __name__ == "__main__":
    main()
