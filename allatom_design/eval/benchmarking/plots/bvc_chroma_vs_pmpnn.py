from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="bvc_chroma_vs_pmpnn")
def main(cfg: DictConfig) -> None:
    """
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Load CSVs
    chroma_boltz_df = pd.read_csv(cfg.chroma_boltz_csv)
    pmpnn_boltz_df = pd.read_csv(cfg.pmpnn_boltz_csv)
    chroma_esmfold_df = pd.read_csv(cfg.chroma_esmfold_csv)
    pmpnn_esmfold_df = pd.read_csv(cfg.pmpnn_esmfold_csv)

    # Handle plddt scaling
    chroma_boltz_df["avg_ca_plddt"] = chroma_boltz_df["avg_ca_plddt"] * 100
    pmpnn_boltz_df["avg_ca_plddt"] = pmpnn_boltz_df["avg_ca_plddt"] * 100

    # Create a function to generate and save scatter plots
    create_scatter_plots(
        chroma_boltz_df,
        pmpnn_boltz_df,
        out_dir=cfg.out_dir,
        prefix="boltz",
        metric="sc_ca_rmsd",
        metric_title="Boltz-1x scRMSD",
        best_is_min=True
    )
    create_scatter_plots(
        chroma_boltz_df,
        pmpnn_boltz_df,
        out_dir=cfg.out_dir,
        prefix="boltz",
        metric="avg_ca_plddt",
        metric_title="Boltz-1x average pLDDT",
        best_is_min=False
    )
    create_scatter_plots(
        chroma_esmfold_df,
        pmpnn_esmfold_df,
        out_dir=cfg.out_dir,
        prefix="esmfold",
        metric="sc_ca_rmsd",
        metric_title="ESMFold scRMSD",
        best_is_min=True
    )
    create_scatter_plots(
        chroma_esmfold_df,
        pmpnn_esmfold_df,
        out_dir=cfg.out_dir,
        prefix="esmfold",
        metric="avg_ca_plddt",
        metric_title="ESMFold average pLDDT",
        best_is_min=False
    )


def create_scatter_plots(
    chroma_df: pd.DataFrame,
    pmpnn_df: pd.DataFrame,
    out_dir: str,
    prefix: str,
    metric: str,
    metric_title: str,
    best_is_min: bool
) -> None:
    # Extract pdb code
    chroma_df["pdb_id"] = chroma_df["record_id"].apply(lambda x: x.split("_sample")[0])
    pmpnn_df["pdb_id"] = pmpnn_df["record_id"].apply(lambda x: x.split("_sample")[0])

    # Aggregate best or worst
    if best_is_min:
        chroma_best = chroma_df.groupby("pdb_id")[metric].min().rename("chroma_best")
        pmpnn_best = pmpnn_df.groupby("pdb_id")[metric].min().rename("pmpnn_best")
    else:
        chroma_best = chroma_df.groupby("pdb_id")[metric].max().rename("chroma_best")
        pmpnn_best = pmpnn_df.groupby("pdb_id")[metric].max().rename("pmpnn_best")
    chroma_median = chroma_df.groupby("pdb_id")[metric].median().rename("chroma_median")
    pmpnn_median = pmpnn_df.groupby("pdb_id")[metric].median().rename("pmpnn_median")

    best_df = pd.merge(chroma_best, pmpnn_best, left_index=True, right_index=True)
    median_df = pd.merge(chroma_median, pmpnn_median, left_index=True, right_index=True)

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
        ax.set_xlabel("Chroma")
        ax.set_ylabel("ProteinMPNN")
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
    # plt.scatter(best_df["chroma_best"], best_df["pmpnn_best"])
    # ax = plt.gca()
    # ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    # plt.xlabel("Chroma")
    # plt.ylabel("ProteinMPNN")
    # plt.title(f"{metric_title} (best of 8, n={len(best_df)})")
    # plt.savefig(f"{out_dir}/{prefix}_{metric}_best.png")
    # plt.close()

    # # Plot median
    # plt.figure(figsize=(10, 10))
    # plt.scatter(median_df["chroma_median"], median_df["pmpnn_median"])
    # ax = plt.gca()
    # ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    # plt.xlabel("Chroma")
    # plt.ylabel("ProteinMPNN")
    # plt.title(f"{metric_title} (median of 8, n={len(median_df)})")
    # plt.savefig(f"{out_dir}/{prefix}_{metric}_median.png")
    # plt.close()


if __name__ == "__main__":
    main()
