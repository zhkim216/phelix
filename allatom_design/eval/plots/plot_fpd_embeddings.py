import pickle
import re
from pathlib import Path
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import yaml
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA

from allatom_design.eval.eval_utils.esm3_utils import load_esm3_embeddings
from allatom_design.eval.eval_utils.eval_metrics import fpd


@hydra.main(version_base="1.3.2", config_path="../../configs/eval/plots", config_name="plot_fpd_embeddings")
def main(cfg: DictConfig):

    # Convert cfg to a dict and store it if desired
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create output directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Save config for record-keeping
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Get sampled embeddings
    sampled_dir = Path(cfg.sampled_embeddings_dir) / cfg.embedding_subdir
    sampled_files = list(sampled_dir.glob("*.pkl"))

    # Get reference embeddings
    ref_files = []
    for sf in sampled_files:
        # Example: sf.name = "step_500000_1qamA01_6_L157.pkl"
        # Remove leading "step_XXXXXX_" and trailing "_L####"
        base = sf.stem  # "step_500000_1qamA01_6_L157"
        no_prefix = re.sub(r"^step_\d+_", "", base)  # "1qamA01_6_L157"
        pdb_key = re.sub(r"_L\d+$", "", no_prefix)   # "1qamA01_6"

        # Reference embedding path
        ref_path = Path(cfg.fpd_embeddings_dir) / cfg.embedding_subdir / f"{pdb_key}.pkl"
        ref_files.append(ref_path)

    # Load the reference embeddings
    ref_embeds, _ = load_esm3_embeddings(ref_files)

    # Load the sampled embeddings
    samp_embeds, _ = load_esm3_embeddings(sampled_files)

    # Plot PCA
    pca_model = plot_pca(ref_embeds, samp_embeds, f"{cfg.out_dir}/esm3_pca.png")

    # Plot KDE of PCA embeddings
    plot_pca_kde(ref_embeds, samp_embeds, pca_model, f"{cfg.out_dir}/esm3_pca_kde.png")

    # If we also want to do a raster layout of reference PDBs in a grid:
    if cfg.raster.plot_raster:
        ref_pca = pca_model.transform(ref_embeds)

        # Build a dataframe of reference PCA coords + corresponding pdb_key
        ref_keys = []
        for rf in ref_files:
            base = rf.stem
            base = re.sub(r"_L\d+$", "", base)
            base = re.sub(r"^step_\d+_", "", base)
            ref_keys.append(base)

        df_ref = pd.DataFrame({
            "x": ref_pca[:, 0],
            "y": ref_pca[:, 1],
            "pdb_key": ref_keys
        })

        create_raster(df_ref, cfg)


def plot_pca(ref_embeds: np.ndarray, samp_embeds: np.ndarray, out_fp: str) -> PCA:
    """
    Creates a single-panel PCA plot comparing reference vs. sample embeddings,
    and saves it to out_fp.
    """
    # Compute FPD
    fpd_score = fpd(ref_embeds, samp_embeds)

    # Fit PCA on reference, transform both
    pca_model = PCA(n_components=2)
    pca_model.fit(ref_embeds)
    ref_pca = pca_model.transform(ref_embeds)
    samp_pca = pca_model.transform(samp_embeds)

    # Build dataframe
    df_pca = pd.DataFrame({
        "x": np.concatenate([ref_pca[:, 0], samp_pca[:, 0]]),
        "y": np.concatenate([ref_pca[:, 1], samp_pca[:, 1]]),
        "Label": ["Reference"] * len(ref_pca) + ["Sample"] * len(samp_pca),
    })

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(
        ax=ax,
        data=df_pca,
        x="x",
        y="y",
        hue="Label",
        alpha=0.5,
        s=2,
        palette="Set2"
    )
    ax.set_title("PCA")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.text(
        0.95,
        0.95,
        f"FPD = {fpd_score:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right"
    )
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close(fig)

    return pca_model


def plot_pca_kde(ref_embeds: np.ndarray, samp_embeds: np.ndarray, pca_model: PCA, out_fp: str,
                 bw_adjust: float = 1,
                 ) -> None:
    """
    Creates a KDE plot of the PCA embeddings comparing reference vs. sample,
    and saves it to out_fp.
    """
    # Transform embeddings using the provided PCA model
    ref_pca = pca_model.transform(ref_embeds)
    samp_pca = pca_model.transform(samp_embeds)

    # Build dataframe
    df_pca = pd.DataFrame({
        "x": np.concatenate([ref_pca[:, 0], samp_pca[:, 0]]),
        "y": np.concatenate([ref_pca[:, 1], samp_pca[:, 1]]),
        "Label": ["Reference"] * len(ref_pca) + ["Sample"] * len(samp_pca),
    })

    # Plot KDE
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(
        data=df_pca[df_pca["Label"] == "Reference"],
        x="x",
        y="y",
        fill=True,
        alpha=0.5,
        color="blue",
        ax=ax,
        label="Reference",
        bw_adjust=bw_adjust
    )
    sns.kdeplot(
        data=df_pca[df_pca["Label"] == "Sample"],
        x="x",
        y="y",
        fill=True,
        alpha=0.5,
        color="orange",
        ax=ax,
        label="Sample",
        bw_adjust=bw_adjust
    )
    ax.set_title("PCA KDE")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.plot([], [], color="blue", label="Reference")
    ax.plot([], [], color="orange", label="Sample")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close(fig)


def create_raster(df_ref: pd.DataFrame, cfg: DictConfig) -> None:
    raster_dir = Path(cfg.out_dir, "raster_pdbs")
    raster_dir.mkdir(parents=True, exist_ok=True)

    n_grid = cfg.raster.get("n_grid", 10)
    xbins = np.linspace(df_ref["x"].min(), df_ref["x"].max(), n_grid + 1)
    ybins = np.linspace(df_ref["y"].min(), df_ref["y"].max(), n_grid + 1)

    idx = 0
    for i in range(n_grid):
        # row_index goes from top (largest Y) to bottom (smallest Y)
        row_index = n_grid - 1 - i
        for j in range(n_grid):
            xlow, xhigh = xbins[j], xbins[j+1]
            ylow, yhigh = ybins[row_index], ybins[row_index+1]
            subset = df_ref[
                (df_ref["x"] >= xlow) & (df_ref["x"] < xhigh) &
                (df_ref["y"] >= ylow) & (df_ref["y"] < yhigh)
            ]

            out_fp = Path(raster_dir, f"pdb_{idx}.pdb")
            idx += 1

            if len(subset) == 0:
                create_dummy_pdb(out_fp)
            else:
                chosen = subset.sample(1).iloc[0]
                pdb_key = chosen["pdb_key"]
                pdb_in = Path(cfg.raster.pdb_dir, f"{pdb_key}.pdb")
                copy_or_dummy_pdb(pdb_in, out_fp)


def copy_or_dummy_pdb(src: Path, dst: Path) -> None:
    if not src.exists():
        create_dummy_pdb(dst)
        return
    with open(src, "r") as fin, open(dst, "w") as fout:
        fout.write(fin.read())


def create_dummy_pdb(out_fp: Path) -> None:
    with open(out_fp, "w") as f:
        f.write("REMARK  Dummy PDB\nEND\n")


if __name__ == "__main__":
    main()
