
import glob
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import transform_sidechain_frame


@hydra.main(config_path="../../configs/eval/plots", config_name="plot_inverse_fold_benchmarking", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Plot self-consistency metrics from pickle files obtained from inverse_fold_benchmarking.py
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Create out dirs and preserve config
    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # Delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    assert len(cfg.line_plots.input_pkl_files) == len(cfg.line_plots.model_names), (
            "Length of input_pkl_files must match length of model_names."
    )

    # Extract metrics from pickle
    model_dfs = {model_name: {} for model_name in cfg.line_plots.model_names}
    for model_name, pkl_file in zip(cfg.line_plots.model_names, cfg.line_plots.input_pkl_files):
        with open(pkl_file, "rb") as f:
            metrics = pickle.load(f)
        model_dfs[model_name]["pdb_name"] = [Path(x).stem for x in metrics.keys()]
        model_dfs[model_name]["length"] = [len(metrics[x]["sc_info"]["struct_preds"]["seq_mask"].squeeze()) for x in metrics.keys()]
        model_dfs[model_name]["sc_ca_rmsd"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_ca_rmsd"].item() for x in metrics.keys()]
        model_dfs[model_name]["sc_tm"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_ca_tm"].item() for x in metrics.keys()]
        model_dfs[model_name]["avg_ca_plddt"] = [metrics[x]["sc_info"]["struct_preds"]["avg_ca_plddt"].item() for x in metrics.keys()]

    model_dfs = {model_name: pd.DataFrame(model_dfs[model_name]) for model_name in cfg.line_plots.model_names}

    # Combine dataframes
    df_list = []
    for model_name, df in model_dfs.items():
        temp = df.copy()
        temp["model"] = model_name
        df_list.append(temp)

    combined_df = pd.concat(df_list, ignore_index=True)

    figsize = (4, 3)
    # colors = ["#CDCDCD", "#8C8C8C", "#5BA4CF", "#22688F", "#0D3D56"]
    # colors = ["#000000", "#CDCDCD", "#8C8C8C", "#5BA4CF", "#22688F"]
    colors = ["#005B96", "#CCCCCC", "#999999", "#555555", "#222222"]  # increasing darkness
    # colors = ["#0D3D56", "#22688F", "#5BA4CF", "#8C8C8C", "#CDCDCD"]

    # median sc_ca_rmsd line plot
    grouped = combined_df.groupby(["model", "length"], as_index=False)["sc_ca_rmsd"].median().rename(columns={"sc_ca_rmsd": "median_val"})
    grouped = grouped.sort_values(["model", "length"])

    plt.figure(figsize=figsize)
    for i, model_name in enumerate(cfg.line_plots.model_names):
        sub_df = grouped[grouped["model"] == model_name]
        plt.plot(sub_df["length"], sub_df["median_val"], label=model_name,
                 marker='o',markersize=4,
                 color=colors[i % len(colors)])
    plt.xlabel("Length", fontsize=12)
    plt.ylabel("Median scRMSD", fontsize=12)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xticks([100, 200, 300, 400, 500])
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/sc_ca_rmsd_med.pdf", dpi=300, transparent=True, bbox_inches="tight")
    plt.savefig(f"{cfg.out_dir}/sc_ca_rmsd_med.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    # median sc_tm line plot
    grouped = combined_df.groupby(["model", "length"], as_index=False)["sc_tm"].median().rename(columns={"sc_tm": "median_val"})
    grouped = grouped.sort_values(["model", "length"])

    plt.figure(figsize=figsize)
    for i, model_name in enumerate(cfg.line_plots.model_names):
        sub_df = grouped[grouped["model"] == model_name]
        plt.plot(sub_df["length"], sub_df["median_val"], label=model_name,
                 marker='o', markersize=3,
                 color=colors[i % len(colors)], alpha=0.8,)
    plt.xlabel("Length", fontsize=12)
    plt.ylabel("Median scTM", fontsize=12)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xticks([100, 200, 300, 400, 500])
    plt.legend(loc="best", )
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/sc_ca_tm_med.pdf", dpi=300, transparent=True, bbox_inches="tight")
    plt.savefig(f"{cfg.out_dir}/sc_ca_tm_med.png", dpi=300, bbox_inches="tight")
    plt.close("all")


    # Create and print LaTeX table
    table_str = create_latex_table(combined_df, cfg.line_plots.model_names)
    print(table_str)


    # #################################### PSCE sweep over validation set ####################################
    # model_dfs = {model_name: {} for model_name in cfg.psce_sweep.model_names}
    # for model_name, pkl_file in zip(cfg.psce_sweep.model_names, cfg.psce_sweep.input_pkl_files):
    #     with open(pkl_file, "rb") as f:
    #         metrics = pickle.load(f)

    #     # Extract metrics
    #     model_dfs[model_name]["pdb_name"] = [Path(x).stem for x in metrics.keys()]
    #     model_dfs[model_name]["length"] = [len(metrics[x]["sc_info"]["struct_preds"]["seq_mask"].squeeze()) for x in metrics.keys()]
    #     model_dfs[model_name]["sc_ca_rmsd"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_ca_rmsd"].item() for x in metrics.keys()]
    #     model_dfs[model_name]["sc_tm"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_ca_tm"].item() for x in metrics.keys()]

    # # Mapping from model name to dataframe
    # model_dfs = {model_name: pd.DataFrame(model_dfs[model_name]) for model_name in cfg.psce_sweep.model_names}

    # for x, df in model_dfs.items():
    #     print(f"mean sc ca rmsd for {x}: {df['sc_ca_rmsd'].mean()}")

    # for x, df in model_dfs.items():
    #     print(f"mean sc ca tm for {x}: {df['sc_tm'].mean()}")

    # for x, df in model_dfs.items():
    #     print(f"median sc ca rmsd for {x}: {df['sc_ca_rmsd'].median()}")

    # for x, df in model_dfs.items():
    #     print(f"median sc ca tm for {x}: {df['sc_tm'].median()}")

    # rmsd_threshold = 3
    # for x, df in model_dfs.items():
    #     print(f"number of structures with sc ca rmsd < {rmsd_threshold} for {x}: {len(df[df['sc_ca_rmsd'] < rmsd_threshold])}")

    # print("TEST")


    #################################### Iterative sampling plots ####################################
    assert len(cfg.iterative_sampling_plots.input_pkl_files) == len(cfg.iterative_sampling_plots.model_names), (
        "Length of iterative_sampling_plots input_pkl_files must match length of model_names."
    )

    # Extract metrics from pickle files
    iter_sampling_dfs = {model_name: {} for model_name in cfg.iterative_sampling_plots.model_names}
    for model_name, pkl_file in zip(cfg.iterative_sampling_plots.model_names, cfg.iterative_sampling_plots.input_pkl_files):
        with open(pkl_file, "rb") as f:
            metrics = pickle.load(f)

        # Extract metrics
        iter_sampling_dfs[model_name]["pdb_name"] = [Path(x).stem for x in metrics.keys()]
        iter_sampling_dfs[model_name]["length"] = [
            len(metrics[x]["sc_info"]["struct_preds"]["seq_mask"].squeeze()) for x in metrics.keys()
        ]
        iter_sampling_dfs[model_name]["sc_ca_rmsd"] = [
            metrics[x]["sc_info"]["sc_metrics"]["sc_ca_rmsd"].item() for x in metrics.keys()
        ]
        iter_sampling_dfs[model_name]["sc_tm"] = [
            metrics[x]["sc_info"]["sc_metrics"]["sc_ca_tm"].item() for x in metrics.keys()
        ]

    # Convert to DataFrame
    iter_sampling_dfs = {model_name: pd.DataFrame(iter_sampling_dfs[model_name]) for model_name in cfg.iterative_sampling_plots.model_names}

    # different shades of blue, keeping the middle color (#005B96) the same,
    # but making lighter blues lighter and darker blues darker for more contrast
    colors = ["#A9D6F3", "#66ADD9", "#005B96", "#00397A", "#001E47"]
    figsize = (4, 3)

    # Extract the names in the original order
    model_order = cfg.iterative_sampling_plots.model_names

    # 1) Median sc_ca_rmsd bar plot
    sc_ca_rmsd_medians = [
        iter_sampling_dfs[m]["sc_ca_rmsd"].median() for m in model_order
    ]

    plt.figure(figsize=figsize)
    plt.bar(
        range(len(model_order)),
        sc_ca_rmsd_medians,
        color=[colors[i % len(colors)] for i in range(len(model_order))],
        alpha=0.8,
        edgecolor="black",
        zorder=2
    )
    plt.ylabel("Median scRMSD", fontsize=12)
    plt.xticks(range(len(model_order)), model_order, rotation=45)
    plt.yticks(range(7))
    plt.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)
    plt.ylim(0, 6.5)
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/iter_sampling_sc_ca_rmsd_median.pdf", dpi=300, transparent=True, bbox_inches="tight")
    plt.savefig(f"{cfg.out_dir}/iter_sampling_sc_ca_rmsd_median.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    # 3) Median sc_tm bar plot
    sc_tm_medians = [
        iter_sampling_dfs[m]["sc_tm"].median() for m in model_order
    ]

    plt.figure(figsize=figsize)
    plt.bar(
        range(len(model_order)),
        sc_tm_medians,
        color=[colors[i % len(colors)] for i in range(len(model_order))],
        alpha=0.8,
        edgecolor="black",
        zorder=2
    )
    plt.ylabel("Median scTM", fontsize=12)
    plt.xticks(range(len(model_order)), model_order, rotation=45)
    plt.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/iter_sampling_sc_tm_median.pdf", dpi=300, transparent=True, bbox_inches="tight")
    plt.savefig(f"{cfg.out_dir}/iter_sampling_sc_tm_median.png", dpi=300, bbox_inches="tight")
    plt.close("all")


    # Confidence plots
    confidence_df = {}
    with open(cfg.confidence_plots.input_pkl_file, "rb") as f:
        metrics = pickle.load(f)

    confidence_df["pdb_name"] = [Path(x).stem for x in metrics.keys()]
    confidence_df["length"] = [len(metrics[x]["sc_info"]["struct_preds"]["seq_mask"].squeeze()) for x in metrics.keys()]
    confidence_df["sc_ca_rmsd"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_ca_rmsd"].item() for x in metrics.keys()]
    confidence_df["sc_aa_rmsd"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_aa_rmsd"].item() for x in metrics.keys()]
    confidence_df["diff"] = np.array(confidence_df["sc_aa_rmsd"]) - np.array(confidence_df["sc_ca_rmsd"])
    confidence_df["sc_tm"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_ca_tm"].item() for x in metrics.keys()]
    confidence_df = pd.DataFrame(confidence_df)
    confidence_df = confidence_df.set_index("pdb_name")

    # structure info
    atom_masks = {Path(x).stem: metrics[x]["sc_info"]["struct_preds"]["atom_mask"][0] for x in metrics.keys()}
    pred_coords = {Path(x).stem: metrics[x]["sc_info"]["struct_preds"]["pred_coords"][0] for x in metrics.keys()}
    sample_coords = {}

    psce_per_res_dict = {}
    for pkl_file in glob.glob(f"{cfg.confidence_plots.sample_pkl_dir}/*.pkl"):
        pdb_name = Path(pkl_file).stem.replace("_sample0", "")
        with open(pkl_file, "rb") as f:
            sample_info = pickle.load(f)

        # Sequence confidence: get perplexities
        aatype = sample_info["pred_aatype"]
        N = aatype.shape[0]
        probs = sample_info["seq_probs"][np.arange(N), aatype]  # select probs for predicted aatype
        log_probs = np.log(probs)
        ppl = np.exp(-np.mean(log_probs))
        confidence_df.loc[pdb_name, "ppl"] = ppl

        # Sidechain confidence: get average pSCE
        atom_mask = rc.STANDARD_ATOM_MASK_WITH_X[aatype][:, rc.non_bb_idxs]  # assume all atom types for a given aatype are present
        avg_psce = (sample_info["psce"] * atom_mask).sum(axis=-1) / atom_mask.sum(axis=-1)
        psce_per_res_dict[pdb_name] = avg_psce
        confidence_df.loc[pdb_name, "avg_psce"] = np.nanmean(avg_psce)  # nanmean to ignore glycines
        confidence_df.loc[pdb_name, "max_avg_psce"] = np.nanmax(avg_psce)

        if pdb_name == "L200_26":
            x = sample_info["x_denoised"]
            y = sample_info["pred_aatype"]

        # Store sample coords
        sample_coords[pdb_name] = torch.tensor(sample_info["x_denoised"])

    # Align each sidechain in its local frame and compute aligned sidechain RMSD
    aligned_scn_rmsds = {}
    aligned_scn_rmsds_per_res = {}
    for pdb_name in sample_coords.keys():
        atom_mask_scn, atom_mask_bb = atom_masks[pdb_name][:, rc.non_bb_idxs], atom_masks[pdb_name][:, rc.bb_idxs]

        # Get predicted sidechains in local frame
        pred = pred_coords[pdb_name]
        pred_scn, pred_bb = pred[None, :, rc.non_bb_idxs], pred[None, :, rc.bb_idxs]

        pred_scn_local, _ = transform_sidechain_frame(pred_scn, pred_bb, atom_mask_scn, atom_mask_bb, to_local=True)

        # Get sample sidechains in local frame
        sample = sample_coords[pdb_name]
        sample_scn, sample_bb = sample[None, :, rc.non_bb_idxs], sample[None, :, rc.bb_idxs]
        sample_scn_local, _ = transform_sidechain_frame(sample_scn, sample_bb, atom_mask_scn, atom_mask_bb, to_local=True)

        # Get aligned RMSD
        pred_scn_local, sample_scn_local = pred_scn_local.squeeze(0), sample_scn_local.squeeze(0)
        aligned_scn_rmsds_per_res[pdb_name] = ((atom_mask_scn[..., None] * (pred_scn_local - sample_scn_local) ** 2).sum(dim=(-1, -2)) / atom_mask_scn.sum(dim=-1).clamp(min=1)).sqrt()
        aligned_scn_rmsds[pdb_name] = aligned_scn_rmsds_per_res[pdb_name].mean().item()

    confidence_df["aligned_scn_rmsd"] = [aligned_scn_rmsds[x] for x in confidence_df.index]
    #################################### Plot aligned sidechain RMSD vs. average PSCE ####################################
    subset_df = confidence_df[confidence_df["sc_ca_rmsd"] < 5.0]
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(subset_df["aligned_scn_rmsd"], subset_df["avg_psce"],
                     c=subset_df["sc_ca_rmsd"], cmap="viridis", s=10, alpha=0.8)

    # Spearman correlation
    spearman_corr_res, _ = spearmanr(subset_df["aligned_scn_rmsd"], subset_df["avg_psce"])
    plt.text(
        0.05, 0.95,
        r"Spearman $\rho$: {0:.3f}".format(spearman_corr_res),
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        color="black"
    )
    max_val = float(max(subset_df["aligned_scn_rmsd"].max().item(), subset_df["avg_psce"].max().item()))
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5)
    plt.xlabel("Aligned sidechain RMSD ($\\mathrm{\\AA}$)", fontsize=12)
    plt.ylabel("Predicted sidechain error ($\\mathrm{\\AA}$)", fontsize=12)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    plt.ylim(0, 1.2)
    plt.xlim(0, 1.2)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="scRMSD")
    plt.savefig(f"{cfg.out_dir}/aligned_scn_rmsd_vs_avg_psce_cutoff_5A.pdf", dpi=300)
    plt.savefig(f"{cfg.out_dir}/aligned_scn_rmsd_vs_avg_psce_cutoff_5A.png", dpi=300)
    plt.close()


    # Highlight L200_24 as an orange star (if it exists in the subset)
    subset_df = confidence_df[confidence_df["sc_ca_rmsd"] < 5.0]
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(subset_df["aligned_scn_rmsd"], subset_df["avg_psce"],
                     c=subset_df["sc_ca_rmsd"], cmap="viridis", s=10, alpha=0.8)

    if "L200_24" in subset_df.index:
        plt.scatter(
            subset_df.loc["L200_24", "aligned_scn_rmsd"],
            subset_df.loc["L200_24", "avg_psce"],
            marker="*",
            color="orange",
            s=100
        )

    # Spearman correlation
    spearman_corr_res, _ = spearmanr(subset_df["aligned_scn_rmsd"], subset_df["avg_psce"])
    plt.text(
        0.05, 0.95,
        r"Spearman $\rho$: {0:.3f}".format(spearman_corr_res),
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        color="black"
    )
    max_val = float(max(subset_df["aligned_scn_rmsd"].max().item(), subset_df["avg_psce"].max().item()))
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5)
    plt.xlabel("Aligned sidechain RMSD ($\\mathrm{\\AA}$)", fontsize=12)
    plt.ylabel("Predicted sidechain error ($\\mathrm{\\AA}$)", fontsize=12)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    plt.ylim(0, 1.2)
    plt.xlim(0, 1.2)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="scRMSD")
    plt.savefig(f"{cfg.out_dir}/aligned_scn_rmsd_vs_avg_psce_cutoff_5A_star.pdf", dpi=300)
    plt.savefig(f"{cfg.out_dir}/aligned_scn_rmsd_vs_avg_psce_cutoff_5A_star.png", dpi=300)
    plt.close()



    #################################### Partial sequence packing plots ####################################
    partial_seq_df = {}
    for input_csv in cfg.partial_seq_packing_plots.input_seq_csvs:
        tseq_val = float(Path(input_csv).parent.name.split("_")[-1])
        partial_seq_df[tseq_val] = pd.read_csv(input_csv).loc[0, "scn_rmsd_avg_all"]
    partial_seq_df = pd.DataFrame(partial_seq_df.items(), columns=["tseq", "scn_rmsd_avg_all"])

    partial_scn_df = {}
    no_context = partial_scn_df[0.0] = partial_seq_df[partial_seq_df["tseq"] == 1.0].iloc[0, 1]
    for input_csv in cfg.partial_seq_packing_plots.input_scn_csvs:
        tscn_val = float(Path(input_csv).parent.name.split("_")[-1])
        partial_scn_df[tscn_val] = pd.read_csv(input_csv).loc[0, "scn_rmsd_avg_all"]

    partial_scn_df = pd.DataFrame(partial_scn_df.items(), columns=["tscn", "scn_rmsd_avg_all"])

    plt.figure(figsize=(6, 4))
    plt.axhline(no_context, color="gray", linestyle="--", label="Full sequence context", linewidth=1.0)
    plt.plot(partial_seq_df["tseq"] * 100, partial_seq_df["scn_rmsd_avg_all"], marker="o", color="black", linewidth=2, label="Partial sequence context")
    plt.plot(partial_scn_df["tscn"] * 100, partial_scn_df["scn_rmsd_avg_all"], marker="o", color="#1f77b4", linewidth=2, label="Partial sidechain context")
    plt.legend()
    plt.xlabel("Partial context given (%)", fontsize=12)
    plt.ylabel("Average packing RMSD\n" r"over known sequence ($\AA$)", fontsize=12)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
    plt.xticks(np.arange(0, 101, 10))
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/packing_vs_partial_seq.pdf", dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(f"{cfg.out_dir}/packing_vs_partial_seq.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    print("DONE")



def create_box_and_whisker_plots(model_dfs: dict, out_dir: Path):
    """
    Create box-and-whisker plots for scRMSD (sc_ca_rmsd) and scTM (sc_tm)
    for each unique length in the data. Saves plots to both PDF and PNG
    with 300 dpi and transparent background.

    Args:
        model_dfs (dict): Dictionary of {model_name: pd.DataFrame}, where each
                          DataFrame has columns ["pdb_name", "length", "sc_ca_rmsd", "sc_tm"].
        out_dir (Path):   Path to the output directory where plots will be saved.
    """

    # 1) Combine DataFrames into one for plotting
    combined_df = []
    for model_name, df in model_dfs.items():
        temp = df.copy()
        temp["model"] = model_name
        combined_df.append(temp)
    combined_df = pd.concat(combined_df, ignore_index=True)

    # 2) Sort lengths to ensure x-axis is in ascending order
    unique_lengths = sorted(combined_df["length"].unique())

    # 3) Helper function to generate and save the boxplot
    def _create_and_save_boxplot(metric: str, df: pd.DataFrame, out_file_prefix: str):
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=df,
            x="length",
            y=metric,
            hue="model",
            order=unique_lengths,     # ensures discrete lengths on the x-axis in ascending order
            showfliers=False,        # whether to show outliers
        )
        plt.title(f"Box and Whisker Plot of {metric} by Length")
        plt.xlabel("Length")
        plt.ylabel(metric)
        plt.legend(title="Model", loc="best")

        # Save to PDF
        pdf_path = out_dir / f"{out_file_prefix}_{metric}.pdf"
        plt.savefig(pdf_path, dpi=300, transparent=True, bbox_inches="tight")

        # Also save to PNG
        png_path = out_dir / f"{out_file_prefix}_{metric}.png"
        plt.savefig(png_path, dpi=300, transparent=True, bbox_inches="tight")
        plt.close()

    # 4) Generate boxplots for sc_ca_rmsd and sc_tm
    _create_and_save_boxplot("sc_ca_rmsd", combined_df, out_file_prefix="boxplot")
    _create_and_save_boxplot("sc_tm", combined_df, out_file_prefix="boxplot")


def plot_sc_medians_line(
    model_dfs,
    metric="sc_ca_rmsd",
    x_label="Length",
    y_label="Median Value",
    x_ticks=None,
    save_file=None
):
    """
    Plots a line with dot markers for the median of `metric` (e.g. sc_ca_rmsd or sc_tm)
    grouped by [model_name, length]. Mimics the style of the read_and_plot() function
    you used previously.

    Args:
        model_dfs (dict):
            Dictionary of {model_name: pd.DataFrame},
            where each DataFrame has at least ['pdb_name', 'length', 'sc_ca_rmsd', 'sc_tm'].
        metric (str): The column name to plot (e.g. "sc_ca_rmsd" or "sc_tm").
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        x_ticks (list): Optional list of x-ticks to display.
        save_file (str): If provided, the plot is saved to this filename (PDF and PNG).
    """

    # 1) Combine data from all models into a single DataFrame
    df_list = []
    for model_name, df in model_dfs.items():
        temp = df.copy()
        temp["model"] = model_name
        df_list.append(temp)
    combined_df = pd.concat(df_list, ignore_index=True)

    # 2) Group by (model, length) and compute the median for the specified metric
    grouped = (
        combined_df
        .groupby(["model", "length"], as_index=False)[metric]
        .median()
        .rename(columns={metric: "median_val"})
    )

    # 3) Sort by length so the line plots go in ascending length order
    grouped = grouped.sort_values(["model", "length"])

    # 4) Create the plot
    plt.figure(figsize=(8, 6))

    # Just use the model names as labels
    model_map = {m: m for m in grouped["model"].unique()}

    # Plot each model's median line
    for model_name in model_map.keys():
        sub_df = grouped[grouped["model"] == model_name]
        label = model_map.get(model_name, model_name)
        plt.plot(
            sub_df["length"],    # x-values
            sub_df["median_val"],# y-values
            label=label,
            marker='o',
            markersize=4
        )

    # 5) Format axes, labels, and grid
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5)  # lighter grid

    # Optional x-ticks
    if x_ticks is not None:
        plt.xticks(x_ticks)

    # 6) Add legend
    plt.legend(loc="best", fontsize=9)

    # 7) Save (PDF + PNG) if requested
    if save_file:
        outpath = Path(save_file)
        plt.savefig(outpath, dpi=300, transparent=True, bbox_inches="tight")
        # Also save as PNG
        png_path = outpath.with_suffix(".png")
        plt.savefig(png_path, dpi=300, transparent=True, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_rmsd_threshold_vs_success(
    model_dfs,
    length_filter: Optional[int] = 500,
    metric="sc_ca_rmsd",
    x_label="scRMSD Threshold",
    y_label="Number of Samples Below Threshold",
    n_thresholds=50,
    threshold_min=None,
    threshold_max=None,
    save_file=None
):
    """
    Plots the success rate of each model at different scRMSD thresholds.
    """
    # Combine all model data
    combined_list = []
    for model_name, df in model_dfs.items():
        temp = df.copy()
        temp["model"] = model_name
        combined_list.append(temp)
    combined_df = pd.concat(combined_list, ignore_index=True)

    # Filter to only rows where length == length_filter
    if length_filter is not None:
        filtered_df = combined_df[combined_df["length"] == length_filter].copy()
        if filtered_df.empty:
            print(f"No samples found for length={length_filter}. Nothing to plot.")
            return
    else:
        filtered_df = combined_df.copy()

    # Determine the threshold range
    data_min = filtered_df[metric].min()
    data_max = filtered_df[metric].max()

    if threshold_min is None:
        threshold_min = data_min
    if threshold_max is None:
        threshold_max = data_max

    threshold_min = min(threshold_min, threshold_max)
    threshold_max = max(threshold_min, threshold_max)
    thresholds = np.linspace(threshold_min, threshold_max, n_thresholds)

    # Handle colormap
    # colors = ["gold", "yellow", "green", "cyan", "slateblue", "violet"]
    # cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", colors, N=256)
    cmap = mcolors.ListedColormap(plt.cm.viridis(np.linspace(0,1,256)))

    # Plot
    plt.figure(figsize=(12, 8))

    for model_idx, model_name in enumerate(model_dfs.keys()):
        # Get the corresponding color from the custom colormap
        curve_color = cmap(model_idx / len(model_dfs))

        # Compute how many samples fall below each threshold
        sub_df = filtered_df[filtered_df["model"] == model_name]
        sub_rmsds = sub_df[metric].values
        y_values = []
        for thr in thresholds:
            pct_below = (sub_rmsds <= thr).sum() / len(sub_rmsds) * 100
            y_values.append(pct_below)

        # Plot the curve
        plt.plot(thresholds, y_values, label=model_name, markersize=3, color=curve_color)

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.legend(loc="best", fontsize=9)

    # Save if requested
    if save_file:
        outpath = Path(save_file)
        plt.savefig(outpath, dpi=300, transparent=True, bbox_inches="tight")
        plt.savefig(outpath.with_suffix(".png"), dpi=300, transparent=True, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_sc_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,                 # e.g. "avg_log_prob", "avg_psce", or "avg_seq_entropy"
    hue_col: Optional[str] = None,    # Column name for color-coding points
    length_filter: Optional[int] = None,
    cutoff: Optional[float] = None,
    x_label: str = "scRMSD",
    y_label: Optional[str] = None,
    save_file: Optional[str] = None
):
    """
    Scatter plot of scRMSD vs. an arbitrary y_col, e.g. avg_log_prob or avg_psce.
    Optionally filters rows by:
      - length_filter: Only includes rows where df['length'] == length_filter (if provided).
      - cutoff:        Only includes rows where df['sc_ca_rmsd'] <= cutoff (if provided).

    Title contains Spearman correlation coefficient (rho) and p-value.

    Args:
        df (pd.DataFrame): Must have columns ['sc_ca_rmsd', y_col, 'length'] as needed.
        y_col (str): Name of the column on the y-axis (e.g., "avg_log_prob", "avg_psce").
        length_filter (Optional[int]): If provided, only plot rows where df['length'] == this value.
        cutoff (Optional[float]): If provided, only plot rows where df['sc_ca_rmsd'] <= cutoff.
        x_label (str): Label for the x-axis. Defaults to "scRMSD".
        y_label (Optional[str]): Label for the y-axis. If None, defaults to y_col.
        save_file (Optional[str]): If provided, saves plot (PDF & PNG) to this path.
    """
    # 1) Filter the DataFrame
    filtered = df.copy()
    if length_filter is not None:
        filtered = filtered[filtered["length"] == length_filter]

    if cutoff is not None:
        filtered = filtered[filtered[x_col] <= cutoff]

    if filtered.empty:
        print(
            f"No data available for length_filter={length_filter} "
            f"and cutoff={cutoff} on {y_col}."
        )
        return

    # 2) Create the scatter plot
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=filtered,
        x=x_col,
        y=y_col,
        hue=hue_col,
        alpha=0.6
    )

    # 3) Calculate Spearman correlation
    rho, pval = spearmanr(filtered[x_col], filtered[y_col])

    # 4) Set the plot title, x-/y-axis labels, grid
    y_axis_label = y_label if y_label else y_col
    plt.title(f"{x_label} vs {y_axis_label} (Spearmanr={rho:.3f}, p={pval:.1e})")
    plt.xlabel(x_label)
    plt.ylabel(y_axis_label)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

    # 5) Save plot if desired
    if save_file:
        outpath = Path(save_file)
        plt.savefig(outpath, dpi=300, transparent=True, bbox_inches="tight")
        plt.savefig(outpath.with_suffix(".png"), dpi=300, transparent=True, bbox_inches="tight")
    plt.close()



def plot_fraction_below_threshold(
    model_dfs,
    thresholds=[1.0, 2.0, 3.0],
    length_filter: Optional[int] = 500,
    metric="sc_ca_rmsd",
    x_label="Step Count",
    y_label="Fraction Below Threshold",
    save_file=None
):
    """
    Plots fraction of samples below each threshold for different step counts.
    Assumes model names contain '(X steps)', e.g. 'FAMPNN (10 steps)'.

    Args:
        model_dfs (dict):
            {model_name: pd.DataFrame}, each DF has columns like:
            ["pdb_name", "length", "sc_ca_rmsd", "sc_tm"].
        thresholds (list[float]):
            One or more thresholds to show. Each threshold results in one curve.
        length_filter (int):
            Only include rows where 'length' == length_filter.
        metric (str):
            Column name to analyze, e.g. "sc_ca_rmsd".
        x_label (str):
            X-axis label. Defaults to "Step Count".
        y_label (str):
            Y-axis label. Defaults to "Fraction Below Threshold".
        save_file (str or Path):
            If provided, saves the figure as both PDF and PNG.
    """
    # --------------------------
    # 1. Aggregate data across all models
    # --------------------------
    import pandas as pd

    df_list = []
    for model_name, df in model_dfs.items():
        temp = df.copy()
        temp["model"] = model_name
        df_list.append(temp)
    combined_df = pd.concat(df_list, ignore_index=True)

    # Filter by length
    if length_filter is not None:
        combined_df = combined_df[combined_df["length"] == length_filter]

    if combined_df.empty:
        print(f"No samples found for length={length_filter}. Nothing to plot.")
        return

    # --------------------------
    # 2. Parse step count from model name, group data by step count
    # --------------------------
    # Example model name: "FAMPNN (10 steps)"
    # We'll store (step_count -> list of sc_ca_rmsd values)
    step_dict = {}  # {step_count (int): [rmsd_vals]}

    for model_name in combined_df["model"].unique():
        # Get rows matching this model name
        sub_df = combined_df[combined_df["model"] == model_name]

        # Attempt to parse steps from model_name
        match = re.search(r"^(\d+)\s*steps?$", model_name)
        if match:
            steps = int(match.group(1))
        else:
            # If no match, default to 0
            steps = 0

        # Collect metric values
        step_dict.setdefault(steps, []).extend(sub_df[metric].values.tolist())

    # --------------------------
    # 3. For each threshold, compute fraction of samples below threshold
    # --------------------------
    # We'll have one line (x=step_count, y=fraction) per threshold
    # Sort step_dict keys so we plot in ascending order of steps
    sorted_steps = sorted(step_dict.keys())

    # fraction_data[threshold] = list of fraction_below across step counts
    fraction_data = {thr: [] for thr in thresholds}

    for step_count in sorted_steps:
        vals = np.array(step_dict[step_count])
        n_total = len(vals)

        for thr in thresholds:
            frac_below = (vals < thr).sum() / n_total if n_total > 0 else 0.0
            fraction_data[thr].append(frac_below)

    # --------------------------
    # 4. Plot results
    # --------------------------
    plt.figure(figsize=(5, 4))

    for thr in thresholds:
        plt.plot(
            sorted_steps,
            fraction_data[thr],
            marker='o',
            linewidth=2,
            label=f"Threshold = {thr} Å"
        )

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xscale("log")
    plt.title(f"Fraction Below Threshold vs Steps (L={length_filter})", fontsize=12)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.8)
    plt.legend(loc="best", fontsize=10)

    # Save figure if requested
    if save_file:
        out_path = Path(save_file)
        plt.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight")
        plt.savefig(out_path.with_suffix(".png"), dpi=300, transparent=True, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_success_rate_by_confidence(
    df: pd.DataFrame,
    conf_col: str,
    sc_rmsd_col: str = "sc_ca_rmsd",
    sc_rmsd_threshold: float = 2.0,
    length_filter: Optional[int] = None,
    bigger_is_better: bool = False,
    n_points: int = 10,
    x_label: str = "Top X% (Confidence Rank)",
    y_label: str = "Success Rate (%)",
    save_file: Optional[str] = None
):
    """
    Plots success rate vs. top fraction/percentile of data sorted by `conf_col`.

    Success is defined as df[sc_rmsd_col] <= sc_rmsd_threshold.

    E.g., if bigger_is_better=True, then "top 10%" means the 10% with the largest conf_col values.
          if bigger_is_better=False, then "top 10%" means the 10% with the smallest conf_col values.

    Args:
        df (pd.DataFrame): Must have columns [conf_col, sc_rmsd_col, 'length'] if length_filter is used.
        conf_col (str): The column name of the confidence measure (e.g., 'ppl' or 'avg_psce').
        sc_rmsd_col (str): The column name for scRMSD. Defaults to "sc_ca_rmsd".
        sc_rmsd_threshold (float): scRMSD <= this value is considered a "success".
        length_filter (Optional[int]): If provided, only use rows where df['length'] == length_filter.
        bigger_is_better (bool): If True, sort by conf_col descending; else ascending.
        n_points (int): Number of percentile points to evaluate (e.g. 10 => [10%, 20%, ... 100%]).
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        save_file (Optional[str]): If provided, saves the plot (PDF & PNG) to this path.
    """
    # 1) Filter data by length if requested
    data = df.copy()
    if length_filter is not None:
        data = data[data["length"] == length_filter]

    if data.empty:
        print(f"No data after applying length_filter={length_filter}.")
        return

    # 2) Determine which entries are successes
    data["success"] = (data[sc_rmsd_col] <= sc_rmsd_threshold)

    # 3) Sort data by confidence measure
    #    If bigger_is_better, we sort descending. Otherwise ascending.
    data = data.sort_values(conf_col, ascending=not bigger_is_better).reset_index(drop=True)

    # 4) We'll evaluate top X% (X in {1,2,...,100} or spaced by n_points)
    #    For example, if n_points=10, we check [10%, 20%, 30%, ..., 100%]
    percentiles = np.linspace(1, 100, n_points, dtype=int)

    x_vals = []
    y_vals = []
    N = len(data)

    for p in percentiles:
        top_count = int(np.ceil(N * (p / 100.0)))
        subset = data.iloc[:top_count]  # top X% by conf_col
        success_rate = 100.0 * subset["success"].mean()  # success rate in %
        x_vals.append(p)
        y_vals.append(success_rate)

    # 5) Create the plot
    plt.figure(figsize=(6, 5))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

    plt.title(
        f"Success vs. {conf_col}\n"
        f"scRMSD <= {sc_rmsd_threshold}, length={length_filter if length_filter else 'ALL'}"
    )

    # 6) Save if requested
    if save_file:
        outpath = Path(save_file)
        plt.savefig(outpath, dpi=300, transparent=True, bbox_inches="tight")
        plt.savefig(outpath.with_suffix(".png"), dpi=300, transparent=True, bbox_inches="tight")
    plt.close()



def create_latex_table(df: pd.DataFrame, model_names: list[str]) -> str:
    # Group data and compute medians
    grouped = df.groupby(["model", "length"], as_index=False)[["sc_tm", "avg_ca_plddt"]].median()
    # grouped = df.groupby(["model", "length"], as_index=False)[["sc_tm", "sc_ca_rmsd", "avg_ca_plddt"]].median()


    # Define lengths to display
    lengths = [100, 200, 300, 400, 500]

    # Start building the LaTeX table string
    # (You can adjust column formatting or alignment as needed.)
    latex = []
    # latex.append("\\begin{tabular}{l" + " ccc"*len(lengths) + "}")
    latex.append("\\begin{tabular}{l" + " cc"*len(lengths) + "}")
    latex.append("\\toprule")
    header_line_1 = ["Method"]
    header_line_2 = [""]
    for length in lengths:
        # header_line_1.append(f"\\multicolumn{{3}}{{c}}{{Length {length}}}")
        # header_line_2.extend(["scTM$\\uparrow$", "scRMSD$\\downarrow$", "pLDDT$\\uparrow$"])
        header_line_1.append(f"\\multicolumn{{2}}{{c}}{{Length {length}}}")
        header_line_2.extend(["scTM$\\uparrow$", "pLDDT$\\uparrow$"])
    latex.append(" & ".join(header_line_1) + " \\\\")
    latex.append(" & ".join(header_line_2) + " \\\\")
    latex.append("\\midrule")

    for model in model_names:
        row_str = [model]
        for length in lengths:
            sub_df = grouped[(grouped["model"] == model) & (grouped["length"] == length)]
            if len(sub_df) == 0:
                # if there's no entry for that length, fill with dashes
                row_str.extend(["-", "-", "-"])
                continue
            sc_tm_val = sub_df["sc_tm"].values[0]
            # sc_rmsd_val = sub_df["sc_ca_rmsd"].values[0]
            plddt_val = sub_df["avg_ca_plddt"].values[0]
            row_str.append(f"{sc_tm_val:.3f}")
            # row_str.append(f"{sc_rmsd_val:.3f}")
            row_str.append(f"{plddt_val:.2f}")
        latex.append(" & ".join(row_str) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    return "\n".join(latex)


if __name__ == "__main__":
    main()