
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import numpy as np


@hydra.main(config_path="../configs/eval", config_name="plot_inverse_fold_benchmarking", version_base="1.3.2")
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

    assert len(cfg.input_pkl_files) == len(cfg.model_names), (
            "Length of input_pkl_files must match length of model_names."
    )

    # Extract metrics from pickle
    model_dfs = {model_name: {} for model_name in cfg.model_names}
    for model_name, pkl_file in zip(cfg.model_names, cfg.input_pkl_files):
        with open(pkl_file, "rb") as f:
            metrics = pickle.load(f)

        # Extract metrics
        model_dfs[model_name]["pdb_name"] = [Path(x).stem for x in metrics.keys()]
        model_dfs[model_name]["length"] = [len(metrics[x]["sc_info"]["struct_preds"]["seq_mask"].squeeze()) for x in metrics.keys()]
        model_dfs[model_name]["sc_ca_rmsd"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_ca_rmsd"].item() for x in metrics.keys()]
        model_dfs[model_name]["sc_tm"] = [metrics[x]["sc_info"]["sc_metrics"]["sc_ca_tm"].item() for x in metrics.keys()]

    # Mapping from model name to dataframe
    model_dfs = {model_name: pd.DataFrame(model_dfs[model_name]) for model_name in cfg.model_names}

    # Box and whisker plots
    create_box_and_whisker_plots(model_dfs, Path(cfg.out_dir))

    # Line plots of median
    plot_sc_medians_line(
        model_dfs,
        metric="sc_ca_rmsd",
        x_label="Length",
        y_label="Median scRMSD",
        x_ticks=[100, 200, 300, 400, 500],
        save_file=f"{cfg.out_dir}/sc_ca_rmsd_med.pdf"
    )

    plot_sc_medians_line(
        model_dfs,
        metric="sc_tm",
        x_label="Length",
        y_label="Median scTM",
        x_ticks=[100, 200, 300, 400, 500],
        save_file=f"{cfg.out_dir}/sc_ca_tm_med.pdf"
    )

    # Plot success rate vs scRMSD threshold
    for length in [400, 500]:
        plot_rmsd_threshold_vs_success(
            model_dfs=model_dfs,
            length_filter=length,
            metric="sc_ca_rmsd",
            x_label="scRMSD Threshold",
            y_label="Number of samples below threshold",
            n_thresholds=50,
            threshold_min=0,
            threshold_max=5,
            save_file=f"{cfg.out_dir}/L{length}_sc_rmsd_success_curve.pdf"
        )


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
    plt.figure(figsize=(6, 4))

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
    length_filter=500,
    metric="sc_ca_rmsd",
    x_label="scRMSD threshold",
    y_label="Designability success rate",
    n_thresholds=100,
    threshold_min=None,
    threshold_max=None,
    save_file=None
):
    """
    Plots a curve for each model:
      - x-axis: RMSD threshold
      - y-axis: number of samples (or fraction) below that threshold

    Only uses rows where length == length_filter.

    Args:
        model_dfs (dict):
            {model_name: pd.DataFrame}, each DF has columns:
            ["pdb_name", "length", "sc_ca_rmsd", "sc_tm"] at minimum.
        length_filter (int):
            Only use rows where `length == length_filter`.
            Default is 500, but can be changed if needed.
        metric (str):
            Column name for the scRMSD metric to use, e.g. "sc_ca_rmsd".
        x_label (str):
            Label for the RMSD threshold axis.
        y_label (str):
            Label for the success rate (or number of samples) axis.
        n_thresholds (int):
            How many threshold points to sample between min and max scRMSD.
        threshold_min (float or None):
            If provided, use this as the lower bound for RMSD thresholds.
            Otherwise, use the min in the data.
        threshold_max (float or None):
            If provided, use this as the upper bound for RMSD thresholds.
            Otherwise, use the max in the data.
        save_file (str or Path):
            If provided, the plot is saved (PDF & PNG).
    """
    # 1) Combine all model data
    combined_list = []
    for model_name, df in model_dfs.items():
        temp = df.copy()
        temp["model"] = model_name
        combined_list.append(temp)
    combined_df = pd.concat(combined_list, ignore_index=True)

    # 2) Filter to only rows where length == length_filter
    filtered_df = combined_df[combined_df["length"] == length_filter].copy()
    if filtered_df.empty:
        print(f"No samples found for length={length_filter}. Nothing to plot.")
        return

    # 3) Determine range of thresholds for the chosen metric
    data_min = filtered_df[metric].min()
    data_max = filtered_df[metric].max()

    if threshold_min is None:
        threshold_min = data_min
    if threshold_max is None:
        threshold_max = data_max

    # Ensure threshold_min <= threshold_max (simple safety check)
    threshold_min = min(threshold_min, threshold_max)
    threshold_max = max(threshold_min, threshold_max)

    thresholds = np.linspace(threshold_min, threshold_max, n_thresholds)

    # 4) Create the plot
    plt.figure(figsize=(6, 4))

    # For each model, compute how many samples fall below each threshold
    for model_name in filtered_df["model"].unique():
        sub_df = filtered_df[filtered_df["model"] == model_name]
        if sub_df.empty:
            continue

        sub_rmsds = sub_df[metric].values
        total_samples = len(sub_rmsds)  # needed if you want fraction
        y_values = []

        for thr in thresholds:
            count_below = (sub_rmsds <= thr).sum()

            # Option 1: Count of samples below threshold
            y_values.append(count_below)

            # Option 2 (if desired): fraction of samples below threshold
            # fraction_below = count_below / total_samples
            # y_values.append(fraction_below)

        plt.plot(
            thresholds,
            y_values,
            label=model_name,
            marker="o",
            markersize=3
        )

    # 5) Label axes, add grid
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5)

    # 6) Add legend
    plt.legend(loc="best", fontsize=9)

    # 7) Save if requested (PDF & PNG)
    if save_file:
        outpath = Path(save_file)
        plt.savefig(outpath, dpi=300, transparent=True, bbox_inches="tight")
        plt.savefig(outpath.with_suffix(".png"), dpi=300, transparent=True, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()