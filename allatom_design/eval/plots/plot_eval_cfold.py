from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colorbar import ColorbarBase
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs/eval/plots", config_name="plot_eval_cfold")
def main(cfg: DictConfig) -> None:
    """
    Iterate over the evaluation directories specified in the config, load the
    metrics for ensemble, state0, and state1, and generate scatter plots
    comparing them.
    """
    # Create the base output directory
    Path(cfg.base_out_dir).mkdir(parents=True, exist_ok=True)

    # Dump the entire config into the base output directory for reference
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.base_out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Optional: read in the manifest CSV for length data
    manifest_df = None
    if cfg.manifest_csv is not None:
        manifest_df = pd.read_csv(cfg.manifest_csv)
        # Prepare for merging by renaming key and dropping duplicates
        manifest_df = manifest_df.rename(columns={'pdb_key': 'pdb_id'})
        manifest_df = manifest_df.drop_duplicates(subset=['pdb_id'])

    # Optional: save a standalone colorbar for length
    length_legend_range = None
    if cfg.length_legend_range is not None:
        length_legend_range = cfg.length_legend_range
        save_length_colorbar(
            vmin=length_legend_range[0],
            vmax=length_legend_range[1],
            out_path=Path(cfg.base_out_dir) / "protein_length_colorbar.png",
        )

    # Iterate over all directories to be processed
    for eval_job in cfg.eval_dirs:
        eval_dir = Path(eval_job.path)
        plot_name = eval_job.plot_name

        # Create a sub-output directory for this specific evaluation
        out_dir_for_job = Path(cfg.base_out_dir) / plot_name
        out_dir_for_job.mkdir(parents=True, exist_ok=True)

        # Define paths to the metrics files
        ensemble_csv = eval_dir / "ensemble" / "self_consistency_metrics.csv"
        state0_csv = eval_dir / "state0" / "self_consistency_metrics.csv"
        state1_csv = eval_dir / "state1" / "self_consistency_metrics.csv"

        # Load the dataframes
        ensemble_df = pd.read_csv(ensemble_csv)
        state0_df = pd.read_csv(state0_csv)
        state1_df = pd.read_csv(state1_csv)

        # Prepare a common key for merging
        ensemble_df["pdb_id"] = ensemble_df["record_id"].apply(lambda x: x.split('_sample')[0])
        state0_df["pdb_id"] = state0_df["record_id"].apply(lambda x: x.split('_sample')[0])
        state1_df["pdb_id"] = state1_df["record_id"].apply(lambda x: x.split('_sample')[0])

        # Select relevant columns and merge the dataframes
        cols_to_keep = ["pdb_id", "avg_ca_plddt", "seq_recovery"]
        merged_df = pd.merge(
            ensemble_df[cols_to_keep],
            state0_df[cols_to_keep],
            on="pdb_id",
            suffixes=("_ensemble", "_state0")
        )
        final_df = pd.merge(
            merged_df,
            state1_df[cols_to_keep],
            on="pdb_id"
        ).rename(columns={
            "avg_ca_plddt": "avg_ca_plddt_state1",
            "seq_recovery": "seq_recovery_state1"
        })

        # Calculate the best sequence recovery between state0 and state1
        final_df['seq_recovery_best_state'] = final_df[
            ['seq_recovery_state0', 'seq_recovery_state1']
        ].max(axis=1)

        # If manifest is available, merge length data
        if manifest_df is not None:
            final_df = pd.merge(
                final_df,
                manifest_df[['pdb_id', 'length']],
                on='pdb_id',
                how='left'
            )

        # Create the pLDDT scatter plot
        plot_combined_scatter(
            df=final_df,
            metric="avg_ca_plddt",
            metric_title="Average C-alpha pLDDT",
            plot_name=plot_name,
            out_path=out_dir_for_job / "plddt_scatter.png"
        )

        # Create the sequence recovery scatter plot
        plot_combined_scatter(
            df=final_df,
            metric="seq_recovery",
            metric_title="Sequence Recovery",
            plot_name=plot_name,
            out_path=out_dir_for_job / "seq_recovery_scatter.png"
        )

        # Create the best-of-states sequence recovery scatter plot
        plot_best_state_scatter(
            df=final_df,
            metric="seq_recovery",
            metric_title="Sequence Recovery",
            plot_name=plot_name,
            out_path=out_dir_for_job / "seq_recovery_best_state_scatter.png",
            length_legend_range=length_legend_range
        )


def plot_combined_scatter(
    df: pd.DataFrame,
    metric: str,
    metric_title: str,
    plot_name: str,
    out_path: Path
) -> None:
    """
    Generates a scatter plot comparing an ensemble metric (y-axis) against
    the same metric from state0 and state1 (x-axis).
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    y_col = f"{metric}_ensemble"
    x_col0 = f"{metric}_state0"
    x_col1 = f"{metric}_state1"

    # Scatter plot for state0 vs ensemble (green)
    ax.scatter(
        df[x_col0],
        df[y_col],
        s=30,
        color="green",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
        label="State 0"
    )

    # Scatter plot for state1 vs ensemble (blue)
    ax.scatter(
        df[x_col1],
        df[y_col],
        s=30,
        color="blue",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
        label="State 1"
    )

    # Determine plot limits to be square and include all data
    all_vals = pd.concat([df[y_col], df[x_col0], df[x_col1]]).dropna()
    lower = all_vals.min() * 0.95
    upper = all_vals.max() * 1.05
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    # Plot diagonal y=x line
    ax.plot([lower, upper], [lower, upper], 'k--', alpha=0.8)

    # Labels, title, legend, and grid
    ax.set_xlabel(f"State 0 / State 1 {metric_title}")
    ax.set_ylabel(f"Ensemble {metric_title}")
    ax.set_title(f"{plot_name}: {metric_title}")
    ax.legend()
    plt.grid(True, alpha=0.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_best_state_scatter(
    df: pd.DataFrame,
    metric: str,
    metric_title: str,
    plot_name: str,
    out_path: Path,
    length_legend_range: list = None
) -> None:
    """
    Generates a scatter plot of an ensemble metric (y-axis) against the
    best of state0/state1 (x-axis), colored by protein length.
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    y_col = f"{metric}_ensemble"
    x_col = f"{metric}_best_state"

    x_vals = df[x_col]
    y_vals = df[y_col]

    # Handle coloring by length if available
    if 'length' in df.columns and not df['length'].isnull().all():
        length_vals = df["length"]
        vmin, vmax = (length_legend_range if length_legend_range is not None else (None, None))
        sc = ax.scatter(
            x_vals, y_vals, s=30, c=length_vals, cmap="viridis",
            alpha=0.8, edgecolor="k", linewidth=0.5, vmin=vmin, vmax=vmax
        )
    else:
        # Default plot if no length data
        ax.scatter(
            x_vals, y_vals, s=30, color="steelblue",
            alpha=0.8, edgecolor="k", linewidth=0.5
        )

    # Determine plot limits
    all_vals = pd.concat([x_vals, y_vals]).dropna()
    lower = all_vals.min() * 0.95
    upper = all_vals.max() * 1.05
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    # Plot diagonal y=x line
    ax.plot([lower, upper], [lower, upper], 'k--', alpha=0.8)

    # Labels, title, and grid
    ax.set_xlabel(f"Best of State 0/1 {metric_title}")
    ax.set_ylabel(f"Ensemble {metric_title}")
    ax.set_title(f"{plot_name}: Best-of-States {metric_title}")
    plt.grid(True, alpha=0.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_length_colorbar(vmin: float, vmax: float, out_path: Path) -> None:
    """
    Creates and saves a standalone horizontal colorbar with the label "Protein length".
    """
    fig, ax = plt.subplots(figsize=(6, 0.5))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    ColorbarBase(
        ax,
        cmap="viridis",
        norm=norm,
        orientation="horizontal",
    ).set_label("Protein length")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()