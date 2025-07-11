from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
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

        # Calculate the best and mean of state0/state1 for each metric
        for metric in ["avg_ca_plddt", "seq_recovery"]:
            state_cols = [f"{metric}_state0", f"{metric}_state1"]
            final_df[f'{metric}_best_state'] = final_df[state_cols].max(axis=1)
            final_df[f'{metric}_mean_state'] = final_df[state_cols].mean(axis=1)

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
        plot_comparison_scatter(
            df=final_df,
            metric="seq_recovery",
            metric_title="Sequence Recovery",
            comparison_type="best_state",
            x_axis_label="Best of State 0/1",
            plot_name=plot_name,
            out_path=out_dir_for_job / "seq_recovery_best_state_scatter.png",
            length_legend_range=length_legend_range
        )

        # Create the mean-of-states plots
        plot_comparison_scatter(
            df=final_df,
            metric="seq_recovery",
            metric_title="Sequence Recovery",
            comparison_type="mean_state",
            x_axis_label="Mean of State 0/1",
            plot_name=plot_name,
            out_path=out_dir_for_job / "seq_recovery_mean_state_scatter.png",
            length_legend_range=length_legend_range
        )
        plot_comparison_scatter(
            df=final_df,
            metric="avg_ca_plddt",
            metric_title="Average C-alpha pLDDT",
            comparison_type="mean_state",
            x_axis_label="Mean of State 0/1",
            plot_name=plot_name,
            out_path=out_dir_for_job / "plddt_mean_state_scatter.png",
            length_legend_range=length_legend_range
        )

        # Define paths to the af2rank metrics files
        af2rank_ensemble_csv = eval_dir / "ensemble" / "af2rank_outputs.csv"
        af2rank_state0_csv = eval_dir / "state0" / "af2rank_outputs.csv"
        af2rank_state1_csv = eval_dir / "state1" / "af2rank_outputs.csv"

        # Load the af2rank dataframes if they exist
        af2rank_ensemble_df = pd.read_csv(af2rank_ensemble_csv) if af2rank_ensemble_csv.exists() else None
        af2rank_state0_df = pd.read_csv(af2rank_state0_csv) if af2rank_state0_csv.exists() else None
        af2rank_state1_df = pd.read_csv(af2rank_state1_csv) if af2rank_state1_csv.exists() else None

        # Create the composite score plots if all data is available
        if all([df is not None for df in [af2rank_ensemble_df, af2rank_state0_df, af2rank_state1_df]]):
            plot_composite_scores(
                ensemble_df=af2rank_ensemble_df,
                state0_df=af2rank_state0_df,
                state1_df=af2rank_state1_df,
                plot_name=plot_name,
                out_path=out_dir_for_job / "composite_scores.png"
            )

            # Filter for and plot high-scoring designs
            x_col = "af2rank_c0_model1_composite"
            y_col = "af2rank_c1_model1_composite"

            # Find record_ids with high scores in each dataframe
            high_scorers_ensemble = af2rank_ensemble_df[
                (af2rank_ensemble_df[x_col] > 0.7) | (af2rank_ensemble_df[y_col] > 0.7)
            ]['record_id']
            high_scorers_state0 = af2rank_state0_df[
                (af2rank_state0_df[x_col] > 0.7) | (af2rank_state0_df[y_col] > 0.7)
            ]['record_id']
            high_scorers_state1 = af2rank_state1_df[
                (af2rank_state1_df[x_col] > 0.7) | (af2rank_state1_df[y_col] > 0.7)
            ]['record_id']

            # Combine all high-scoring record_ids into a unique set
            all_high_scorer_ids = set(high_scorers_ensemble) | set(high_scorers_state0) | set(high_scorers_state1)

            if all_high_scorer_ids:
                # Filter the original dataframes
                ensemble_df_filtered = af2rank_ensemble_df[af2rank_ensemble_df['record_id'].isin(all_high_scorer_ids)]
                state0_df_filtered = af2rank_state0_df[af2rank_state0_df['record_id'].isin(all_high_scorer_ids)]
                state1_df_filtered = af2rank_state1_df[af2rank_state1_df['record_id'].isin(all_high_scorer_ids)]

                # Call the new plotting function
                plot_composite_scores_high_scorers(
                    ensemble_df=ensemble_df_filtered,
                    state0_df=state0_df_filtered,
                    state1_df=state1_df_filtered,
                    plot_name=plot_name,
                    out_path=out_dir_for_job / "composite_scores_high_scorers.png"
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

    # Optionally annotate each point with its pdb_id
    # (Comment out if it clutters the plot)
    for pdb_id, x, y in zip(df["pdb_id"], df[x_col0], df[y_col]):
        ax.annotate(
            pdb_id.split("_")[0],
            (x, y),
            textcoords="offset points",
            xytext=(0, -8),
            ha='center',
            fontsize=4
        )
    for pdb_id, x, y in zip(df["pdb_id"], df[x_col1], df[y_col]):
        ax.annotate(
            pdb_id.split("_")[0],
            (x, y),
            textcoords="offset points",
            xytext=(0, -8),
            ha='center',
            fontsize=4
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


def plot_comparison_scatter(
    df: pd.DataFrame,
    metric: str,
    metric_title: str,
    comparison_type: str,
    x_axis_label: str,
    plot_name: str,
    out_path: Path,
    length_legend_range: list = None,
) -> None:
    """
    Generates a scatter plot of an ensemble metric vs. a comparison
    (e.g., 'best_state', 'mean_state'), colored by protein length.
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    y_col = f"{metric}_ensemble"
    x_col = f"{metric}_{comparison_type}"

    x_vals = df[x_col]
    y_vals = df[y_col]

    # Handle coloring by length if available
    if 'length' in df.columns and not df['length'].isnull().all():
        length_vals = df["length"]
        vmin, vmax = (length_legend_range if length_legend_range is not None else (None, None))
        ax.scatter(
            x_vals, y_vals, s=30, c=length_vals, cmap="viridis",
            alpha=0.8, edgecolor="k", linewidth=0.5, vmin=vmin, vmax=vmax
        )
    else:
        # Default plot if no length data
        ax.scatter(
            x_vals, y_vals, s=30, color="steelblue",
            alpha=0.8, edgecolor="k", linewidth=0.5
        )

    # Optionally annotate each point with its pdb_id
    # (Comment out if it clutters the plot)
    for pdb_id, xi, yi in zip(df["pdb_id"], x_vals, y_vals):
        ax.annotate(
            pdb_id.split("_")[0],
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -8),
            ha='center',
            fontsize=4
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
    ax.set_xlabel(f"{x_axis_label} {metric_title}")
    ax.set_ylabel(f"Ensemble {metric_title}")
    ax.set_title(f"{plot_name}: {x_axis_label} {metric_title}")
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


def plot_composite_scores(
    ensemble_df: pd.DataFrame,
    state0_df: pd.DataFrame,
    state1_df: pd.DataFrame,
    plot_name: str,
    out_path: Path
) -> None:
    """
    Generates a scatter plot of af2rank composite scores, comparing c1 vs c0.
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    x_col = "af2rank_c0_model1_composite"
    y_col = "af2rank_c1_model1_composite"

    # Scatter plot for ensemble (black)
    ax.scatter(
        ensemble_df[x_col],
        ensemble_df[y_col],
        s=30,
        color="black",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
        label="Ensemble"
    )

    # Scatter plot for state0 (green)
    ax.scatter(
        state0_df[x_col],
        state0_df[y_col],
        s=30,
        color="green",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
        label="State 0"
    )

    # Scatter plot for state1 (blue)
    ax.scatter(
        state1_df[x_col],
        state1_df[y_col],
        s=30,
        color="blue",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
        label="State 1"
    )

    # Determine plot limits to be square and include all data
    all_x_vals = pd.concat([ensemble_df[x_col], state0_df[x_col], state1_df[x_col]]).dropna()
    all_y_vals = pd.concat([ensemble_df[y_col], state0_df[y_col], state1_df[y_col]]).dropna()
    all_vals = pd.concat([all_x_vals, all_y_vals])
    lower = all_vals.min() * 0.95
    upper = all_vals.max() * 1.05
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    # Plot diagonal y=x line
    ax.plot([lower, upper], [lower, upper], 'k--', alpha=0.8)

    # Labels, title, legend, and grid
    ax.set_xlabel("Composite Score (state 0)")
    ax.set_ylabel("Composite Score (state 1)")
    ax.set_title(f"{plot_name}: AF2Rank Composite Score")
    ax.legend()
    plt.grid(True, alpha=0.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_composite_scores_high_scorers(
    ensemble_df: pd.DataFrame,
    state0_df: pd.DataFrame,
    state1_df: pd.DataFrame,
    plot_name: str,
    out_path: Path
) -> None:
    """
    Generates a scatter plot of af2rank composite scores for designs
    with at least one composite score > 0.7.

    Uses markers to denote state and color to denote PDB ID.
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    x_col = "af2rank_c0_model1_composite"
    y_col = "af2rank_c1_model1_composite"

    # Get unique PDB IDs from the record_id to assign colors
    unique_pdb_ids = sorted(ensemble_df['record_id'].apply(lambda r: r.split('_sample')[0]).unique())

    # Create a color map for PDBs using a high-contrast colormap
    colors = plt.cm.get_cmap('tab20', len(unique_pdb_ids))
    color_map = {pdb_id: colors(i) for i, pdb_id in enumerate(unique_pdb_ids)}

    # Plot each PDB with a unique color, using markers for state
    for pdb_id, color in color_map.items():
        # Filter data for the current PDB
        ens_subset = ensemble_df[ensemble_df['record_id'].str.startswith(pdb_id)]
        s0_subset = state0_df[state0_df['record_id'].str.startswith(pdb_id)]
        s1_subset = state1_df[state1_df['record_id'].str.startswith(pdb_id)]

        # Plot points for this PDB
        if not ens_subset.empty:
            ax.scatter(ens_subset[x_col], ens_subset[y_col], marker='X', s=50, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        if not s0_subset.empty:
            ax.scatter(s0_subset[x_col], s0_subset[y_col], marker='s', s=50, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        if not s1_subset.empty:
            ax.scatter(s1_subset[x_col], s1_subset[y_col], marker='o', s=50, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

    # --- Annotation Logic ---
    # Annotate ensemble points
    for _, row in ensemble_df.iterrows():
        label = row['record_id'].split('_sample')[0].split('_')[0]
        ax.annotate(label, (row[x_col], row[y_col]), textcoords="offset points", xytext=(0, 2), ha='center', fontsize=3)
    # Annotate state0 points
    for _, row in state0_df.iterrows():
        label = row['record_id'].split('_sample')[0].split('_')[0]
        ax.annotate(label, (row[x_col], row[y_col]), textcoords="offset points", xytext=(0, 2), ha='right', fontsize=3)
    # Annotate state1 points
    for _, row in state1_df.iterrows():
        label = row['record_id'].split('_sample')[0].split('_')[0]
        ax.annotate(label, (row[x_col], row[y_col]), textcoords="offset points", xytext=(0, 2), ha='left', fontsize=3)

    # --- Plot Limits and Formatting ---
    all_x_vals = pd.concat([ensemble_df[x_col], state0_df[x_col], state1_df[x_col]]).dropna()
    all_y_vals = pd.concat([ensemble_df[y_col], state0_df[y_col], state1_df[y_col]]).dropna()
    all_vals = pd.concat([all_x_vals, all_y_vals])
    lower = all_vals.min() * 0.95
    upper = all_vals.max() * 1.05
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.plot([lower, upper], [lower, upper], 'k--', alpha=0.8)

    # --- Custom Legend for Markers ---
    legend_elements = [
        Line2D([0], [0], marker='X', color='grey', label='Ensemble', linestyle='None', markersize=4),
        Line2D([0], [0], marker='s', color='grey', label='State 0', linestyle='None', markersize=4),
        Line2D([0], [0], marker='o', color='grey', label='State 1', linestyle='None', markersize=4)
    ]
    ax.legend(handles=legend_elements, title="States")

    ax.set_xlabel("Composite Score (state 0)")
    ax.set_ylabel("Composite Score (state 1)")
    ax.set_title(f"{plot_name}: AF2Rank Composite Score (> 0.7)")
    plt.grid(True, alpha=0.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()