from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colorbar import ColorbarBase
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="compare_self_consistency")
def main(cfg: DictConfig) -> None:
    """
    Iterate over pairs in cfg.comparisons, flipping the axes so the first
    element of each pair becomes the y-axis and the second becomes the x-axis.
    """
    # Create the base output directory
    Path(cfg.base_out_dir).mkdir(parents=True, exist_ok=True)

    # Dump the entire config into the base output directory for reference
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.base_out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)
        
    # Optional: read in the length CSV (skip if None)
    length_df = None
    if getattr(cfg, "length_csv", None) is not None:
        length_df = pd.read_csv(
            cfg.length_csv,
            header=None,
            names=["pdb_id", "length"],
            sep=","
        )

    # Retrieve optional length_legend_range
    length_legend_range = None
    if cfg.length_legend_range is not None:
        length_legend_range = cfg.length_legend_range
        save_length_colorbar(
            vmin=length_legend_range[0],
            vmax=length_legend_range[1],
            out_path=Path(cfg.base_out_dir) / "protein_length_colorbar.png",
        )
    

    # Get all models that we need to load
    required_models = set()
    for comparison in cfg.comparisons:
        required_models.update(comparison)
    required_models = list(required_models)

    # Build a dictionary for all models defined in `model_csvs` that we require for the comparisons
    model_data = {}
    for model_cfg in cfg.model_csvs:
        if model_cfg["model_name"] not in required_models:
            continue

        mname = model_cfg["model_name"]
        mplot = model_cfg["plot_name"]
        mcsv = model_cfg["csv"]

        df = pd.read_csv(mcsv)

        # pLDDT scaling
        if "avg_ca_plddt" in df.columns:
            df["avg_ca_plddt"] = df["avg_ca_plddt"] * 100

        # Extract "pdb_name" from record_id
        df["pdb_name"] = df["record_id"].apply(lambda x: f'{x.split("_sample")[0]}.cif')

        # If this model is in the list of subset models, filter by subset_pdb_names
        if cfg.subset_pdb_names is not None and mname in cfg.use_subset_models:
            with open(cfg.subset_pdb_names, "r") as f:
                subset_pdbs = [line.strip() for line in f.readlines()]
            df = df[df["pdb_name"].isin(subset_pdbs)]

        model_data[mname] = {
            "df": df,
            "plot_name": mplot
        }

    # Iterate over all comparisons
    for comparison in cfg.comparisons:
        y_model_name, x_model_name = comparison

        # Grab the dataframes + plot names
        x_df = model_data[x_model_name]["df"].copy()
        x_plot_name = model_data[x_model_name]["plot_name"]
        y_df = model_data[y_model_name]["df"].copy()
        y_plot_name = model_data[y_model_name]["plot_name"]

        # Create a sub-output directory for this pair
        out_dir_for_comp = Path(cfg.base_out_dir) / f"{y_model_name}-vs-{x_model_name}"
        out_dir_for_comp.mkdir(parents=True, exist_ok=True)

        # For scRMSD (best is min) and pLDDT (best is max)
        create_scatter_plots(
            model1_df=x_df,        # x-axis
            model2_df=y_df,        # y-axis
            model1_name=x_plot_name,
            model2_name=y_plot_name,
            out_dir=out_dir_for_comp,
            prefix="boltz",
            metric="sc_ca_rmsd",
            metric_title="Boltz-1x scRMSD",
            best_is_min=True,
            length_df=length_df,
            length_legend_range=length_legend_range
        )
        create_scatter_plots(
            model1_df=x_df,        # x-axis
            model2_df=y_df,        # y-axis
            model1_name=x_plot_name,
            model2_name=y_plot_name,
            out_dir=out_dir_for_comp,
            prefix="boltz",
            metric="avg_ca_plddt",
            metric_title="Boltz-1x average pLDDT",
            best_is_min=False,
            length_df=length_df,
            length_legend_range=length_legend_range
        )


def create_scatter_plots(
    model1_df: pd.DataFrame,
    model2_df: pd.DataFrame,
    model1_name: str,
    model2_name: str,
    out_dir: Path,
    prefix: str,
    metric: str,
    metric_title: str,
    best_is_min: bool,
    length_df: pd.DataFrame = None,
    length_legend_range: list = None
) -> None:
    """
    model1 => x-axis
    model2 => y-axis
    best_is_min => whether a lower metric is better (e.g. RMSD)
    
    length_df: optional pd.DataFrame with columns ["pdb_id", "length"].
    length_legend_range: optional list with [min_length, max_length].
    """
    # Extract a simpler "pdb_id"
    model1_df["pdb_id"] = model1_df["record_id"].apply(lambda x: x.split("_sample")[0])
    model2_df["pdb_id"] = model2_df["record_id"].apply(lambda x: x.split("_sample")[0])

    # Aggregate best (min) or worst (max)
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

    # If length_df is available, merge length information so we can color by length
    if length_df is not None:
        best_df = best_df.reset_index().merge(length_df, on="pdb_id", how="left").set_index("pdb_id")
        median_df = median_df.reset_index().merge(length_df, on="pdb_id", how="left").set_index("pdb_id")
    else:
        # If not provided, just add a dummy length column with NaN or zero
        best_df = best_df.reset_index().assign(length=np.nan).set_index("pdb_id")
        median_df = median_df.reset_index().assign(length=np.nan).set_index("pdb_id")

    def plot_and_annotate(df: pd.DataFrame, kind: str):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # x = first metric column => model1
        # y = second metric column => model2
        col_names = df.columns[:2]
        x_vals = df[col_names[0]].values
        y_vals = df[col_names[1]].values

        # Length values for coloring, or None
        length_vals = df["length"].values

        # Scatter plot
        if length_df is not None:
            # If we have length data, color by length
            if length_legend_range is not None:
                vmin, vmax = length_legend_range
            else:
                # If no explicit range is set, let matplotlib pick the range
                vmin = None
                vmax = None

            sc = ax.scatter(
                x_vals,
                y_vals,
                s=30,
                c=length_vals,
                cmap="viridis",
                alpha=0.8,
                edgecolor="k",
                linewidth=0.5,
                vmin=vmin,
                vmax=vmax
            )
            # # Put a colorbar below the plot
            # cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.15, fraction=0.046)
            # cbar.set_label("Protein Length")
        else:
            # If no length info, just use a single color (e.g., steelblue)
            ax.scatter(
                x_vals,
                y_vals,
                s=30,
                color="steelblue",
                alpha=0.8,
                edgecolor="k",
                linewidth=0.5
            )

        # Compute unified plot limits
        lower = 0.0
        upper = max(np.max(x_vals), np.max(y_vals)) * 1.05  # some padding
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.plot([lower, upper], [lower, upper], 'k--')

        # Optionally annotate each point with its pdb_id
        # (Comment out if it clutters the plot)
        for pdb_id, xi, yi in zip(df.index, x_vals, y_vals):
            ax.annotate(
                pdb_id,
                (xi, yi),
                textcoords="offset points",
                xytext=(0, -8),
                ha='center',
                fontsize=6
            )

        # Labels, title, and grid
        ax.set_xlabel(model1_name)
        ax.set_ylabel(model2_name)
        n = len(df)
        ax.set_title(f"{metric_title} ({kind} of 8, n={n})")
        plt.grid(True, alpha=0.5)

        # Adjust tick spacing for RMSD vs pLDDT
        if metric == "sc_ca_rmsd":
            step = 2.0
        else:
            step = 5.0
        ticks = np.arange(0, upper + 1e-6, step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        plt.savefig(out_dir / f"{prefix}_{metric}_{kind}.png", dpi=300)
        plt.close()

    # Make best and median scatter plots
    plot_and_annotate(best_df, "best")
    plot_and_annotate(median_df, "median")


def save_length_colorbar(vmin: float, vmax: float, out_path: Path) -> None:
    """
    Creates and saves a standalone horizontal colorbar (no other axes) with the label "Protein length".
    """
    # 1×1 colorbar figure (you can tweak figsize as needed)
    fig, ax = plt.subplots(figsize=(6, 0.5))

    # Normalize from vmin→vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Create a ColorbarBase on that single Axes
    ColorbarBase(
        ax,
        cmap="viridis",
        norm=norm,
        orientation="horizontal",
    ).set_label("Protein length")

    # Save, tight around the colorbar only
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    main()
