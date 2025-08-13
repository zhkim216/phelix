from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="plot_bindcraft_traj_benchmark")
def main(cfg: DictConfig) -> None:
    """
    Generates scatter plots for key performance metrics (iPAE, pLDDT, iPTM)
    to compare different protein design models.
    """
    # Create the base output directory
    base_out_dir = Path(cfg.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the entire config into the base output directory for reference
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(base_out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Get all models that we need to load
    required_models = set()
    for comparison in cfg.comparisons:
        required_models.update(comparison)

    # Build a dictionary for all models defined in `model_csvs`
    model_data = {}
    for model_cfg in cfg.model_csvs:
        if model_cfg["model_name"] not in required_models:
            continue

        mname = model_cfg["model_name"]
        mplot = model_cfg["plot_name"]
        mcsv = model_cfg["csv"]

        df = pd.read_csv(mcsv)

        # Calculate the required metrics across model0 and model1 for each sample
        df["min_complex_iPAE"] = df[["complex_B_model0_i_pae", "complex_B_model1_i_pae"]].min(axis=1)

        # pLDDT values need to be scaled to 0-100 range
        df["max_binder_pLDDT"] = df[["binder_B_model0_plddt", "binder_B_model1_plddt"]].max(axis=1) * 100

        df["max_complex_iPTM"] = df[["complex_B_model0_i_ptm", "complex_B_model1_i_ptm"]].max(axis=1)

        # Extract "protein_id" from record_id to aggregate samples
        df["protein_id"] = df["record_id"].apply(lambda x: x.split("_sample")[0])

        # Extract "target" from protein_id for color faceting
        df["target"] = df["protein_id"].apply(lambda x: "_".join(x.split("_")[:-2]))

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
        out_dir_for_comp = base_out_dir / f"{y_model_name}_vs_{x_model_name}"
        out_dir_for_comp.mkdir(parents=True, exist_ok=True)

        # Generate scatter plots for the three key metrics
        create_scatter_plot(
            model1_df=x_df,
            model2_df=y_df,
            model1_name=x_plot_name,
            model2_name=y_plot_name,
            out_dir=out_dir_for_comp,
            metric="min_complex_iPAE",
            metric_title="Best Complex iPAE",
            best_is_min=True,
        )
        create_scatter_plot(
            model1_df=x_df,
            model2_df=y_df,
            model1_name=x_plot_name,
            model2_name=y_plot_name,
            out_dir=out_dir_for_comp,
            metric="max_binder_pLDDT",
            metric_title="Best Binder pLDDT",
            best_is_min=False,
        )
        create_scatter_plot(
            model1_df=x_df,
            model2_df=y_df,
            model1_name=x_plot_name,
            model2_name=y_plot_name,
            out_dir=out_dir_for_comp,
            metric="max_complex_iPTM",
            metric_title="Best Complex iPTM",
            best_is_min=False,
        )


def create_scatter_plot(
    model1_df: pd.DataFrame,
    model2_df: pd.DataFrame,
    model1_name: str,
    model2_name: str,
    out_dir: Path,
    metric: str,
    metric_title: str,
    best_is_min: bool,
) -> None:
    """
    Creates a scatter plot comparing the best value of a given metric between two models.
    model1 => x-axis
    model2 => y-axis
    best_is_min => whether a lower metric is better (e.g., iPAE)
    """
    # Aggregate to find the best value for each protein across all its samples
    if best_is_min:
        model1_best = model1_df.groupby("protein_id")[metric].min().rename(f"{model1_name}_best")
        model2_best = model2_df.groupby("protein_id")[metric].min().rename(f"{model2_name}_best")
    else:
        model1_best = model1_df.groupby("protein_id")[metric].max().rename(f"{model1_name}_best")
        model2_best = model2_df.groupby("protein_id")[metric].max().rename(f"{model2_name}_best")

    # Merge the aggregated data for both models into a single DataFrame for plotting
    plot_df = pd.merge(model1_best, model2_best, left_index=True, right_index=True)

    # Add target information for coloring
    target_info = model1_df[["protein_id", "target"]].drop_duplicates()
    plot_df = plot_df.reset_index().merge(target_info, on="protein_id").set_index("protein_id")

    # Set up the plot aesthetics, making space for the legend
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    x_vals = plot_df[f"{model1_name}_best"]
    y_vals = plot_df[f"{model2_name}_best"]

    # Create a color map for the unique targets
    unique_targets = sorted(plot_df["target"].unique())
    cmap = plt.get_cmap('tab20', len(unique_targets))
    color_map = {target: cmap(i) for i, target in enumerate(unique_targets)}
    colors_for_plot = plot_df["target"].map(color_map)

    # Create the scatter plot, colored by target
    ax.scatter(
        x_vals,
        y_vals,
        s=30,
        c=colors_for_plot,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.5
    )

    # Determine unified plot limits to keep scales consistent
    all_vals = pd.concat([x_vals, y_vals])
    lower = 0.0
    upper = all_vals.max() * 1.05

    # Set specific ranges for known metrics
    if "pLDDT" in metric:
        lower, upper = 0, 100
    elif "iPTM" in metric or "iPAE" in metric:
        lower, upper = 0, 1

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.plot([lower, upper], [lower, upper], 'k--', alpha=0.7) # Add y=x line

    # Set labels, title, and grid
    ax.set_xlabel(model1_name)
    ax.set_ylabel(model2_name)
    n = len(plot_df)
    ax.set_title(f"{metric_title} (n={n})")
    ax.grid(True, alpha=0.5)

    # Add a legend for the targets outside the plot area
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=target,
                              markerfacecolor=color_map[target], markersize=8)
                       for target in unique_targets]
    ax.legend(handles=legend_elements, title="Target", bbox_to_anchor=(1.04, 1), loc="upper left")

    # Save the figure with a tight bounding box to include the legend
    plt.savefig(out_dir / f"{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()