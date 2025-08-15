from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="plot_denovo500")
def main(cfg: DictConfig) -> None:
    """
    Generates line plots of median/mean performance metrics (pLDDT, scRMSD)
    vs. protein length for different model groups.
    """
    # Create the base output directory
    base_out_dir = Path(cfg.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the entire config into the base output directory for reference
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(base_out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Create a lookup map for model configurations for easy access
    model_info_map = {model["model_name"]: model for model in cfg.model_csvs}

    for comparison in cfg.comparisons:
        # Create a specific output directory for this comparison
        out_dir = base_out_dir / comparison["struct_model_name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create a set of the model names for efficient lookup
        models_to_plot = set(comparison["models"])

        # Load and concatenate data for the selected models into a single DataFrame
        all_dfs = []
        for model_cfg in cfg.model_csvs:
            # Only process the models specified in the 'comparisons' list
            if model_cfg["model_name"] in models_to_plot:
                df = pd.read_csv(model_cfg["csv"])

                # pLDDT scaling
                if "avg_ca_plddt" in df.columns:
                    df["avg_ca_plddt"] = df["avg_ca_plddt"] * 100

                # Add the plot_name for grouping and legends
                df["plot_name"] = model_cfg["plot_name"]

                # Extract protein length from the record_id
                df["length"] = df["record_id"].str.split("_").str[0].str[1:].astype(int)

                all_dfs.append(df)

        # Check if any data was loaded before proceeding
        if not all_dfs:
            print(f"Warning: No models for comparison '{comparison['struct_model_name']}' were found in 'model_csvs'. No plots will be generated for this group.")
            continue

        master_df = pd.concat(all_dfs, ignore_index=True)

        # Get the ordered list of plot names for consistent legend and coloring
        plot_names_in_order = []
        for model_name in comparison["models"]:
            if model_name in model_info_map:
                plot_name = model_info_map[model_name]["plot_name"]
                if plot_name not in plot_names_in_order:
                    plot_names_in_order.append(plot_name)

        # Generate scRMSD plots
        plot_line_metric(
            df=master_df,
            plot_names_in_order=plot_names_in_order,
            metric="sc_ca_rmsd",
            aggregation="median",
            ylabel="Median scRMSD",
            out_path=out_dir / "sc_ca_rmsd_med.png",
        )
        plot_line_metric(
            df=master_df,
            plot_names_in_order=plot_names_in_order,
            metric="sc_ca_rmsd",
            aggregation="mean",
            ylabel="Mean scRMSD",
            out_path=out_dir / "sc_ca_rmsd_mean.png",
        )

        # Generate pLDDT plots
        plot_line_metric(
            df=master_df,
            plot_names_in_order=plot_names_in_order,
            metric="avg_ca_plddt",
            aggregation="median",
            ylabel="Median pLDDT",
            out_path=out_dir / "avg_ca_plddt_med.png",
        )
        plot_line_metric(
            df=master_df,
            plot_names_in_order=plot_names_in_order,
            metric="avg_ca_plddt",
            aggregation="mean",
            ylabel="Mean pLDDT",
            out_path=out_dir / "avg_ca_plddt_mean.png",
        )


def plot_line_metric(
    df: pd.DataFrame,
    plot_names_in_order: list[str],
    metric: str,
    aggregation: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Creates and saves a line plot for a given metric vs. protein length.
    """
    # Aggregate the data (median or mean) by model and length
    grouped = (
        df.groupby(["plot_name", "length"])[metric]
        .agg(aggregation)
        .reset_index()
        .rename(columns={metric: "value"})
    )

    # Set up plot aesthetics based on the provided example
    figsize = (4, 3)
    colors = ["#005B96", "#CCCCCC", "#999999", "#555555", "#222222"]
    plt.figure(figsize=figsize)

    # Plot a line for each model
    for i, model_name in enumerate(plot_names_in_order):
        sub_df = grouped[grouped["plot_name"] == model_name].sort_values("length")
        if not sub_df.empty:
            plt.plot(
                sub_df["length"],
                sub_df["value"],
                label=model_name,
                marker="o",
                markersize=4,
                color=colors[i % len(colors)],
            )

    # Set labels and grid
    plt.xlabel("Length", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
    plt.xticks([100, 200, 300, 400, 500])
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    # Save the figure in both PNG and PDF formats
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()