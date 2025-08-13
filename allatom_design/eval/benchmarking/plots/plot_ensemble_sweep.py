from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="plot_ensemble_sweep")
def main(cfg: DictConfig) -> None:
    """
    Generates box and whisker plots comparing performance metrics (pLDDT, scRMSD)
    across different model temperatures ('t').
    """
    # Create the base output directory
    base_out_dir = Path(cfg.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the entire config into the base output directory for reference
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(base_out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    for comparison in cfg.comparisons:
        # Create a specific output directory for this comparison
        out_dir = base_out_dir / comparison["plot_name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        model_name_for_title = comparison["model_name"]

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

                # Add the temperature 't' value from the config
                df["t"] = model_cfg["t"]

                # Extract "pdb_id" from record_id to identify unique proteins
                df["pdb_id"] = df["record_id"].apply(lambda x: x.split("_sample")[0])

                all_dfs.append(df)

        # Check if any data was loaded before proceeding
        if not all_dfs:
            print(f"Warning: No models for comparison '{model_name_for_title}' were found in 'model_csvs'. No plots will be generated for this group.")
            continue

        master_df = pd.concat(all_dfs, ignore_index=True)

        # Generate plot for all pLDDT samples
        plot_metric_vs_t(
            df=master_df,
            metric="avg_ca_plddt",
            ylabel="Average CA pLDDT",
            title=f"Distribution of pLDDT across Temperatures (All Samples), {model_name_for_title}",
            out_path=out_dir / "plddt_all_samples_vs_t.png",
            is_plddt=True,
        )

        # Generate plot for all scRMSD samples
        plot_metric_vs_t(
            df=master_df,
            metric="sc_ca_rmsd",
            ylabel="scRMSD (Å)",
            title=f"Distribution of scRMSD across Temperatures (All Samples), {model_name_for_title}",
            out_path=out_dir / "scrmsd_all_samples_vs_t.png",
            is_plddt=False,
        )

        # Determine the best sample for each protein at each temperature
        plddt_best_df = master_df.loc[
            master_df.groupby(["pdb_id", "t"])["avg_ca_plddt"].idxmax()
        ]
        rmsd_best_df = master_df.loc[
            master_df.groupby(["pdb_id", "t"])["sc_ca_rmsd"].idxmin()
        ]

        # Generate plot for the best pLDDT samples
        plot_metric_vs_t(
            df=plddt_best_df,
            metric="avg_ca_plddt",
            ylabel="Average CA pLDDT",
            title=f"Distribution of pLDDT across Temperatures (Best of 8 Samples), {model_name_for_title}",
            out_path=out_dir / "plddt_best_of_8_vs_t.png",
            is_plddt=True,
        )

        # Generate plot for the best scRMSD samples
        plot_metric_vs_t(
            df=rmsd_best_df,
            metric="sc_ca_rmsd",
            ylabel="scRMSD (Å)",
            title=f"Distribution of scRMSD across Temperatures (Best of 8 Samples), {model_name_for_title}",
            out_path=out_dir / "scrmsd_best_of_8_vs_t.png",
            is_plddt=False,
        )


def plot_metric_vs_t(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    is_plddt: bool,
) -> None:
    """
    Creates and saves a box and whisker plot for a given metric vs. temperature.
    """
    # Set up the plot aesthetics
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Sort by 't' to ensure the x-axis is ordered correctly
    sorted_t = sorted(df["t"].unique())

    # Create the box plot using seaborn for easy grouping
    sns.boxplot(
        x="t",
        y=metric,
        data=df,
        ax=ax,
        order=sorted_t,
        palette="viridis",
        fliersize=0,  # Hide standard outlier points, we'll plot them with stripplot
        width=0.5,
    )

    # Overlay individual data points for a comprehensive view
    sns.stripplot(
        x="t",
        y=metric,
        data=df,
        ax=ax,
        order=sorted_t,
        color=".25",
        size=2.5,
        alpha=0.6,
    )

    # Set labels, title, and grid
    ax.set_xlabel("Temperature (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits specifically for pLDDT
    if is_plddt:
        ax.set_ylim(50, 100)
    else:
        ax.set_ylim(0, 8)

    # Save the figure with a tight bounding box
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()