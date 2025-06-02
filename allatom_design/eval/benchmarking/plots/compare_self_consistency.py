from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
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
            best_is_min=True
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
            best_is_min=False
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
    best_is_min: bool
) -> None:
    """
    model1 => x-axis
    model2 => y-axis
    best_is_min => whether a lower metric is better (e.g. RMSD)
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

    def plot_and_annotate(df: pd.DataFrame, kind: str):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # x = first column in df => model1's best/median
        # y = second column in df => model2's best/median
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values

        ax.scatter(x, y, s=4)

        # Compute unified plot limits
        lower = 0.0
        upper = max(np.max(x), np.max(y)) * 1.05  # some padding
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.plot([lower, upper], [lower, upper], 'k--')

        # Annotate each point with its pdb_id
        for pdb_id, xi, yi in zip(df.index, x, y):
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

        # Ticks
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


if __name__ == "__main__":
    main()
