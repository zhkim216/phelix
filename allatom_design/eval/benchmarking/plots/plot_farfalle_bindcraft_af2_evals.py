from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="plot_farfalle_bindcraft_af2_evals")
def main(cfg: DictConfig) -> None:
    """
    Iterate over comparison groups in cfg.comparisons, and for each group,
    create bar charts for specified metrics, grouped by motif.
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
        required_models.update(comparison["models"])
    required_models = list(required_models)

    # Build a dictionary for all models defined in `model_csvs` that we require
    model_data = {}
    for model_cfg in cfg.model_csvs:
        if model_cfg["model_name"] not in required_models:
            continue

        mname = model_cfg["model_name"]
        mplot = model_cfg["plot_name"]
        mcsv = model_cfg["csv"]

        if not Path(mcsv).exists():
            print(f"Warning: CSV file {mcsv} does not exist. Skipping.")
            continue

        df = pd.read_csv(mcsv)

        df['rmsd_per_run'] = df[['model_0_binder_rmsd', 'model_1_binder_rmsd']].min(axis=1)
        df['iptm_per_run'] = df[['model_0_complex_i_ptm', 'model_1_complex_i_ptm']].max(axis=1)
        df['ipae_per_run'] = df[['model_0_complex_i_pae', 'model_1_complex_i_pae']].min(axis=1)
        df['plddt_per_run'] = df[['model_0_complex_plddt', 'model_1_complex_plddt']].max(axis=1) * 100

        df = df.groupby('pdb_path').agg(
            rmsd=('rmsd_per_run', 'min'),
            iptm=('iptm_per_run', 'max'),
            ipae=('ipae_per_run', 'min'),
            plddt=('plddt_per_run', 'max')
        ).reset_index()

        df['motif_name'] = df['pdb_path'].apply(lambda p: Path(p).parent.name)

        model_data[mname] = {
            "df": df,
            "plot_name": mplot
        }

    # Iterate over all comparison groups
    for comparison in cfg.comparisons:
        comp_name = comparison["name"]
        models_in_group = comparison["models"]

        out_dir_for_comp = base_out_dir / comp_name
        out_dir_for_comp.mkdir(parents=True, exist_ok=True)

        all_dfs = []
        plot_names_in_group = []
        for model_name in models_in_group:
            if model_name not in model_data:
                print(f"Warning: Model '{model_name}' not found in model_csvs. Skipping.")
                continue

            model_df = model_data[model_name]["df"].copy()
            plot_name = model_data[model_name]["plot_name"]
            plot_names_in_group.append(plot_name)

            model_df['model_name'] = plot_name
            all_dfs.append(model_df)

        if not all_dfs:
            print(f"No data to plot for comparison group {comp_name}. Skipping.")
            continue

        group_df = pd.concat(all_dfs, ignore_index=True)

        # Plot min/max aggregated metrics with custom plot settings
        min_max_metric_info = {
            "rmsd":  {"agg": "min", "title": "min RMSD",  "ylim": (0, 10)},
            "iptm":  {"agg": "max", "title": "max ipTM",  "ylim": (0.4, 1.0)},
            "ipae":  {"agg": "min", "title": "min ipAE",  "ylim": (0, 1.0)},
            "plddt": {"agg": "max", "title": "max pLDDT", "ylim": (50, 100)},
        }
        agg_dict_minmax = {metric: info["agg"] for metric, info in min_max_metric_info.items()}
        summary_minmax = group_df.groupby(['model_name', 'motif_name']).agg(agg_dict_minmax).reset_index()

        for metric, info in min_max_metric_info.items():
            create_bar_chart(
                data=summary_minmax,
                models_in_group=plot_names_in_group,
                metric=metric,
                y_label=info["title"],
                title=f"{comp_name}, {info['title']}",
                out_path=out_dir_for_comp / f"{'_'.join(info['title'].split()).lower()}_comparison.png",
                ylim=info.get("ylim") # Pass ylim
            )

        # Plot mean aggregated metrics with custom plot settings
        mean_metric_info = {
            "rmsd":  {"agg": "mean", "title": "mean RMSD",  "ylim": (0, 15)},
            "iptm":  {"agg": "mean", "title": "mean ipTM",  "ylim": (0, 1.0)},
            "ipae":  {"agg": "mean", "title": "mean ipAE",  "ylim": (0, 1.0)},
            "plddt": {"agg": "mean", "title": "mean pLDDT", "ylim": (50, 100)},
        }
        agg_dict_mean = {metric: info["agg"] for metric, info in mean_metric_info.items()}
        summary_mean = group_df.groupby(['model_name', 'motif_name']).agg(agg_dict_mean).reset_index()

        for metric, info in mean_metric_info.items():
            create_bar_chart(
                data=summary_mean,
                models_in_group=plot_names_in_group,
                metric=metric,
                y_label=info["title"],
                title=f"{comp_name}, {info['title']}",
                out_path=out_dir_for_comp / f"mean_{metric}_comparison.png",
                ylim=info.get("ylim") # Pass ylim
            )

        # Plot number of successes, defined by ipae < 0.35, iptm > 0.5
        group_df['success'] = (group_df['ipae'] < 0.35) & (group_df['iptm'] > 0.5)
        success_summary = group_df.groupby(['model_name', 'motif_name'])['success'].sum().reset_index()

        create_bar_chart(
            data=success_summary,
            models_in_group=plot_names_in_group,
            metric='success',
            y_label="Number of Successes",
            title=f"{comp_name}, Number of Successes \n (ipAE < 0.35, ipTM > 0.5)",
            out_path=out_dir_for_comp / "success_count_comparison.png",
            bar_total_width=0.7, # Example of custom bar width
            ylim=(0, None) # Example: set lower bound to 0, let upper be automatic
        )


def create_bar_chart(
    data: pd.DataFrame,
    models_in_group: list[str],
    metric: str,
    y_label: str,
    title: str,
    out_path: Path,
    bar_total_width: float = 0.8,
    ylim: tuple[float | None, float | None] | None = None,
    xlim: tuple[float | None, float | None] | None = None,
) -> None:
    """
    Creates a grouped bar chart for a given metric, with motifs on the x-axis.

    Args:
        data: DataFrame containing the data to plot.
        models_in_group: List of model names to include in the plot.
        metric: The column name of the metric to plot.
        y_label: The label for the y-axis.
        title: The title of the plot.
        out_path: The path to save the plot image.
        bar_total_width: The total width that all bars for a single motif should occupy.
        ylim: A tuple (min, max) for the y-axis limit. Use None for auto-scaling.
        xlim: A tuple (min, max) for the x-axis limit. Use None for auto-scaling.
    """
    motifs = natsorted(data['motif_name'].unique())
    n_models = len(models_in_group)

    x = np.arange(len(motifs))  # the label locations
    width = bar_total_width / n_models  # the width of an individual bar

    fig, ax = plt.subplots(figsize=(0.5 * len(motifs), 6))

    for i, model_plot_name in enumerate(models_in_group):
        model_data = data[data['model_name'] == model_plot_name]
        metric_map = model_data.set_index('motif_name')[metric]
        values = [metric_map.get(motif, np.nan) for motif in motifs]
        position = x - (bar_total_width / 2) + (i * width) + (width / 2)
        ax.bar(position, values, width, label=model_plot_name)

    # Add some text for labels, title and axes ticks
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x, motifs, rotation=45, ha="right")
    # ax.legend(title="Models", bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Set custom axis limits if provided
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    # Set custom y-tick spacing based on metric type
    current_ymin, current_ymax = ax.get_ylim()
    step = None
    if metric in ["ipae", "iptm"]:
        step = 0.05
    elif metric == "rmsd":
        step = 1.0
    elif metric == "plddt":
        step = 10.0

    if step is not None:
        start = np.floor(current_ymin / step) * step
        ticks = np.arange(start, current_ymax + step, step)
        ax.set_yticks(ticks)
    elif metric == "success":
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if ylim is None or ylim[0] is None:
             ax.set_ylim(bottom=0)

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()