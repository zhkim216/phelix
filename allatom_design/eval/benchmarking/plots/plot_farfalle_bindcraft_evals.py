from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="plot_farfalle_bindcraft_evals")
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

        df = pd.read_csv(mcsv)

        model_data[mname] = {
            "df": df,
            "plot_name": mplot
        }

    # Iterate over all comparison groups
    for comparison in cfg.comparisons:
        comp_name = comparison["name"]
        models_in_group = comparison["models"]

        # Create a sub-output directory for this group
        out_dir_for_comp = base_out_dir / comp_name
        out_dir_for_comp.mkdir(parents=True, exist_ok=True)

        # Collect raw dataframes for all models in the current comparison group
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

        # Plot min/max aggregated metrics
        min_max_metric_info = {
            "rmsd": {"agg": "min", "title": "min RMSD"},
            "iptm": {"agg": "max", "title": "max ipTM"},
            "ipae": {"agg": "min", "title": "min ipAE"},
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
                out_path=out_dir_for_comp / f"{'_'.join(info['title'].split()).lower()}_comparison.png"
            )

        # Plot mean aggregated metrics
        mean_metric_info = {
            "rmsd": {"agg": "mean", "title": "mean RMSD"},
            "iptm": {"agg": "mean", "title": "mean ipTM"},
            "ipae": {"agg": "mean", "title": "mean ipAE"},
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
                out_path=out_dir_for_comp / f"mean_{metric}_comparison.png"
            )

        # Plot number of successes, defined by ipae < 0.35, iptm > 0.5, and rmsd < 2
        group_df['success'] = (
            (group_df['ipae'] < 0.35) &
            (group_df['iptm'] > 0.5)
            # (group_df['rmsd'] < 2)
        )
        success_summary = group_df.groupby(['model_name', 'motif_name'])['success'].sum().reset_index()
        create_bar_chart(
            data=success_summary,
            models_in_group=plot_names_in_group,
            metric='success',
            y_label="Number of Successes",
            title=f"{comp_name}, Number of Successes \n (ipAE < 0.35, ipTM > 0.5)",
            out_path=out_dir_for_comp / "success_count_comparison.png"
        )


def create_bar_chart(
    data: pd.DataFrame,
    models_in_group: list[str],
    metric: str,
    y_label: str,
    title: str,
    out_path: Path
) -> None:
    """
    Creates a grouped bar chart for a given metric, with motifs on the x-axis.
    """
    motifs = natsorted(data['motif_name'].unique())
    n_models = len(models_in_group)

    x = np.arange(len(motifs))  # the label locations
    total_width = 0.8
    width = total_width / n_models

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(motifs)), 6))

    for i, model_plot_name in enumerate(models_in_group):
        model_data = data[data['model_name'] == model_plot_name]

        # Create a map of motif to metric value for quick lookup
        metric_map = model_data.set_index('motif_name')[metric]

        # Get values in the correct order of motifs, filling with NaN if a motif is missing
        values = [metric_map.get(motif, np.nan) for motif in motifs]

        # Calculate position for each bar
        position = x - (total_width / 2) + (i * width) + (width / 2)
        ax.bar(position, values, width, label=model_plot_name)

    # Add some text for labels, title and axes ticks
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x, motifs, rotation=45, ha="right")
    ax.legend(title="Models", bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Set custom y-tick spacing
    ymin, ymax = ax.get_ylim()
    step = None
    if metric in ["ipae", "iptm"]:
        step = 0.05
    elif metric == "rmsd":
        ax.set_ylim(top=10)
        ymin, ymax = ax.get_ylim()
        step = 1.0

    if step is not None:
        start = np.floor(ymin / step) * step
        ticks = np.arange(start, ymax + step, step)
        ax.set_yticks(ticks)
    elif metric == "success":
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()