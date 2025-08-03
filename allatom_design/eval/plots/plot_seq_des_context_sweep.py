from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colorbar import ColorbarBase
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.eval_utils.eval_metrics import compute_seq_recovery


@hydra.main(version_base=None, config_path="../../configs/eval/plots", config_name="plot_seq_des_context_sweep")
def main(cfg: DictConfig) -> None:
    """
    Plot the results of sequence design context sweeps against each other.
    """
    # Create the base output directory
    base_out_dir = Path(cfg.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the entire config into the base output directory for reference
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(base_out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Process each model specified in the config
    all_model_data = {}
    for model_cfg in cfg.model_csvs:
        model_name = model_cfg["model_name"]
        if model_name not in cfg.comparisons:
            continue

        base_dir = Path(model_cfg["base_dir"])
        plot_name = model_cfg["plot_name"]

        timestep_results = []

        # Find all timestep directories (e.g., t0.0, t0.1, ...)
        for timestep_dir in sorted(base_dir.glob("t*")):
            if not timestep_dir.is_dir():
                continue

            # Extract float from timestep directory name (e.g., "t0.1" -> 0.1)
            try:
                timestep = float(timestep_dir.name[1:])
            except ValueError:
                continue

            sc_csv_path = timestep_dir / "self_consistency_metrics.csv"
            sd_csv_path = timestep_dir / "seq_des_outputs.csv"

            if not sc_csv_path.exists() or not sd_csv_path.exists():
                continue

            # Load data for the current timestep
            sc_df = pd.read_csv(sc_csv_path)
            sd_df = pd.read_csv(sd_csv_path)

            # Calculate average pLDDT (and scale by 100)
            avg_plddt = (sc_df["avg_ca_plddt"] * 100).mean()

            # Calculate average sequence recovery
            recoveries = sd_df.apply(
                lambda row: compute_seq_recovery(row["input_seq"], row["seq"]),
                axis=1
            )
            avg_seq_rec = recoveries.mean()

            timestep_results.append({
                "timestep": timestep,
                "avg_ca_plddt": avg_plddt,
                "seq_recovery": avg_seq_rec
            })

        # Store results for this model in a DataFrame
        if timestep_results:
            model_df = pd.DataFrame(timestep_results).sort_values("timestep")
            all_model_data[model_name] = {
                "df": model_df,
                "plot_name": plot_name
            }

    # Generate plots
    plot_metrics_vs_timestep(
        all_model_data=all_model_data,
        models_to_plot=cfg.comparisons,
        metric_col="seq_recovery",
        y_label="Sequence Recovery",
        title="Sequence Recovery vs. Timestep",
        out_path=base_out_dir / "seq_recovery_vs_timestep.png"
    )

    plot_metrics_vs_timestep(
        all_model_data=all_model_data,
        models_to_plot=cfg.comparisons,
        metric_col="avg_ca_plddt",
        y_label="Average CA pLDDT",
        title="Average pLDDT vs. Timestep",
        out_path=base_out_dir / "plddt_vs_timestep.png"
    )


def plot_metrics_vs_timestep(
    all_model_data: dict,
    models_to_plot: list,
    metric_col: str,
    y_label: str,
    title: str,
    out_path: Path
) -> None:
    """
    Creates a line plot of a given metric vs. timestep for multiple models.
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for model_name in models_to_plot:
        if model_name in all_model_data:
            data = all_model_data[model_name]
            df = data["df"]
            plot_name = data["plot_name"]
            ax.plot(df["timestep"], df[metric_col], marker='o', linestyle='-', label=plot_name)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.5)
    ax.legend()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
