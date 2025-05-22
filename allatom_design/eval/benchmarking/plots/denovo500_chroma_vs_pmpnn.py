import re
import pickle
from pathlib import Path

import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="denovo500_chroma_vs_pmpnn")
def main(cfg: DictConfig) -> None:
    """
    Reads Chroma and ProteinMPNN pickle files containing sc_ca_rmsd/sc_ca_tm data,
    then plots scatter comparisons for each length (L100, L200, L300, L400, L500).
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(cfg.chroma_pickle, "rb") as f:
        chroma_data = pickle.load(f)

    with open(cfg.pmpnn_pickle, "rb") as f:
        pmpnn_data = pickle.load(f)

    def parse_pickle(data_dict):
        """
        Each key in data_dict is expected to look like:
          '/.../L500_99.pdb'
        We'll parse 'L500', '99', and extract sc_ca_rmsd, sc_ca_tm.
        """
        rows = []
        for full_path, val in data_dict.items():
            # Example: full_path = "/scratch/.../L500_99.pdb"
            match = re.search(r"L(\d+)_(\d+)\.pdb", Path(full_path).name)
            if not match:
                continue

            length = int(match.group(1))
            sample_idx = int(match.group(2))

            # Each metric is stored in a 1-element tensor, e.g. tensor([16.2422])
            # Use .item() to get the float.
            sc_ca_rmsd = float(val["sc_info"]["sc_metrics"]["sc_ca_rmsd"].item())
            sc_ca_tm   = float(val["sc_info"]["sc_metrics"]["sc_ca_tm"].item())

            rows.append({
                "length": length,
                "sample_idx": sample_idx,
                "sc_ca_rmsd": sc_ca_rmsd,
                "sc_ca_tm": sc_ca_tm
            })

        return pd.DataFrame(rows)

    df_chroma = parse_pickle(chroma_data)
    df_pmpnn  = parse_pickle(pmpnn_data)

    df_chroma.rename(columns={
        "sc_ca_rmsd": "sc_ca_rmsd_chroma",
        "sc_ca_tm":   "sc_ca_tm_chroma"
    }, inplace=True)

    df_pmpnn.rename(columns={
        "sc_ca_rmsd": "sc_ca_rmsd_pmpnn",
        "sc_ca_tm":   "sc_ca_tm_pmpnn"
    }, inplace=True)

    merged_df = pd.merge(
        df_chroma,
        df_pmpnn,
        on=["length", "sample_idx"],
        how="inner"
    )

    # --- Generate scatter plots ---
    # We'll plot two metrics (sc_ca_rmsd and sc_ca_tm), for each length in {100,200,300,400,500}.
    metrics = ["sc_ca_rmsd", "sc_ca_tm"]

    for metric in metrics:
        x_col = f"{metric}_chroma"
        y_col = f"{metric}_pmpnn"

        if metric == "sc_ca_rmsd":
            title = "scRMSD"
            step = 2.0
        elif metric == "sc_ca_tm":
            title = "scTM"
            step = 0.1

        # Generate a plot per length
        for length in sorted(merged_df["length"].unique()):
            sub_df = merged_df[merged_df["length"] == length].copy()

            # Prepare figure
            plt.figure(figsize=(8, 8))
            ax = plt.gca()

            # Scatter
            x = sub_df[x_col].values
            y = sub_df[y_col].values
            ax.scatter(x, y, s=4)

            # Calculate plot limits
            lower_lim = 0.0
            upper_lim = max(np.max(x), np.max(y))
            ax.set_xlim(lower_lim, upper_lim)
            ax.set_ylim(lower_lim, upper_lim)

            # Diagonal reference line
            ax.plot([lower_lim, upper_lim], [lower_lim, upper_lim], 'k--')

            # Annotate with the sample index only
            for _, row in sub_df.iterrows():
                sample_number = int(row["sample_idx"])
                ax.annotate(
                    str(sample_number),
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(0, -8),
                    ha='center',
                    fontsize=6
                )

            # Axis labels and title
            ax.set_xlabel("Chroma")
            ax.set_ylabel("ProteinMPNN")
            ax.set_title(f"{title}, L={length}, n={len(sub_df)}")
            plt.grid(True, alpha=0.5)

            ax.set_xticks(np.arange(0, upper_lim + 1e-6, step))
            ax.set_yticks(np.arange(0, upper_lim + 1e-6, step))

            # Save plot
            out_name = f"L{length}_{metric}.png"
            plt.savefig(str(Path(cfg.out_dir) / out_name), dpi=300)
            plt.close()

    print(f"Plots saved to: {cfg.out_dir}")

if __name__ == "__main__":
    main()
