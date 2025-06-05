from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colorbar import ColorbarBase
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs/eval/plots", config_name="plot_potts_energies")
def main(cfg: DictConfig) -> None:
    """
    Plot potts energies
    """
    # Create the base output directory
    Path(cfg.base_out_dir).mkdir(parents=True, exist_ok=True)

    # Dump the entire config into the base output directory for reference
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.base_out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)
        
    # Load in score csv
    score_df = pd.read_csv(cfg.score_csv)
    
    # Subset to subset_pdb_names
    if cfg.subset_pdb_names is not None:
        with open(cfg.subset_pdb_names, "r") as f:
            subset_pdbs = [line.strip() for line in f.readlines()]
        subset_pdbs = [Path(pdb_name).stem for pdb_name in subset_pdbs]
        score_df = score_df[score_df["pdb_name"].isin(subset_pdbs)]
    
    # load in s6_a0 sc csv
    s6_a0_model_csv_data = [x for x in cfg.model_csvs if x["model_name"] == "s6_a0_t0.7"]
    s6_a0_sc_df = pd.read_csv(s6_a0_model_csv_data[0]["csv"])
    
    # merge to get self-consistency metrics
    score_df = score_df.merge(s6_a0_sc_df, left_on="sample_pdb_key", right_on="record_id", how="left")
    score_df["conformer_num"] = score_df["bb_pdb_key"].apply(lambda x: int(x.split("_")[-1]) if "_" in x else -1)
    # only use conformers up to 15 exclusive
    score_df = score_df[score_df["conformer_num"] < 15]
    
    idx_min = score_df.groupby("sample_pdb_key")["U"].idxmin()
    min_U_rows = score_df.loc[idx_min]
    result = min_U_rows[["sample_pdb_key", "bb_pdb_key", "U"]]
    
    # Plot
    plot_potts_energies(score_df, cfg.base_out_dir)
    
    # Save
    score_df.to_csv(Path(cfg.base_out_dir, "score_df.csv"), index=False)
    

def plot_potts_energies(score_df: pd.DataFrame, out_dir: str):
    # Get unique pdb names
    pdb_names = score_df["pdb_name"].unique()
    num_pdbs = len(pdb_names)

    # Create subplots; one column per unique pdb_name
    fig, axes = plt.subplots(
        nrows=1, ncols=num_pdbs, figsize=(5 * num_pdbs, 5), sharey=False
    )

    # If there's only one pdb_name, axes might not be iterable
    if num_pdbs == 1:
        axes = [axes]

    # A color cycle or palette for the sample_pdb_keys
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, pdb_name in zip(axes, pdb_names):
        # Subset rows for this pdb_name
        sub_df = score_df[score_df["pdb_name"] == pdb_name]

        # Collect all unique sample_pdb_keys for this pdb_name
        sample_keys = sub_df["sample_pdb_key"].unique()

        for i, sample_key in enumerate(sample_keys):
            # Further subset by sample_pdb_key
            sample_sub_df = sub_df[sub_df["sample_pdb_key"] == sample_key]

            # Split out row(s) where bb_pdb_key == pdb_name (to mark with a star)
            star_rows = sample_sub_df[sample_sub_df["bb_pdb_key"] == pdb_name]
            circle_rows = sample_sub_df[sample_sub_df["bb_pdb_key"] != pdb_name]
            
            # plot average as a square
            mean_U = sub_df[sub_df["sample_pdb_key"] == sample_key]["U"].mean()
            plddt = sample_sub_df["avg_ca_plddt"].iloc[0]


            # Pick a color from the color cycle
            color = color_cycle[i % len(color_cycle)]

            # Plot normal rows as circles
            ax.scatter(
                circle_rows["avg_ca_plddt"],
                circle_rows["U"],
                color=color,
                label=sample_key,
                alpha=0.7,
                edgecolor="none",
                s=40,
            )
            
            # Plot a filled square at (plddt, mean_U)
            ax.scatter(
                plddt,
                mean_U,
                color=color,
                marker="s",
                s=100,
                edgecolor="k",
                linewidth=0.7,
                alpha=1.0,
                zorder=5,
            )

            # Plot star rows (if any)
            if len(star_rows) > 0:
                ax.scatter(
                    star_rows["avg_ca_plddt"],
                    star_rows["U"],
                    color=color,
                    marker="*",
                    s=250,  # Make the star bigger
                    edgecolor="k",
                    linewidth=0.5,
                    alpha=0.9,
                )

        # Title, labels, etc.
        ax.set_title(pdb_name)
        ax.set_xlabel("avg_ca_plddt (↓ to the right)")
        ax.set_ylabel("Potts Energy (U)")

        # Invert the x-axis so that 1.0 is on the left and 0.0 on the right
        ax.set_xlim(1.0, 0.0)
        # set ylim to min and max of U in this pdb
        ax.set_ylim(min(sub_df["U"]) * 1.1, max(sub_df["U"]) * 0.9)

    plt.tight_layout()
    plt.savefig(Path(out_dir, "potts_energies.png"))
    plt.close()
    
    
if __name__ == "__main__":
    main()
