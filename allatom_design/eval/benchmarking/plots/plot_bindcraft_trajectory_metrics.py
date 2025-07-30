import glob
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

target_name_to_bc = {
    "PD-L1": "PD-L1",
    "IFNAR2": "IFNAR2",
    "CD45": "CD45",
    "BBF-14": "BBF-14",
    "CrSAS-6": "SAS6",
    "DerF7": "DerF7",
    "DerF21": "DerF21",
    "SpCas9": "SpCas9",
}


target_name_to_farfalle = {
    "PD-L1": ["1_PD-L1"],
    "IFNAR2": ["2_IFNAR2"],
    "CD45": ["3_CD45(d2)", "4_CD45(d3-4)"],
    "BBF-14": ["6_BBF-14(1)", "7_BBF-14(2)"],
    "CrSAS-6": ["8_CrSAS-6(1)", "9_CrSAS-6(2)"],
    "DerF7": ["10_Derf7"],
    "DerF21": ["11_Derf21"],
    "SpCas9": ["13_SpCas9"],
}

@hydra.main(version_base=None, config_path="../../../configs/eval/benchmarking/plots", config_name="plot_bindcraft_trajectory_metrics")
def main(cfg: DictConfig) -> None:
    """
    Plot trajectory metrics for BindCraft data.
    """
    # Create the base output directory
    base_out_dir = Path(cfg.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(base_out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    target_names = ["PD-L1", "IFNAR2", "CD45", "BBF-14", "CrSAS-6", "DerF7", "DerF21", "SpCas9"]

    # Load in BindCraft results
    df_list = []
    for target_name in target_names:
        target_dir = target_name_to_bc[target_name]
        df = pd.read_csv(f"{cfg.bindcraft_data_dir}/{target_dir}/mpnn_design_stats.csv")
        df["target_name"] = target_name
        df = get_first_k_mpnn_design(df, k=2)
        df_list.append(df)
    mpnn_design_stats_df = pd.concat(df_list)

    mpnn_design_stats_df["binder_rmsd"] = mpnn_design_stats_df[["1_Binder_RMSD", "2_Binder_RMSD"]].min(axis=1)
    mpnn_design_stats_df["complex_i_ptm"] = mpnn_design_stats_df[["1_i_pTM", "2_i_pTM"]].max(axis=1)
    mpnn_design_stats_df["complex_i_pae"] = mpnn_design_stats_df[["1_i_pAE", "2_i_pAE"]].min(axis=1)
    mpnn_design_stats_df["complex_plddt"] = mpnn_design_stats_df[["1_pLDDT", "2_pLDDT"]].max(axis=1)

    # get the best design across 2 MPNN targets
    mpnn_design_stats_df = mpnn_design_stats_df.groupby("design_name").agg({
        "binder_rmsd": "min",
        "complex_i_ptm": "max",
        "complex_i_pae": "min",
        "complex_plddt": "max",
        "target_name": "first",
    }).reset_index()

    # Load in Protpardelle-1c results
    farfalle_df_list = []
    for target_name, farfalle_files in target_name_to_farfalle.items():
        for farfalle_file in farfalle_files:
            df = pd.read_csv(f"{cfg.farfalle_bindcraft_dir}/{farfalle_file}_alphafold_results_all.csv")
            df["target_name"] = target_name
            farfalle_df_list.append(df)
    farfalle_df = pd.concat(farfalle_df_list)

    farfalle_df["binder_rmsd"] = farfalle_df[["model_0_binder_rmsd", "model_1_binder_rmsd"]].min(axis=1)
    farfalle_df["complex_i_ptm"] = farfalle_df[["model_0_complex_i_ptm", "model_1_complex_i_ptm"]].max(axis=1)
    farfalle_df["complex_i_pae"] = farfalle_df[["model_0_complex_i_pae", "model_1_complex_i_pae"]].min(axis=1)
    farfalle_df["complex_plddt"] = farfalle_df[["model_0_complex_plddt", "model_1_complex_plddt"]].max(axis=1)

    # take best across 2 MPNN targets
    farfalle_df = farfalle_df.groupby("pdb_path").agg({
        "binder_rmsd": "min",
        "complex_i_ptm": "max",
        "complex_i_pae": "min",
        "complex_plddt": "max",
        "target_name": "first",
    }).reset_index()

    # Plotting
    metrics_to_plot = ["binder_rmsd", "complex_i_ptm", "complex_i_pae", "complex_plddt"]

    plot_settings = {
        "colors": {
            "BindCraft": "#88CCEE",
            "Protpardelle-1c": "#44AA99",
        },
        "xlabel": "Target name",
        "titles": {
            "binder_rmsd": "Binder RMSD per target",
            "complex_i_ptm": "i_pTM per target",
            "complex_i_pae": "i_pAE per target",
            "complex_plddt": "pLDDT per target",
        },
        "ylabels": {
            "binder_rmsd": "Binder RMSD (Å)",
            "complex_i_ptm": "i_pTM",
            "complex_i_pae": "i_pAE",
            "complex_plddt": "pLDDT",
        },
        "ylims": {
            "binder_rmsd": [0, 10],
            "complex_i_ptm": [0.0, 1.0],
            "complex_i_pae": [0.0, 1.0],
            "complex_plddt": [0.0, 1.0],
        },
        "yticks": {
            "binder_rmsd": np.arange(0, 11, 1),
            "complex_i_ptm": np.arange(0, 1.1, 0.1),
            "complex_i_pae": np.arange(0, 1.1, 0.1),
            "complex_plddt": np.arange(0, 1.1, 0.1),
        },
    }

    thresholds = {
        "binder_rmsd": 3.5,
        "complex_i_ptm": 0.5,
        "complex_i_pae": 0.35,
        "complex_plddt": 0.8,
    }

    # Plotting
    metrics_to_plot = ["binder_rmsd", "complex_i_ptm", "complex_i_pae", "complex_plddt"]
    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 6))

        # Prepare data for plotting
        bindcraft_plot_df = mpnn_design_stats_df[["target_name", metric]].copy()
        bindcraft_plot_df["Method"] = "BindCraft"

        farfalle_plot_df = farfalle_df[["target_name", metric]].copy()
        farfalle_plot_df["Method"] = "Protpardelle-1c"

        combined_df = pd.concat([bindcraft_plot_df, farfalle_plot_df])

        # Create the plot
        ax = sns.boxplot(
            x="target_name",
            y=metric,
            hue="Method",
            data=combined_df,
            palette=plot_settings["colors"],
            width=0.6,
            order=target_names,
        )
        ax.get_legend().set_title(None)

        # Add threshold lines
        if metric in thresholds:
            plt.axhline(y=thresholds[metric], color='#CC6677', linestyle='--', linewidth=1.5)

        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.title(plot_settings["titles"].get(metric, f"{metric.replace('_', ' ').title()} Comparison"), fontsize=14)
        # plt.xlabel(plot_settings["xlabel"])
        plt.xlabel("")
        plt.ylabel(plot_settings["ylabels"].get(metric, metric.replace("_", " ").title()), fontsize=12)

        # Set optional y-limits and y-ticks from config
        if metric in plot_settings["ylims"]:
            plt.ylim(plot_settings["ylims"][metric])
        if metric in plot_settings["yticks"]:
            plt.yticks(plot_settings["yticks"][metric], fontsize=12)

        plt.xticks(rotation=0, ha="center", fontsize=12)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{base_out_dir}/{metric}_comparison.png", dpi=300)
        plt.savefig(f"{base_out_dir}/{metric}_comparison.pdf", dpi=300)
        plt.close()



def get_first_k_mpnn_design(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Get the first k mpnn designs from the BindCraft data.
    """
    df["mpnn_design_num"] = df["Design"].str.rsplit("_").str[-1].str.replace("mpnn", "").astype(int)
    df["design_name"] = df["Design"].str.rsplit("_").str[:-1].str.join("_")
    df = df[df["mpnn_design_num"] <= k]
    return df


    # 1: PD-L1
    # pd_l1_df = pd.read_csv(f"{cfg.bindcraft_data_dir}/PD-L1/mpnn_design_stats.csv")
    # pd_l1_df = pd_l1_df[pd_l1_df["Target_Hotspot"] == "A54,A56,A66,A115"].reset_index(drop=True)  # subset to correct hotspot
    # pd_l1_df = get_first_mpnn_design(pd_l1_df)

    # # 2: IFNAR2
    # ifnar2_df = pd.read_csv(f"{cfg.bindcraft_data_dir}/IFNAR2/mpnn_design_stats.csv")
    # ifnar2_df = ifnar2_df[ifnar2_df["Target_Hotspot"] == "A52,A80,A82,A84,A96,A98"].reset_index(drop=True)  # subset to correct hotspot
    # ifnar2_df = get_first_mpnn_design(ifnar2_df)

    # # 3: CD45(d2)
    # cd45_d2_df = pd.read_csv(f"{cfg.bindcraft_data_dir}/CD45/mpnn_design_stats.csv")
    # cd45_d2_df = cd45_d2_df[cd45_d2_df["TargetSettings"] == "CD45_d2"].reset_index(drop=True)
    # cd45_d2_df = get_first_mpnn_design(cd45_d2_df)

    # # 4: CD45(d3-4)
    # cd45_d3_4_df = pd.read_csv(f"{cfg.bindcraft_data_dir}/CD45/mpnn_design_stats.csv")
    # cd45_d3_4_df = cd45_d3_4_df[cd45_d3_4_df["TargetSettings"] == "CD45_d3-d4"].reset_index(drop=True)
    # cd45_d3_4_df = get_first_mpnn_design(cd45_d3_4_df)

    # # 6. BBF-14
    # bbf14_df = pd.read_csv(f"{cfg.bindcraft_data_dir}/BBF-14/mpnn_design_stats.csv")


tol_cblind = [
    "#332288",  # dark blue
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
]

tol_extended_16 = [
    "#332288",  # Tol - dark blue
    "#88CCEE",  # Tol - light blue
    "#44AA99",  # Tol - teal
    "#117733",  # Tol - green
    "#999933",  # Tol - olive
    "#DDCC77",  # Tol - sand
    "#CC6677",  # Tol - rose
    "#882255",  # Tol - wine
    "#AA4499",  # Tol-inspired magenta
    "#661100",  # deep brown
    "#6699CC",  # muted blue
    "#888888",  # grey (neutral)
    "#E69F00",  # Okabe-Ito - orange
    "#56B4E9",  # Okabe-Ito - sky blue
    "#009E73",  # Okabe-Ito - bluish green
    "#F0E442",  # Okabe-Ito - yellow
]


if __name__ == "__main__":
    main()