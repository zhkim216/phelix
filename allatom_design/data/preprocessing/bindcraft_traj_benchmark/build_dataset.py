#!/usr/bin/env python3
import glob
import shutil
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../../../configs/data/preprocessing/bindcraft_traj_benchmark", config_name="build_dataset", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given the BindCraft data directory, create a dataset of trajectory pdbs for benchmarking sequence design models.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    L.seed_everything(cfg.seed)  # set seed

    # Get all target dirs
    target_dirs = natsorted(glob.glob(f"{cfg.bindcraft_data_dir}/*"))

    # Subsample and combine all trajectory dfs
    df = []
    for target_dir in target_dirs:
        target_name = Path(target_dir).name
        traj_df = pd.read_csv(f"{target_dir}/trajectory_stats.csv")

        # Subsample
        traj_df["target_name"] = target_name
        n_take = min(cfg.n_subsample, len(traj_df))
        df.append(traj_df.sample(n=n_take, random_state=cfg.seed))
    df = pd.concat(df)
    df.to_csv(f"{cfg.out_dir}/subsampled_trajectory_stats_all.csv", index=False)

    # Copy pdbs for sampled designs
    pdb_out_dir = f"{cfg.out_dir}/pdbs"
    Path(pdb_out_dir).mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        design = row["Design"]
        target_name = row["target_name"]

        traj_dir = f"{cfg.bindcraft_data_dir}/{target_name}/Trajectory/Relaxed"
        src = f"{traj_dir}/{design}.pdb"
        dst = f"{pdb_out_dir}/{design}.pdb"
        shutil.copyfile(src, dst)

    # Make pdb_name_list for subsampling to 20, 50 for each target
    pdb_name_list_out_dir = f"{cfg.out_dir}/pdb_name_lists"
    Path(pdb_name_list_out_dir).mkdir(parents=True, exist_ok=True)

    df_25 = df.groupby("target_name").sample(n=20, random_state=cfg.seed)
    df_25["pdb_name"] = df_25["Design"].apply(lambda x: f"{x}.pdb")
    df_25["pdb_name"].to_csv(f"{pdb_name_list_out_dir}/subset_20.txt", index=False, header=False)

    df_50 = df.groupby("target_name").sample(n=50, random_state=cfg.seed)
    df_50["pdb_name"] = df_50["Design"].apply(lambda x: f"{x}.pdb")
    df_50["pdb_name"].to_csv(f"{pdb_name_list_out_dir}/subset_50.txt", index=False, header=False)


if __name__ == "__main__":
    main()
