from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
import glob


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

    # Load in each mpnn_design_stats file
    target_dirs = glob.glob(f"{cfg.bindcraft_data_dir}/*/")
    # DEBUG
    target_dirs = [x for x in target_dirs if "PD" in x]

    mpnn_design_stats_df = pd.concat([pd.read_csv(f"{target_dir}/mpnn_design_stats.csv") for target_dir in target_dirs])
    # print(mpnn_design_stats_df.head())




if __name__ == "__main__":
    main()