import argparse
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../configs/data/preprocessing", config_name="filter_afdb", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create output directory and preserver config
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Read the dataframe
    df = pd.read_csv(cfg.csv_file)

    # Filter
    if cfg.plddt_column is not None and cfg.plddt_threshold is not None:
        df = df[df[cfg.plddt_column] >= cfg.plddt_threshold]  # pLDDT threshold
    if cfg.rog_threshold is not None:
        df = df[df["radius_of_gyration"] <= cfg.rog_threshold]  # radius of gyration threshold

    if cfg.min_seq_len is not None:
        df = df[df["seq_len"] >= cfg.min_seq_len]  # min sequence length threshold

    if cfg.max_seq_len is not None:
        df = df[df["seq_len"] <= cfg.max_seq_len]  # max sequence length threshold

    # Save the full dataframe as a csv and the mmcif_file stems as a txt file
    df.to_csv(f"{cfg.out_dir}/afdb_filtered.csv", index=False)
    df["mmcif"] = df["mmcif_file"].apply(lambda x: Path(x).stem)
    df["mmcif"].to_csv(f"{cfg.out_dir}/afdb_filtered.txt", index=False, header=False)


if __name__ == "__main__":
    main()