#!/usr/bin/env python3
import glob
import shutil
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.types import Manifest


@hydra.main(config_path="../../../configs/data/preprocessing/cfold", config_name="get_cfold_conformers", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given train and test conformations from Cfold, create a dataset of conformers by merging train and test files.
    """
    # Create dataset directory
    conformer_out_dir = f"{cfg.out_dir}/conformers"
    Path(conformer_out_dir).mkdir(parents=True, exist_ok=True)

    # Get train files
    train_files = glob.glob(f"{cfg.cfold_train_confs_dir}/*.pdb")

    # For each train file, get the corresponding test file
    mapping_data = []
    for train_file in tqdm(train_files, desc="Processing train files"):
        train_id, test_id = Path(train_file).stem.split("_")

        test_file = f"{cfg.cfold_test_confs_dir}/{test_id}_{train_id}.pdb"
        if Path(test_file).exists():
            # Make conformer directory and save train/test files to it
            conformer_dir = f"{conformer_out_dir}/{train_id.lower()}"
            Path(conformer_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(train_file, f"{conformer_dir}/{train_id.lower()}.pdb")
            shutil.copy(test_file, f"{conformer_dir}/{test_id.lower()}.pdb")
            mapping_data.append({"pdb_key": train_id.lower(), "conformer_dir": Path(conformer_dir).name})
            mapping_data.append({"pdb_key": test_id.lower(), "conformer_dir": Path(conformer_dir).name})
        else:
            print(f"Test file {test_file} does not exist")

    print(f"Found {len(mapping_data) // 2} conformers")

    mapping_df = pd.DataFrame(mapping_data).sort_values(by="conformer_dir").reset_index(drop=True)
    mapping_df.to_csv(f"{cfg.out_dir}/conformer_mapping.csv", index=False)

    # Save pdb_name_list
    pdb_name_list_dir = f"{cfg.out_dir}/pdb_name_lists"
    Path(pdb_name_list_dir).mkdir(parents=True, exist_ok=True)
    mapping_df["pdb_name"] = mapping_df["pdb_key"] + ".pdb"
    mapping_df["pdb_name"].to_csv(f"{pdb_name_list_dir}/all_pdb_names.txt", index=False, header=False)

    # Get pdb names to hold out for testing based on boltz_v2 manifest
    manifest = Manifest.load(Path(cfg.boltz_v2_manifest))
    records = manifest.records

    test_holdout_ids = mapping_df["pdb_key"].tolist()

    # only hold out pdb names that are in the boltz_v2 manifest
    record_ids = [r.id for r in records]
    test_holdout_ids = set([id for id in test_holdout_ids if id in record_ids])
    print(f"Writing {len(test_holdout_ids)} holdout record IDs to file")

    with open(f"{cfg.out_dir}/pdb_name_lists/holdout_pdb_names.txt", "w") as f:
        for id in test_holdout_ids:
            f.write(f"{id}.pdb\n")


if __name__ == "__main__":
    main()