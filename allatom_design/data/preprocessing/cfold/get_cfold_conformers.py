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

    # Save both the train and test files to a conformer directory
    mapping_data = []
    for train_file in tqdm(train_files, desc="Processing train files"):
        train_id, test_id = Path(train_file).stem.split("_")
        test_file = f"{cfg.cfold_test_confs_dir}/{test_id}_{train_id}.pdb"

        if Path(test_file).exists():
            # Make conformer directory and save train + test files to it
            train_stem = Path(train_file).stem.lower()
            test_stem = Path(test_file).stem.lower()
            conformer_dir = f"{conformer_out_dir}/{train_stem}"
            Path(conformer_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(train_file, f"{conformer_dir}/{train_stem}.pdb")
            shutil.copy(test_file, f"{conformer_dir}/{test_stem}.pdb")
            mapping_data.append({"pdb_key": train_stem, "conformer_dir": Path(conformer_dir).name})
            mapping_data.append({"pdb_key": test_stem, "conformer_dir": Path(conformer_dir).name})

            # Also save all PDB files to a single directory
            all_pdbs_dir = f"{cfg.out_dir}/pdbs"
            Path(all_pdbs_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(train_file, f"{all_pdbs_dir}/{train_stem}.pdb")
            shutil.copy(test_file, f"{all_pdbs_dir}/{test_stem}.pdb")
        else:
            print(f"Matching test file {test_file} does not exist, skipping...")

    print(f"Found {len(mapping_data) // 2} conformers")

    mapping_df = pd.DataFrame(mapping_data).sort_values(by="conformer_dir").reset_index(drop=True)
    mapping_df["source_pdb_key"] = mapping_df["pdb_key"].str.split("_").str[0]
    mapping_df.to_csv(f"{cfg.out_dir}/conformer_mapping.csv", index=False)

    # Get pdb names to hold out for testing based on boltz_v2 manifest
    holdout_pdb_keys_dir = f"{cfg.out_dir}/holdout_pdb_keys"
    Path(holdout_pdb_keys_dir).mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(Path(cfg.boltz_v2_manifest))
    records = manifest.records

    test_holdout_pdb_keys = mapping_df["source_pdb_key"].tolist()

    # only hold out pdb names that are in the boltz_v2 manifest
    record_ids = [r.id for r in records]
    test_holdout_pdb_keys = set([id for id in test_holdout_pdb_keys if id in record_ids])
    print(f"Writing {len(test_holdout_pdb_keys)} holdout record IDs to file")

    with open(f"{holdout_pdb_keys_dir}/holdout_pdb_keys.txt", "w") as f:
        for key in test_holdout_pdb_keys:
            f.write(f"{key}\n")


if __name__ == "__main__":
    main()