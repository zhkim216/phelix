#!/usr/bin/env python
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from allatom_design.data.preprocessing.preprocessing_utils import (
    cache_examples, get_pdb_file_from_key, get_radius_of_gyration_from_cached)


@hydra.main(config_path="../../../configs/data/preprocessing/augmented_af3_monomer_v2", config_name="cache_and_build_manifest", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Reads PDB keys from an input CSV file, caches the examples,
    and writes a pdb manifest CSV containing useful information.

    Here, we also build eval2 keys from train keys.
    """
    df = pd.read_csv(cfg.input_pdb_key_csv)
    df["pdb_key"] = df["pdb_name"].apply(lambda x: Path(x).stem)

    # Cache examples
    pdb_keys = df["pdb_key"].tolist()
    phases = df["phase"].tolist()
    pdb_key_to_pdb_file = {pdb_key: get_pdb_file_from_key(cfg.pdb_path, phase, pdb_key) for pdb_key, phase in zip(pdb_keys, phases)}
    cache_dir = cache_examples(
        pdb_key_to_pdb_file=pdb_key_to_pdb_file,
        pdb_path=cfg.pdb_path,
        overwrite_cache=cfg.overwrite_cache,
        num_workers=cfg.num_workers
    )

    # Compute relative radius of gyration
    pdb_key_to_rog = get_radius_of_gyration_from_cached(pdb_keys, cache_dir, num_workers=cfg.num_workers)
    df["radius_of_gyration"] = df["pdb_key"].map(pdb_key_to_rog)
    df["ideal_rad"] = 2.24 * (df["seq_length"] ** 0.392)  # Dill et al. https://www.pnas.org/doi/full/10.1073/pnas.1114477108
    df["rel_rog"] = df["radius_of_gyration"] / df["ideal_rad"]  # Verkuil et al.  https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1

    # Build eval2 keys
    train_df = df[df["phase"] == "train"]
    eval2_df = train_df.sample(n=cfg.n_eval2, random_state=cfg.seed)
    df.loc[eval2_df.index, "phase"] = "eval2"

    # Build manifest
    manifest_df = df.copy()

    # Write out manifest to CSV
    manifest_csv = f"{cfg.pdb_path}/pdb_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Wrote dataset manifest to {manifest_csv}")

    # Save out the original PDB names for train, eval, and eval2 as separate lists
    for phase in ["train", "eval", "eval2"]:
        pdb_names = df[df["phase"] == phase]["pdb_name"]
        pdb_names.to_csv(f"{cfg.pdb_path}/{phase}_pdb_names.list", index=False, header=False)
        print(f"Wrote {phase} pdb names to {cfg.pdb_path}/{phase}_pdb_names.list")


if __name__ == "__main__":
    main()
