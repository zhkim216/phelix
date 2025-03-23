#!/usr/bin/env python
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from allatom_design.data.preprocessing.caching_utils import (
    cache_examples, get_pdb_file_from_key)


@hydra.main(config_path="../../../configs/data/preprocessing/augmented_af3_monomer_v1", config_name="cache_and_build_manifest", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Reads PDB keys from an input CSV file, caches the examples,
    and writes a pdb manifest CSV containing useful information.

    Here, we also build eval2 keys from train keys.
    """
    input_pdb_key_df = pd.read_csv(cfg.input_pdb_key_csv)

    # add cluster id to input_pdb_key_df
    pdb_key_to_cluster_id = get_cluster_ids(input_pdb_key_df["pdb_key"].tolist())
    input_pdb_key_df["cluster_id"] = input_pdb_key_df["pdb_key"].map(pdb_key_to_cluster_id)

    # remove cluster id from pdb key
    input_pdb_key_df["pdb_key"] = input_pdb_key_df["pdb_key"].str.rsplit("_", n=1).str[0]

    # Cache examples
    pdb_keys = input_pdb_key_df["pdb_key"].tolist()
    phases = input_pdb_key_df["phase"].tolist()
    pdb_key_to_pdb_file = {pdb_key: get_pdb_file_from_key(cfg.pdb_path, phase, pdb_key) for pdb_key, phase in zip(pdb_keys, phases)}
    cache_dir = cache_examples(
        pdb_key_to_pdb_file=pdb_key_to_pdb_file,
        pdb_path=cfg.pdb_path,
        overwrite_cache=cfg.overwrite_cache,
        num_workers=cfg.num_workers
    )

    # Also save names of the original PDB files
    input_pdb_key_df["pdb_name"] = input_pdb_key_df["pdb_key"].map(pdb_key_to_pdb_file).apply(lambda x: Path(x).name)

    # Build eval2 keys
    train_df = input_pdb_key_df[input_pdb_key_df["phase"] == "train"]
    eval2_df = train_df.sample(n=cfg.n_eval2, random_state=cfg.seed)
    input_pdb_key_df.loc[eval2_df.index, "phase"] = "eval2"

    # Build manifest
    manifest_df = input_pdb_key_df.copy()

    # Write out manifest to CSV
    manifest_csv = f"{cfg.pdb_path}/pdb_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Wrote dataset manifest to {manifest_csv}")

    # Save out the original PDB names for train, eval, and eval2 as separate lists
    for phase in ["train", "eval", "eval2"]:
        pdb_names = input_pdb_key_df[input_pdb_key_df["phase"] == phase]["pdb_name"]
        pdb_names.to_csv(f"{cfg.pdb_path}/{phase}_pdb_names.list", index=False, header=False)
        print(f"Wrote {phase} pdb names to {cfg.pdb_path}/{phase}_pdb_names.list")


def get_cluster_ids(pdb_keys: list[str]) -> dict[str, int]:
    """
    Get cluster ID from each pdb key. In augmented_af3_monomer_v1, we stored the cluster ID in the pdb key.
    """
    return {pdb_key: pdb_key.split('_')[-1] for pdb_key in pdb_keys}


if __name__ == "__main__":
    main()
