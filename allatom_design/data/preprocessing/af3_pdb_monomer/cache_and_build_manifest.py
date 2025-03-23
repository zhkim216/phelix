#!/usr/bin/env python
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from allatom_design.data.preprocessing.caching_utils import (
    cache_examples, get_lengths_from_cached, get_pdb_file_from_key)


@hydra.main(config_path="../../../configs/data/preprocessing/af3_pdb_monomer", config_name="cache_and_build_manifest", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Reads PDB keys from an input CSV file, caches the examples,
    and writes a pdb manifest CSV containing useful information, including pdb_key, phase, and seq_length.

    Here, we also build eval2 keys from train keys.
    """
    input_pdb_key_df = pd.read_csv(cfg.input_pdb_key_csv)

    pdb_keys = input_pdb_key_df["pdb_key"].tolist()
    phases = input_pdb_key_df["phase"].tolist()

    # Cache examples
    pdb_key_to_pdb_file = {pdb_key: get_pdb_file_from_key(cfg.pdb_path, phase, pdb_key) for pdb_key, phase in zip(pdb_keys, phases)}
    cache_dir = cache_examples(
        pdb_key_to_pdb_file=pdb_key_to_pdb_file,
        pdb_path=cfg.pdb_path,
        overwrite_cache=cfg.overwrite_cache,
        num_workers=cfg.num_workers
    )

    # Also save names of the original PDB files
    input_pdb_key_df["pdb_name"] = input_pdb_key_df["pdb_key"].map(pdb_key_to_pdb_file).apply(lambda x: Path(x).name)

    # Get sequence lengths
    pdb_key_to_length = get_lengths_from_cached(pdb_keys, cache_dir, num_workers=cfg.num_workers)

    # Build eval2 keys
    train_df = input_pdb_key_df[input_pdb_key_df["phase"] == "train"]
    eval2_df = train_df.sample(n=cfg.n_eval2, random_state=cfg.seed)
    input_pdb_key_df.loc[eval2_df.index, "phase"] = "eval2"

    # Build manifest
    manifest_df = input_pdb_key_df.copy()
    manifest_df["seq_length"] = pdb_key_to_length

    # Write out manifest to CSV
    manifest_csv = f"{cfg.pdb_path}/pdb_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Wrote dataset manifest to {manifest_csv}")

    # Save out the original PDB names for train, eval, and eval2 as separate lists
    for phase in ["train", "eval", "eval2"]:
        pdb_names = input_pdb_key_df[input_pdb_key_df["phase"] == phase]["pdb_name"]
        pdb_names.to_csv(f"{cfg.pdb_path}/{phase}_pdb_names.list", index=False, header=False)
        print(f"Wrote {phase} pdb names to {cfg.pdb_path}/{phase}_pdb_names.list")


if __name__ == "__main__":
    main()
