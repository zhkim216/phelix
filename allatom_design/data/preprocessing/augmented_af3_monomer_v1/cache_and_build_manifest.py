#!/usr/bin/env python
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
    input_pdb_key_df["pdb_key"] = input_pdb_key_df["pdb_key"].str.rsplit("_", n=1).str[0]  # get rid of cluster id from pdb key

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

    # Build eval2 keys
    train_df = input_pdb_key_df[input_pdb_key_df["phase"] == "train"]
    eval2_df = train_df.sample(n=cfg.n_eval2, random_state=cfg.seed)
    input_pdb_key_df.loc[eval2_df.index, "phase"] = "eval2"

    # Build manifest
    manifest_df = input_pdb_key_df.copy()

    # Write out manifest to CSV
    out_csv = f"{cfg.pdb_path}/pdb_manifest.csv"
    manifest_df.to_csv(out_csv, index=False)
    print(f"Wrote dataset manifest to {out_csv}")


if __name__ == "__main__":
    main()
