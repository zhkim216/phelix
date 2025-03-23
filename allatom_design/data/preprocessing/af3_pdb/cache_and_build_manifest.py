#!/usr/bin/env python
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allatom_design.data.preprocessing.preprocessing_utils import (
    cache_examples, get_lengths_from_cached, get_pdb_file_from_key)


@hydra.main(config_path="../../../configs/data/preprocessing/af3_pdb", config_name="cache_and_build_manifest", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Reads PDB keys from .list files for each phase, caches the examples,
    and writes a single CSV containing pdb_name, phase, seq_length, and cluster_id.
    """
    # Collect all entries in a list of dicts to concatenate later
    manifest_info = []

    # For each phase, read pdb_keys, cache them, then compute lengths
    for phase in cfg.phases:
        pdb_keys_file = f"{cfg.pdb_path}/{phase}_pdb_keys.list"
        with open(pdb_keys_file) as f:
            pdb_keys = np.array(f.read().splitlines())

        pdb_key_to_pdb_file = {pdb_key: get_pdb_file_from_key(cfg.pdb_path, phase, pdb_key) for pdb_key in pdb_keys}

        cache_dir = cache_examples(
            pdb_key_to_pdb_file=pdb_key_to_pdb_file,
            pdb_path=cfg.pdb_path,
            overwrite_cache=cfg.overwrite_cache,
            num_workers=cfg.num_workers
        )

        pdb_key_to_length = get_lengths_from_cached(pdb_keys, cache_dir, num_workers=cfg.num_workers)
        pdb_key_to_cluster_id = get_cluster_ids(pdb_keys)
        for pdb_key in pdb_keys:
            manifest_info.append({"pdb_key": pdb_key, "phase": phase, "seq_length": pdb_key_to_length[pdb_key], "cluster_id": pdb_key_to_cluster_id[pdb_key]})

    # Write out manifest to CSV
    df = pd.DataFrame(manifest_info)
    out_csv = f"{cfg.pdb_path}/pdb_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote dataset manifest to {out_csv}")


def get_cluster_ids(pdb_keys: list[str]) -> Dict[str, int]:
    """
    Get cluster ID from each pdb key. In AF3 PDB, we stored the cluster ID in the pdb key.
    """
    return {pdb_key: pdb_key.split('_')[-1] for pdb_key in pdb_keys}



if __name__ == "__main__":
    main()
