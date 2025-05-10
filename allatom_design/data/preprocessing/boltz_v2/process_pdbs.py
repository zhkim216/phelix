#!/usr/bin/env python3
import glob
import json
import pickle
import shutil
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import rdkit
from joblib import Parallel, delayed
from omegaconf import DictConfig
from p_tqdm import p_umap
from tqdm import tqdm

from allatom_design.data.filter.static.ligand import ExcludedLigands
from allatom_design.data.filter.static.polymer import (ClashingChainsFilter,
                                                       ConsecutiveCA,
                                                       MinimumLengthFilter,
                                                       UnknownFilter)
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    Resource, fetch, finalize, pdb_to_mmcif, process_structure)
from allatom_design.eval.eval_utils.eval_setup_utils import start_redis


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_v2", config_name="process_pdbs", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Re-process the Boltz-1 RCSB dataset to get resolution info.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Static filters
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ConsecutiveCA(max_dist=10.0),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Set up CCD resource in Redis
    ## set default pickle properties
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)

    redis_host, redis_port = "localhost", 7777
    start_redis(redis_host, redis_port, cfg.software_path, cfg.ccd_rdb_path)
    resource = Resource(host=redis_host, port=redis_port)

    # Set up cluster resource in Redis
    redis_host, redis_port = "localhost", 7778
    start_redis(redis_host, redis_port, cfg.software_path, cfg.cluster_rdb_path)
    clusters_resource = Resource(host=redis_host, port=redis_port)

    # Fetch data
    mmcif_files = Path(cfg.mmcif_dir).rglob("*.cif")
    data = fetch(mmcif_files, max_file_size=cfg.max_file_size)

    use_parallel = cfg.num_workers > 1

    # Run processing
    processed_targets_dir = f"{cfg.out_dir}/processed_targets"
    Path(processed_targets_dir).mkdir(parents=True, exist_ok=True)
    if use_parallel:
        fn = partial(
            process_structure,
            resource=resource,
            outdir=Path(processed_targets_dir),
            filters=filters,
            clusters=clusters_resource,
        )
        p_umap(fn, data, num_cpus=cfg.num_workers, desc="Processing mmCIFs")
    else:
        for pdb in tqdm(data, desc="Processing mmCIFs"):
            process_structure(
                pdb,
                resource=resource,
                outdir=Path(processed_targets_dir),
                filters=filters,
                clusters=clusters_resource,
            )

    # Post‑processing to create manifest.json
    finalize(outdir=Path(processed_targets_dir))


if __name__ == "__main__":
    main()
