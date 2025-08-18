#!/usr/bin/env python3
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import hydra
import torch
import yaml
from atomworks.ml.datasets.datasets import StructuralDatasetWrapper
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from allatom_design.data.transform.featurizer import featurizer

# tame BLAS threads inside each worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@hydra.main(config_path="../../../configs/data/preprocessing/atomworks", config_name="cache_features", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Process a set of mmCIFs using AtomWorks.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve the original config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Setup
    use_parallel = cfg.num_workers > 1
    dataset_name = Path(cfg.out_dir).stem

    ### Cache features ###
    cached_structure_dir = f"{cfg.out_dir}/cached_structures"
    Path(cached_structure_dir).mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.dataset.cif_parser_args["cache_dir"] = cached_structure_dir
        cfg.dataset.dataset.name = dataset_name

    cached_feats_dir = f"{cfg.out_dir}/cached_feats"
    Path(cached_feats_dir).mkdir(parents=True, exist_ok=True)
    cache_fn = partial(_cache_feats, cached_feats_dir=cached_feats_dir)

    # iterate over the dataset, and the caching will happen automatically
    struct_dataset = hydra.utils.instantiate(cfg.dataset, transform=featurizer())
    if use_parallel:
        indices = range(len(struct_dataset))
        with ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=mp.get_context("forkserver"),
                                 initializer=_init_dataset, initargs=(cfg.dataset,)) as executor:
            for _ in tqdm(executor.map(cache_fn, indices), total=len(struct_dataset), desc="Caching features"):
                pass
    else:
        for idx in tqdm(range(len(struct_dataset)), desc="Caching features"):
            cache_fn(idx, dataset=struct_dataset)

def _cache_feats(idx: int,
                 cached_feats_dir: str,
                 *,
                 dataset: StructuralDatasetWrapper | None = None) -> str:
    feats = dataset[idx] if dataset is not None else _DATASET[idx]  # indexing the dataset triggers structure caching

    # save feats to disk
    pdb_id = feats["extra_info"]["pdb_id"]
    torch.save(feats, f"{cached_feats_dir}/{pdb_id}.pt")
    return pdb_id

# Initialize the dataset in each worker so that the dataset is not pickled
_DATASET: StructuralDatasetWrapper | None = None

def _init_dataset(dataset_cfg: DictConfig):
    global _DATASET
    _DATASET = hydra.utils.instantiate(dataset_cfg, transform=featurizer())



if __name__ == "__main__":
    main()
