#!/usr/bin/env python3
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

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

    ### Cache examples ###
    cache_dir = f"{cfg.out_dir}/cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.dataset.cif_parser_args["cache_dir"] = cache_dir
        cfg.dataset.dataset.name = dataset_name

    # iterate over the dataset, and the caching will happen automatically
    struct_dataset = hydra.utils.instantiate(cfg.dataset)
    if use_parallel:
        indices = range(len(struct_dataset))
        with ProcessPoolExecutor(max_workers=cfg.num_workers, initializer=_init_dataset, initargs=(cfg.dataset,)) as executor:
            for _ in tqdm(executor.map(_touch, indices), total=len(struct_dataset), desc="Caching examples"):
                pass

    else:
        list(tqdm(struct_dataset, desc="Caching examples"))

_DATASET = None
def _init_dataset(dataset_cfg: DictConfig):
    # initialize the dataset in each worker so that the dataset is not pickled
    global _DATASET
    _DATASET = hydra.utils.instantiate(dataset_cfg)

def _touch(idx: int) -> int:
    _ = _DATASET[idx]  # indexing the dataset triggers caching
    return 0


if __name__ == "__main__":
    main()
