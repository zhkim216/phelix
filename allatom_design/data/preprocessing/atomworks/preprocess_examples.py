#!/usr/bin/env python3
"""
Preprocess examples from mmCIF files using AtomWorks.

Features:
- Batch processing with resume support for OOM resilience
- Sequential fallback when parallel processing fails
"""
from __future__ import annotations
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import hydra
import torch
import yaml
from atomworks.ml.datasets import PandasDataset
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from allatom_design.data.preprocessing.atomworks.sharding_utils import \
    take_shard, use_sharding
from allatom_design.data.transform.preprocess import preprocess_transform

# tame BLAS threads inside each worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@hydra.main(config_path="../../../configs_local/data/preprocessing/atomworks", config_name="preprocess_examples", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Process a set of mmCIFs using AtomWorks.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve the original config
    if not use_sharding(cfg.shard_id, cfg.num_shards) or (cfg.shard_id == 0):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg_dict, f)

    # Setup
    use_parallel = cfg.num_workers > 1

    ### Cache features ###
    cached_structure_dir = f"{cfg.out_dir}/cached_structures"
    Path(cached_structure_dir).mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.dataset.loader.cif_parser_args["cache_dir"] = cached_structure_dir

    cached_example_dir = f"{cfg.out_dir}/cached_examples"
    Path(cached_example_dir).mkdir(parents=True, exist_ok=True)
    cache_fn = partial(_cache_examples, cached_example_dir=cached_example_dir)

    # iterate over the dataset, and the caching will happen automatically
    struct_dataset = hydra.utils.instantiate(cfg.dataset, transform=preprocess_transform(**cfg.preprocess_cfg))
    indices = list(range(len(struct_dataset)))
    indices = take_shard(indices, shard_id=cfg.shard_id, num_shards=cfg.num_shards)
    
    print(f"Shard {cfg.shard_id}/{cfg.num_shards}: {len(indices)} examples to process.")
    
    # Progress tracking for resume
    progress_dir = Path(cfg.out_dir) / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / f"shard_{cfg.shard_id:05d}_progress.json"
    log_file = progress_dir / f"shard_{cfg.shard_id:05d}.log"
    
    # Load progress if resuming
    completed_indices: set[int] = set()
    completed_batches: set[int] = set()
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_indices = set(progress_data.get('completed_indices', []))
                completed_batches = set(progress_data.get('completed_batches', []))
            print(f"Resuming: found {len(completed_indices)} already processed, {len(completed_batches)} completed batches.")
        except Exception as e:
            print(f"Warning: could not load progress file: {e}")
    
    # Batch processing settings
    batch_size = getattr(cfg, 'batch_size', 100)
    max_tasks_per_child = getattr(cfg, 'max_tasks_per_child', 50)
    total_batches = (len(indices) + batch_size - 1) // batch_size
    
    # Track failed indices
    failed_indices: list[tuple[int, str]] = []

    if use_parallel:
        for batch_idx in range(total_batches):
            batch_num = batch_idx + 1
            
            # Skip already completed batches
            if batch_num in completed_batches:
                print(f"Skipping already completed batch {batch_num}/{total_batches}")
                continue
            
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            try:
                with ProcessPoolExecutor(
                    max_workers=cfg.num_workers, 
                    mp_context=mp.get_context("forkserver"),
                    initializer=_init_dataset, 
                    initargs=(cfg.dataset, cfg.preprocess_cfg),
                    max_tasks_per_child=max_tasks_per_child
                ) as executor:
                    for idx, result in zip(
                        batch_indices,
                        tqdm(executor.map(cache_fn, batch_indices, chunksize=1), 
                             total=len(batch_indices), 
                             desc=f"Caching examples (shard {cfg.shard_id}, batch {batch_num}/{total_batches})")
                    ):
                        if result is None:
                            failed_indices.append((idx, "processing_error"))
                        else:
                            completed_indices.add(idx)
                            
            except Exception as e:
                # Handle BrokenProcessPool or other pool errors gracefully
                print(f"WARNING: Batch {batch_num} failed with error: {repr(e)}")
                print(f"Falling back to sequential processing for this batch...")
                
                # Reinitialize dataset for fallback
                fallback_dataset = hydra.utils.instantiate(
                    cfg.dataset, transform=preprocess_transform(**cfg.preprocess_cfg)
                )
                
                for idx in batch_indices:
                    if idx in completed_indices:
                        continue  # Already processed
                    try:
                        result = cache_fn(idx, dataset=fallback_dataset)
                        if result is None:
                            failed_indices.append((idx, "fallback_error"))
                        else:
                            completed_indices.add(idx)
                    except Exception as fallback_e:
                        failed_indices.append((idx, repr(fallback_e)))
            
            # Update progress after each batch
            completed_batches.add(batch_num)
            try:
                with open(progress_file, 'w') as f:
                    json.dump({
                        'completed_indices': list(completed_indices),
                        'completed_batches': list(completed_batches)
                    }, f)
            except Exception:
                pass
    else:
        for idx in tqdm(indices, desc=f"Caching examples (shard {cfg.shard_id})"):
            if idx in completed_indices:
                continue
            try:
                result = cache_fn(idx, dataset=struct_dataset)
                if result is None:
                    failed_indices.append((idx, "processing_error"))
                else:
                    completed_indices.add(idx)
            except Exception as e:
                failed_indices.append((idx, repr(e)))
        
        # Update progress
        try:
            with open(progress_file, 'w') as f:
                json.dump({'completed_indices': list(completed_indices), 'completed_batches': []}, f)
        except Exception:
            pass
    
    # Log summary
    try:
        with open(log_file, 'w') as f:
            f.write(f"Shard {cfg.shard_id}: total={len(indices)}, completed={len(completed_indices)}, failed={len(failed_indices)}\n")
            if failed_indices:
                f.write("Failed indices:\n")
                for idx, reason in failed_indices:
                    f.write(f"  {idx}\t{reason}\n")
    except Exception:
        pass
    
    print(f"Shard {cfg.shard_id}: completed {len(completed_indices)}/{len(indices)}, failed {len(failed_indices)}")


def _cache_examples(idx: int,
                    cached_example_dir: str,
                    *,
                    dataset: PandasDataset | None = None) -> str:
    try:
        feats = dataset[idx] if dataset is not None else _DATASET[idx]  # indexing the dataset triggers structure caching
    except Exception as e:
        print(f"Error caching example {idx}: {e}")
        return None

    # save feats to disk
    pdb_id = feats["extra_info"]["pdb_id"]
    torch.save(feats, f"{cached_example_dir}/{pdb_id}.pt")
    return pdb_id

# Initialize the dataset in each worker so that the dataset is not pickled
_DATASET: PandasDataset | None = None

def _init_dataset(dataset_cfg: DictConfig = None,
                  preprocess_cfg: DictConfig = None):
    global _DATASET
    _DATASET = hydra.utils.instantiate(dataset_cfg, transform=preprocess_transform(**preprocess_cfg))


if __name__ == "__main__":
    main()
