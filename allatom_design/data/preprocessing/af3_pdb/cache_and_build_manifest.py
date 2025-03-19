#!/usr/bin/env python
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.data import load_feats_from_pdb
from allatom_design.data.pdb_utils import write_to_pdb


@hydra.main(config_path="../../../configs/data/preprocessing/af3_pdb", config_name="cache_examples", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Reads PDB keys from .list files for each phase, caches the examples,
    and writes a single CSV containing pdb_name, phase, seq_length, and cluster_id.
    """
    # Create the cache directory
    cache_dir = f"{cfg.pdb_path}/cached_examples"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Collect all entries in a list of dicts to concatenate later
    pdb_info = []

    # For each phase, read pdb_keys, cache them, then compute lengths
    for phase in cfg.phases:
        pdb_keys_file = f"{cfg.pdb_path}/{phase}_pdb_keys.list"
        with open(pdb_keys_file) as f:
            pdb_keys = np.array(f.read().splitlines())

        cache_examples(
            pdb_keys=pdb_keys,
            phase=phase,
            pdb_path=cfg.pdb_path,
            overwrite_cache=cfg.overwrite_cache,
            num_workers=cfg.num_workers
        )

        pdb_key_to_length = get_lengths(pdb_keys, cache_dir)
        pdb_key_to_cluster_id = get_cluster_ids(pdb_keys)
        for pdb_key in pdb_keys:
            pdb_info.append({"pdb_key": pdb_key, "phase": phase, "seq_length": pdb_key_to_length[pdb_key], "cluster_id": pdb_key_to_cluster_id[pdb_key]})

    # Combine into a single DataFrame and write out
    df = pd.DataFrame(pdb_info)
    out_csv = f"{cfg.pdb_path}/pdb_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote dataset manifest to {out_csv}")


def cache_examples(
    pdb_keys: list[str],
    phase: str,
    pdb_path: str,
    overwrite_cache: bool,
    num_workers: int
):
    """
    Reads in PDB files and caches the examples to disk.
    Cached files are stored in cached_examples/ in the pdb_path.
    """
    cache_dir = f"{pdb_path}/cached_examples"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"Caching examples to {cache_dir}...")

    print(f"Using {num_workers} Workers!")

    # Prepare arguments as tuples (pdb_key, cache_dir, overwrite_cache, pdb_path, phase) for process_pdb_key
    task_args = [(pdb_key, cache_dir, overwrite_cache, pdb_path, phase) for pdb_key in pdb_keys]

    # Use a Pool for parallel processing
    with Pool(processes=num_workers) as pool:
        # Use tqdm to display progress
        for _ in tqdm(pool.imap_unordered(process_pdb_key, task_args), total=len(task_args), desc="Caching PDBs"):
            pass

    print("Caching completed.")


def get_cluster_ids(pdb_keys: list[str]) -> Dict[str, int]:
    """
    Get cluster ID from each pdb key. In AF3 PDB, we stored the cluster ID in the pdb key.
    """
    return {pdb_key: pdb_key.split('_')[-1] for pdb_key in pdb_keys}


def get_lengths(pdb_keys: List[str], cache_dir: str) -> Dict[str, int]:
    """
    Computes sequence lengths for given PDB keys in parallel using joblib.
    Args:
        pdb_keys: List of PDB keys to process
        cache_dir: Directory containing cached examples
    Returns:
        Dictionary mapping PDB keys to their sequence lengths.
    """
    num_workers = 8
    print(f"Computing sequence lengths using {num_workers} workers...")
    parallel = Parallel(n_jobs=num_workers, verbose=0)
    jobs = [delayed(get_seq_length_from_cached)(pdb_key, cache_dir) for pdb_key in pdb_keys]
    results = parallel(tqdm(jobs, desc="Getting lengths", total=len(jobs)))
    return dict(results)


def get_seq_length_from_cached(pdb_key: str, cache_dir: str) -> Tuple[str, int]:
    """
    Helper function for parallel processing of sequence lengths.
    Args:
        pdb_key: The PDB key to process
        cache_dir: Directory containing cached examples
    Returns:
        Tuple of (pdb_key, sequence_length)
    """
    data_file = f"{cache_dir}/{pdb_key}.pt"
    example = torch.load(data_file, weights_only=True)
    seq_len = example["seq_mask"].sum().long().item()
    return (pdb_key, seq_len)


def get_pdb_data_file(pdb_path: str, phase: str, pdb_key: str) -> str:
    """
    Based on the pdb_key, return the path to the PDB file.
    """
    pdb_data_file = f"{pdb_path}/{phase}_mmcifs/{pdb_key[1:3]}/{pdb_key[:4]}-assembly1.cif"
    return pdb_data_file


def process_pdb_key(args):
    pdb_key, cache_dir, overwrite_cache, pdb_path, phase = args
    out_file = f"{cache_dir}/{pdb_key}.pt"
    if Path(out_file).exists() and not overwrite_cache:
        return  # Skip caching if file exists and overwrite_cache is False

    pdb_data_file = get_pdb_data_file(pdb_path, phase, pdb_key)

    chain_ids_override = None
    if Path(pdb_path).stem == "af3_pdb":
        chain_ids_override = pdb_key.split('_')[1]

    example = load_feats_from_pdb(pdb_data_file, chain_ids_override=chain_ids_override, max_conformers=1)
    torch.save(example, out_file)


def cached_example_to_pdb(pt_file: str, out_pdb_file: str, mode: str = "aa", conect: bool = False):
    """
    Load a cached PyTorch file (pt_file) with the expected keys:
      - "aatype"
      - "all_atom_positions"
      - "all_atom_mask"
      - "residue_index"
      - "chain_index"
      - optionally "b_factors"

    Write the structure to 'out_pdb_file' as a PDB file.
    """
    # Load the .pt file
    data = torch.load(pt_file, weights_only=True)

    # Extract required fields
    aatype = data["aatype"]  # shape [n]
    atom_positions = data["all_atom_positions"]  # shape [n, 37, 3]
    atom_mask = data["all_atom_mask"]  # shape [n, 37]
    residue_index = data["residue_index"]  # shape [n]
    chain_index = data["chain_index"]      # shape [n]

    # b_factors might not exist
    b_factors = data.get("b_factors", None)

    # Call the write_to_pdb function
    write_to_pdb(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
        filename=out_pdb_file,
        mode=mode,
        conect=conect,
    )


if __name__ == "__main__":
    main()
