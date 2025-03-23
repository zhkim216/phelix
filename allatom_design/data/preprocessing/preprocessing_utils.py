#!/usr/bin/env python
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio.PDB.MMCIFParser import FastMMCIFParser
from joblib import Parallel, delayed
from tqdm import tqdm

from allatom_design.data.data import load_feats_from_pdb
from allatom_design.data.pdb_utils import write_to_pdb

mmcif_parser = FastMMCIFParser(auth_chains=True, auth_residues=False, QUIET=True)


def get_pdb_file_from_key(pdb_path: str, phase: str, pdb_key: str) -> str:
    """
    Based on the pdb_key, return the path to the PDB file.
    """
    dataset_name = Path(pdb_path).stem

    if dataset_name == "af3_pdb":
        # AF3 PDB dataset used for FAMPNN training
        pdb_file = f"{pdb_path}/{phase}_mmcifs/{pdb_key[1:3]}/{pdb_key[:4]}-assembly1.cif"
    elif dataset_name == "af3_pdb_monomer":
        # AF3 PDB monomer dataset
        pdb_file = f"{pdb_path}/preprocessing/residx_quality_control_af3_monomer/filtered_mmcifs/{pdb_key}.cif"
    elif dataset_name == "augmented_af3_monomer_v1":
        # Augmented AF3 monomer dataset
        pdb_file = f"{pdb_path}/esmfold_preds/{pdb_key}.pdb"
    elif dataset_name == "augmented_ingraham_cath_bugfree":
        # Tianyu's augmented dataset
        pdb_file = f"{pdb_path}/mpnn_esmfold/{pdb_key}"
        if not Path(pdb_file).exists():
            pdb_file = f"{pdb_path}/dne_mpnn/{pdb_key}"
    elif dataset_name == "augmented_af3_monomer_v2":
        # Augmented AF3 monomer dataset
        pdb_file = f"{pdb_path}/fampnn_multi/S20_nl0.1_n4_L32_256/preds/ca_aligned_struct_preds/esmfold_{pdb_key}.pdb"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if not Path(pdb_file).exists():
        raise FileNotFoundError(f"PDB file {pdb_file} not found")
    return pdb_file


def cache_examples(
    pdb_key_to_pdb_file: dict[str, str],
    pdb_path: str,
    overwrite_cache: bool,
    num_workers: int
) -> str:
    """
    Reads in PDB files and caches the examples to disk.
    Cached files are stored as {pdb_key}.pt in {pdb_path}/cached_examples.
    """
    cache_dir = f"{pdb_path}/cached_examples"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Caching examples to {cache_dir} with {num_workers} workers...")
    parallel = Parallel(n_jobs=num_workers, verbose=0)
    jobs = [delayed(cache_pdb_key)(pdb_key, pdb_file, pdb_path, cache_dir, overwrite_cache) for pdb_key, pdb_file in pdb_key_to_pdb_file.items()]
    list(parallel(tqdm(jobs, total=len(jobs), desc="Caching PDBs")))
    print("Caching completed.")
    return cache_dir


def cache_pdb_key(pdb_key: str, pdb_file: str, pdb_path: str, cache_dir: str, overwrite_cache: bool):
    out_file = f"{cache_dir}/{pdb_key}.pt"
    if Path(out_file).exists() and not overwrite_cache:
        return  # Skip caching if file exists and overwrite_cache is False

    chain_ids_override = None
    if Path(pdb_path).stem == "af3_pdb":
        # for AF3 PDB, we only load in the chains in the pdb key
        chain_ids_override = pdb_key.split('_')[1]
    example = load_feats_from_pdb(pdb_file, chain_ids_override=chain_ids_override, max_conformers=1)
    torch.save(example, out_file)


def get_lengths_from_cached(pdb_keys: list[str], cache_dir: str, num_workers: int) -> dict[str, int]:
    """
    Computes sequence lengths for given PDB keys in parallel using joblib.
    Args:
        pdb_keys: List of PDB keys to process
        cache_dir: Directory containing cached examples
    Returns:
        Dictionary mapping PDB keys to their sequence lengths.
    """
    print(f"Computing sequence lengths using {num_workers} workers...")
    cache_files = [f"{cache_dir}/{pdb_key}.pt" for pdb_key in pdb_keys]
    parallel = Parallel(n_jobs=num_workers, verbose=0)
    jobs = [delayed(_get_seq_length_from_cached)(cache_file) for cache_file in cache_files]
    results = parallel(tqdm(jobs, desc="Getting lengths", total=len(jobs)))
    return dict(zip(pdb_keys, results))


def _get_seq_length_from_cached(cache_file: str) -> int:
    """
    Helper function for parallel processing of sequence lengths.
    Args:
        cache_file: The cache file to process
    Returns:
        sequence_length
    """
    example = torch.load(cache_file, weights_only=True)
    seq_len = example["seq_mask"].sum().long().item()
    return seq_len


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


def get_radius_of_gyration_from_cached(pdb_keys: list[str], cache_dir: str, num_workers: int) -> dict[str, int]:
    """
    Computes radius of gyration for given PDB keys in parallel using joblib.
    """
    print(f"Computing radius of gyration using {num_workers} workers...")
    cache_files = [f"{cache_dir}/{pdb_key}.pt" for pdb_key in pdb_keys]
    parallel = Parallel(n_jobs=num_workers, verbose=0)
    jobs = [delayed(_compute_radius_of_gyration_from_cached)(cache_file) for cache_file in cache_files]
    results = parallel(tqdm(jobs, desc="Computing radius of gyration", total=len(jobs)))
    return dict(zip(pdb_keys, results))


def _compute_radius_of_gyration_from_cached(cache_file: str, min_dist=1e-3) -> float:
    """
    Computes the radius of gyration for a chain. Adapted from RFDiffusion code: https://github.com/RosettaCommons/RFdiffusion/blob/820bfdfaded8c260b962dc40a3171eae316b6ce0/rfdiffusion/potentials/potentials.py#L24
    """
    data = torch.load(cache_file, weights_only=True)

    # Extract CA atom coordinates
    Ca_coords = data["all_atom_positions"][:, 1, :]  # shape [n, 3]
    Ca_mask = data["all_atom_mask"][:, 1]  # shape [n]
    Ca = torch.tensor(Ca_coords, dtype=torch.float32)

    centroid = torch.sum(Ca, dim=0, keepdim=True) / torch.sum(Ca_mask, dim=0, keepdim=True)[..., None]

    dists = torch.norm((centroid - Ca), dim=-1)
    dists = dists.clamp(min=min_dist)

    rad_of_gyration = torch.sqrt(torch.sum(torch.square(dists) * Ca_mask) / torch.sum(Ca_mask))
    return rad_of_gyration.item()


