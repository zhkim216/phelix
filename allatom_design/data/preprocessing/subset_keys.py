#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

def main(
    pdb_path: str = ".",
    phase: str = "val",
    min_len: int = 64,
    max_len: int = 512,
    n_chains: int = 1,
):
    """
    This script:
      1. Reads a file called {phase}_pdb_keys.list from `pdb_path`.
      2. Loads each file {pdb_key}.pt from {pdb_path}/cached_examples/.
      3. Checks if the protein is single-chain and length in [min_len, max_len].
      4. Writes valid keys to {phase}_pdb_keys_L{min_len}_{max_len}_single_chain.list.
    """

    # Location of the input pdb_keys list
    pdb_keys_file = Path(f"{pdb_path}/{phase}_pdb_keys.list")
    if not pdb_keys_file.exists():
        raise FileNotFoundError(f"Could not find {pdb_keys_file}!")

    # Read all pdb keys
    with open(pdb_keys_file, "r") as f:
        pdb_keys = [line.strip() for line in f if line.strip()]

    # Directory containing the cached .pt files
    cache_dir = Path(pdb_path) / "cached_examples"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Could not find cached_examples directory at {cache_dir}!")

    # Parallel processing of pdb_keys
    valid_pdb_keys_results = Parallel(n_jobs=-1)(
        delayed(_process_pdb_key)(pdb_key, cache_dir, n_chains, min_len, max_len)
        for pdb_key in tqdm(pdb_keys, desc=f"Filtering for {n_chains}-chain proteins")
    )
    valid_pdb_keys = [x for x in valid_pdb_keys_results if x is not None]

    # Write out valid keys
    out_file = Path(f"{pdb_path}/{phase}_pdb_keys_L{min_len}_{max_len}_nchain_{n_chains}.list")
    with open(out_file, "w") as fout:
        for k in valid_pdb_keys:
            fout.write(f"{k}\n")

    print(f"Done! Wrote {len(valid_pdb_keys)} keys to {out_file}")


def _process_pdb_key(pdb_key, cache_dir, n_chains, min_len, max_len):
    data_file = cache_dir / f"{pdb_key}.pt"
    if not data_file.exists():
        # If cached file does not exist, skip
        return None

    # Load the cached data
    example = torch.load(data_file, weights_only=True)

    # Check single-chain condition
    chain_index = example["chain_index"]  # [n], each residue’s chain index
    unique_chains = torch.unique(chain_index)

    if len(unique_chains) != n_chains:
        # Skip if not desired number of chains
        return None

    # Check length condition
    seq_mask = example["seq_mask"]  # [n]
    seq_len = seq_mask.sum().item()
    if min_len <= seq_len <= max_len:
        return pdb_key
    return None


if __name__ == "__main__":
    # Example usage:
    #   python subset_single_chain.py --pdb_path /path/to/data --phase val
    import argparse

    parser = argparse.ArgumentParser(description="Subset single-chain proteins of a specific length range.")
    parser.add_argument("--pdb_path", type=str, default=".", help="Path to the dataset directory containing cached_examples.")
    parser.add_argument("--phase", type=str, default="val", help="Which phase pdb_keys file to read (e.g., val).")
    parser.add_argument("--min_len", type=int, default=64, help="Minimum length threshold.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum length threshold.")
    parser.add_argument("--n_chains", type=int, default=1, help="number of chains to subset for")
    args = parser.parse_args()

    main(
        pdb_path=args.pdb_path,
        phase=args.phase,
        min_len=args.min_len,
        max_len=args.max_len,
        n_chains=args.n_chains
    )
