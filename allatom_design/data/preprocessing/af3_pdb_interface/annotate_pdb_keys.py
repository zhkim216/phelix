#!/usr/bin/env python3
"""
Add useful information to the pdb keys, and create an eval2 subset.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB.MMCIFParser import FastMMCIFParser
from joblib import Parallel, delayed
from tqdm import tqdm

mmcif_parser = FastMMCIFParser(auth_chains=True, auth_residues=False, QUIET=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_keys_list", type=str, help="Path to train keys list.")
    parser.add_argument("--eval_keys_list", type=str, help="Path to eval keys list.")
    parser.add_argument("--n_eval2", type=int, help="Number of eval2 keys to create.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out_dir", type=str, help="Output directory.")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory.")
    args = parser.parse_args()

    # Merge train and eval keys and create a phase column
    train_df = pd.read_csv(args.train_keys_list, header=None, names=['pdb_key'])
    eval_df = pd.read_csv(args.eval_keys_list, header=None, names=['pdb_key'])
    train_df["phase"] = "train"
    eval_df["phase"] = "eval"
    df = pd.concat([train_df, eval_df], ignore_index=True)

    # Collect some useful information
    chain_lengths = Parallel(n_jobs=-1)(
        delayed(compute_chain_lengths)(row, args.dataset_dir)
        for _, row in tqdm(df.iterrows(), total=len(df))
    )
    df["chain_1_seq_length"] = [r[0] for r in chain_lengths]
    df["chain_2_seq_length"] = [r[1] for r in chain_lengths]

    # Get rid of problematic entries
    problematic_df = df[(df["chain_1_seq_length"].isna()) | (df["chain_2_seq_length"].isna())]
    df = df.drop(problematic_df.index).reset_index(drop=True)
    print(f"Removed {len(problematic_df)} problematic entries.")

    # Randomly select n_eval2 rows from train_df
    train_df = df[df["phase"] == "train"]
    eval2_df = train_df.sample(n=args.n_eval2, random_state=args.seed)
    df.loc[eval2_df.index, "phase"] = "eval2"

    # Save outputs
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.out_dir}/annotated_pdb_keys.csv", index=False, header=True)
    df[df["phase"] == "train"]["pdb_key"].to_csv(f"{args.out_dir}/train_pdb_keys_for_eval2.list", index=False, header=False)
    df[df["phase"] == "eval"]["pdb_key"].to_csv(f"{args.out_dir}/eval_pdb_keys_for_eval2.list", index=False, header=False)
    df[df["phase"] == "eval2"]["pdb_key"].to_csv(f"{args.out_dir}/eval2_pdb_keys_for_eval2.list", index=False, header=False)
    problematic_df.to_csv(f"{args.out_dir}/problematic_pdb_keys.csv", index=False, header=True)  # save problematic entries to a separate file


def compute_chain_lengths(row, dataset_dir) -> tuple[int, int]:
    try:
        mmcif_dir = f"{dataset_dir}/{row['phase']}_mmcifs"
        mmcif_path = f"{mmcif_dir}/{row['pdb_key']}.cif"
        model_num = 0
        structure = mmcif_parser.get_structure("s", mmcif_path)[model_num]
        chains = list(structure.get_chains())
        if len(chains) != 2:
            raise ValueError(f"Expected exactly 2 chains in {mmcif_path}, found {len(chains)}")
        chain_lengths = []
        for chain in chains:
            Ca_coords = [atom for residue in chain for atom in residue if atom.get_name() == "CA"]
            chain_lengths.append(len(Ca_coords))
        return chain_lengths[0], chain_lengths[1]
    except Exception as e:
        print(f"Error processing {mmcif_path}: {e}")
        return np.nan, np.nan


if __name__ == "__main__":
    main()
