#!/usr/bin/env python3
"""
Create eval2 from a random subset of train keys.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Create eval2 from a random subset of train keys.")
    parser.add_argument("--annotated_pdb_keys_csv", type=str, help="Path to annotated PDBs csv.")
    parser.add_argument("--n_eval2", type=int, help="Number of eval2 keys to create.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out_dir", type=str, help="Output directory.")
    args = parser.parse_args()

    df = pd.read_csv(args.annotated_pdb_keys_csv)

    # Randomly select n_eval2 rows from train_df
    train_df = df[df["phase"] == "train"]
    eval2_df = train_df.sample(n=args.n_eval2, random_state=args.seed)

    # Change the phase of selected rows to "eval2"
    df.loc[eval2_df.index, "phase"] = "eval2"

    # Save outputs
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.out_dir}/annotated_pdb_keys.csv", index=False, header=True)
    df[df["phase"] == "train"]["pdb_key"].to_csv(f"{args.out_dir}/train_pdb_keys_for_eval2.list", index=False, header=False)
    df[df["phase"] == "eval"]["pdb_key"].to_csv(f"{args.out_dir}/eval_pdb_keys_for_eval2.list", index=False, header=False)
    df[df["phase"] == "eval2"]["pdb_key"].to_csv(f"{args.out_dir}/eval2_pdb_keys_for_eval2.list", index=False, header=False)


if __name__ == "__main__":
    main()
