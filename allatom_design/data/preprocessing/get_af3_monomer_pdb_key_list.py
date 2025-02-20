#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Subset single-chain proteins of a specific length range.")
    parser.add_argument("--chain_clustering_csv", type=str, help="Path to AF3 chain-based clustering CSV.")
    parser.add_argument("--phase", type=str, choices=["train", "eval", "test"], help="Phase to subset.")
    parser.add_argument("--out_dir", type=str, help="Output directory.")
    args = parser.parse_args()

    # Read clustered chains csv
    chain_df = pd.read_csv(args.chain_clustering_csv)
    chain_df["pdb_key"] = chain_df["pdb_id"].str[:4] + "_"  + chain_df["chain_id"] + "_" + chain_df["cluster_id"].astype(str)

    # For validation and test, only take first PDB in the cluster for deterministic evaluation
    if args.phase in ["val", "test"]:
        chain_df = chain_df.groupby("chain_id").first().reset_index()

    # Save out pdb_keys
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_file = f"{args.out_dir}/{args.phase}_pdb_keys.list"
    with open(out_file, "w") as fout:
        for k in chain_df["pdb_key"].tolist():
            fout.write(f"{k}\n")
    print(f"Done! Wrote {len(chain_df)} keys to {out_file}")

if __name__ == "__main__":
    main()