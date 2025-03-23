#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Parse AF3 clustered chains (monomers) as a pdb_key list.")
    parser.add_argument("--interface_clustering_csv", type=str, help="Path to AF3 chain-based clustering CSV.")
    parser.add_argument("--phase", type=str, choices=["train", "eval", "test"], help="Phase to subset.")
    parser.add_argument("--out_file", type=str, help="Output file.")
    args = parser.parse_args()

    # Read clustered interfaces csv
    interface_df = pd.read_csv(args.interface_clustering_csv, keep_default_na=False)
    interface_df = interface_df[(interface_df["interface_molecule_id_1"] == "protein") & (interface_df["interface_molecule_id_2"] == "protein")]  # extract only protein-protein interfaces
    interface_df["pdb_key"] = interface_df.apply(lambda x: f"{x['pdb_id'][:4]}_{x['interface_chain_id_1']}_{x['interface_chain_id_2']}_{x['interface_cluster_id']}", axis=1)

    # Save out pdb_keys
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as fout:
        for k in interface_df["pdb_key"].tolist():
            fout.write(f"{k}\n")
    print(f"Done! Wrote {len(interface_df)} keys to {args.out_file}")


if __name__ == "__main__":
    main()