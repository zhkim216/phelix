#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Parse AF3 clustered chains (monomers) as a pdb_key list.")
    parser.add_argument("--af3_data_cache_dir", type=str, help="Path to AF3 data cache directory.")
    parser.add_argument("--out_dir", type=str, help="Output file.")
    args = parser.parse_args()

    # Read clustered chains csv
    phases = ["eval", "train"]
    phase_to_df = {}
    for phase in phases:
        if phase == "train":
            chain_clustering_csv = f"{args.af3_data_cache_dir}/train_clusterings/protein_chain_cluster_mapping.csv"
        elif phase == "eval":
            chain_clustering_csv = f"{args.af3_data_cache_dir}/val_clusterings/protein_chain_cluster_mapping.csv"
        else:
            raise ValueError(f"Invalid phase: {phase}")

        chain_df = pd.read_csv(chain_clustering_csv, keep_default_na=False)
        chain_df["pdb_key"] = chain_df["pdb_id"].str[:4] + "_"  + chain_df["chain_id"]

        # get dataframe with just pdb_key and phase columns
        phase_df = pd.DataFrame({
            "pdb_key": chain_df["pdb_key"],
            "phase": phase,
            "cluster_id": chain_df["cluster_id"]
        })

        phase_df = phase_df.drop_duplicates(subset=["pdb_key"])  # in validation, some pdb_id+chain_id+cluster_id are duplicated for some reason
        phase_to_df[phase] = phase_df

    # Save out all pdb_keys as a csv
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_keys_df = pd.concat(list(phase_to_df.values()), ignore_index=True)
    all_keys_csv = f"{args.out_dir}/all_pdb_keys.csv"
    all_keys_df.to_csv(all_keys_csv, index=False)
    print(f"Wrote {len(all_keys_df)} keys to {all_keys_csv}")
    for phase in phases:
        phase_df = phase_to_df[phase]
        phase_csv = f"{args.out_dir}/{phase}_pdb_keys.csv"
        phase_df.to_csv(phase_csv, index=False)
        print(f"Wrote {len(phase_df)} keys to {phase_csv}")


if __name__ == "__main__":
    main()