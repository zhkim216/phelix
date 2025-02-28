#!/usr/bin/env python3
"""
Parse the self_consistency_metrics.csv output from construct_sc_dataset.py to get FAMPNN1 designability statistics at T=0.1.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Parse self_consistency_metrics csv")
    parser.add_argument("--train_sc_csv", type=str, help="Path to self_consistency_metrics csv.")
    parser.add_argument("--eval_sc_csv", type=str, help="Path to self_consistency_metrics csv.")
    parser.add_argument("--out_dir", type=str, help="Output directory.")
    args = parser.parse_args()

    # First, combine train and eval csvs, keeping track of the phase
    train_df = pd.read_csv(args.train_sc_csv)
    train_df["phase"] = "train"
    eval_df = pd.read_csv(args.eval_sc_csv)
    eval_df["phase"] = "eval"
    sc_df = pd.concat([train_df, eval_df], ignore_index=True)

    # Filter for T=0.1, take only the first entry for each pdb_key
    sc_df = sc_df[sc_df["temperature"] == 0.1].drop_duplicates(subset=["pdb_key"], keep="first").reset_index(drop=True)
    df_out = sc_df[["pdb_key", "temperature", "sc_ca_rmsd", "sc_aa_rmsd", "sc_ca_tm", "avg_plddt", "phase"]]

    # Additional info
    df_out["sample_name"] = sc_df["pdb_name"]  # in case we want to know which FAMPNN sample it is
    df_out["seq_length"] = sc_df["pred_seq"].str.len()

    # Save to out directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df_out.to_csv(f"{args.out_dir}/designability_stats.csv", index=False, header=True)


if __name__ == "__main__":
    main()