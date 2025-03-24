#!/usr/bin/env python3
"""
Given the output of fampnn_multi.py:
- parse the designability statistics csv
- annotate the designability statistics csv with useful information from the original AF3 monomer dataset
"""
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../../../configs/data/preprocessing/augmented_af3_monomer_v2", config_name="parse_fampnn_multi", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Annotate the designability statistics csv with useful information from the original AF3 monomer dataset.
    """
    # Parse designability statistics csv  TODO: will need to extend this to take in multiple fampnn multi outputs
    designability_stats_df = pd.read_csv(f"{cfg.fampnn_multi_dir}/self_consistency_metrics.csv")

    # If we had to re-run with fixed_missing, concatenate the results
    if Path(f"{cfg.fampnn_multi_dir}/fampnn_outputs_fixed_missing.csv").exists():
        designability_stats_df = pd.concat([designability_stats_df,
                                            pd.read_csv(f"{cfg.fampnn_multi_dir}/fampnn_outputs_fixed_missing.csv")],
                                            ignore_index=True)

    # Annotate designability statistics csv with useful information from the original AF3 monomer dataset
    # Read in AF3 monomer manifest
    af3_monomer_manifest = pd.read_csv(cfg.af3_monomer_manifest)
    af3_monomer_manifest = af3_monomer_manifest[["pdb_name", "cluster_id", "phase", "seq_length"]].rename(columns={"pdb_name": "input_pdb_name"})

    # Merge designability statistics csv with AF3 monomer manifest
    designability_stats_df = designability_stats_df.merge(af3_monomer_manifest, left_on="input_pdb_name", right_on="input_pdb_name", how="left",
                                                          validate="many_to_one")
    # check for unannotated rows
    unannotated_rows = designability_stats_df[designability_stats_df["cluster_id"].isna()]
    if len(unannotated_rows) > 0:
        print(f"WARNING: {len(unannotated_rows)} rows were not annotated with cluster_id")
        print(unannotated_rows[["input_pdb_name", "pdb_name", "phase", "seq_length"]])

    # Set any eval2 phase back to train; we want to take the original evals, but we will re-sample eval2 keys later
    designability_stats_df.loc[designability_stats_df["phase"] == "eval2", "phase"] = "train"

    # Write out annotated designability statistics csv
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    designability_stats_df.to_csv(f"{cfg.out_dir}/pdb_keys.csv", index=False)
    print(f"Wrote annotated pdb keys to {cfg.out_dir}/pdb_keys.csv")


if __name__ == "__main__":
    main()
