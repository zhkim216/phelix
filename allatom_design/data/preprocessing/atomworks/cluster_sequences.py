"""
Adapted from: https://github.com/jwohlwend/boltz/blob/main/scripts/process/cluster.py

Cluster the sequences in the metadata parquet file.
"""

import json
import os
import subprocess
from pathlib import Path

import atomworks.enums as aw_enums
import hydra
import numpy as np
import pandas as pd
import logging
from atomworks.ml.utils.misc import hash_sequence
import atomworks.ml.preprocessing.constants as aw_const
from atomworks.ml.preprocessing.constants import TRAINING_SUPPORTED_CHAIN_TYPES_INTS
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

def apply_query(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a single query to the data.

    Args:
        query (str): A query string to apply to the data.
        df (pd.DataFrame): The DataFrame to filter.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    original_num_rows = len(df)
    df = df.query(query).copy()  # .copy() to avoid SettingWithCopyWarning
    filtered_num_rows = len(df)
    _validate_filter_impact(query, original_num_rows, filtered_num_rows)
    return df


def _validate_filter_impact(query: str, original_num_rows: int, filtered_num_rows: int) -> None:
    """
    Validate the impact of the filter.

    Args:
        query (str): The query string that was applied.
        original_num_rows (int): The number of rows before applying the filter.
        filtered_num_rows (int): The number of rows after applying the filter.

    Raises:
        Warning: If the filter did not remove any rows.
        ValueError: If the filter removed all rows.
    """
    rows_removed = original_num_rows - filtered_num_rows
    percent_removed = (rows_removed / original_num_rows) * 100
    percent_remaining = (filtered_num_rows / original_num_rows) * 100

    if filtered_num_rows == original_num_rows:
        logger.warning(f"Query '{query}' did not remove any rows.")
    elif filtered_num_rows == 0:
        raise ValueError(f"Query '{query}' removed all rows.")
    else:
        logger.info(
            f"\n+-------------------------------------------+\n"
            f"Query '{query}':\n"
            f"  - Started with: {original_num_rows:,} rows\n"
            f"  - Removed: {rows_removed:,} rows ({percent_removed:.2f}%)\n"
            f"  - Remaining: {filtered_num_rows:,} rows ({percent_remaining:.2f}%)\n"
            f"+-------------------------------------------+\n"
        )


def apply_filters(filters: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a list of query filters to the DataFrame.

    Args:
        filters (list[str]): List of query strings to apply.
        df (pd.DataFrame): The DataFrame to filter.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    for query in filters:
        df = apply_query(query, df)
    return df

@hydra.main(config_path="../../../configs/data/preprocessing/atomworks", config_name="cluster_sequences", version_base="1.3.2",)
def main(cfg: DictConfig) -> None:
    """
    Cluster the sequences in the metadata parquet file.
    """
    df = pd.read_parquet(cfg.parquet_path)
    
    # exclude chain types by ChainType enums
    df = apply_query(f"q_pn_unit_type in {TRAINING_SUPPORTED_CHAIN_TYPES_INTS}", df)
    
    # Annotate protein and peptide chains
    is_polypeptide_l = df['q_pn_unit_type'] == aw_enums.ChainType.POLYPEPTIDE_L.value
    protein_seq_len = df['q_pn_unit_processed_entity_canonical_sequence'].str.len()
    df['q_pn_unit_is_protein'] = is_polypeptide_l & (protein_seq_len >= aw_const.PEPTIDE_MAX_RESIDUES)
    df['q_pn_unit_is_peptide'] = is_polypeptide_l & (protein_seq_len < aw_const.PEPTIDE_MAX_RESIDUES)
    
    # Seq threshold for clustering
    seq_id_threshold = cfg.seq_id_threshold
    print(f"Using sequence identity threshold: {seq_id_threshold}")
    subdir_name = f"clustering_thres_{str(seq_id_threshold).replace(".", "")}"

    clustering_dir = Path(cfg.pdb_path) / subdir_name
    clustering_dir.mkdir(parents=True, exist_ok=True)

    # Get all sequences by type
    proteins = set()
    peptides = set()
    nucleic_acids = set()    
    nonpolymer_seqs = set()

    nucleic_acid_type_values = [aw_enums.ChainType.DNA.value, aw_enums.ChainType.RNA.value, aw_enums.ChainType.DNA_RNA_HYBRID.value]
    nonpolymer_type_values = [aw_enums.ChainType.BRANCHED.value, aw_enums.ChainType.MACROLIDE.value, aw_enums.ChainType.NON_POLYMER.value, aw_enums.ChainType.WATER.value]

    for _, row in tqdm(df.iterrows(), desc="Sorting sequences by type", total=len(df)):
        if row["q_pn_unit_is_protein"]:            
            proteins.add(row["q_pn_unit_processed_entity_canonical_sequence"])
        elif row["q_pn_unit_is_peptide"]:
            peptides.add(row["q_pn_unit_processed_entity_canonical_sequence"])
        elif row["q_pn_unit_type"] in nucleic_acid_type_values:
            nucleic_acids.add(row["q_pn_unit_processed_entity_canonical_sequence"])
        elif row["q_pn_unit_type"] in nonpolymer_type_values:
            res_names = row["q_pn_unit_non_polymer_res_names"]
            normalized_res_names = "".join(sorted(res_names.split(",")))
            nonpolymer_seqs.add(normalized_res_names)        
            
    # Run mmseqs on the protein data
    proteins = [f">{hash_sequence(seq)}\n{seq}" for seq in proteins]
    with (clustering_dir / "proteins.fasta").open("w") as f:
        f.write("\n".join(proteins))
    
    subprocess.run(
        f"{os.environ['SOFTWARE_PATH']}/mmseqs/bin/mmseqs easy-cluster {clustering_dir / 'proteins.fasta'} {clustering_dir / 'clust_prot'} {clustering_dir / 'tmp'} --min-seq-id {seq_id_threshold}",
        shell=True,
        check=True,
    )

    # Load protein clusters
    clustering_path = clustering_dir / "clust_prot_cluster.tsv"
    protein_data = pd.read_csv(clustering_path, sep="\t", header=None)
    clusters = protein_data[0]
    items = protein_data[1]
    clustering = dict(zip(list(items), list(clusters)))

    # Each peptide sequence is given an id
    for peptide in peptides:
        peptide_id = hash_sequence(peptide)
        clustering[peptide_id] = peptide_id
    
    # Each unique rna, dna sequences are given an id
    for nucl in nucleic_acids:
        nucl_id = hash_sequence(nucl)
        clustering[nucl_id] = nucl_id

    # Each unique sequence of CCD codes is given an id
    for nonpolymer_seq in nonpolymer_seqs:
        nonpolymer_id = hash_sequence(nonpolymer_seq)
        clustering[nonpolymer_id] = nonpolymer_id

    # Assign each hash a cluster ID from 0 to num_clusters - 1
    cluster_hashes = list(set(clustering.values()))
    cluster_ids = {h: i for i, h in enumerate(cluster_hashes)}
    clustering = {k: cluster_ids[v] for k, v in clustering.items()}

    # Save clustering
    with (clustering_dir / "clustering.json").open("w") as handle:
        json.dump(clustering, handle)

    # Add cluster IDs to metadata df
    conditions = [
        df["q_pn_unit_is_protein"],
        df["q_pn_unit_is_peptide"],
        df["q_pn_unit_type"].isin(nucleic_acid_type_values),
        df["q_pn_unit_type"].isin(nonpolymer_type_values),
    ]
    choices = [
        df['q_pn_unit_processed_entity_canonical_sequence_hash'].map(clustering),
        df['q_pn_unit_processed_entity_canonical_sequence_hash'].map(clustering),
        df['q_pn_unit_processed_entity_canonical_sequence_hash'].map(clustering),
        df['q_pn_unit_non_polymer_res_names'].apply(hash_sequence).map(clustering),
    ]
        
    df["q_pn_unit_cluster_id"] = np.select(conditions, choices)

    # Sanity check that we have no missing values
    if df["q_pn_unit_cluster_id"].isna().any():
        print(f"WARNING: {df['q_pn_unit_cluster_id'].isna().sum()} missing values in q_pn_unit_cluster_id")

    df["q_pn_unit_cluster_id"] = df["q_pn_unit_cluster_id"].fillna(-1).astype(np.int32)  # fill missing cluster IDs with -1
    df.to_parquet(cfg.parquet_path.replace(".parquet", f"_seq_clustered_{str(seq_id_threshold).replace(".", "")}.parquet"))
    print(f"Saved clustered metadata to {cfg.parquet_path.replace('.parquet', f'_seq_clustered_{str(seq_id_threshold).replace(".", "")}.parquet')}")

if __name__ == "__main__":
    main()
