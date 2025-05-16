"""
Adapted from: https://github.com/jwohlwend/boltz/blob/main/scripts/process/cluster.py
Cluster the sequences of the processed targets and update the manifest with the cluster IDs.
"""

import glob
import hashlib
import json
import pickle
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import gemmi
import hydra
import pandas as pd
import rdkit
import redis
from Bio import SeqIO
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data import const
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import \
    load_input
from allatom_design.data.types import Manifest, Record


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_v2", config_name="cluster", version_base="1.3.2",)
def main(cfg: DictConfig) -> None:
    """
    Create clustering.json file along with its .rdb file.
    """
    # Create output directory
    outdir = Path(f"{cfg.processed_targets_dir}/clustering")
    outdir.mkdir(parents=True, exist_ok=True)

    # Get all polymer sequences among processed targets
    structure_files = glob.glob(f"{cfg.processed_targets_dir}/structures/*.npz")

    proteins = set()
    shorts = set()
    nucleotides = set()
    nonpolymer_seqs = set()

    key_to_seq = {}
    for structure_file in tqdm(structure_files, desc="Parsing polymer sequences"):
        struct = load_input(structure_file).structure
        pdb_id = Path(structure_file).stem.lower()

        for chain in struct.chains:
            key = f"{pdb_id}_{chain['name']}"
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]

            if chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]:
                # For non-polymer chains, use the CCD code as the sequence
                ccd_seq = "".join(struct.residues[res_start:res_end]["name"].tolist())
                key_to_seq[key] = ccd_seq
                nonpolymer_seqs.add(ccd_seq)
                continue

            seq = gemmi.one_letter_code(struct.residues[res_start:res_end]["name"])
            key_to_seq[key] = seq

            # Separate the sequences into proteins, nucleotides and short sequences
            if set(seq).issubset({"A", "C", "G", "T", "U", "N"}):
                nucleotides.add(seq)
            elif len(seq) < 10:
                shorts.add(seq)
            else:
                proteins.add(seq)

    # Run mmseqs on the protein data
    proteins = [f">{hash_sequence(seq)}\n{seq}" for seq in proteins]
    with (outdir / "proteins.fasta").open("w") as f:
        f.write("\n".join(proteins))

    subprocess.run(
        f"mmseqs easy-cluster {outdir / 'proteins.fasta'} {outdir / 'clust_prot'} {outdir / 'tmp'} --min-seq-id 0.4",  # noqa: E501
        shell=True,  # noqa: S602
        check=True,
    )

    # Load protein clusters
    clustering_path = outdir / "clust_prot_cluster.tsv"
    protein_data = pd.read_csv(clustering_path, sep="\t", header=None)
    clusters = protein_data[0]
    items = protein_data[1]
    clustering = dict(zip(list(items), list(clusters)))

    # Each short sequence is given an id
    for short in shorts:
        short_id = hash_sequence(short)
        clustering[short_id] = short_id

    # Each unique rna sequence is given an id
    for nucl in nucleotides:
        nucl_id = hash_sequence(nucl)
        clustering[nucl_id] = nucl_id

    # Add non-polymer sequences to clustering
    for nonpolymer_seq in nonpolymer_seqs:
        nonpolymer_id = hash_sequence(nonpolymer_seq)
        clustering[nonpolymer_id] = nonpolymer_id

    # Assign each hash a cluster ID from 0 to num_clusters - 1
    cluster_hashes = list(set(clustering.values()))
    cluster_ids = {h: i for i, h in enumerate(cluster_hashes)}
    clustering = {k: cluster_ids[v] for k, v in clustering.items()}

    # Load ligand data
    with Path(cfg.ccd).open("rb") as handle:
        ligand_data = pickle.load(handle)  # noqa: S301

    # Each unique ligand CCD is given an id
    for ccd_code in ligand_data:
        clustering[ccd_code] = ccd_code

    # Save clustering
    with (outdir / "clustering.json").open("w") as handle:
        json.dump(clustering, handle)

    # Load in manifest_unclustered.json and add cluster IDs to the manifest
    manifest = Manifest.load(f"{cfg.processed_targets_dir}/manifest_unclustered.json")
    new_records = []
    for record in manifest.records:
        struct = load_input(f"{cfg.processed_targets_dir}/structures/{record.id}.npz").structure
        name_to_chain = {c['name']: c for c in struct.chains}

        new_chain_infos = []
        for chain_info in record.chains:
            # Recompute the same key we used above
            chain = name_to_chain[chain_info.chain_name]
            key = f"{record.id.lower()}_{chain['name']}"

            seq_hash = hash_sequence(key_to_seq[key])
            if seq_hash in clustering:
                cluster_id = clustering[seq_hash]
            else:
                print(f"WARNING: {key} not found in clustering")
                cluster_id = -1

            new_chain_infos.append(replace(chain_info, cluster_id=cluster_id))
        new_records.append(replace(record, chains=new_chain_infos))

    new_records = [asdict(r) for r in new_records]
    with open(f"{cfg.processed_targets_dir}/manifest.json", "w") as f:
        json.dump(new_records, f)


def hash_sequence(seq: str) -> str:
    """Hash a sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()


if __name__ == "__main__":
    main()
