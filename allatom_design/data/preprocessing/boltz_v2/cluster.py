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
from functools import partial
from pathlib import Path

import gemmi
import hydra
import pandas as pd
import redis
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data import const
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    Resource, load_input)
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

    use_parallel = cfg.num_workers > 1
    if use_parallel:
        # process in parallel
        results = Parallel(n_jobs=cfg.num_workers)(
            delayed(process_structure_file)(structure_file)
            for structure_file in tqdm(structure_files, desc="Parsing polymer sequences")
        )
        # merge all results
        for p, s, n, npoly, k2s in results:
            proteins.update(p)
            shorts.update(s)
            nucleotides.update(n)
            nonpolymer_seqs.update(npoly)
            key_to_seq.update(k2s)
    else:
        # process sequentially
        for structure_file in tqdm(structure_files, desc="Parsing polymer sequences"):
            p, s, n, npoly, k2s = process_structure_file(structure_file)
            proteins.update(p)
            shorts.update(s)
            nucleotides.update(n)
            nonpolymer_seqs.update(npoly)
            key_to_seq.update(k2s)


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

    fn = partial(add_cluster_id_to_record, clustering=clustering, key_to_seq=key_to_seq, processed_targets_dir=cfg.processed_targets_dir)
    new_records = [fn(record) for record in tqdm(manifest.records, desc="Adding cluster IDs to manifest")]
    new_records = [asdict(r) for r in new_records]

    with open(f"{cfg.processed_targets_dir}/manifest.json", "w") as f:
        json.dump(new_records, f)


def hash_sequence(seq: str) -> str:
    """Hash a sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()


def process_structure_file(structure_file: str) -> tuple[set[str], set[str], set[str], set[str], dict[str, str]]:
    """
    Parses a single structure file and returns:
        - sets of proteins, shorts, nucleotides, nonpolymer_seqs
        - the key_to_seq mapping for that file
    """
    struct = load_input(structure_file).structure
    pdb_id = Path(structure_file).stem.lower()

    proteins = set()
    shorts = set()
    nucleotides = set()
    nonpolymer_seqs = set()
    key_to_seq = {}

    for chain in struct.chains:
        key = f"{pdb_id}_{chain['name']}"
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]

        # For non-polymer chains, use the sequence of CCD codes as the sequence
        if chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]:
            ccd_seq = "".join(struct.residues[res_start:res_end]["name"].tolist())
            key_to_seq[key] = ccd_seq
            nonpolymer_seqs.add(ccd_seq)
            continue

        # For polymers, separate the sequences into proteins, nucleotides and short sequences
        seq = gemmi.one_letter_code(struct.residues[res_start:res_end]["name"])
        key_to_seq[key] = seq
        if set(seq).issubset({"A", "C", "G", "T", "U", "N"}):
            nucleotides.add(seq)
        elif len(seq) < 10:
            shorts.add(seq)
        else:
            proteins.add(seq)

    return (
        proteins,
        shorts,
        nucleotides,
        nonpolymer_seqs,
        key_to_seq,
    )


def add_cluster_id_to_record(record: Record, clustering: dict[str, int], key_to_seq: dict[str, str],
                             processed_targets_dir: str) -> Record:
    """Returns a new record with the cluster ID added."""
    new_chain_infos = []
    for chain_info in record.chains:
        key = f"{record.id.lower()}_{chain_info.chain_name}"
        seq_hash = hash_sequence(key_to_seq[key])
        cluster_id = clustering.get(seq_hash, -1)
        if cluster_id == -1:
            print(f"WARNING: {key} not found in clustering")
        new_chain_infos.append(replace(chain_info, cluster_id=cluster_id))
    return replace(record, chains=new_chain_infos)


if __name__ == "__main__":
    main()
