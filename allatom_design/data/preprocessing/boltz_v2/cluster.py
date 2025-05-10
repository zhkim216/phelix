"""
Adapted from: https://github.com/jwohlwend/boltz/blob/main/scripts/process/cluster.py
Create a mapping from structure and chain ID to MSA indices.
"""

import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import hydra
import pandas as pd
import rdkit
import redis
from Bio import SeqIO
from omegaconf import DictConfig

from allatom_design.eval.eval_utils.eval_setup_utils import start_redis


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_v2", config_name="cluster", version_base="1.3.2",)
def main(cfg: DictConfig) -> None:
    """
    Create clustering.json file along with its .rdb file.
    """
    # Create output directory
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Split the sequences into proteins and nucleotides
    with Path(cfg.sequences).open("r") as f:
        data = list(SeqIO.parse(f, "fasta"))

    proteins = set()
    shorts = set()
    nucleotides = set()

    # Separate the sequences into proteins, nucleotides and short sequences
    # Short sequences cause a bug in the clustering, so they are separated
    for seq in data:
        if set(str(seq.seq)).issubset({"A", "C", "G", "T", "U", "N"}):
            nucleotides.add(str(seq.seq).strip())
        elif len(str(seq.seq).strip()) < 10:  # noqa: PLR2004
            shorts.add(str(seq.seq).strip())
        else:
            proteins.add(str(seq.seq).strip())

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

    # Load ligand data
    with Path(cfg.ccd).open("rb") as handle:
        ligand_data = pickle.load(handle)  # noqa: S301

    # Each unique ligand CCD is given an id
    for ccd_code in ligand_data:
        clustering[ccd_code] = ccd_code

    # Save clustering
    with (outdir / "clustering.json").open("w") as handle:
        json.dump(clustering, handle)

    # Create clustering.rdb
    redis_host, redis_port = "localhost", 7778
    start_redis(redis_host, redis_port, cfg.software_path, f"{outdir}/clustering.rdb")

    r = redis.Redis(host=redis_host, port=redis_port)
    r.flushall()
    for k, v in clustering.items():
        r.set(k, v)
    r.save()
    print(f"Redis clustering.rdb saved to {outdir}/clustering.rdb")


def hash_sequence(seq: str) -> str:
    """Hash a sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()


if __name__ == "__main__":
    main()
