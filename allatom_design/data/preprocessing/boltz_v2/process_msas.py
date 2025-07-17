#!/usr/bin/env python3
import glob
import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any

import gemmi
import hydra
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig
from p_tqdm import p_umap
from redis import Redis
from tqdm import tqdm

from allatom_design.data import const
from allatom_design.data.preprocessing.boltz_utils.a3m import parse_a3m
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    hash_sequence, load_input)
from allatom_design.data.types import Manifest, Record
from allatom_design.eval.eval_utils.eval_setup_utils import start_redis


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_v2", config_name="process_msas", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Process pre-downloaded MSAs from Boltz-1.
    Rehashes the query sequence of each MSA and saves it to a new file with the new hash.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # === Add MSA hashes to the manifest === #
    ## get all polymer sequences among processed targets
    processed_targets_dir = f"{cfg.out_dir}/processed_targets"
    structure_files = glob.glob(f"{processed_targets_dir}/structures/*.npz")

    key_to_hash = {}

    use_parallel = cfg.num_workers > 1
    if use_parallel:
        # process in parallel
        results = Parallel(n_jobs=cfg.num_workers)(
            delayed(get_protein_hashes)(structure_file)
            for structure_file in tqdm(structure_files, desc="Parsing protein sequences")
        )
        # merge all results
        for k2h in results:
            key_to_hash.update(k2h)
    else:
        # process sequentially
        for structure_file in tqdm(structure_files, desc="Parsing protein sequences"):
            k2h = get_protein_hashes(structure_file)
            key_to_hash.update(k2h)

    ## load in manifest and add MSA IDs
    manifest = Manifest.load(f"{processed_targets_dir}/manifest.json")
    new_records = [add_msa_id_to_record(record, key_to_hash) for record in tqdm(manifest.records, desc="Adding MSA IDs to manifest")]
    new_records = [asdict(r) for r in new_records]
    with open(f"{processed_targets_dir}/manifest.json", "w") as f:
        json.dump(new_records, f)

    # === Preprocess MSAs === #
    # Load in taxonomy database
    redis_host, redis_port = "localhost", 7777
    start_redis(redis_host, redis_port, cfg.taxonomy_rdb_path)
    resource = MSAResource(host=redis_host, port=redis_port)

    # Fetch data
    print("Fetching data...")
    data = list(Path(cfg.raw_msa_dir).rglob("*.a3m*"))
    print(f"Found {len(data)} MSA's.")

    # Run processing
    processed_msas_dir = f"{cfg.out_dir}/processed_msas"
    Path(processed_msas_dir).mkdir(parents=True, exist_ok=True)
    if use_parallel:
        # Create processing function
        fn = partial(
            process_msa,
            outdir=processed_msas_dir,
            max_seqs=cfg.max_seqs,
            resource=resource,
        )

        # Run in parallel
        p_umap(fn, data, num_cpus=cfg.num_workers)
    else:
        for path in tqdm(data):
            process_msa(
                path,
                outdir=processed_msas_dir,
                max_seqs=cfg.max_seqs,
                resource=resource,
            )


class MSAResource:
    """A shared resource for processing MSAs."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize the redis database."""
        self._redis = Redis(host=host, port=port)

    def get(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the Redis database."""
        return self._redis.get(key)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the resource."""
        out = self.get(key)
        if out is None:
            raise KeyError(key)
        return out


def process_msa(
    path: Path,
    outdir: str,
    max_seqs: int,
    resource: MSAResource,
) -> None:
    """Run processing in a worker thread."""
    outdir = Path(outdir)
    msa, query_hash = parse_a3m(path, resource, max_seqs)
    out_path = f"{outdir}/{query_hash}.npz"
    np.savez_compressed(out_path, **asdict(msa))


def get_protein_hashes(structure_file: str) -> dict[str, str]:
    """
    Parses a single structure file and returns a mapping from each key (record_id + chain_name) to its hash for protein-only.
    """
    struct = load_input(structure_file).structure
    pdb_id = Path(structure_file).stem.lower()

    key_to_hash = {}

    for chain in struct.chains:
        key = f"{pdb_id}_{chain['name']}"
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]

        if chain["mol_type"] != const.chain_type_ids["PROTEIN"]:
            continue

        seq = gemmi.one_letter_code(struct.residues[res_start:res_end]["name"])
        key_to_hash[key] = hash_sequence(seq)

    return key_to_hash


def add_msa_id_to_record(record: Record, key_to_hash: dict[str, str]) -> Record:
    """Add the MSA ID to the record."""
    new_chain_infos = []
    for chain in record.chains:
        key = f"{record.id.lower()}_{chain.chain_name}"
        chain.msa_id = key_to_hash.get(key, -1)
        new_chain_infos.append(chain)
    return replace(record, chains=new_chain_infos)


if __name__ == "__main__":
    main()
