#!/usr/bin/env python3
import glob
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data import const
from allatom_design.data.feature.seq_des_featurizer import \
    SequenceDesignFeaturizer
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import \
    load_input
from allatom_design.data.tokenize.tokenizer import Tokenizer
from allatom_design.data.types import (Connection, Input, InterfaceInfo,
                                       Manifest, Record, Structure)
from allatom_design.data.write.mmcif import to_mmcif
from collections import defaultdict


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_protein_interfaces", config_name="build_dataset", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given processed structures, apply various filters and save them as mmCIFs to disk.
    """
    # Create output directory
    interface_out_dir = f"{cfg.pdb_path}/interface_cifs"
    Path(interface_out_dir).mkdir(parents=True, exist_ok=True)

    # Load manifest.json
    manifest_path = f"{cfg.processed_targets_dir}/manifest.json"
    print(f"Loading in manifest from {manifest_path}...")
    manifest = Manifest.load(Path(manifest_path))

    # Load in original boltz manifest
    original_boltz_manifest_path = cfg.original_boltz_manifest
    print(f"Loading in original boltz manifest from {original_boltz_manifest_path}...")
    original_boltz_manifest = Manifest.load(Path(original_boltz_manifest_path))

    # TEMP: fix for cluster ids
    id_to_record = {r.id: r for r in original_boltz_manifest.records}
    fixed_records = []
    for record in tqdm(manifest.records, desc="Fixing cluster ids"):
        try:
            if record.id not in id_to_record:
                print(f"WARNING: {record.id} not found in original boltz manifest, skipping...")
                continue
            original_chain_name_to_chain = {c.chain_name: c for c in id_to_record[record.id].chains}

            fixed_chains = []
            for chain in record.chains:
                if chain.num_residues != original_chain_name_to_chain[chain.chain_name].num_residues:
                    # sanity check
                    print(f"WARNING: In {record.id}, chain {chain.chain_name} has {chain.num_residues} residues in the new manifest, but {original_chain_name_to_chain[chain.chain_name].num_residues} residues in the original manifest, setting to -1...")
                    chain = replace(chain, cluster_id=-1)

                else:
                    # fix cluster ids
                    chain = replace(chain, cluster_id=original_chain_name_to_chain[chain.chain_name].cluster_id)
                fixed_chains.append(chain)
            record = replace(record, chains=fixed_chains)
            fixed_records.append(record)
        except Exception as e:
            print(f"WARNING: Error in {record.id}, skipping...")
            print(e)
            continue

    manifest = replace(manifest, records=fixed_records)

    # Filter for chain pairs
    chain_type_filter = lambda c1, c2: c1.mol_type == const.chain_type_ids[cfg.interface_chain_type] and c2.mol_type == const.chain_type_ids[cfg.interface_chain_type]
    chain_size_filter = lambda c1, c2: c1.num_residues >= cfg.chain_min_residues and c2.num_residues >= cfg.chain_min_residues
    interface_size_filter = lambda c1, c2: c1.num_residues + c2.num_residues <= cfg.interface_max_residues
    valid_cluster_id_filter = lambda c1, c2: (c1.cluster_id != -1) and (c2.cluster_id != -1)
    filters = [
        chain_type_filter,
        chain_size_filter,
        interface_size_filter,
        valid_cluster_id_filter,
    ]

    filtered_records = []
    for record in manifest.records:
        filtered_interfaces = [i for i in record.interfaces if all(f(record.chains[i.chain_1], record.chains[i.chain_2]) for f in filters)]
        filtered_interfaces = [i for i in filtered_interfaces if i.valid]
        record = replace(record, interfaces=filtered_interfaces)
        if len(record.interfaces) > 0:
            filtered_records.append(record)

    # Filter records by resolution
    filtered_records = [r for r in filtered_records if r.structure.resolution <= cfg.resolution_cutoff]

    # Convert records to CSV
    csv_out = f"{cfg.pdb_path}/interface_info.csv"
    interface_info = defaultdict(list)
    for record in filtered_records:
        for interface in record.interfaces:
            interface_info["record_id"].append(record.id)
            interface_info["resolution"].append(record.structure.resolution)

            chain_1_record = record.chains[interface.chain_1]
            chain_2_record = record.chains[interface.chain_2]
            interface_info["chain_1_name"].append(chain_1_record.chain_name)
            interface_info["chain_1_cluster_id"].append(chain_1_record.cluster_id)
            interface_info["chain_1_num_residues"].append(chain_1_record.num_residues)

            interface_info["chain_2_name"].append(chain_2_record.chain_name)
            interface_info["chain_2_cluster_id"].append(chain_2_record.cluster_id)
            interface_info["chain_2_num_residues"].append(chain_2_record.num_residues)

    interface_info = pd.DataFrame(interface_info)
    interface_info.to_csv(csv_out, index=False)

    # Save each interface as a mmCIF file
    use_parallel = cfg.num_workers > 1
    save_interface_fn = partial(save_interface_as_mmcif,
                                processed_structure_dir=f"{cfg.processed_targets_dir}/structures",
                                interface_out_dir=interface_out_dir)
    if use_parallel:
        with Parallel(n_jobs=cfg.num_workers) as parallel_pool:
            jobs = [delayed(save_interface_fn)(record) for record in filtered_records]
            list(parallel_pool(tqdm(jobs, total=len(jobs), desc="Saving interfaces")))
    else:
        for record in tqdm(filtered_records, desc="Saving interfaces"):
            save_interface_fn(record)


def save_interface_as_mmcif(record: Record,
                            processed_structure_dir: str,
                            interface_out_dir: str) -> None:
    """
    Given a record, load in the processed structure. For each interface, filter for the interface and save it as a mmCIF file.
    """
    processed_structure_file = f"{processed_structure_dir}/{record.id}.npz"
    structure = load_input(processed_structure_file).structure

    for interface in record.interfaces:
        chain_record_1, chain_record_2 = record.chains[interface.chain_1], record.chains[interface.chain_2]
        chain_name_1, chain_name_2 = chain_record_1.chain_name, chain_record_2.chain_name
        chain_1, chain_2 = structure.chains[interface.chain_1], structure.chains[interface.chain_2]
        interface_structure = replace(structure, chains=[chain_1, chain_2])

        out_file = f"{interface_out_dir}/{Path(processed_structure_file).stem}_{chain_name_1}_{chain_name_2}.cif"
        mmcif_str = to_mmcif(interface_structure)
        with open(out_file, "w") as f:
            f.write(mmcif_str)


if __name__ == "__main__":
    main()
