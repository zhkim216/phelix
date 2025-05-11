#!/usr/bin/env python3
import glob
import json
import pickle
import shutil
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import gemmi
import hydra
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from p_tqdm import p_umap
from tqdm import tqdm

from allatom_design.data.filter.static.ligand import ExcludedLigands
from allatom_design.data.filter.static.polymer import (ClashingChainsFilter,
                                                       MinimumLengthFilter,
                                                       UnknownFilter)
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    Resource, fetch, finalize, process_structure)
from allatom_design.data.types import Manifest


@hydra.main(config_path="../../../configs/data/preprocessing/af3_pdb_monomer_boltz", config_name="process_pdbs", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Parse af3_pdb_monomer dataset into boltz format.

    Note: we set SEQRES records to the sequence in the model, so we cannot trust label_seq_id here and should instead use auth_seq_id.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Read in previous manifest
    manifest_df = pd.read_csv(f"{cfg.af3_pdb_monomer_path}/pdb_manifest.csv")

    # Copy over previous manifest and pdb_names lists
    shutil.copy(f"{cfg.af3_pdb_monomer_path}/pdb_manifest.csv", f"{cfg.pdb_path}/manifest.csv")
    for phase in ["train", "eval", "eval2"]:
        shutil.copy(f"{cfg.af3_pdb_monomer_path}/{phase}_pdb_names.list", f"{cfg.pdb_path}/{phase}_pdb_names.list")

    manifest_df = pd.read_csv(f"{cfg.pdb_path}/manifest.csv")

    # ───────────────────────────── Boltz processing ────────────────────────────
    # First, set up entities in all cif files with gemmi
    use_parallel = cfg.num_workers > 1
    input_dir = f"{cfg.af3_pdb_monomer_path}/preprocessing/residx_quality_control_af3_monomer/filtered_mmcifs"
    mmcif_dir = f"{cfg.out_dir}/mmcifs"
    Path(mmcif_dir).mkdir(parents=True, exist_ok=True)
    input_cif_files = glob.glob(f"{input_dir}/*.cif")
    if use_parallel:
        with Parallel(n_jobs=cfg.num_workers) as parallel_pool:
            jobs = [delayed(setup_entities_in_cif)(str(cif_file), str(Path(mmcif_dir, Path(cif_file).name))) for cif_file in input_cif_files]
            list(parallel_pool(tqdm(jobs, total=len(jobs), desc="Setting up entities and SEQRES records in mmCIFs")))
    else:
        for cif_file in tqdm(input_cif_files, desc="Setting up entities and SEQRES records in mmCIFs"):
            out_file = str(Path(mmcif_dir, Path(cif_file).name))
            setup_entities_in_cif(str(cif_file), out_file)


    # For this dataset, we'll manually assign clusters based on the manifest.csv file
    clusters = {}

    # Static filters
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Load or seed CCD resource in Redis
    if cfg.redis_host is not None:
        resource = Resource(host=cfg.redis_host, port=cfg.redis_port)
    else:
        resource = pickle.load(open(cfg.ccd_pkl_path, "rb"))

    # Fetch data
    mmcif_files = glob.glob(f"{mmcif_dir}/*.cif")
    data = fetch(mmcif_files, max_file_size=None)

    # Run processing
    processed_targets_dir = f"{cfg.pdb_path}/processed_targets"
    Path(processed_targets_dir).mkdir(parents=True, exist_ok=True)
    if use_parallel:
        fn = partial(
            process_structure,
            resource=resource,
            outdir=Path(processed_targets_dir),
            filters=filters,
            clusters=clusters,
        )
        p_umap(fn, data, num_cpus=cfg.num_workers, desc="Processing mmCIFs")
    else:
        for pdb in tqdm(data, desc="Processing mmCIFs"):
            process_structure(
                pdb,
                resource=resource,
                outdir=Path(processed_targets_dir),
                filters=filters,
                clusters=clusters,
            )

    # Post‑processing to create manifest.json
    finalize(outdir=Path(processed_targets_dir))

    # Based on manifest.csv, load in designability info and add to manifest.json
    update_manifest_from_csv(manifest_path=f"{processed_targets_dir}/manifest.json", manifest_df=manifest_df)
    print("Updated manifest records from CSV.")


def update_manifest_from_csv(manifest_path: str, manifest_df: pd.DataFrame) -> None:
    """
    Add cluster ID and phase from manifest_df to records in manifest_path.
    """

    # Load manifest
    manifest = Manifest.load(Path(manifest_path))

    # Add cluster ID from manifest_df to records
    manifest_df["id"] = manifest_df["pdb_name"].apply(lambda x: Path(x).stem.lower())
    manifest_df = manifest_df.set_index("id")

    new_records = []
    for record in manifest.records:
        row = manifest_df.loc[record.id].to_dict()

        # Add cluster ID to chain, assuming monomeric structure
        if len(record.chains) != 1:
            raise ValueError(f"Expected monomeric structure, got {len(record.chains)} chains for {record.id}")
        chains = [replace(chain, cluster_id=row["cluster_id"]) for chain in record.chains]
        record = replace(record, chains=chains)

        # Add phase
        record = replace(record, phase=row["phase"])

        new_records.append(asdict(record))

    # Save manifest records back to file
    with open(manifest_path, "w") as f:
        json.dump(new_records, f)


def setup_entities_in_cif(cif_file: str, out_file: str) -> None:
    """
    Setup entities in a cif file with gemmi. Writes out a new mmcif file with entities set.

    Also, we set SEQRES records to the sequence in the model, so we cannot trust label_seq_id here.
    """
    structure = gemmi.read_structure(cif_file)
    structure.setup_entities()

    # Set sequence for each entity based on the sequence in the model, since we do not have SEQRES in these files
    # create mapping from subchain id to entity
    entities: dict[str, gemmi.Entity] = {}
    for entity in structure.entities:
        entity: gemmi.Entity
        if entity.entity_type.name == "Water":
            continue
        for subchain_id in entity.subchains:
            entities[subchain_id] = entity

    # set sequence for each entity
    for raw_chain in structure[0].subchains():
        model_sequence = raw_chain.extract_sequence()
        subchain_id = raw_chain.subchain_id()
        entities[subchain_id].full_sequence = model_sequence

    structure.make_mmcif_document().write_file(out_file)


if __name__ == "__main__":
    main()
