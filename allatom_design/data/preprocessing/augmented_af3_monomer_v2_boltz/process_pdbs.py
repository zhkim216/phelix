#!/usr/bin/env python3
import glob
import json
import pickle
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import gemmi
import hydra
import pandas as pd
import rdkit
import rdkit.Chem
from boltz.data.filter.static.ligand import ExcludedLigands
from boltz.data.filter.static.polymer import (ClashingChainsFilter,
                                              ConsecutiveCA,
                                              MinimumLengthFilter,
                                              UnknownFilter)
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    fetch, finalize, process_structure)
from allatom_design.data.types import DesignabilityInfo, Manifest


@hydra.main(config_path="../../../configs/data/preprocessing/augmented_af3_monomer_v2_boltz", config_name="process_pdbs", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Parse augmented_af3_monomer_v2 dataset into boltz format.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Copy over previous manifest and pdb_names lists
    shutil.copy(f"{cfg.augmented_af3_monomer_v2_path}/pdb_manifest.csv", f"{cfg.pdb_path}/manifest.csv")
    for phase in ["train", "eval", "eval2"]:
        shutil.copy(f"{cfg.augmented_af3_monomer_v2_path}/{phase}_pdb_names.list", f"{cfg.pdb_path}/{phase}_pdb_names.list")

    manifest_df = pd.read_csv(f"{cfg.pdb_path}/manifest.csv")

    ### Boltz processing ###
    # First, convert all pdbs into mmcif format
    use_parallel = cfg.num_workers > 1
    pdb_paths = glob.glob(f"{cfg.augmented_af3_monomer_v2_path}/esmfold_preds/*.pdb")
    mmcif_dir = f"{cfg.out_dir}/mmcifs"
    Path(mmcif_dir).mkdir(parents=True, exist_ok=True)
    if use_parallel:
        with Parallel(n_jobs=cfg.num_workers) as parallel_pool:
            jobs = [delayed(pdb_to_mmcif)(pdb_path, Path(mmcif_dir, Path(pdb_path).name.replace(".pdb", ".cif"))) for pdb_path in pdb_paths]
            list(parallel_pool(tqdm(jobs, total=len(jobs), desc="Converting PDBs to mmCIFs")))
    else:
        for pdb_path in tqdm(pdb_paths, desc="Converting PDBs to mmCIFs"):
            mmcif_out = Path(mmcif_dir, Path(pdb_path).name.replace(".pdb", ".cif"))
            pdb_to_mmcif(pdb_path, mmcif_out)

    # For this dataset, we'll manually assign clusters based on the manifest.csv file
    clusters = {}

    # Construct static filters
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ConsecutiveCA(max_dist=10.0),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Load in CCD resource
    pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
    rdkit.Chem.SetDefaultPickleProperties(pickle_option)
    with open(cfg.ccd_pkl, "rb") as f:
        ccd_resource = pickle.load(f)

    # Fetch data
    data = fetch(datadir=Path(mmcif_dir), max_file_size=None)

    # Run processing
    processed_targets_dir = f"{cfg.pdb_path}/processed_targets"
    Path(processed_targets_dir).mkdir(parents=True, exist_ok=True)

    if use_parallel:
        with Parallel(n_jobs=cfg.num_workers) as parallel_pool:
            jobs = [delayed(process_structure)(pdb, ccd_resource, processed_targets_dir, filters, clusters) for pdb in data]
            list(parallel_pool(tqdm(jobs, total=len(jobs), desc="Processing mmCIFs")))
    else:
        for pdb in tqdm(data, desc="Processing mmCIFs"):
            process_structure(pdb, ccd_resource, processed_targets_dir, filters, clusters)

    # Run post-processing to create manifest.json
    finalize(outdir=Path(processed_targets_dir))

    # Based on manifest.csv, load in designability info and add to manifest.json
    update_manifest_from_csv(manifest_path=f"{processed_targets_dir}/manifest.json", manifest_df=manifest_df)
    print("Updated manifest records from CSV.")


def update_manifest_from_csv(manifest_path: str, manifest_df: pd.DataFrame) -> None:
    """
    Add designability info, cluster ID, and phase from manifest_df to records in manifest_path.
    """
    # Load manifest
    manifest = Manifest.load(Path(manifest_path))

    # Add designability info from manifest_df to records
    manifest_df["id"] = manifest_df["pdb_name"].apply(lambda x: Path(x).stem.lower())
    manifest_df = manifest_df.set_index("id")

    new_records = []
    for record in manifest.records:
        row = manifest_df.loc[record.id].to_dict()

        # Add designability info
        designability_info = DesignabilityInfo(
            sc_ca_rmsd=row["sc_ca_rmsd"],
            sc_ca_tm=row["sc_ca_tm"],
            sc_aa_rmsd=row["sc_aa_rmsd"],
            avg_ca_plddt=row["avg_ca_plddt"],
            radius_of_gyration=row["radius_of_gyration"],
            ideal_rad=row["ideal_rad"],
            rel_rog=row["rel_rog"],
        )
        record = replace(record, designability_info=designability_info)

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


def pdb_to_mmcif(pdb_path: str, mmcif_out: Path) -> None:
    """
    Convert a PDB file to mmCIF format using gemmi.
    """
    if Path(mmcif_out).exists():
        return

    structure = gemmi.read_structure(pdb_path)
    structure.setup_entities()

    # Create mapping from subchain id to entity
    entities: dict[str, gemmi.Entity] = {}
    for entity in structure.entities:
        entity: gemmi.Entity
        if entity.entity_type.name == "Water":
            continue
        for subchain_id in entity.subchains:
            entities[subchain_id] = entity

    # Set sequence for each entity based on the sequence in the model, since we do not have SEQRES in these files
    for raw_chain in structure[0].subchains():
        model_sequence = raw_chain.extract_sequence()

        subchain_id = raw_chain.subchain_id()
        entities[subchain_id].full_sequence = model_sequence

    # Write mmCIF file
    mmcif_doc = structure.make_mmcif_document()
    mmcif_doc.write_file(str(mmcif_out))


if __name__ == "__main__":
    main()
