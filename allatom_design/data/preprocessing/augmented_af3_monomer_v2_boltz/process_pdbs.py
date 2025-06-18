#!/usr/bin/env python3
import glob
import json
import pickle
import shutil
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import rdkit.Chem
from allatom_design.data.filter.static.ligand import ExcludedLigands
from allatom_design.data.filter.static.polymer import (ClashingChainsFilter,
                                                       MinimumLengthFilter,
                                                       UnknownFilter)
from joblib import Parallel, delayed
from omegaconf import DictConfig
from p_tqdm import p_umap
from tqdm import tqdm

from allatom_design.data.preprocessing.boltz_utils.parsing_utils import (
    fetch, finalize, process_structure, pdb_to_mmcif, Resource)


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

    # ───────────────────────────── Boltz processing ────────────────────────────
    # Convert all pdbs into mmcif format first
    use_parallel = cfg.num_workers > 1
    pdb_paths = glob.glob(f"{cfg.augmented_af3_monomer_v2_path}/esmfold_preds/*.pdb")
    mmcif_dir = f"{cfg.out_dir}/mmcifs"
    Path(mmcif_dir).mkdir(parents=True, exist_ok=True)
    if use_parallel:
        with Parallel(n_jobs=cfg.num_workers) as parallel_pool:
            jobs = [delayed(pdb_to_mmcif)(pdb_path, Path(mmcif_dir, Path(pdb_path).name.replace(".pdb", ".cif")), assign_label_seq_id=False) for pdb_path in pdb_paths]
            list(parallel_pool(tqdm(jobs, total=len(jobs), desc="Converting PDBs to mmCIFs")))
    else:
        for pdb_path in tqdm(pdb_paths, desc="Converting PDBs to mmCIFs"):
            mmcif_out = Path(mmcif_dir, Path(pdb_path).name.replace(".pdb", ".cif"))
            pdb_to_mmcif(pdb_path, mmcif_out, assign_label_seq_id=False)

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
    mmcif_files = Path(mmcif_dir).rglob("*.cif")
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
    update_manifest_from_csv(manifest_path=f"{processed_targets_dir}/manifest_unclustered.json", manifest_df=manifest_df)
    print("Updated manifest records from CSV.")


def update_manifest_from_csv(manifest_path: str, manifest_df: pd.DataFrame) -> None:
    """
    Add designability info, cluster ID, and phase from manifest_df to records in manifest_path.
    """
    from allatom_design.data.types import DesignabilityInfo, Manifest

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
    new_manifest_path = manifest_path.replace("_unclustered", "")
    with open(new_manifest_path, "w") as f:
        json.dump(new_records, f)


if __name__ == "__main__":
    main()
