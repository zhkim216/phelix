#!/usr/bin/env python3
import glob
import os
import shutil
from pathlib import Path

import gemmi
import hydra
import lightning as L
import torch
from Bio import PDB
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_val_cifs", config_name="make_md_pd_conformers", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script to process partial diffusion PDBs from Petr's MDCATH-trained model.
    """
    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Create output directories
    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Gather partial diffusion directories
    pd_dirs = natsorted(glob.glob(f"{cfg.input_dir}/*"))
    # Build parallel jobs
    jobs = [
        delayed(process_pd_dir)(pd_dir, out_dir, cfg.boltz_val_cifs_dir)
        for pd_dir in pd_dirs
    ]

    # Run parallel or not
    use_parallel = cfg.num_workers > 1
    if use_parallel:
        parallel = Parallel(n_jobs=cfg.num_workers)
        list(parallel(tqdm(jobs, total=len(jobs), desc="Processing pd_dirs")))
    else:
        for job in tqdm(jobs, total=len(jobs), desc="Processing pd_dirs"):
            job()


def process_pd_dir(pd_dir: str, out_dir: Path, boltz_val_cifs_dir: str):
    """
    Processes one partial-diffusion directory:
      1. Finds all PDB samples
      2. Splits them into models
      3. Fixes SEQRES using gemmi
    """
    samples = glob.glob(f"{pd_dir}/samples/*.pdb")
    pd_out_dir = f"{out_dir}/{Path(pd_dir).stem}"
    Path(pd_out_dir).mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, desc="Splitting partial diffusion PDBs", leave=False):
        record_id = Path(sample).stem.split("__")[1]
        sample_out_dir = f"{pd_out_dir}/{record_id}"
        Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

        # Copy native PDB to output directory
        native_pdb_path = f"{boltz_val_cifs_dir}/{record_id}.cif"
        shutil.copy(native_pdb_path, f"{sample_out_dir}/{record_id}.cif")

        # Split the PDB into models
        out_files = split_pdb_models_biopython(sample, record_id, sample_out_dir)

        # Finally, we need to use gemmi to fix the seqids in the PDBs to match the native PDB
        fix_pdb_seqids(native_pdb_path, out_files, sample_out_dir)


def split_pdb_models_biopython(pdb_path: str, record_id: str, sample_out_dir: str):
    """
    Splits a multi-model PDB into separate files using Biopython.
    Biopython is used because gemmi throws an error when reading these PDBs.

    :param pdb_path: Path to the multi-model PDB file.
    :param out_dir: Output directory for individual model PDBs.
    """
    # Parse the structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("my_structure", pdb_path)

    # Iterate over models, save each to a new file
    out_files = []
    for i, model in enumerate(structure):
        out_file = f"{sample_out_dir}/{record_id}_model_{i}.pdb"
        io = PDB.PDBIO()
        io.set_structure(model)
        io.save(out_file)
        out_files.append(out_file)

    return out_files


def fix_pdb_seqids(native_pdb_path: str, pdb_files: list[str], sample_out_dir: str):
    """
    Fixes the seqids in the PDBs to match the native PDB.
    Assumes single chain per model.
    """
    native_structure = gemmi.read_structure(native_pdb_path)
    native_structure.setup_entities()
    native_chain: gemmi.ResidueSpan = native_structure[0].subchains()[0]

    for pdb_file in pdb_files:
        structure = gemmi.read_structure(pdb_file)
        structure.setup_entities()

        # create mapping from subchain id to entity
        entities: dict[str, gemmi.Entity] = {}
        for entity in structure.entities:
            entity: gemmi.Entity
            if entity.entity_type.name == "Water":
                continue
            for subchain_id in entity.subchains:
                entities[subchain_id] = entity


        # Set seqid to match native PDB
        for chain in structure[0].subchains():
            chain: gemmi.ResidueSpan
            for i in range(len(native_chain)):
                chain[i].seqid = native_chain[i].seqid

            # Set sequence for this entity
            subchain_id = chain.subchain_id()
            entities[subchain_id].full_sequence = chain.extract_sequence()


        # Write PDB file (needs to be PDB, not mmCIF, since we add X's to gaps in PDB to CIF conversion later)
        pdb_str = structure.make_pdb_string()
        with open(pdb_file, "w") as f:
            f.write(pdb_str)


if __name__ == "__main__":
    main()
