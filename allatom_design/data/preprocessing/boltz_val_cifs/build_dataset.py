#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.filter.dynamic.chain_type_size import \
    ChainTypeSizeFilter
from allatom_design.data.filter.dynamic.max_residues import MaxResiduesFilter
from allatom_design.data.preprocessing.boltz_utils.parsing_utils import \
    load_input
from allatom_design.data.types import Record
from allatom_design.data.write.mmcif import write_sd_feats_to_mmcif
from allatom_design.eval.eval_utils.eval_setup_utils import process_pdb_files
from allatom_design.eval.eval_utils.seq_des_utils import get_sd_batch


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_val_cifs", config_name="build_dataset", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given the Boltz-1 validation split, retrieve the mmCIF files from the downloaded mmCIF directory from RCSB.
    Also create some pdb names lists for various subsets of the validation set.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Read in validation ids
    with open(cfg.val_ids_txt, "r") as f:
        val_ids = [line.strip() for line in f.readlines()]
    val_ids = set([id.lower() for id in val_ids])

    # Retrieve mmCIF files
    out_mmcif_files = []
    pdb_dir = f"{cfg.out_dir}/pdbs"
    Path(pdb_dir).mkdir(parents=True, exist_ok=True)
    mmcif_files = Path(cfg.mmcif_dir).rglob("*.cif")

    for mmcif_file in tqdm(list(mmcif_files), desc="Copying mmCIF files to output directory"):
        pdb_id = mmcif_file.stem.lower()
        if pdb_id in val_ids:
            out_mmcif_file = f"{pdb_dir}/{pdb_id}.cif"
            shutil.copy(mmcif_file, out_mmcif_file)

            out_mmcif_files.append(out_mmcif_file)
            val_ids.remove(pdb_id)

    if len(val_ids) > 0:
        print(f"Warning: did not find the following PDB IDs in the mmCIF directory: {val_ids}")
    else:
        print("Successfully found all PDB IDs in the mmCIF directory.")

    # Process structures to get info about them
    processed_struct_dir = f"{cfg.out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(out_mmcif_files, processed_struct_dir=processed_struct_dir, **cfg.pdb_processing_cfg)

    # Load in records
    record_dir = f"{processed_struct_dir}/records"
    records = []
    for processed_struct_file in tqdm(processed_struct_files, desc="Loading input data"):
        # Read in record
        with open(f"{record_dir}/{Path(processed_struct_file).stem}.json", "r") as f:
            records.append(Record.from_dict(json.load(f)))

    # Filter for single protein chain, total residues between [32, 512]
    filters = [
        ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=1, max_chains=1, min_residues=None, max_residues=None),
        MaxResiduesFilter(min_residues=32, max_residues=512),
    ]
    filtered_records = [r for r in records if all(f.filter(r) for f in filters)]

    # Save PDB names to txt
    lists_dir = f"{cfg.out_dir}/pdb_name_lists"
    Path(lists_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{lists_dir}/protein_monomer_32_512.txt", "w") as f:
        for record in filtered_records:
            f.write(f"{record.id}.cif\n")

    # TEMP: save cifs to new directory for convenience
    # Initialize tokenizer and featurizer
    data_cfg = hydra.utils.instantiate(cfg.data_cfg)

    protein_monomer_32_512_dir = f"{cfg.out_dir}/protein_monomer_32_512"
    Path(protein_monomer_32_512_dir).mkdir(parents=True, exist_ok=True)
    for record in filtered_records:
        processed_struct_file = f"{processed_struct_dir}/structures/{record.id}.npz"
        example, input_structure = get_sd_batch([processed_struct_file], device="cpu", data_cfg=data_cfg, parallel_pool=None)
        write_sd_feats_to_mmcif(example, input_structure, [f"{protein_monomer_32_512_dir}/{record.id}.cif"])


if __name__ == "__main__":
    main()
