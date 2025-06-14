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
from allatom_design.data.filter.dynamic.size import SizeFilter
from allatom_design.data.types import Record
from allatom_design.data.write.mmcif import write_feats_to_mmcif
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

    # Read in validation ids and retrieve mmCIF files
    with open(cfg.val_ids_txt, "r") as f:
        val_ids = [line.strip() for line in f.readlines()]
    val_ids = set([id.lower() for id in val_ids])

    mmcif_files = Path(cfg.mmcif_dir).rglob("*.cif")
    mmcif_files = [f for f in mmcif_files if Path(f).stem.lower() in val_ids]

    # Check if we found all the mmCIF files
    found_mmcif_files = set([Path(f).stem.lower() for f in mmcif_files])
    if found_mmcif_files != val_ids:
        print(f"Warning: did not find the following PDB IDs in the mmCIF directory: {val_ids - found_mmcif_files}")
    else:
        print("Successfully found all PDB IDs in the mmCIF directory.")

    # Process mmCIF files to get info about them (e.g. taking only first bioassembly)
    processed_struct_dir = f"{cfg.out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(mmcif_files, processed_struct_dir=processed_struct_dir, **cfg.pdb_processing_cfg)

    # Copy mmCIF files to a new directory
    pdb_dir = f"{cfg.out_dir}/pdbs"
    Path(pdb_dir).mkdir(parents=True, exist_ok=True)
    data_cfg = hydra.utils.instantiate(cfg.data_cfg)
    for processed_struct_file in tqdm(processed_struct_files, desc="Copying mmCIF files to output directory"):
        # TODO: this can be easily parallelized
        record = Record.from_dict(json.load(open(f"{processed_struct_dir}/records/{Path(processed_struct_file).stem}.json")))
        processed_struct_file = f"{processed_struct_dir}/structures/{record.id}.npz"
        example, input_structure = get_sd_batch([processed_struct_file], device="cpu", data_cfg=data_cfg, parallel_pool=None)
        write_feats_to_mmcif(example, input_structure, f"{pdb_dir}/{record.id}.cif")

    # Load in records and filter
    record_dir = f"{processed_struct_dir}/records"
    records = []
    for processed_struct_file in tqdm(processed_struct_files, desc="Loading input data"):
        with open(f"{record_dir}/{Path(processed_struct_file).stem}.json", "r") as f:
            records.append(Record.from_dict(json.load(f)))

    val_subset_filters = {
        "protein_monomer_32_512": [
            ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=1, max_chains=1, min_residues=None, max_residues=None),
            MaxResiduesFilter(min_residues=32, max_residues=512),
        ],
        "protein_monomer_32_256": [
            ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=1, max_chains=1, min_residues=None, max_residues=None),
            MaxResiduesFilter(min_residues=32, max_residues=256),
        ],
        "protein_monomer_32_512_no_ligand": [
            ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=1, max_chains=1, min_residues=None, max_residues=None),
            MaxResiduesFilter(min_residues=32, max_residues=512),
            SizeFilter(max_chains=1),
        ],
        "protein_monomer_32_256_no_ligand": [
            ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=1, max_chains=1, min_residues=None, max_residues=None),
            MaxResiduesFilter(min_residues=32, max_residues=256),
            SizeFilter(max_chains=1),
        ],
    }

    for val_subset_name, filters in val_subset_filters.items():
        filtered_records = [r for r in records if all(f.filter(r) for f in filters)]
        print(f"Found {len(filtered_records)} records for {val_subset_name}")

        # Save PDB names to txt
        lists_dir = f"{cfg.out_dir}/pdb_name_lists"
        Path(lists_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{lists_dir}/{val_subset_name}.txt", "w") as f:
            for record in filtered_records:
                f.write(f"{record.id}.cif\n")

        # Also save csv of PDB names with their lengths
        length_list_dir = f"{cfg.out_dir}/length_lists"
        Path(length_list_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{length_list_dir}/{val_subset_name}.csv", "w") as f:
            for record in filtered_records:
                num_residues = sum(chain.num_residues for chain in record.chains)
                f.write(f"{record.id},{num_residues}\n")

        if cfg.copy_subset_cifs:
            subset_cif_dir = f"{cfg.out_dir}/subset_cifs/{val_subset_name}"
            Path(subset_cif_dir).mkdir(parents=True, exist_ok=True)

            for record in tqdm(filtered_records, desc=f"Copying over {val_subset_name} cifs"):
                shutil.copy(f"{pdb_dir}/{record.id}.cif", f"{subset_cif_dir}/{record.id}.cif")


if __name__ == "__main__":
    main()
