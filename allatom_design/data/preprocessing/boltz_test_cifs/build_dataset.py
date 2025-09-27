#!/usr/bin/env python3
import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.filter.dynamic.chain_type_size import \
    ChainTypeSizeFilter
from allatom_design.data.filter.dynamic.max_residues import MaxResiduesFilter
from allatom_design.data.filter.dynamic.size import SizeFilter
from allatom_design.data.types import Record
from allatom_design.data.write.mmcif import batch_write_feats_to_mmcif
from allatom_design.eval.eval_utils.eval_setup_utils import process_pdb_files
from allatom_design.eval.eval_utils.seq_des_utils import get_sd_batch
from allatom_design.utils.feature_utils import unbatch_feats


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_test_cifs", config_name="build_dataset", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given the Boltz-1 test split, retrieve the mmCIF files from the downloaded mmCIF directory from RCSB.
    Also create some pdb names lists for various subsets of the test set.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Read in test ids and retrieve mmCIF files
    with open(cfg.test_ids_txt, "r") as f:
        test_ids = [line.strip() for line in f.readlines()]
    test_ids = set([id.lower() for id in test_ids])

    mmcif_files = Path(cfg.mmcif_dir).rglob("*.cif")
    mmcif_files = [f for f in mmcif_files if Path(f).stem.lower() in test_ids]

    # Check if we found all the mmCIF files
    found_mmcif_files = set([Path(f).stem.lower() for f in mmcif_files])
    if found_mmcif_files != test_ids:
        print(f"Warning: did not find the following PDB IDs in the mmCIF directory: {test_ids - found_mmcif_files}")
    else:
        print("Successfully found all PDB IDs in the mmCIF directory.")

    # Exclude PDBs
    if cfg.exclude_pdb_keys is not None:
        print(f"Excluding PDBs: {cfg.exclude_pdb_keys}, len(mmcif_files): {len(mmcif_files)}")
        mmcif_files = [f for f in mmcif_files if Path(f).stem.lower() not in cfg.exclude_pdb_keys]
        print(f"Found {len(mmcif_files)} mmCIF files after excluding PDBs")

    # Process mmCIF files to get info about them (e.g. taking only first bioassembly)
    processed_struct_dir = f"{cfg.out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(mmcif_files, processed_struct_dir=processed_struct_dir, **cfg.pdb_processing_cfg)

    # Copy mmCIF files to a new directory
    pdb_dir = f"{cfg.out_dir}/pdbs"
    Path(pdb_dir).mkdir(parents=True, exist_ok=True)
    data_cfg = hydra.utils.instantiate(cfg.data_cfg)
    parallel_context = Parallel(n_jobs=cfg.num_workers) if cfg.num_workers > 1 else nullcontext()  # for loading PDBs in parallel

    # Store features in memory
    record_id_to_feats = {}
    with parallel_context:
        B = 32
        for i in tqdm(range(0, len(processed_struct_files), B), desc="Copying mmCIF files to output directory"):
            batch_struct_files = processed_struct_files[i:i+B]
            batch, input_structs = get_sd_batch(batch_struct_files, device="cpu", data_cfg=data_cfg, parallel_pool=None)
            filenames = [f"{pdb_dir}/{Path(struct_file).stem}.cif" for struct_file in batch_struct_files]
            # batch_write_feats_to_mmcif(batch, input_structs, filenames)

            feats_list = unbatch_feats(batch)
            for bi in range(len(batch["pdb_key"])):
                record_id = batch["pdb_key"][bi]
                record_id_to_feats[record_id] = feats_list[bi]

    # Load in records and filter
    record_dir = f"{processed_struct_dir}/records"
    records = []
    for processed_struct_file in tqdm(processed_struct_files, desc="Loading input data"):
        with open(f"{record_dir}/{Path(processed_struct_file).stem}.json", "r") as f:
            records.append(Record.from_dict(json.load(f)))

    test_subset_filters = {
        # "protein_monomer_32_512": [
        #     ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=1, max_chains=1, min_residues=None, max_residues=None),
        #     MaxResiduesFilter(min_residues=32, max_residues=512),
        # ],
        # "protein_monomer_32_256": [
        #     ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=1, max_chains=1, min_residues=None, max_residues=None),
        #     MaxResiduesFilter(min_residues=32, max_residues=256),
        # ],
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
        # "protein_interface_32_256_no_ligand": [
        #     ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=2, max_chains=2, min_residues=None, max_residues=None),
        #     MaxResiduesFilter(min_residues=32, max_residues=256),
        #     SizeFilter(max_chains=2),
        # ],
        # "protein_interface_32_512_no_ligand": [
        #     ChainTypeSizeFilter(chain_type="PROTEIN", min_chains=2, max_chains=2, min_residues=None, max_residues=None),
        #     MaxResiduesFilter(min_residues=32, max_residues=512),
        #     SizeFilter(max_chains=2),
        # ]
    }

    for test_subset_name, filters in test_subset_filters.items():
        filtered_records = [r for r in records if all(f.filter(r) for f in filters)]
        print(f"Found {len(filtered_records)} records for {test_subset_name}")

        # Save PDB names to txt
        lists_dir = f"{cfg.out_dir}/pdb_name_lists"
        Path(lists_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{lists_dir}/{test_subset_name}.txt", "w") as f:
            for record in filtered_records:
                f.write(f"{record.id}.cif\n")

        # Also save csv of PDB names with their lengths
        length_list_dir = f"{cfg.out_dir}/length_lists"
        Path(length_list_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{length_list_dir}/{test_subset_name}.csv", "w") as f:
            for record in filtered_records:
                num_residues = sum(chain.num_residues for chain in record.chains)
                f.write(f"{record.id},{num_residues}\n")

        # Save sse to .pt file
        sse_anno_dir = f"{cfg.out_dir}/sse_anno"
        Path(sse_anno_dir).mkdir(parents=True, exist_ok=True)
        record_id_to_sse = {}
        for record in filtered_records:
            record_id = record.id
            sse = record_id_to_feats[record_id]["sse"]
            token_pad_mask = record_id_to_feats[record_id]["token_pad_mask"]
            token_resolved_mask = record_id_to_feats[record_id]["token_resolved_mask"]
            record_id_to_sse[record_id] = sse[token_resolved_mask.bool() & token_pad_mask.bool()]
        torch.save(record_id_to_sse, f"{sse_anno_dir}/{test_subset_name}.pt")

        if cfg.copy_subset_cifs:
            subset_cif_dir = f"{cfg.out_dir}/subset_cifs/{test_subset_name}"
            Path(subset_cif_dir).mkdir(parents=True, exist_ok=True)

            for record in tqdm(filtered_records, desc=f"Copying over {test_subset_name} cifs"):
                shutil.copy(f"{pdb_dir}/{record.id}.cif", f"{subset_cif_dir}/{record.id}.cif")


if __name__ == "__main__":
    main()
