#!/usr/bin/env python3
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.eval.eval_utils.eval_setup_utils import process_pdb_files


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
    process_pdb_files(out_mmcif_files, processed_struct_dir=f"{cfg.out_dir}/processed_structures", **cfg.pdb_processing_cfg)




if __name__ == "__main__":
    main()
