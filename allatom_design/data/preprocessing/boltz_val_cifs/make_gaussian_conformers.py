#!/usr/bin/env python3
import copy
import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import hydra
import lightning as L
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.data.types import Record
from allatom_design.data.write.mmcif import write_feats_to_mmcif
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             process_pdb_files)
from allatom_design.eval.eval_utils.seq_des_utils import get_sd_batch


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_val_cifs", config_name="make_gaussian_conformers", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given PDBs from boltz_val_cifs, generate multiple "Gaussian conformers" for each cif by adding Gaussian noise to the coordinates.
    """
    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Create output directories
    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for eps in cfg.eps_values:
        Path(f"{out_dir}/eps{eps}").mkdir(parents=True, exist_ok=True)

    # Load in PDB files to make conformers from
    pdb_files = get_pdb_files(**cfg.input_cfg)
    temp_processed_struct_dir = f"{out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=temp_processed_struct_dir, **cfg.pdb_processing_cfg)

    data_cfg = hydra.utils.instantiate(cfg.data_cfg)

    use_parallel = cfg.num_workers > 1
    if use_parallel:
        parallel = Parallel(n_jobs=cfg.num_workers)
        jobs = [delayed(generate_gaussian_conformers)(processed_struct_file, data_cfg, cfg.eps_values, cfg.num_conformers, out_dir) for processed_struct_file in processed_struct_files]
        list(parallel(tqdm(jobs, total=len(jobs), desc="Generating Gaussian conformers")))
    else:
        for processed_struct_file in tqdm(processed_struct_files, desc="Generating Gaussian conformers"):
            generate_gaussian_conformers(processed_struct_file, data_cfg, cfg.eps_values, cfg.num_conformers, out_dir)

    # Clean up temp directory
    shutil.rmtree(temp_processed_struct_dir)


def generate_gaussian_conformers(processed_struct_file: str, data_cfg: DictConfig,
                                 eps_values: list[float], num_conformers: int, out_dir: str):
    """
    Given a processed structure directory, generate multiple "Gaussian conformers" for each cif by adding Gaussian noise to the coordinates.
    """
    processed_struct_dir = Path(processed_struct_file).parent.parent

    record = Record.from_dict(json.load(open(f"{processed_struct_dir}/records/{Path(processed_struct_file).stem}.json")))
    processed_struct_file = f"{processed_struct_dir}/structures/{record.id}.npz"
    example, input_structure = get_sd_batch([processed_struct_file], device="cpu", data_cfg=data_cfg, parallel_pool=None)

    for eps in eps_values:
        # Save original cif
        conformer_out_dir = f"{out_dir}/eps{eps}/{record.id}"
        Path(conformer_out_dir).mkdir(parents=True, exist_ok=True)
        write_feats_to_mmcif(example, input_structure, f"{conformer_out_dir}/{record.id}.cif")

        for i in range(num_conformers):
            example_noised = copy.deepcopy(example)
            # Add Gaussian noise to the coordinates
            example_noised["coords"] = example_noised["coords"] + torch.randn_like(example_noised["coords"]) * eps
            example_noised["coords"] = example_noised["coords"] * example_noised["atom_pad_mask"].unsqueeze(-1) * example_noised["atom_resolved_mask"].unsqueeze(-1)
            write_feats_to_mmcif(example_noised, input_structure, f"{conformer_out_dir}/{record.id}_conf{i}.cif")


if __name__ == "__main__":
    main()
