import glob
import itertools
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.eval_utils.eval_setup_utils import process_pdb_files
from allatom_design.eval.eval_utils.seq_des_utils import (get_seq_des_model,
                                                          score_samples)


@hydra.main(config_path="../../configs/eval/sampling", config_name="score_conformers", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for scoring sequences from input PDBs against input backbones.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up output directory
    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in input backbones to score with
    pdb_names = [Path(x).name for x in glob.glob(f"{cfg.conformer_dir}/*")]

    # first, collect per-PDB conformer lists
    conformer_groups = []
    for pdb_name in pdb_names:
        all_conformers = natsorted(glob.glob(f"{cfg.conformer_dir}/{pdb_name}/*"))
        conformer_groups.append((pdb_name, all_conformers))

    # flatten and process everything in one go
    all_confs_flat = [c for _, group in conformer_groups for c in group]
    processed_flat = process_pdb_files(all_confs_flat, processed_struct_dir=f"{out_dir}/processed_structures", **cfg.pdb_processing_cfg, keep_order=True)

    # then split the flat list back into per-PDB results
    conformer_struct_files = []
    offset = 0
    for pdb_name, group in conformer_groups:
        n = len(group)
        conformer_struct_files.append((pdb_name, processed_flat[offset:offset + n]))
        offset += n

    # filter out conformers that failed to process
    conformer_struct_files = [(pdb_name, [x for x in struct_files if x is not None]) for pdb_name, struct_files in conformer_struct_files]
    
    # Load in sampled cifs, assuming they take form of {pdb_name}_sample{i}.cif
    pdb_name_to_samples = []  # list of (pdb_name, [sample_cif_paths])
    for pdb_name in pdb_names:
        pdb_name_to_samples.append((pdb_name, natsorted(glob.glob(f"{cfg.sample_dir}/{pdb_name}_sample*.cif"))))
    
    # flatten and process everything in one go
    samples_flat = [c for _, group in pdb_name_to_samples for c in group]
    processed_sampled_cifs = process_pdb_files(samples_flat, processed_struct_dir=f"{out_dir}/processed_sampled_cifs", **cfg.pdb_processing_cfg, keep_order=True)

    # then split the flat list back into per-PDB results
    sample_struct_files = []
    offset = 0
    for pdb_name, group in pdb_name_to_samples:
        n = len(group)
        sample_struct_files.append((pdb_name, processed_sampled_cifs[offset:offset + n]))
        offset += n
        
    # filter out samples that failed to process
    sample_struct_files = [(pdb_name, [x for x in samples if x is not None]) for pdb_name, samples in sample_struct_files]
    
    # convert to dict
    sample_struct_files = {pdb_name: struct_files for pdb_name, struct_files in sample_struct_files}
    
    # For each input backbone, we score all samples from the pdb_name 
    bb_to_sample_files = {}
    for pdb_name, input_backbone_files in conformer_struct_files:
        sample_files = sample_struct_files[pdb_name]
        for input_backbone_file in input_backbone_files:
            bb_to_sample_files[input_backbone_file] = sample_files

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Run sequence design model to get potts parameters for each input backbone
    score_outputs = score_samples(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                                  bb_to_sample_files=bb_to_sample_files, device=device)
    
    
    # Parse score outputs into a flattened dataframe
    df = defaultdict(list)
    for pdb_name, input_backbone_files in conformer_struct_files:
        for input_backbone_file in input_backbone_files:
            score_outputs_i = score_outputs[input_backbone_file]
            df["pdb_name"].extend([pdb_name] * len(score_outputs_i["bb_pdb_key"]))
            df["bb_pdb_key"].extend(score_outputs_i["bb_pdb_key"])
            df["sample_pdb_key"].extend(score_outputs_i["sample_pdb_key"])
            df["U"].extend(score_outputs_i["U"].tolist())
    df = pd.DataFrame(df)
    
    # Save to csv
    df.to_csv(f"{out_dir}/score_outputs.csv", index=False)
    
    print("DEBUG")


if __name__ == "__main__":
    main()
