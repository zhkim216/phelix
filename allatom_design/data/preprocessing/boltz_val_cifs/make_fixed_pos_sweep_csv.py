#!/usr/bin/env python3
import random
import shutil
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from collections import defaultdict
from allatom_design.data.feature.seq_des_featurizer import crop_sd_feats
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             process_pdb_files)
from allatom_design.eval.eval_utils.seq_des_utils import get_sd_example


@hydra.main(config_path="../../../configs/data/preprocessing/boltz_val_cifs", config_name="make_fixed_pos_sweep_csv", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Given PDBs from boltz_val_cifs, generate a dataframe of fixed positions in random order to provide random sequence context for each PDB.
    """
    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Create output directories
    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load in PDB files
    pdb_files = get_pdb_files(**cfg.input_cfg)
    temp_processed_struct_dir = f"{out_dir}/processed_structures"
    processed_struct_files = process_pdb_files(pdb_files, processed_struct_dir=temp_processed_struct_dir, **cfg.pdb_processing_cfg)

    # Process each PDB file
    data_cfg = hydra.utils.instantiate(cfg.data_cfg)
    fixed_pos_df = defaultdict(list)
    for struct_file in tqdm(processed_struct_files, desc="Retrieving fixed positions for each PDB"):
        example, input_structure = get_sd_example(struct_file, data_cfg)

        # get <auth_asym_name+residue_index> for all positions where the token is resolved
        example = crop_sd_feats(example, example["token_resolved_mask"], max_tokens=None, max_atoms=None, max_seqs=None)
        asym_id_to_chain = {c["asym_id"]: c["auth_asym_name"] for c in input_structure.chains}
        residue_index, asym_id = example["residue_index"].tolist(), example["asym_id"].tolist()
        auth_asym_name = [asym_id_to_chain[i] for i in asym_id]
        fixed_pos_list = [f"{i}{j}" for i, j in zip(auth_asym_name, residue_index)]

        # randomize sequence context order
        random.shuffle(fixed_pos_list)


        # add fixed positions to df
        fixed_pos_df["pdb_key"].append(example["pdb_key"])
        fixed_pos_df["fixed_pos_seq"].append(",".join(fixed_pos_list))


    # Save to csv
    df = pd.DataFrame(fixed_pos_df)
    df.to_csv(f"{out_dir}/fixed_pos_sweep.csv", index=False)

    # Clean up
    shutil.rmtree(temp_processed_struct_dir)


if __name__ == "__main__":
    main()