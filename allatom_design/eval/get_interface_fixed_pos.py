import csv
import glob
import os
from pathlib import Path
from typing import List

import hydra
import lightning as L
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from natsort import natsorted
from joblib import Parallel, delayed

from allatom_design.data.data import load_feats_from_pdb


@hydra.main(config_path="../configs/eval/", config_name="get_interface_fixed_pos", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Constructs two CSV files containing fixed residue positions (sequence or sidechain)
    for each t_seq in cfg.t_seqs:
      - t_seq = 1.0 => fix all interface residues
      - t_seq < 1.0 => fix t_seq fraction of interface residues
    Each CSV file has 3 columns: [pdb_name, fixed_pos_seq, fixed_pos_scn].
    One CSV has fixed_pos_seq populated; the other has fixed_pos_scn populated.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make output directories
    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Read PDB keys
    with open(cfg.pdb_key_list, "r") as f:
        pdb_keys = f.read().splitlines()

    pdb_files = [f"{cfg.pdb_dir}/{key}{cfg.pdb_key_ext}" for key in pdb_keys]

    # Process for each t_seq
    for t_seq in cfg.t_seqs:
        # Use joblib to parallelize
        results = Parallel(n_jobs=cfg.num_workers)(
            delayed(fix_pos_single_pdb)(pdb_file, key, t_seq, cfg.interface_only)
            for pdb_file, key in tqdm(zip(pdb_files, pdb_keys), total=len(pdb_files))
        )

        # Separate the seq and scn rows
        seq_csv_rows = [r[0] for r in results]
        scn_csv_rows = [r[1] for r in results]

        # Write output CSVs
        with open(f"{out_dir}/fixed_positions_seq_{t_seq}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for row in seq_csv_rows:
                writer.writerow(row)

        with open(f"{out_dir}/fixed_positions_scn_{t_seq}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for row in scn_csv_rows:
                writer.writerow(row)


def fix_pos_single_pdb(
    pdb_file: str,
    key: str,
    t_seq: float,
    interface_only: bool
):
    """
    Process a single PDB file to find which residues to fix based on t_seq fraction
    of interface residues. Returns rows for the sequence CSV and sidechain CSV.
    """
    data = load_feats_from_pdb(pdb_file)

    if interface_only:
        possible_indices = np.where(data["interface_residue_mask"])[0]
    else:
        possible_indices = np.arange(len(data["aatype"]))
    num_to_fix = int(round(len(possible_indices) * t_seq))
    fix_indices = np.random.choice(possible_indices, size=num_to_fix, replace=False)

    chain_index = data["chain_index"]
    residue_index = data["residue_index"]  # Typically 1-based PDB numbering

    # Invert chain_id_mapping: idx -> chain_letter
    chain_id_mapping = data["chain_id_mapping"]
    idx_to_chain = {v: k for k, v in chain_id_mapping.items()}

    # Build the string of fixed positions in parse_fixed_positions format
    # e.g. "A1,A2,B10"
    pos_list = []
    for i_res in fix_indices:
        chain_letter = idx_to_chain[int(chain_index[i_res])]
        res_no = int(residue_index[i_res])
        pos_list.append(f"{chain_letter}{res_no}")
    fix_str = ",".join(natsorted(pos_list))

    # seq_csv_rows: fix_str in fixed_pos_seq, empty in fixed_pos_scn
    seq_csv_row = [key, fix_str, ""]

    # scn_csv_rows: fix_str in fixed_pos_seq, fix_str in fixed_pos_scn
    scn_csv_row = [key, fix_str, fix_str]

    return seq_csv_row, scn_csv_row



if __name__ == "__main__":
    main()
