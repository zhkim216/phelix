import fcntl
import glob
import math
import os
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data.data import get_length_from_pdb
from allatom_design.eval import eval_metrics
from allatom_design.eval.fampnn_utils import get_seq_des_model, run_fampnn
from allatom_design.eval.folding_utils import get_struct_pred_model


@hydra.main(config_path="../../configs/eval/sampling", config_name="fampnn_multi", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for all PDBs in a directory.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make output directories
    out_dir = cfg.out_dir  # base output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # create output directory

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Device setup
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Load structure prediction model
    if cfg.run_self_consistency_eval:
        pred_out_dir = f"{out_dir}/preds"  # directory for structure predictions (if running folding)
        Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)
        out_df_path = f"{out_dir}/self_consistency_metrics.csv"
    else:
        out_df_path = f"{out_dir}/fampnn_outputs.csv"

    # Read in fixed positions
    if cfg.pos_constraint_csv is not None:
        pos_constraint_df = pd.read_csv(cfg.pos_constraint_csv)
    else:
        pos_constraint_df = pd.DataFrame(columns=["pdb_name"])

    ### Load in PDB files ###
    if cfg.pdb_key_list is not None:
        # Get PDBs with keys in the list
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdb_files = [f"{cfg.pdb_dir}/{key}{cfg.pdb_key_ext}" for key in pdb_keys]
    else:
        # Get all PDBs with .pdb_key_ext extension in the directory
        pdb_files = natsorted(list(glob.glob(f"{cfg.pdb_dir}/*")))
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {cfg.pdb_dir}")

    # Re-run missing PDBs
    if cfg.fix_missing:
        out_df = pd.read_csv(out_df_path)
        pdb_keys = out_df["pdb_key"].unique()
        pdb_files = [f for f in pdb_files if Path(f).stem not in pdb_keys]
        out_df_path = f"{out_dir}/fampnn_outputs_missing.csv"  # save to a different file

    # Parallelization
    if cfg.array_id is not None:
        # Determine chunk size
        array_id = cfg.array_id
        num_arrays = cfg.num_arrays
        chunk_size = math.ceil(len(pdb_files) / num_arrays)

        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(pdb_files))
        pdb_files = pdb_files[start_idx:end_idx]

    # If specified, pre-sort by length (descending)
    if cfg.presort_by_length:
        # determine lengths
        results = Parallel(n_jobs=-1)(delayed(get_length_from_pdb)(f) for f in tqdm(pdb_files, desc="Loading PDBs to determine lengths"))
        pdb_to_length = dict(results)

        # sort by length, longest first
        pdb_files = sorted(pdb_files, key=lambda x: pdb_to_length[x], reverse=True)

    ### SAMPLING ###
    # Run FAMPNN
    _, aux = run_fampnn(seq_des_model["fampnn_model"], seq_des_model["fampnn_cfg"],
                        pdb_paths=pdb_files, device=device, pos_constraint_df=pos_constraint_df,
                        out_dir=cfg.out_dir)
    sampled_pdbs = aux["out_pdbs"]
    input_pdb_names = aux["input_pdb_names"]  # original PDB names
    pred_seqs = aux["pred_seqs"]

    # Run self-consistency evaluation
    out_metrics = defaultdict(list)  # to store results

    if cfg.run_self_consistency_eval:
        sc_info = eval_metrics.run_self_consistency_eval(
            sampled_pdbs,
            None,
            struct_pred_model,
            device,
            out_dir=pred_out_dir,
            eval_codesign=True,
            temp_dir=f"{pred_out_dir}/tmp",
            override_metrics_to_compute=["sc_ca_rmsd", "sc_aa_rmsd", "sc_ca_tm"]
        )

        # Aggregate results
        for j, pdb in enumerate(sampled_pdbs):
            out_metrics["pdb_name"].append(Path(pdb).stem)
            out_metrics["pdb_key"].append(Path(input_pdb_names[j]).stem)
            out_metrics["pred_seq"].append(pred_seqs[j])

            for k, v in sc_info[pdb]["sc_metrics"].items():
                out_metrics[f"{k}"].append(v.item())
            out_metrics["avg_plddt"].append(sc_info[pdb]["struct_preds"]["avg_plddt"].item())
    else:
        # If not running self-consistency evaluation, just append basic metrics to a CSV
        for j, pdb in enumerate(sampled_pdbs):
            out_metrics["pdb_name"].append(Path(pdb).stem)
            out_metrics["pdb_key"].append(Path(input_pdb_names[j]).stem)
            out_metrics["pred_seq"].append(pred_seqs[j])
            out_metrics["design_number"].append(Path(pdb).stem.split("_")[-1])  # extract design number from filename

    out_df = pd.DataFrame(out_metrics)

    # Safely append to CSV using a file lock
    with open(out_df_path, "a+") as f:
        # Acquire exclusive lock
        fcntl.flock(f, fcntl.LOCK_EX)

        # Check if file is empty
        f.seek(0, os.SEEK_END)
        file_empty = (f.tell() == 0)

        # Write DataFrame
        out_df.to_csv(f, index=False, header=file_empty)

        # Release lock
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
