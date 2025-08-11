import ast
import glob
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from Bio.PDB import PDBParser, Selection
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree

from allatom_design.eval.eval_utils import bindcraft_utils, eval_metrics
from allatom_design.eval.eval_utils.eval_setup_utils import (get_pdb_files,
                                                             process_pdb_files,
                                                             wandb_setup)
from allatom_design.eval.eval_utils.folding_utils import get_struct_pred_model
from allatom_design.eval.eval_utils.seq_des_utils import (get_seq_des_model,
                                                          run_seq_des)


@hydra.main(config_path="../configs/eval", config_name="eval_seq_des_bindcraft_ensemble", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for evaluating ensemble sequence design on BindCraft designs.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging / output directory
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)

    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # # Load structure prediction model for self-consistency evaluation
    if cfg.run_self_consistency_eval:
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    target_chain, binder_chain = "A", "B"  # for bindcraft targets, target is always chain A, and binder is chain B

    # Load and preprocess the trajectory pdbs
    traj_pdbs = get_pdb_files(**cfg.input_cfg)
    processed_struct_files = process_pdb_files(traj_pdbs, processed_struct_dir=f"{log_dir}/processed_structures", **cfg.pdb_processing_cfg)
    processed_struct_files = natsorted(processed_struct_files)

    # Recompute interface residues
    key_to_fixed_pos_seq = {}
    for traj_pdb in traj_pdbs:
        pdb_key = Path(traj_pdb).stem
        interface_residues_map = hotspot_residues(traj_pdb, binder_chain)
        fixed_pos_seq = ",".join([f"{binder_chain}{i}" for i in interface_residues_map])

        # also fix the target chain
        fixed_pos_seq = f"{target_chain},{fixed_pos_seq}"

        # add to key_to_fixed_pos_seq
        key_to_fixed_pos_seq[pdb_key] = fixed_pos_seq

    # Run our sequence design model
    # create fixed position df
    pos_constraint_df = pd.DataFrame(key_to_fixed_pos_seq.items(), columns=["pdb_key", "fixed_pos_seq"])

    outputs = run_seq_des(seq_des_model["model"], seq_des_model["data_cfg"], seq_des_model["sampling_cfg"],
                            struct_file_paths=processed_struct_files, device=device, out_dir=log_dir,
                            pos_constraint_df=pos_constraint_df)
    sampled_pdbs = outputs["out_pdbs"]

    # Save outputs to CSV
    record_ids = [Path(x).stem.lower() for x in outputs["out_pdbs"]]
    output_df = pd.DataFrame({"record_id": record_ids, "pdb_key": outputs["pdb_keys"], "seq": outputs["seqs"], "input_seq": outputs["input_seqs"]})
    output_df.to_csv(f"{log_dir}/seq_des_outputs.csv", index=False)

    if cfg.run_self_consistency_eval:
        id_to_metrics = eval_metrics.run_af2_interface_eval(
            sampled_pdbs,
            binder_chain_ids=[binder_chain] * len(sampled_pdbs),
            struct_pred_model=struct_pred_model,
            out_dir=log_dir)

# Log self-consistency metrics
if cfg.run_self_consistency_eval:
    # Save metrics as CSV
    metrics_df = pd.DataFrame([{"record_id": rid, **m} for rid, m in id_to_metrics.items()])
    metrics_df.to_csv(f"{log_dir}/self_consistency_metrics.csv", index=False)

    if not cfg.wandb.no_wandb:
        # Aggregate results
        sc_metrics = defaultdict(list)
        for metrics in id_to_metrics.values():
            for k, v in metrics.items():
                sc_metrics[f"{k}"].append(v)

        # Update metrics
        out_metrics = {f"seq_des/mean/{k}": np.nanmean(v) for k, v in sc_metrics.items() if k != "record_id"}
        out_metrics.update({f"seq_des/median/{k}": np.nanmedian(v) for k, v in sc_metrics.items() if k != "record_id"})

        # Log metrics to wandb
        wandb.log(out_metrics)


# identify interacting residues at the binder interface
three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0) -> dict[str, str]:
    """
    From BindCraft: https://github.com/martinpacesa/BindCraft/blob/main/functions/biopython_utils.py#L138
    """
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues


if __name__ == "__main__":
    main()
