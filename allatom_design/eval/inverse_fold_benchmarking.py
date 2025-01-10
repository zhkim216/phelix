import glob
import os
import pickle
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
import wandb
import yaml
from Bio import SeqIO
from Bio.PDB import PDBIO, PDBParser
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

import allatom_design.data.residue_constants as rc
from allatom_design.data.data import load_feats_from_pdb
from allatom_design.data.pdb_utils import write_to_pdb
from allatom_design.eval import eval_metrics
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.eval.proteinmpnn_utils import load_mpnn


@hydra.main(config_path="../configs/eval", config_name="inverse_fold_benchmarking", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Evaluate self-consistency metrics using other sequence design models for benchmarking.
    If input_fasta_dir is null, use ProteinMPNN to generate sequences. Otherwise, read in sequences from input_fasta_dir.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Create out dirs and preserve config
    if cfg.overwrite_out_dir and Path(cfg.out_dir).exists():
        # Delete existing out_dir
        print(f"Deleting pre-existing out_dir: {cfg.out_dir}")
        shutil.rmtree(cfg.out_dir)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    ### CALCULATE STRUCTURE METRICS ###
    all_metrics = defaultdict(dict)

    if cfg.pdb_key_list is not None:
        # Get PDBs with keys in the list
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdbs = [f"{cfg.sample_dir}/{key}" for key in pdb_keys]
    else:
        # Get all PDBs in the sample directory
        pdbs = natsorted(glob.glob(f"{cfg.sample_dir}/*.pdb"))

    # Copy over original samples
    shutil.copytree(cfg.sample_dir, f"{cfg.out_dir}/samples")

    # Set up models (in eval mode)
    device = torch.device("cuda" if cfg.cuda else "cpu")
    torch.set_grad_enabled(False)
    struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)

    use_input_pdbs = cfg.input_pdb_dir is not None
    use_mpnn = not (cfg.input_fasta_dir or cfg.input_pdb_dir)
    if use_mpnn:
        # If no input fasta or pdb dir, use MPNN to generate sequences
        mpnn_cfg = OmegaConf.load(cfg.mpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.mpnn.overrides)  # override base mpnn config with mpnn.overrides
        mpnn_model = load_mpnn(cfg.mpnn.mpnn_params_dir, mpnn_cfg, device=device)
    else:
        mpnn_model, mpnn_cfg = None, None  # no MPNN
        designs_dir = f"{cfg.out_dir}/designs"
        Path(designs_dir).mkdir(parents=True, exist_ok=True)
        design_pdbs = []

        for pdb_path in pdbs:
            stem = Path(pdb_path).stem
            out_pdb_path = f"{designs_dir}/{stem}.pdb"
            design_pdbs.append(out_pdb_path)
            if use_input_pdbs:
                # load in input pdbs with sequences already on them
                sample_pdb_path = find_sample_pdb(cfg.input_pdb_dir, stem)
                shutil.copy(sample_pdb_path, out_pdb_path)
            else:
                # thread sequence onto backbone
                fasta_path = find_sample_fasta(cfg.input_fasta_dir, stem)
                assert Path(fasta_path).exists(), f"No corresponding FASTA found for {pdb_path} at {fasta_path}"
                thread_sequence_onto_backbone(pdb_path, fasta_path, out_pdb_path)
        pdbs = design_pdbs  # use designed pdbs for evaluation

    # Run self-consistency evaluation
    sc_output = eval_metrics.run_self_consistency_eval(pdbs,
                                                       mpnn_model, mpnn_cfg,
                                                       struct_pred_model,
                                                       device,
                                                       out_dir=cfg.out_dir,
                                                       eval_codesign=(not use_mpnn),
                                                       temp_dir=f"{cfg.out_dir}/tmp")
    for pdb, v in sc_output.items():
        all_metrics[pdb]["sc_info"] = v

    ### SAVE METRICS ###
    # Save all metrics to pickle file
    with open(f"{cfg.out_dir}/all_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)

    # Stratify by sequence length
    for pdb in all_metrics:
        all_metrics[pdb]["length"] = all_metrics[pdb]["sc_info"]["struct_preds"]["pred_coords"].shape[1]
    unique_lengths = set([all_metrics[pdb]["length"] for pdb in all_metrics])

    L_to_metrics = {}
    for length in unique_lengths:
        # Store metrics for each pdb in this length group
        pdbs_l = [pdb for pdb in all_metrics if all_metrics[pdb]["length"] == length]
        metrics_l = {}
        sc_metrics_l = defaultdict(list)
        for pdb in pdbs_l:
            sc_info = all_metrics[pdb]["sc_info"]
            for k, v in sc_info["sc_metrics"].items():
                sc_metrics_l[k].append(v.item())

            struct_preds = sc_info["struct_preds"]
            sc_metrics_l["avg_plddt"].append(struct_preds["avg_plddt"].item())

        # Aggregate self-consistency metrics for this length group
        for k, v in sc_metrics_l.items():
            metrics_l[f"{k}_mean"] = np.mean(v)
            metrics_l[f"{k}_med"] = np.median(v)

        L_to_metrics[length] = metrics_l

    # Set up wandb logging
    if not cfg.no_wandb:
        # Create wandb dir
        wandb_dir = str(Path(cfg.out_dir))
        Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)

        # Set wandb cache directory
        wandb_cache_dir = str(Path(cfg.out_dir, "cache", "wandb"))
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

        wandb.init(
            project=cfg.project,
            entity=cfg.wandb_id,
            name=cfg.exp_name,
            group=cfg.group,
            config=cfg_dict,
            dir=wandb_dir,
        )

        # Log metrics
        for length in sorted(list(unique_lengths)):
            metrics_l = L_to_metrics[length]
            metrics_l["length"] = length
            wandb.log(metrics_l)
        wandb.finish()


def thread_sequence_onto_backbone(pdb_path, fasta_path, output_pdb_path):
    """
    Given a PDB and a corresponding FASTA with a single sequence,
    thread the sequence onto the backbone of the PDB and save the new PDB.
    """
    # Read the FASTA
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    assert len(records) == 1, f"Expected exactly one sequence in {fasta_path}, found {len(records)}"
    seq = str(records[0].seq)

    example = load_feats_from_pdb(pdb_path)

    aatype = torch.tensor([rc.restype_order[x] for x in seq])
    example["aatype"] = aatype

    write_to_pdb(aatype, example["all_atom_positions"],
                example["all_atom_mask"],
                example["residue_index"].long(),
                example["chain_index"].long(),
                example["b_factors"],
                filename=output_pdb_path,
                mode="bb")


def find_sample_fasta(sample_dir: str, pdb_key: str) -> str:
    # Look for an exact match first
    exact_match = f"{sample_dir}/{pdb_key}.fasta"
    if Path(exact_match).exists():
        return exact_match

    # Otherwise, search for a file matching {pdb_key}_*.fasta
    pattern = f"{sample_dir}/{pdb_key}_*.fasta"
    matched_fastas = glob.glob(pattern)

    if len(matched_fastas) == 1:
        return matched_fastas[0]
    elif len(matched_fastas) == 0:
        raise FileNotFoundError(f"No sample PDB file found for key '{pdb_key}' in '{sample_dir}'.")
    else:
        raise ValueError(
            f"Multiple sample PDB files found for key '{pdb_key}' in '{sample_dir}': {matched_fastas}"
        )


def find_sample_pdb(sample_dir: str, pdb_key: str) -> str:
    # Look for an exact match first
    exact_match = f"{sample_dir}/{pdb_key}.pdb"
    if Path(exact_match).exists():
        return exact_match

    # Otherwise, search for a file matching {pdb_key}_*.pdb
    pattern = f"{sample_dir}/{pdb_key}_*.pdb"
    matched_pdbs = glob.glob(pattern)

    if len(matched_pdbs) == 1:
        return matched_pdbs[0]
    elif len(matched_pdbs) == 0:
        raise FileNotFoundError(f"No sample PDB file found for key '{pdb_key}' in '{sample_dir}'.")
    else:
        raise ValueError(
            f"Multiple sample PDB files found for key '{pdb_key}' in '{sample_dir}': {matched_pdbs}"
        )


if __name__ == "__main__":
    main()
