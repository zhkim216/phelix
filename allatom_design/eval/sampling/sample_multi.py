import fcntl
import glob
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.conditioning_labels as cl
from allatom_design.data import residue_constants as rc
from allatom_design.data.data import (load_feats_from_pdb, pad_to_max_len,
                                      process_single_pdb)
from allatom_design.eval import eval_metrics, sampling_utils
from allatom_design.eval.folding_utils import get_struct_pred_model
from allatom_design.eval.sampling.sample_single import parse_fixed_positions
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


@hydra.main(config_path="../../configs/eval/sampling", config_name="sample_multi", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for all PDBs in a directory.
    For each batch of PDBs, we produce one designed sequence per PDB.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make output directories
    out_dir = cfg.out_dir  # base output directory
    sample_out_dir = f"{out_dir}/samples"
    fasta_out_dir = f"{out_dir}/fastas"  # directory for sequences in FASTA format
    sample_pkl_dir = f"{out_dir}/sample_pkls"  # directory for pkls containing various sample info

    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    Path(fasta_out_dir).mkdir(parents=True, exist_ok=True)
    Path(sample_pkl_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Device setup
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load denoiser model
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.checkpoint_path).eval()

    # Load structure prediction model
    if cfg.run_self_consistency_eval:
        pred_out_dir = f"{out_dir}/preds"  # directory for structure predictions (if running folding)
        Path(pred_out_dir).mkdir(parents=True, exist_ok=True)
        struct_pred_model = get_struct_pred_model(cfg.struct_pred_cfg, device=device)
        self_consistency_path = f"{out_dir}/self_consistency_metrics.csv"

    # Setup sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs_template = {
        "num_steps": cfg.scn_diffusion.num_steps,
        "timesteps": None,  # will be filled per batch
        "noise_schedule": noise_schedule,
        "churn_cfg": churn_cfg,
        "return_scn_diffusion_aux": False
    }

    # Read in fixed positions
    if cfg.fixed_pos_csv is not None:
        fixed_pos_df = pd.read_csv(cfg.fixed_pos_csv, names=["fixed_pos_seq", "fixed_pos_scn"], index_col=0)
        fixed_pos_df = fixed_pos_df.fillna("")
        fixed_pos_df.index = fixed_pos_df.index.str.replace(".pdb", "").str.replace(".cif", "")  # remove extension from index if present
    else:
        fixed_pos_df = pd.DataFrame(columns=["fixed_pos_seq", "fixed_pos_scn"])

    # Print omitted amino acids
    if cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {cfg.omit_aas}")

    ### Load in PDB files ###
    if cfg.pdb_key_list is not None:
        # Get PDBs with keys in the list
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdb_files = [f"{cfg.pdb_dir}/{key}{cfg.pdb_key_ext}" for key in pdb_keys]
    else:
        # Get all PDBs with .pdb extension in the directory
        pdb_files = natsorted(list(glob.glob(f"{cfg.pdb_dir}/*.pdb")))
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {cfg.pdb_dir}")

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
        results = Parallel(n_jobs=-1)(delayed(get_length)(f) for f in tqdm(pdb_files, desc="Loading PDBs to determine lengths"))
        pdb_to_length = dict(results)

        # sort by length, longest first
        pdb_files = sorted(pdb_files, key=lambda x: pdb_to_length[x], reverse=True)

    ### SAMPLING ###
    print(f"Sampling with num denoising steps S={cfg.num_steps} on {len(pdb_files)} PDBs (array_id={cfg.array_id})")

    cfg.timestep_schedule.num_steps = cfg.num_steps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    # Process PDBs in batches of size B
    pdb_files_repeated = np.repeat(pdb_files, cfg.num_seqs_per_pdb)

    pbar = tqdm(total=len(pdb_files_repeated))
    for i in range(0, len(pdb_files_repeated), cfg.batch_size):
        pdb_batch_files = pdb_files_repeated[i:i+cfg.batch_size]
        B = len(pdb_batch_files)

        # Load and process all PDBs in this batch
        batch_list = []
        batch_chain_id_mapping = []
        for pdb_file in pdb_batch_files:
            data = load_feats_from_pdb(pdb_file)
            single = process_single_pdb(data)
            batch_list.append(single)

            # store chain ID mapping for parsing fixed positions
            batch_chain_id_mapping.append(data["chain_id_mapping"])

        pdb_names = [Path(pdb_file).stem for pdb_file in pdb_batch_files]

        # Create a batch dictionary from batch_list by stacking
        model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index", "interface_residue_mask"]
        max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
        batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len)for b in batch_list]  # pad each batch to max length
        batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}  # stack the padded batches

        # Move to device
        batch = {k: batch[k].to(device) for k in model_input_keys}

        # Prepare scd_inputs for this batch
        scd_inputs = dict(scd_inputs_template)
        scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(device)

        # Prepare sampling timesteps
        timesteps = t_seq[None].expand(B, -1).to(device)

        cond_labels_in = {
            "crop_aug": torch.Tensor([cl.DEFAULT_TOKEN_ID['crop_aug']]*B).to(device),
            "dataset_source": torch.Tensor([cl.DEFAULT_TOKEN_ID['dataset_source']]*B).to(device),
            "designability": torch.Tensor([cl.PLACEHOLDER_TOKEN_ID]*B).to(device)
        }

        aatype_override_mask, scn_override_mask = get_override_masks(batch, pdb_names, batch_chain_id_mapping, fixed_pos_df, verbose=cfg.fixed_pos_verbose)

        # Run sampling
        x_denoised, aatype_denoised, aux = lit_sd_model.model.sample(
            batch["x"],
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            missing_atom_mask=batch["missing_atom_mask"],
            residue_index=batch["residue_index"],
            chain_index=batch["chain_index"],
            cond_labels=cond_labels_in,
            timesteps=timesteps,
            aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
            num_corrector_steps=cfg.num_corrector_steps,
            corrector_step_ratio=cfg.corrector_step_ratio,
            seq_only=cfg.seq_only,
            temperature=cfg.temperature,
            repack_last=cfg.repack_last,
            repack_every_step=cfg.repack_every_step,
            psce_threshold=cfg.psce_threshold,
            noise_labels=cfg.noise_labels,
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=scn_override_mask,
            omit_aas=cfg.omit_aas,
            scd_inputs=scd_inputs,
        )

        samples = {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
            "seq_probs": aux["seq_probs"],
            # save other useful info
            "original_aatype": batch["aatype"],
            "interface_residue_mask": batch["interface_residue_mask"],
            "aatype_override_mask": aatype_override_mask,
            "scn_override_mask": scn_override_mask,
        }

        # Save outputs
        # Save to PDB
        pdb_keys = [f"{pdb_name}_sample{(i+j) % cfg.num_seqs_per_pdb}" for j, pdb_name in enumerate(pdb_names)]
        pdbs = [f"{sample_out_dir}/{pdb_key}.pdb" for pdb_key in pdb_keys]
        fastas = [f"{fasta_out_dir}/{pdb_key}.fasta" for pdb_key in pdb_keys]
        pred_seqs = []
        SeqDenoiser.save_samples_to_pdb(samples, pdbs)

        for j, pdb_file in enumerate(pdb_batch_files):
            # Extract the sequence
            seq_mask_i = samples["seq_mask"][j].cpu()
            pred_aatype_i = samples["pred_aatype"][j].cpu()
            pred_aatype_i = pred_aatype_i[seq_mask_i.bool()]
            pred_seq_i = "".join(rc.restypes_with_x[a] for a in pred_aatype_i)
            pred_seqs.append(pred_seq_i)

            # Save fasta
            fasta_out = fastas[j]
            with open(fasta_out, "w") as f:
                f.write(f">{pdb_keys[j]}\n{pred_seq_i}\n")

        # Save samples as pkl
        for j in range(B):
            sample_j = {k: v[j].cpu().numpy() for k, v in samples.items()}
            # crop to the actual sequence length
            seq_mask_j = sample_j["seq_mask"]
            sample_j = {k: v[seq_mask_j.astype(bool)] for k, v in sample_j.items()}
            with open(f"{sample_pkl_dir}/{pdb_keys[j]}.pkl", "wb") as f:
                pickle.dump(sample_j, f)

        # Run self-consistency evaluation
        if cfg.run_self_consistency_eval:
            sc_info = eval_metrics.run_self_consistency_eval(
                pdbs,
                None, None,  # no MPNN model for co-design eval
                struct_pred_model,
                device,
                out_dir=pred_out_dir,
                eval_codesign=True,
                temp_dir=f"{pred_out_dir}/tmp",
                override_metrics_to_compute=["sc_ca_rmsd", "sc_aa_rmsd", "sc_ca_tm"]
            )

            # Aggregate results
            sc_metrics = defaultdict(list)
            for j, pdb in enumerate(pdbs):
                sc_metrics["pdb_name"].append(Path(pdb).stem)
                sc_metrics["pdb_key"].append(pdb_names[j])

                for k, v in sc_info[pdb]["sc_metrics"].items():
                    sc_metrics[f"{k}"].append(v.item())

            sc_metrics["pred_seq"].extend(pred_seqs)
            out_df = pd.DataFrame(sc_metrics)

            # Safely append to CSV using a file lock
            with open(self_consistency_path, "a+") as f:
                # Acquire exclusive lock
                fcntl.flock(f, fcntl.LOCK_EX)

                # Check if file is empty
                f.seek(0, os.SEEK_END)
                file_empty = (f.tell() == 0)

                # Write DataFrame
                out_df.to_csv(f, index=False, header=file_empty)

                # Release lock
                fcntl.flock(f, fcntl.LOCK_UN)

        pbar.update(B)

    pbar.close()


def get_length(pdb_file: str) -> Tuple[str, int]:
    data = load_feats_from_pdb(pdb_file)
    return pdb_file, len(data["aatype"])


def get_override_masks(batch: Dict[str, TensorType["b ..."]],
                       pdb_names: List[str],
                       batch_chain_id_mapping: List[Dict[str, int]],  # maps chain letter to chain index
                       fixed_pos_df: pd.DataFrame,
                       verbose: bool = False) -> Tuple[Optional[TensorType["b n", int]],
                                                       Optional[TensorType["b n", int]]]:
    aatype_override_mask, scn_override_mask = torch.zeros_like(batch["residue_index"]), torch.zeros_like(batch["residue_index"])

    for i, pdb_name in enumerate(pdb_names):
        if pdb_name not in fixed_pos_df.index:
            if verbose:
                print(f"No fixed positions found for {pdb_name}")
            continue
        # Get fixed positions from df
        row = fixed_pos_df.loc[pdb_name]
        fixed_pos_seq, fixed_pos_scn = row["fixed_pos_seq"], row["fixed_pos_scn"]

        example = {k: v[i] for k, v in batch.items()}
        chain_id_mapping = batch_chain_id_mapping[i]

        if fixed_pos_seq:
            # sequence override
            if verbose:
                print(f"{pdb_name}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_positions(fixed_pos_seq, chain_id_mapping, example["residue_index"], example["chain_index"])
            aatype_override_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            if verbose:
                fixed_seq_viz = "".join([rc.restypes_with_x[example["aatype"][j]] if aatype_override_mask[i, j] else "-" for j in range(aatype_override_mask.shape[1])])
                print(f"Fixed sequence: {fixed_seq_viz}")
        else:
            if verbose:
                print("No fixed sequence positions specified.")

        if fixed_pos_scn:
            # sidechain override
            if verbose:
                print(f"{pdb_name}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_positions(fixed_pos_scn, chain_id_mapping, example["residue_index"], example["chain_index"])
            scn_override_mask[i, abs_fixed_pos_scn] = 1

            # print fixed sidechains
            if verbose:
                fixed_scn_viz = "".join([rc.restypes_with_x[example["aatype"][j]] if scn_override_mask[i, j] else "-" for j in range(scn_override_mask.shape[1])])
                print(f"Fixed sidechains: {fixed_scn_viz}")
        else:
            if verbose:
                print("No fixed sidechain positions specified.")

    return aatype_override_mask, scn_override_mask


if __name__ == "__main__":
    main()
