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


@hydra.main(config_path="../configs/eval", config_name="eval_partial_t_seq_design", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences with a partial_t mask applied.
    partial_t=1 => fully unmasked, partial_t=0 => fully masked.
    We run 3 modes:
        1) partial on sequence only,
        2) partial on both sequence and packed sidechains.
        3) partial on both sequence and ground truth sidechains
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

    # Device setup
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load denoiser model
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.checkpoint_path).eval()

    # Setup sidechain diffusion inputs
    t_scd = sampling_utils.get_timesteps_from_schedule(**cfg.scn_diffusion.timestep_schedule)
    noise_schedule = NoiseSchedule(cfg.scn_diffusion.noise_schedule)
    churn_cfg = dict(cfg.scn_diffusion.churn_cfg)
    scd_inputs_template = {
        "num_steps": cfg.scn_diffusion.num_steps,
        "timesteps": None,
        "noise_schedule": noise_schedule,
        "churn_cfg": churn_cfg,
        "return_scn_diffusion_aux": False
    }

    # Gather PDBs
    if cfg.pdb_key_list is not None:
        with open(cfg.pdb_key_list, "r") as f:
            pdb_keys = f.read().splitlines()
        pdb_files = [f"{cfg.pdb_dir}/{key}" for key in pdb_keys]
    else:
        pdb_files = natsorted(list(glob.glob(f"{cfg.pdb_dir}/*.pdb")))
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {cfg.pdb_dir}")

    if cfg.subset_length_range is not None:
        # determine lengths
        results = Parallel(n_jobs=-1)(delayed(get_length)(f) for f in tqdm(pdb_files, desc="Loading PDBs to determine lengths"))
        pdb_to_length = dict(results)

        pdb_files = []
        for pdb_file, length in pdb_to_length.items():
            if cfg.subset_length_range[0] <= length <= cfg.subset_length_range[1]:
                pdb_files.append(pdb_file)

    print(f"Sampling with num denoising steps S={cfg.num_steps} on {len(pdb_files)} PDBs")
    cfg.timestep_schedule.num_steps = cfg.num_steps
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    # Dictionary to track sequence accuracy for each mode
    seq_acc_dict = {
        "seq_only": [],
        "packed_sidechains": [],
        "gt_sidechains": []
    }

    for i in tqdm(range(0, len(pdb_files), cfg.batch_size)):
        pdb_batch_files = pdb_files[i:i+cfg.batch_size]
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
        model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index"]
        max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
        batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len) for b in batch_list]
        batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}    # stack the padded batches

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
        }

        # Randomly mask out a portion of the sequence (and sidechain context)
        seq_mask = batch["seq_mask"]
        lengths = seq_mask.sum(dim=-1).long()
        num_to_mask = torch.floor(lengths * (1 - cfg.partial_t)).long()
        aatype_override_mask = seq_mask.clone()
        for i in range(len(lengths)):
            mask_indices = torch.randperm(lengths[i])[:num_to_mask[i]]
            aatype_override_mask[i, mask_indices] = 0
        scn_override_mask = aatype_override_mask.clone()
        seq_recovery_mask = seq_mask - aatype_override_mask  # denotes the residues that we're actually designing

        # 1) Design with partial sequence only
        L.seed_everything(cfg.seed)
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
            seq_only=True,
            temperature=cfg.temperature,
            repack_last=cfg.repack_last,
            psce_threshold=cfg.psce_threshold,
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=None,
            scd_inputs=scd_inputs,
        )

        samples_seq_only =  {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
            "seq_probs": aux["seq_probs"],
        }

        # 2) Design with partial sequence and partial packed sidechains
        L.seed_everything(cfg.seed)
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
            seq_only=False,
            temperature=cfg.temperature,
            repack_last=cfg.repack_last,
            psce_threshold=cfg.psce_threshold,
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=None,
            scd_inputs=scd_inputs,
        )

        samples_packed_sidechains =  {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
            "seq_probs": aux["seq_probs"],
        }

        # 3) Design with partial sequence and partial ground truth sidechains
        L.seed_everything(cfg.seed)
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
            seq_only=False,
            temperature=cfg.temperature,
            repack_last=cfg.repack_last,
            psce_threshold=cfg.psce_threshold,
            aatype_override_mask=aatype_override_mask,
            scn_override_mask=scn_override_mask,
            scd_inputs=scd_inputs,
        )

        samples_gt_sidechains =  {
            "x_denoised": x_denoised,
            "seq_mask": batch["seq_mask"],
            "missing_atom_mask": batch["missing_atom_mask"],
            "residue_index": batch["residue_index"],
            "chain_index": batch["chain_index"],
            "pred_aatype": aatype_denoised,
            "psce": aux["psce"],
            "seq_probs": aux["seq_probs"],
        }

        # Track sequence accuracy for each mode
        for b_idx in range(B):
            # ground truth at designed positions
            gt_aatype = batch["aatype"][b_idx][seq_recovery_mask[b_idx].bool()]
            gt_seq = "".join([rc.restypes_with_x[x.item()] for x in gt_aatype])

            # partial sequence only
            pred_seq_seq_only = samples_seq_only["pred_aatype"][b_idx][seq_recovery_mask[b_idx].bool()]
            pred_seq_seq_only = "".join([rc.restypes_with_x[x.item()] for x in pred_seq_seq_only])
            seq_acc_dict["seq_only"].append(
                np.mean(np.array(list(pred_seq_seq_only)) == np.array(list(gt_seq)))
            )

            # partial sequence + packed sidechains
            pred_seq_packed = samples_packed_sidechains["pred_aatype"][b_idx][seq_recovery_mask[b_idx].bool()]
            pred_seq_packed = "".join([rc.restypes_with_x[x.item()] for x in pred_seq_packed])
            seq_acc_dict["packed_sidechains"].append(
                np.mean(np.array(list(pred_seq_packed)) == np.array(list(gt_seq)))
            )

            # partial sequence + ground truth sidechains
            pred_seq_gt = samples_gt_sidechains["pred_aatype"][b_idx][seq_recovery_mask[b_idx].bool()]
            pred_seq_gt = "".join([rc.restypes_with_x[x.item()] for x in pred_seq_gt])
            seq_acc_dict["gt_sidechains"].append(
                np.mean(np.array(list(pred_seq_gt)) == np.array(list(gt_seq)))
            )

    print("DONE")

    # print median sequence accuracy for each mode
    for mode, acc_list in seq_acc_dict.items():
        print(f"{mode} median sequence accuracy: {np.median(acc_list)}")
        print(f"{mode} mean sequence accuracy: {np.mean(acc_list)}")


def get_length(pdb_file: str) -> Tuple[str, int]:
    data = load_feats_from_pdb(pdb_file)
    return pdb_file, len(data["aatype"])


if __name__ == "__main__":
    main()

