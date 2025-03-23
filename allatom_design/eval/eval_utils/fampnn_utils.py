"""
Utils for sampling from FAMPNN.
"""
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import residue_constants as rc
from allatom_design.data.data import load_feats_from_pdb, pad_to_max_len
from allatom_design.data.datasets.sd_dataset import process_single_pdb_sd
from allatom_design.eval.eval_utils import sampling_utils
from allatom_design.eval.eval_utils.proteinmpnn_utils import load_mpnn
from allatom_design.interpolants.ad_interpolants.sampling_schedule import \
    NoiseSchedule
from allatom_design.model.seq_denoiser.denoisers.fampnn_denoiser import \
    FAMPNNDenoiser
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


def get_seq_des_model(cfg: DictConfig, device: str) -> Dict[str, Any]:
    """
    Load in a sequence design model. Similar to get_struct_pred_model()
    Example config:

    seq_des_cfg:
    # MPNN args
    model_name: "fampnn"  # ["proteinmpnn", "fampnn"]
    proteinmpnn:
        mpnn_cfg: allatom_design/configs/seq_des/proteinmpnn.yaml
        mpnn_params_dir: /media/scratch/huang_lab/allatom_design/model_params/mpnn
        overrides:
        # num seqs per structure will be batch_size * number_of_batches
        batch_size: 1
        number_of_batches: 1
        verbose: false
    fampnn:
        # FAMPNN args
        fampnn_cfg: allatom_design/configs/seq_des/fampnn_inference.yaml
        fampnn_ckpt:
    """
    model_name = cfg.model_name
    seq_des_model = {"model_name": model_name, "cfg": cfg, "device": device}

    if model_name == "fampnn":
        fampnn_cfg = OmegaConf.load(cfg.fampnn.fampnn_cfg)
        fampnn_cfg = OmegaConf.merge(fampnn_cfg, cfg.fampnn.overrides)
        lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.fampnn.fampnn_ckpt).eval()
        seq_des_model["fampnn_model"] = lit_sd_model.model
        seq_des_model["fampnn_cfg"] = fampnn_cfg

    elif model_name == "proteinmpnn":
        mpnn_cfg = OmegaConf.load(cfg.proteinmpnn.mpnn_cfg)
        mpnn_cfg = OmegaConf.merge(mpnn_cfg, cfg.proteinmpnn.overrides)  # override base mpnn config with mpnn.overrides
        mpnn_model = load_mpnn(cfg.proteinmpnn.mpnn_params_dir, mpnn_cfg, device=device)
        seq_des_model["mpnn_model"] = mpnn_model
        seq_des_model["mpnn_cfg"] = mpnn_cfg

    return seq_des_model


def run_fampnn(model: SeqDenoiser,
               cfg: DictConfig,
               pdb_paths: List[str],
               device: str,
               pos_constraint_df: Optional[pd.DataFrame] = None,  # optional df for specifying fixed positions for a given pdb name (including extensions)
               out_dir: Optional[str] = None,
               ) -> Tuple[Dict[str, Dict[str, torch.Tensor]],
                          Dict]:
    """
    Given a list of PDB files, run FAMPNN on them.

    Returns a dictionary mapping from PDB paths to dictionaries containing samples for that PDB, including keys:
    - x_denoised: denoised coordinates
    - seq_mask: sequence mask
    - missing_atom_mask: missing atom mask
    - residue_index: residue index
    - chain_index: chain index
    - pred_aatype: predicted amino acid types
    - pred_seq: predicted sequences

    Also returns a run_aux:
    - If out_dir is specified, save the samples to the given directory and return the paths to the samples in aux.
    """
    # Set up output directory
    run_aux = {}
    if out_dir is not None:
        sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
        sample_pt_out_dir = f"{out_dir}/sample_pts"  # directory for pts containing various sample info
        Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
        Path(sample_pt_out_dir).mkdir(parents=True, exist_ok=True)  # create output directory for samples

        run_aux["out_pdbs"] = []  # store paths to all output PDBs
        run_aux["input_pdb_names"] = []  # store names of all input pdbs
        run_aux["out_pts"] = []  # store paths to all output pts
        run_aux["pred_seqs"] = []  # store predicted sequences as a string for each sample

    # Validate pos_constraint_df
    if pos_constraint_df is not None:
        valid_columns = ["pdb_name", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # columns in input df must be a subset of valid columns
            raise ValueError(f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. Found: {pos_constraint_df.columns}")
        pos_constraint_df = pos_constraint_df.set_index("pdb_name")  # set index to pdb name

        # set empty string to NaN for easier parsing
        pos_constraint_df = pos_constraint_df.replace("", np.nan)

    # Sequence design timestep schedule
    t_seq = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)

    # Set up sidechain diffusion inputs
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

    # Print omitted amino acids
    if cfg.verbose and cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {cfg.omit_aas}")

    # Process PDBs in batches of size B
    pdb_paths_repeated = np.repeat(pdb_paths, cfg.num_seqs_per_pdb)
    pbar = tqdm(total=len(pdb_paths_repeated), desc=f"FAMPNN: sampling S={cfg.num_steps} steps, {len(pdb_paths)} PDBs, {cfg.num_seqs_per_pdb} sequences per PDB...")

    input_pdb_to_samples = defaultdict(list)  # maps from a given input pdb path to its samples
    with Parallel(n_jobs=cfg.num_workers) as parallel_pool:  # for loading PDBs in parallel
        for i in range(0, len(pdb_paths_repeated), cfg.batch_size):
            pdb_batch_files = pdb_paths_repeated[i:i+cfg.batch_size]
            B = len(pdb_batch_files)
            batch, pdb_names, batch_chain_id_mapping = get_fampnn_batch(pdb_batch_files, device=device, parallel_pool=parallel_pool)

            # Prepare scd_inputs for this batch
            scd_inputs = dict(scd_inputs_template)
            scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(device)

            # Prepare sampling timesteps
            timesteps = t_seq[None].expand(B, -1).to(device)

            # Handle fixed positions and conditioning; update batch with override masks and overridden aatypes
            batch = parse_fixed_pos_info(batch, pdb_names, batch_chain_id_mapping, pos_constraint_df, verbose=cfg.verbose)

            # Restrict aatype sampling at certain positions
            pos_restrict_aatype = parse_pos_restrict_aatype_info(batch, pdb_names, batch_chain_id_mapping, pos_constraint_df, verbose=cfg.verbose)

            # Run sampling
            x_denoised, aatype_denoised, aux = model.sample(
                batch["x"],
                aatype=batch["aatype"],
                seq_mask=batch["seq_mask"],
                missing_atom_mask=batch["missing_atom_mask"],
                residue_index=batch["residue_index"],
                chain_index=batch["chain_index"],
                timesteps=timesteps,
                temperature=cfg.temperature,
                aatype_decoding_order_mode=cfg.aatype_decoding_order_mode,
                seq_only=cfg.seq_only,
                repack_last=cfg.repack_last,
                psce_threshold=cfg.psce_threshold,
                omit_aas=cfg.omit_aas,
                noise_labels=cfg.noise_labels,
                add_noise=cfg.add_noise,
                aatype_override_mask=batch["aatype_override_mask"],
                scn_override_mask=batch["scn_override_mask"],
                pos_restrict_aatype=pos_restrict_aatype,
                scd_inputs=scd_inputs,
            )

            batch_samples = {
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
                "aatype_override_mask": batch["aatype_override_mask"],
                "scn_override_mask": batch["scn_override_mask"],
            }

            batch_samples = {k: v.cpu() for k, v in batch_samples.items()}

            # Store samples and remove padding
            for j in range(B):
                length_j = batch["seq_mask"][j].sum().long().item()
                sample_j = {k: v[j, :length_j].clone() for k, v in batch_samples.items()}
                pdb_path = pdb_batch_files[j]
                input_pdb_to_samples[pdb_path].append(sample_j)

            # Save outputs to disk
            if out_dir is not None:
                # Save as PDB
                sample_stems = [f"{Path(pdb_name).stem}_sample{(i+j) % cfg.num_seqs_per_pdb}" for j, pdb_name in enumerate(pdb_names)]
                batch_out_pdbs = [f"{sample_out_dir}/{sample_stem}.pdb" for sample_stem in sample_stems]  # output PDBs
                SeqDenoiser.save_samples_to_pdb(batch_samples, batch_out_pdbs)
                run_aux["out_pdbs"].extend(batch_out_pdbs)
                run_aux["input_pdb_names"].extend(pdb_names)

                # Save samples as pt
                for j in range(B):
                    length_j = batch["seq_mask"][j].sum().long().item()
                    sample_j = {k: v[j, :length_j].clone() for k, v in batch_samples.items()}
                    pt_file = f"{sample_pt_out_dir}/{sample_stems[j]}.pt"
                    with open(pt_file, "wb") as f:
                        torch.save(sample_j, f)
                    run_aux["out_pts"].append(pt_file)

                    # Keep explicit track of pred seqs for convenience
                    run_aux["pred_seqs"].append("".join([rc.restypes_with_x[aa] for aa in sample_j["pred_aatype"]]))


            pbar.update(B)
    pbar.close()

    # For each input pdb, aggregate all FAMPNN samples
    preds = defaultdict(dict)
    for pdb, samples_list in input_pdb_to_samples.items():
        for k in samples_list[0].keys():
            preds[pdb][k] = torch.stack([s[k] for s in samples_list])

        # Get sampled sequences for this PDB as a list of strings
        aatype_denoised = preds[pdb]["pred_aatype"]

        pred_seqs = []
        for i in range(aatype_denoised.shape[0]):
            pred_seq = "".join([rc.restypes_with_x[aatype_denoised[i, j]] for j in range(aatype_denoised.shape[1])])
            pred_seqs.append(pred_seq)
        preds[pdb]["pred_seqs"] = pred_seqs

    return preds, run_aux


def run_fampnn_packing(model: SeqDenoiser,
                       cfg: DictConfig,
                       pdb_paths: list[str],
                       device: str,
                       pos_constraint_df: pd.DataFrame | None = None,  # optional df for specifying fixed positions for a given pdb name (including extensions)
                       out_dir: str | None = None,
                       ) -> Tuple[Dict[str, Dict[str, torch.Tensor]],
                                  Dict]:
    """
    Given a list of PDB files, run FAMPNN sidechain packing on them.
    """
    if pos_constraint_df is not None:
        raise NotImplementedError("Fixed positions are not yet supported for sidechain packing, but shouldn't be too hard to implement.")

    # Set up output directory
    run_aux = {}
    run_aux["sample_info"] = []
    if out_dir is not None:
        sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
        Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

        run_aux["out_pdbs"] = []  # store paths to all output PDBs
        run_aux["input_pdb_names"] = []  # store names of all input pdbs

    # Set up sidechain diffusion inputs
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

    # Process PDBs in batches of size B
    pdb_paths_repeated = np.repeat(pdb_paths, cfg.num_seqs_per_pdb)
    pbar = tqdm(total=len(pdb_paths_repeated), desc=f"FAMPNN: packing with S_scd={cfg.scn_diffusion.num_steps} steps, {len(pdb_paths)} PDBs, {cfg.num_seqs_per_pdb} samples per PDB...")

    input_pdb_to_samples = defaultdict(list)  # maps from a given input pdb path to its samples
    with Parallel(n_jobs=cfg.num_workers) as parallel_pool:  # for loading PDBs in parallel
        for i in range(0, len(pdb_paths_repeated), cfg.batch_size):
            pdb_batch_files = pdb_paths_repeated[i:i+cfg.batch_size]
            B = len(pdb_batch_files)
            batch, pdb_names, batch_chain_id_mapping = get_fampnn_batch(pdb_batch_files, device=device, parallel_pool=parallel_pool)

            # Prepare scd_inputs for this batch
            scd_inputs = dict(scd_inputs_template)
            scd_inputs["timesteps"] = t_scd[None].expand(B, -1).to(device)

            # Handle fixed positions and conditioning; update batch with override masks and overridden aatypes
            batch = parse_fixed_pos_info(batch, pdb_names, batch_chain_id_mapping, pos_constraint_df, verbose=cfg.verbose)

            # Run sidechain packing
            x_denoised, _, aux = model.sidechain_pack(
                x=batch["x"],
                aatype=batch["aatype"],
                seq_mask=batch["seq_mask"],
                missing_atom_mask=batch["missing_atom_mask"],
                residue_index=batch["residue_index"],
                chain_index=batch["chain_index"],
                scd_inputs=scd_inputs,
            )
            batch_samples = {"x_denoised": x_denoised,
                            "seq_mask": batch["seq_mask"],
                            "missing_atom_mask": batch["missing_atom_mask"],
                            "pred_aatype": batch["aatype"],
                            "psce": aux["psce"],
                            "residue_index": batch["residue_index"],
                            "chain_index": batch["chain_index"]}
            batch_samples = {k: v.cpu() for k, v in batch_samples.items()}

            # Store samples and remove padding
            for j in range(B):
                length_j = batch["seq_mask"][j].sum().long().item()
                sample_j = {k: v[j, :length_j].clone() for k, v in batch_samples.items()}
                pdb_path = pdb_batch_files[j]
                input_pdb_to_samples[pdb_path].append(sample_j)

            # Save outputs to disk
            if out_dir is not None:
                # Save as PDB
                sample_stems = [f"{Path(pdb_name).stem}_sample{(i+j) % cfg.num_seqs_per_pdb}" for j, pdb_name in enumerate(pdb_names)]
                batch_out_pdbs = [f"{sample_out_dir}/{sample_stem}.pdb" for sample_stem in sample_stems]  # output PDBs
                SeqDenoiser.save_samples_to_pdb(batch_samples, batch_out_pdbs)

                run_aux["out_pdbs"].extend(batch_out_pdbs)
                run_aux["input_pdb_names"].extend(pdb_names)

            # Also save useful sample info
            aatype, seq_mask = batch["aatype"].cpu(), batch["seq_mask"].cpu()
            atom_mask = torch.tensor(rc.STANDARD_ATOM_MASK)[aatype] * seq_mask[..., None]
            atom_mask = atom_mask * (1 - batch["missing_atom_mask"].cpu())  # handle atoms missing from the ground truth PDB
            batch_sample_info = {"x_denoised": x_denoised,
                                "x_in": batch["x"],
                                "seq_mask": seq_mask,
                                "atom_mask": atom_mask,
                                "aatype": aatype}
            batch_sample_info = {k: v.cpu() for k, v in batch_sample_info.items()}
            run_aux["sample_info"].append(batch_sample_info)

            pbar.update(B)
    pbar.close()

    # For each input pdb, aggregate all packed samples
    preds = defaultdict(dict)
    for pdb, samples_list in input_pdb_to_samples.items():
        for k in samples_list[0].keys():
            preds[pdb][k] = torch.stack([s[k] for s in samples_list])

    # Also return padded batch info of all samples
    max_len = max([v["x_denoised"].shape[1] for v in preds.values()])
    run_aux["sample_info"] = [pad_to_max_len(batch_i, max_len) for batch_i in run_aux["sample_info"]]
    run_aux["sample_info"] = {k: torch.cat([v[k] for v in run_aux["sample_info"]], dim=0) for k in run_aux["sample_info"][0].keys()}

    return preds, run_aux


def get_fampnn_batch(pdb_batch_files: List[str], device: str,
                     parallel_pool: Parallel | None
                     ) -> Tuple[Dict[str, TensorType["b n ..."]],
                                List[str],
                                List[Dict[str, int]],  # maps chain letters to chain index
                                ]:
    if parallel_pool is None:
        # Load PDBs sequentially
        batch_data = [load_feats_from_pdb(pdb_file) for pdb_file in pdb_batch_files]
    else:
        # Load PDBs in parallel
        batch_data = parallel_pool(delayed(load_feats_from_pdb)(pdb_file) for pdb_file in pdb_batch_files)

    # Load and process all PDBs in this batch
    batch_list = []
    batch_chain_id_mapping = []
    for data in batch_data:
        single = process_single_pdb_sd(data)
        batch_list.append(single)

        # store chain ID mapping for parsing fixed positions
        batch_chain_id_mapping.append(data["chain_id_mapping"])

        # Ensure that input PDB does not have insertion codes
        if (data["insertion_code_offsets"] > 0).any():
            raise ValueError("Input PDB has insertion codes, which is not handled by fixed_pos specifications. Please renumber your input PDB before running sampling.")

    pdb_names = [Path(pdb_file).name for pdb_file in pdb_batch_files]  # include extension

    # Create a batch dictionary from batch_list by stacking
    model_input_keys = ["x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index", "interface_residue_mask"]
    max_len = max(b["x"].shape[0] for b in batch_list)  # determine the max_len (max number of residues across the batch)
    batch_list = [pad_to_max_len({k: b[k].unsqueeze(0) for k in model_input_keys}, max_len)for b in batch_list]  # pad each batch to max length
    batch = {k: torch.cat([b[k] for b in batch_list], dim=0) for k in model_input_keys}  # stack the padded batches

    # Move to device
    batch = {k: batch[k].to(device) for k in model_input_keys}

    return batch, pdb_names, batch_chain_id_mapping


def create_fampnn_embeddings(model: FAMPNNDenoiser,
                             pdb_paths: List[str],
                             backbone_only: bool,
                             batch_size: int,
                             device: str,
                             out_dir: str):
    """
    Create FAMPNN embeddings for a list of PDB files.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(pdb_paths), desc="Creating FAMPNN embeddings")
    for i in range(0, len(pdb_paths), batch_size):
        pdb_batch_files = pdb_paths[i:i + batch_size]
        B = len(pdb_batch_files)

        batch, pdb_names, _ = get_fampnn_batch(pdb_batch_files, device)
        with torch.no_grad():
            x, aatype, seq_mask, missing_atom_mask, residue_index, chain_index = batch["x"], batch["aatype"], batch["seq_mask"], batch["missing_atom_mask"], batch["residue_index"], batch["chain_index"]
            if backbone_only:
                # Zero out aatype and sidechains
                aatype = torch.full_like(residue_index, fill_value=rc.restype_order_with_x["X"]) * seq_mask.long()
                seq_mlm_mask = torch.zeros_like(seq_mask)
                scn_mlm_mask = torch.zeros_like(seq_mask)
            else:
                seq_mlm_mask = torch.ones_like(seq_mask)
                scn_mlm_mask = torch.ones_like(seq_mask)

            _, mpnn_feature_dict = model.score(x=x,
                                               aatype=aatype,
                                               seq_mask=seq_mask,
                                               missing_atom_mask=missing_atom_mask,
                                               scn_mlm_mask=scn_mlm_mask,
                                               residue_index=residue_index,
                                               chain_index=chain_index,
                                               return_embeddings=True)

        # Save FAMPNN feature dict to output directory
        mpnn_feature_dict = {k: v.cpu() for k, v in mpnn_feature_dict.items() if k in ["h_V", "h_V_enc"]}  # prune to node embeddings only
        lengths = seq_mask.sum(dim=-1).long()
        for j in range(B):
            name = Path(pdb_names[j]).stem  # strip off extension
            out_file = Path(out_dir) / f"{name}.pt"
            length_j = lengths[j].item()
            mpnn_feature_dict_j = {k: v[j, :length_j].clone() for k, v in mpnn_feature_dict.items()}  # clone to avoid extra disk space usage from slice view
            torch.save(mpnn_feature_dict_j, out_file)

        pbar.update(B)
    pbar.close()


def parse_fixed_pos_info(batch: Dict[str, TensorType["b ..."]],
                         pdb_names: List[str],
                         batch_chain_id_mapping: List[Dict[str, int]],  # maps chain letter to chain index
                         pos_constraint_df: Optional[pd.DataFrame],
                         verbose: bool = False) -> Dict[str, torch.Tensor]:

    """
    Given a pos_constraint_df containing fixed positions for each PDB, return a batch updated with:
    - a mask for the aatype and sidechain overrides.
    - possibly overridden "aatype"


    The pos_constraint_df should have the following format:
    index: PDB name (including extension)
    columns: ["fixed_pos_seq", "fixed_pos_scn"]
    where each entry is a comma-separated string of positions in the format "A1-100,B1-100", "A1-10,A15-20", or np.nan.
    """

    aatype_override_mask, scn_override_mask = torch.zeros_like(batch["residue_index"]), torch.zeros_like(batch["residue_index"])

    if pos_constraint_df is None:
        if verbose:
            print("No fixed positions specified, redesigning all positions.")
        batch["aatype_override_mask"] = aatype_override_mask
        batch["scn_override_mask"] = scn_override_mask
        return batch

    for i, pdb_name in enumerate(pdb_names):
        if verbose:
            print(f"\n======================== {pdb_name} ========================")

        if pdb_name not in pos_constraint_df.index:
            if verbose:
                print(f"No fixed positions found for {pdb_name}")
            continue

        ### Get fixed positions from df ###
        row = pos_constraint_df.loc[pdb_name]
        fixed_pos_seq, fixed_pos_scn = row.get("fixed_pos_seq", np.nan), row.get("fixed_pos_scn", np.nan)  # get fixed positions for this PDB
        example = {k: v[i] for k, v in batch.items()}
        chain_id_mapping = batch_chain_id_mapping[i]

        ### Override sequence at specified positions and condition on them ###
        fixed_pos_override_seq = row.get("fixed_pos_override_seq", np.nan)
        if not pd.isna(fixed_pos_override_seq):
            if verbose:
                print(f"{pdb_name}: Overriding sequence at positions {fixed_pos_override_seq}")

            # parse the override string into a list of positions and aatypes
            pdb_pos, abs_pos, override_aatypes = parse_fixed_pos_override_seq_str(fixed_pos_override_seq, chain_id_mapping, example["residue_index"], example["chain_index"])
            for abs_pos_i, aa in zip(abs_pos, override_aatypes):
                batch["aatype"][i, abs_pos_i] = rc.restype_order_with_x[aa]  # override the aatype at the specified position

            # add to fixed_pos_seq
            fixed_pos_seq = f"{fixed_pos_seq}," if not pd.isna(fixed_pos_seq) else ""
            fixed_pos_seq += ",".join(pdb_pos)  # add the positions to the fixed_pos_seq to condition on them

        ### Create override masks based on fixed sequence and sidechain positions ###
        if not pd.isna(fixed_pos_seq):
            # sequence override
            if verbose:
                print(f"{pdb_name}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_pos_str(fixed_pos_seq, chain_id_mapping, example["residue_index"], example["chain_index"])
            aatype_override_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            if verbose:
                fixed_seq_viz = "".join([rc.restypes_with_x[example["aatype"][j]] if aatype_override_mask[i, j] else "-" for j in range(aatype_override_mask.shape[1])])
                print(f"Fixed sequence: {fixed_seq_viz}")
        else:
            if verbose:
                print(f"{pdb_name}: No fixed sequence positions specified.")

        if not pd.isna(fixed_pos_scn):
            # sidechain override
            if verbose:
                print(f"{pdb_name}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_pos_str(fixed_pos_scn, chain_id_mapping, example["residue_index"], example["chain_index"])
            scn_override_mask[i, abs_fixed_pos_scn] = 1

            if not pd.isna(fixed_pos_override_seq):
                # ensure that we're not fixing sidechains when we override the PDB sequence
                assert scn_override_mask[i, abs_fixed_pos_seq].sum() == 0, "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

            # print fixed sidechains
            if verbose:
                fixed_scn_viz = "".join([rc.restypes_with_x[example["aatype"][j]] if scn_override_mask[i, j] else "-" for j in range(scn_override_mask.shape[1])])
                print(f"Fixed sidechains: {fixed_scn_viz}")
        else:
            if verbose:
                print(f"{pdb_name}: No fixed sidechain positions specified.")

    # Update batch
    batch["aatype_override_mask"] = aatype_override_mask
    batch["scn_override_mask"] = scn_override_mask

    return batch


def parse_pos_restrict_aatype_info(batch: Dict[str, TensorType["b ..."]],
                                  pdb_names: List[str],
                                  batch_chain_id_mapping: List[Dict[str, int]],  # maps chain letter to chain index
                                  pos_constraint_df: Optional[pd.DataFrame],
                                  verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """
    Given a pos_constraint_df containing position restrictions for each PDB, return:
    - a mask indicating which positions have restricted amino acid sampling
    - a mask indicating which amino acids are allowed at each position

    The pos_constraint_df should have the following format:
    index: PDB name (including extension)
    columns: ["pos_restrict_aatype"]
    where each entry is a comma-separated string of positions in the format "A1:AVG,B10:ILMV", or None.
    """
    B, N = batch["seq_mask"].shape
    K = len(rc.restype_order_with_x)

    if pos_constraint_df is None:
        if verbose:
            print("No amino acid restrictions specified, allowing all amino acids at all positions.")
        return None

    # Initialize masks for the entire batch
    restrict_pos_mask = torch.zeros((B, N), dtype=torch.float32, device=batch["seq_mask"].device)
    allowed_aatype_mask = torch.ones((B, N, K), dtype=torch.float32, device=batch["seq_mask"].device)

    if verbose:
        print("\n======================== Position-wise amino acid restrictions ========================")

    for i, pdb_name in enumerate(pdb_names):
        if pdb_name not in pos_constraint_df.index:
            if verbose:
                print(f"{pdb_name}: No amino acid restrictions specified.")
            continue

        # Get position restrictions from df
        row = pos_constraint_df.loc[pdb_name]
        pos_restrict_aatype = row.get("pos_restrict_aatype", np.nan)

        if pd.isna(pos_restrict_aatype):
            if verbose:
                print(f"{pdb_name}: No amino acid restrictions specified.")
            continue

        example = {k: v[i] for k, v in batch.items()}
        chain_id_mapping = batch_chain_id_mapping[i]

        if verbose:
            print(f"{pdb_name}: Restricting amino acid sampling at positions {pos_restrict_aatype}")

        # Parse the restriction string into lists of positions and allowed amino acids
        pdb_pos, abs_pos, allowed_aatypes = parse_pos_restrict_aatype_str(
            pos_restrict_aatype,
            chain_id_mapping,
            example["residue_index"],
            example["chain_index"]
        )

        # Mark positions with restrictions
        restrict_pos_mask[i, abs_pos] = 1.0

        # Apply restrictions for each position
        for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
            # First, disallow all amino acids at this position
            allowed_aatype_mask[i, pos_idx, :] = 0.0

            # Then allow only the specified amino acids
            for aa in allowed_aa:
                if aa in rc.restype_order_with_x:
                    allowed_aatype_mask[i, pos_idx, rc.restype_order_with_x[aa]] = 1.0
                else:
                    print(f"Warning: Unknown amino acid '{aa}' in restriction for {pdb_name} at position {pdb_pos[abs_pos.index(pos_idx)]}")

        if verbose:
            # Print a summary of the restrictions
            for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
                pos_str = pdb_pos[abs_pos.index(pos_idx)]
                print(f"  Position {pos_str}: Restricted to {allowed_aa}")
            print("\n========================\n")

    return restrict_pos_mask, allowed_aatype_mask


def parse_fixed_pos_str(fixed_pos_str: str,
                        chain_id_mapping: Dict[str, int],
                        residue_index: TensorType["n", int],
                        chain_index: TensorType["n", int]) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A1", "A10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A1,A10-25").
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        List[int]: List of absolute indices to set to 1 in the masks.
    """
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices  # no positions specified

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        if not match:
            raise ValueError(f"Invalid position format: {pos}")

        chain_letter = match.group(1)
        start_residue = int(match.group(2))
        end_residue = int(match.group(3)) if match.group(3) else start_residue

        if chain_letter not in chain_id_mapping:
            raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

        # For the given chain, create a mask for all residues in the desired range
        chain_i = chain_id_mapping[chain_letter]
        range_mask = (chain_index == chain_i) & (residue_index >= start_residue) & (residue_index <= end_residue)
        matching_indices = torch.where(range_mask)[0]

        # Check that each residue in the requested range; warn if not found
        found_residues = residue_index[matching_indices].tolist()
        found_residues_set = set(found_residues)

        for r in range(start_residue, end_residue + 1):
            if r not in found_residues_set:
                print(f"Warning: Requested position {chain_letter}{r} not found in structure.")

        # Extend our fixed indices with whatever we did find
        fixed_indices.extend(matching_indices.tolist())

    return fixed_indices


def parse_fixed_pos_override_seq_str(override_str: str,
                                 chain_id_mapping: dict[str, int],
                                 residue_index: TensorType["n", int],
                                 chain_index: TensorType["n", int]
                                 ) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a fixed position sequence override string in the format "A26:A,A27:L" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and override amino acids (e.g., ["A", "L"]).

    Args:
        override_str (str): Comma-separated string of position overrides
                           in the format "<chain+residue>:<desired aatype>"
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        tuple: (pdb_pos, abs_pos, override_aatypes) - lists with corresponding entries
    """
    if not override_str or override_str.strip() == "":
        return [], [], []

    pdb_pos = []
    override_aatypes = []

    # Split by comma and process each override
    overrides = [o.strip() for o in override_str.split(",") if o.strip()]

    for override in overrides:
        # Split by colon to get position and override aatype
        parts = override.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid override format: {override}. Expected format: 'A26:A'")

        pos, aatype = parts[0].strip(), parts[1].strip()

        if len(aatype) != 1 or aatype not in rc.restypes_with_x:
            raise ValueError(f"Invalid aatype: {aatype} in {override}. Expected single letter amino acid code.")

        pdb_pos.append(pos)
        override_aatypes.append(aatype)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), chain_id_mapping, residue_index, chain_index)

    return pdb_pos, abs_pos, override_aatypes


def parse_pos_restrict_aatype_str(pos_restrict_str: str,
                                  chain_id_mapping: dict[str, int],  # maps chain letter to chain index
                                  residue_index: TensorType["n", int],
                                  chain_index: TensorType["n", int]) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a position restriction string in the format "A26:AVG,A27:VG" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and allowed aatypes (e.g., ["AVG", "VG"]).

    Args:
        pos_restrict_str (str): Comma-separated string of position restrictions
                               in the format "<chain+residue>:<allowed aatypes>"
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        tuple: (pdb_pos, abs_pos, allowed_aatypes) - lists with corresponding entries
    """
    if not pos_restrict_str or pos_restrict_str.strip() == "":
        return [], [], []

    pdb_pos = []
    allowed_aatypes = []

    # Split by comma and process each restriction
    restrictions = [r.strip() for r in pos_restrict_str.split(",") if r.strip()]

    for restriction in restrictions:
        # Split by colon to get position and allowed aatypes
        parts = restriction.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid restriction format: {restriction}. Expected format: 'A26:AVG'")

        pos, aatypes = parts[0].strip(), parts[1].strip()
        pdb_pos.append(pos)
        allowed_aatypes.append(aatypes)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), chain_id_mapping, residue_index, chain_index)

    return pdb_pos, abs_pos, allowed_aatypes
