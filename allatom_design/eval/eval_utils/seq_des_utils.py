"""
Utils for sampling from sequence design models.
"""

import re
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor
from atomworks.io.parser import parse as aw_parse
from atomworks.io.utils import non_rcsb
from atomworks.io.utils.io_utils import to_cif_string, to_cif_file
from atomworks.ml.utils.token import apply_token_wise, get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
from allatom_design.data.const import AF3_ENCODING
from allatom_design.checkpoint_utils import get_cfg_from_ckpt
from allatom_design.data.data import to
from allatom_design.data.datasets.atomworks_sd_dataset import sd_collator
from allatom_design.data.transform.preprocess import preprocess_transform
from allatom_design.data.transform.sd_featurizer import sd_featurizer
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.data.transform.pad import pad_dim

def get_seq_des_model(cfg: DictConfig, device: str) -> dict[str, Any]:
    """
    Load in a sequence design model.
    Example config:

    seq_des_cfg:
        # MPNN args
        model_name: "atom_mpnn"  # ["atom_mpnn"]
            atom_mpnn:
                # Atom MPNN args
                atom_mpnn_cfg: caliby/configs/seq_des/atom_mpnn_inference.yaml
                atom_mpnn_ckpt:
    """
    model_name = cfg.model_name
    seq_des_model = {"model_name": model_name, "cfg": cfg, "device": device}

    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(cfg.atom_mpnn.ckpt_path).eval()
    model_cfg, _ = get_cfg_from_ckpt(cfg.atom_mpnn.ckpt_path)
    data_cfg = hydra.utils.instantiate(model_cfg.data)
    sampling_cfg = OmegaConf.load(cfg.atom_mpnn.sampling_cfg)
    sampling_cfg = OmegaConf.merge(sampling_cfg, OmegaConf.to_container(cfg.atom_mpnn.overrides, resolve=True))
    seq_des_model["model"] = lit_sd_model.model
    seq_des_model["data_cfg"] = data_cfg
    seq_des_model["sampling_cfg"] = sampling_cfg

    return seq_des_model


def run_seq_des(
    *,
    model: SeqDenoiser = None,
    data_cfg: DictConfig = None,
    sampling_cfg: DictConfig = None,
    pdb_paths: list[str] = None,
    device: str = None,
    out_dir: str = None,
    pos_constraint_df: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """
    Given a list of processed structure files, run sequence design on them.

    If out_dir is not None, PDBs with sampled sequences will be saved to the provided directory. In this case, run_aux
    will be a dictionary with the following keys:
        - "out_pdb": list of output PDB paths
        - "pred_seqs": list of predicted sequences as a string for each sample
    """
    # Set up outputs.
    outputs = defaultdict(list)
    sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df.
    if pos_constraint_df is not None:
        valid_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # Columns in input df must be a subset of valid columns.
            raise ValueError(
                f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. "
                f"Found: {pos_constraint_df.columns}"
            )
        # Set index to pdb name.
        pos_constraint_df = pos_constraint_df.set_index("pdb_key")

        # Set empty string to NaN for easier parsing.
        pos_constraint_df = pos_constraint_df.replace("", np.nan)

    # Print omitted amino acids.
    if sampling_cfg.verbose and sampling_cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {sampling_cfg.omit_aas}")

    # Process PDBs in parallel.
    parallel_context = Parallel(n_jobs=sampling_cfg.num_workers) if sampling_cfg.num_workers > 1 else nullcontext()

    # Begin sampling.
    pbar = tqdm(
        total=len(pdb_paths),
        desc=f"Sampling {len(pdb_paths)} PDBs, {sampling_cfg.num_seqs_per_pdb} sequences per PDB...",
    )
    total_avg_seq_recovery = 0.0
    total_avg_lp_seq_recovery = 0.0
    with parallel_context as parallel_pool:
        for pi in range(0, len(pdb_paths), sampling_cfg.batch_size):
            batch_pdb_paths = pdb_paths[pi : pi + sampling_cfg.batch_size]
            B = len(batch_pdb_paths)
            batch = get_sd_batch(batch_pdb_paths, data_cfg=data_cfg, device=device, parallel_pool=parallel_pool)

            # Initialize seq_cond and atom_cond masks.
            batch = initialize_sampling_masks(batch)
            
            # Initialize ligand pocket mask if ligand conditioning is enabled.
            if sampling_cfg.ligand_conditioning:
                batch = initialize_ligand_pocket_mask(batch, ligand_pocket_dist_cutoff=sampling_cfg.ligand_pocket_dist_cutoff,
                                                      small_molecule_only=sampling_cfg.small_molecule_only)

            # Parse fixed positions.
            batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=sampling_cfg.verbose)

            # Restrict aatype sampling at certain positions.
            sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
            sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(
                batch, pos_constraint_df, verbose=sampling_cfg.verbose
            )

            # Run sampling.
            id_to_atom_arrays, id_to_aux = model.sample(batch, sampling_inputs=sampling_inputs)

            # Save outputs.
            example_id_to_batch_idx = {eid: idx for idx, eid in enumerate(batch["example_id"])}
            for si, (example_id, atom_arrays) in enumerate(id_to_atom_arrays.items()):
                aux = id_to_aux[example_id]
                sample_stems = [f"{example_id}_sample{si}" for si in range(len(atom_arrays))]

                # Save output atom arrays to cif files.
                for ai, sample_stem in enumerate(sample_stems):
                    out_file = f"{sample_out_dir}/{sample_stem}.cif"
                    atom_array = atom_arrays[ai]
                    with open(out_file, "w") as f:
                        f.write(to_cif_string(atom_array, include_nan_coords=False))
                    
                    outputs["example_id"].append(example_id)
                    # outputs["out_pdb"].append(out_file)
                    outputs["U"].append(aux[ai]["U"])

                # Get sampled sequences as a string, with ":" to separate chains.
                for ai in range(len(atom_arrays)):
                    chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_arrays[ai])
                    outputs["seq"].append(
                        ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                    )
                    
                # Compute sequence recovery metrics
                bi = example_id_to_batch_idx[example_id]
                orig_res_types = batch["restype"][bi].argmax(dim=-1)          
                seq_mask = (1 - batch["seq_cond_mask"][bi]) * batch["token_pad_mask"][bi] * batch["token_resolved_mask"][bi]
                lp_seq_mask = (1 - batch["seq_cond_mask"][bi]) * batch["token_pad_mask"][bi] * batch["sm_pocket_token_mask"][bi]
                
                total_seq_recovery = 0.0
                total_lp_seq_recovery = 0.0
                for ai in range(len(atom_arrays)):
                    samp_atom_array = atom_arrays[ai]
                    samp_token_starts = get_token_starts(samp_atom_array)
                    samp_res_names = samp_atom_array.res_name[samp_token_starts]
                    samp_res_types = AF3_ENCODING.encode(samp_res_names)                    
                    samp_res_types = torch.tensor(samp_res_types, device=device)
                    samp_res_types = pad_dim(samp_res_types, 0, len(orig_res_types) - len(samp_res_types))
                    
                    # Compute sequence recovery
                    seq_recovery = (samp_res_types == orig_res_types).float() * seq_mask
                    seq_recovery = seq_recovery.sum() / seq_mask.sum().clamp(min=1e-8)
                    total_seq_recovery += seq_recovery
                                                                    
                    # Compute LP sequence recovery
                    lp_seq_recovery = (samp_res_types == orig_res_types).float() * lp_seq_mask
                    lp_seq_recovery = lp_seq_recovery.sum() / lp_seq_mask.sum().clamp(min=1e-8)
                    total_lp_seq_recovery += lp_seq_recovery
                    
                    print (f"sample {ai} of {example_id}: seq recovery: {seq_recovery}, lp seq recovery: {lp_seq_recovery}")                                        
                                        
                avg_seq_recovery = total_seq_recovery / len(atom_arrays)
                avg_lp_seq_recovery = total_lp_seq_recovery / len(atom_arrays)
                outputs["avg_seq_recovery"].append(avg_seq_recovery)
                outputs["avg_lp_seq_recovery"].append(avg_lp_seq_recovery)
                
                
                total_avg_seq_recovery += avg_seq_recovery
                total_avg_lp_seq_recovery += avg_lp_seq_recovery

                print (f"{example_id} avg seq recovery: {seq_recovery}, avg lp seq recovery: {lp_seq_recovery} out of {len(atom_arrays)} samples")                   

            pbar.update(B)
    pbar.close()

    return outputs


def run_seq_des_ensemble(
    *,
    model: SeqDenoiser,
    data_cfg: DictConfig,
    sampling_cfg: DictConfig,
    pdb_to_conformers: dict[str, list[str]],  # maps from a given pdb name to its conformer pdb files
    device: str,
    out_dir: str,
    pos_constraint_df: pd.DataFrame | None = None,
    use_primary_res_type: bool = True,  # use res_type from primary structure. Otherwise use res_type from conformer pdb
) -> dict[str, Any]:
    """
    Given a list of processed structure files, run sequence design on them.
    """
    # Set up outputs.
    outputs = defaultdict(list)
    sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df.
    if pos_constraint_df is not None:
        valid_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # Columns in input df must be a subset of valid columns.
            raise ValueError(
                f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. Found: {pos_constraint_df.columns}"
            )
        # Set index to pdb name.
        pos_constraint_df = pos_constraint_df.set_index("pdb_key")

        # Set empty string to NaN for easier parsing.
        pos_constraint_df = pos_constraint_df.replace("", np.nan)

    # Print omitted amino acids.
    if sampling_cfg.verbose and sampling_cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {sampling_cfg.omit_aas}")

    # Process PDBs in parallel.
    parallel_context = Parallel(n_jobs=sampling_cfg.num_workers) if sampling_cfg.num_workers > 1 else nullcontext()

    # Begin sampling.
    with parallel_context as parallel_pool:
        for pdb_name, pdb_paths in tqdm(
            pdb_to_conformers.items(),
            desc=f"Sampling {len(pdb_to_conformers)} PDBs, {sampling_cfg.num_seqs_per_pdb} sequences per PDB...",
        ):
            # Create tied_sampling_ids by tying all samples together.
            batch = get_sd_batch(pdb_paths, device=device, data_cfg=data_cfg, parallel_pool=parallel_pool)
            batch["tied_sampling_ids"] = torch.zeros(len(pdb_paths), device=device, dtype=torch.long)

            # Use res_type from primary structure
            if use_primary_res_type:
                # Update restype in batch.
                batch["restype"] = batch["restype"][0:1].expand(len(pdb_paths), *((batch["restype"].ndim - 1) * (-1,)))

                # Update atom array annotations.
                for i in range(1, len(batch["atom_array"])):
                    atomwise_resnames = spread_token_wise(
                        batch["atom_array"][i],
                        const.AF3_ENCODING.idx_to_token[batch["restype"][0].argmax(dim=-1).cpu().numpy()],
                    )
                    batch["atom_array"][i].set_annotation("res_name", atomwise_resnames)

            # Ensure that all entries in the batch have the same residue and chain index so that they're aligned.
            if not sampling_cfg["ensemble_ignore_res_idx_mismatch"]:
                _validate_ensemble_alignment(batch)

            # Initialize seq_cond and atom_cond masks.
            batch = initialize_sampling_masks(batch)

            # Parse fixed positions.
            batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=sampling_cfg.verbose)

            # Restrict aatype sampling at certain positions.
            sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
            sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(
                batch, pos_constraint_df, verbose=sampling_cfg.verbose
            )

            # Run sampling.
            id_to_atom_arrays, id_to_aux = model.sample(batch, sampling_inputs=sampling_inputs)

            # Save outputs.
            for example_id, atom_arrays in id_to_atom_arrays.items():
                aux = id_to_aux[example_id]
                sample_stems = [f"{example_id}_sample{si}" for si in range(len(atom_arrays))]

                # Save output atom arrays to cif files.
                for si in range(len(atom_arrays)):
                    out_file = f"{sample_out_dir}/{sample_stems[si]}.cif"
                    atom_array = atom_arrays[si]
                    with open(out_file, "w") as f:
                        f.write(to_cif_string(atom_array, include_nan_coords=False))

                    outputs["example_id"].append(example_id)
                    outputs["out_pdb"].append(out_file)
                    outputs["U"].append(aux[si]["U"])

                # Get sampled sequences as a string, with ":" to separate chains.
                for si in range(len(atom_arrays)):
                    chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_arrays[si])
                    outputs["seq"].append(
                        ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                    )

    return outputs

def run_lc_seq_des(
    *,
    model: SeqDenoiser = None,
    load_from_cache: bool = True,
    featurizer_cfg: DictConfig = None,
    sampling_cfg: DictConfig = None,
    metadata: pd.DataFrame = None,
    pdb_paths: list[str] = None,
    device: str = None,
    out_dir: str = None,
    pos_constraint_df: pd.DataFrame | None = None,
    io_cfg: DictConfig = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """
    Given a list of processed structure files, run sequence design on them.

    If out_dir is not None, PDBs with sampled sequences will be saved to the provided directory. In this case, run_aux
    will be a dictionary with the following keys:
        - "out_pdb": list of output PDB paths
        - "pred_seqs": list of predicted sequences as a string for each sample
    """
    # Set up outputs.
    outputs = defaultdict(list)
    sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df.
    if pos_constraint_df is not None:
        valid_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # Columns in input df must be a subset of valid columns.
            raise ValueError(
                f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. "
                f"Found: {pos_constraint_df.columns}"
            )
        # Set index to pdb name.
        pos_constraint_df = pos_constraint_df.set_index("pdb_key")

        # Set empty string to NaN for easier parsing.
        pos_constraint_df = pos_constraint_df.replace("", np.nan)

    # Print omitted amino acids.
    if sampling_cfg.verbose and sampling_cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {sampling_cfg.omit_aas}")

    # Process PDBs in parallel.
    parallel_context = Parallel(n_jobs=sampling_cfg.num_workers) if sampling_cfg.num_workers > 1 else nullcontext()

    # Begin sampling.
    pbar = tqdm(
        total=len(pdb_paths),
        desc=f"Sampling {len(pdb_paths)} PDBs, {sampling_cfg.num_seqs_per_pdb} sequences per PDB...",
    )
    
    total_avg_seq_recovery = 0.0
    total_avg_sp_seq_recovery = 0.0 #! small molecule pocket
    # total_avg_mp_seq_recovery = 0.0 #! metal pocket
    # total_avg_np_seq_recovery = 0.0 #! nucleotide pocket
    with parallel_context as parallel_pool:
        for pi in range(0, len(pdb_paths), sampling_cfg.batch_size):
            batch_pdb_paths = pdb_paths[pi : pi + sampling_cfg.batch_size]
            B = len(batch_pdb_paths)
            batch = get_sd_batch(batch_pdb_paths, featurizer_cfg=featurizer_cfg, device=device, parallel_pool=parallel_pool, 
                                 load_from_cache=load_from_cache, metadata=metadata)

            # Initialize seq_cond and atom_cond masks.
            batch = initialize_sampling_masks(batch)
            
            # Initialize ligand pocket mask if ligand conditioning is enabled.
            if sampling_cfg.ligand_conditioning:
                batch = initialize_pocket_mask(batch, ligand_pocket_dist_cutoff=sampling_cfg.ligand_pocket_dist_cutoff,
                                                      small_molecule_only=sampling_cfg.small_molecule_only)

            # Parse fixed positions.
            batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=sampling_cfg.verbose)

            # Restrict aatype sampling at certain positions.
            sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
            sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(
                batch, pos_constraint_df, verbose=sampling_cfg.verbose
            )

            # Run sampling.
            id_to_atom_arrays, id_to_aux = model.sample(batch, sampling_inputs=sampling_inputs)

            # Save outputs.
            example_id_to_batch_idx = {eid: idx for idx, eid in enumerate(batch["example_id"])}
            for si, (example_id, atom_arrays) in enumerate(id_to_atom_arrays.items()):
                aux = id_to_aux[example_id]
                
                sample_stems = [convert_stem(f"{example_id}_sample{si}") for si in range(len(atom_arrays))]
                
                DUMMY_DATE = "1959-01-08" 
                extra_categories = {
                    "pdbx_audit_revision_history": {
                        "ordinal": [1],
                        "revision_id": [1], 
                        "revision_date": [DUMMY_DATE],
                        "major_revision": [1],
                        "minor_revision": [0],
                        "revision_description": ["Dummy date for template-conditioning AF3"]
                    }
                }

                # Save output atom arrays to cif files.
                for ai, sample_stem in enumerate(sample_stems):
                    out_file = f"{sample_out_dir}/{sample_stem}.cif"
                    atom_array = atom_arrays[ai]
                    
                    # with open(out_file, "w") as f:
                    #     f.write(to_cif_string(atom_array, include_nan_coords=False))
                    out_file = to_cif_file(atom_arrays[ai], 
                                           out_file,
                                           file_type="cif",
                                           date = DUMMY_DATE,
                                           include_entity_poly=True,
                                           include_entity_nonpoly=True,
                                           include_nan_coords=False,
                                           include_bonds=True,
                                           extra_categories=extra_categories)
                    
                    # Fix pdbx_formal_charge format for OpenStructure compatibility
                    # OpenStructure expects integer (e.g., "1") not signed (e.g., "+1")
                    _fix_cif_formal_charge(out_file)

                    outputs["example_id"].append(example_id)
                    outputs["out_pdb"].append(out_file)
                    outputs["U"].append(aux[ai]["U"])

                # Get sampled sequences as a string, with ":" to separate chains.
                for ai in range(len(atom_arrays)):
                    chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_arrays[ai])
                    outputs["seq"].append(
                        ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                    )
                    outputs["atom_array"].append(atom_arrays[ai])
                    
                    
                # Compute sequence recovery metrics            
                bi = example_id_to_batch_idx[example_id]
                orig_res_types = batch["restype"][bi].argmax(dim=-1)          
                seq_mask = (1 - batch["seq_cond_mask"][bi]) * batch["token_pad_mask"][bi] * batch["token_resolved_mask"][bi]
                lp_seq_mask = (1 - batch["seq_cond_mask"][bi]) * batch["token_pad_mask"][bi] * batch["pocket_token_mask"][bi]
                
                total_seq_recovery = 0.0
                total_sp_seq_recovery = 0.0
                # total_mp_seq_recovery = 0.0
                # total_np_seq_recovery = 0.0
                for ai in range(len(atom_arrays)):
                    samp_atom_array = atom_arrays[ai]
                    samp_token_starts = get_token_starts(samp_atom_array)
                    samp_res_names = samp_atom_array.res_name[samp_token_starts]
                    samp_res_types = AF3_ENCODING.encode(samp_res_names)                    
                    samp_res_types = torch.tensor(samp_res_types, device=device)
                    samp_res_types = pad_dim(samp_res_types, 0, len(orig_res_types) - len(samp_res_types))
                    
                    # Compute sequence recovery
                    seq_recovery = (samp_res_types == orig_res_types).float() * seq_mask
                    seq_recovery = seq_recovery.sum() / seq_mask.sum().clamp(min=1e-8)
                    outputs["sample_seq_recovery"].append(seq_recovery)
                    
                    total_seq_recovery += seq_recovery
                                                                    
                    # Compute sp sequence recovery
                    sp_seq_recovery = (samp_res_types == orig_res_types).float() * lp_seq_mask
                    sp_seq_recovery = sp_seq_recovery.sum() / lp_seq_mask.sum().clamp(min=1e-8)
                    outputs["sample_sp_seq_recovery"].append(sp_seq_recovery)
                    total_sp_seq_recovery += sp_seq_recovery
                    
                    # Todo: Compute mp and np sequence recovery
                    
                    print (f"sample {ai} of {example_id}: seq recovery: {seq_recovery}, sp seq recovery: {sp_seq_recovery}")                                        
                                        
                avg_seq_recovery = total_seq_recovery / len(atom_arrays)
                avg_sp_seq_recovery = total_sp_seq_recovery / len(atom_arrays)
                outputs["sample_avg_seq_recovery"].append(avg_seq_recovery)
                outputs["sample_avg_sp_seq_recovery"].append(avg_sp_seq_recovery)
                                
                total_avg_seq_recovery += avg_seq_recovery
                total_avg_sp_seq_recovery += avg_sp_seq_recovery

                print (f"{example_id} avg seq recovery: {seq_recovery}, avg sp seq recovery: {sp_seq_recovery} out of {len(atom_arrays)} samples")                   

            pbar.update(B)
    pbar.close()
    total_avg_seq_recovery /= len(pdb_paths)
    total_avg_sp_seq_recovery /= len(pdb_paths)
    outputs["total_avg_seq_recovery"] = total_avg_seq_recovery
    outputs["total_avg_sp_seq_recovery"] = total_avg_sp_seq_recovery
    print (f"Total avg seq recovery: {total_avg_seq_recovery}, Total avg sp seq recovery: {total_avg_sp_seq_recovery}")
    
    for k, v in outputs.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            outputs[k] = [t.detach().cpu().item() for t in v]
        elif isinstance(v, torch.Tensor):
            outputs[k] = v.detach().cpu().item()
        else:
            outputs[k] = v

    return outputs


def score_samples(
    *,
    model: SeqDenoiser,
    data_cfg: DictConfig,
    sampling_cfg: DictConfig,
    pdb_paths: list[str],
    device: str,
) -> dict[str, Any]:
    """
    Given a list of processed structure files, score the sequences on them.
    """
    # Set up outputs.
    outputs = defaultdict(list)

    # Process PDBs in parallel.
    pbar = tqdm(total=len(pdb_paths), desc=f"Scoring {len(pdb_paths)} PDBs...")
    parallel_context = Parallel(n_jobs=sampling_cfg.num_workers) if sampling_cfg.num_workers > 1 else nullcontext()

    # Begin scoring.
    with parallel_context as parallel_pool:
        for i in range(0, len(pdb_paths), sampling_cfg.batch_size):
            batch_pdb_paths = pdb_paths[i : i + sampling_cfg.batch_size]
            B = len(batch_pdb_paths)
            batch = get_sd_batch(batch_pdb_paths, device=device, data_cfg=data_cfg, parallel_pool=parallel_pool)

            # Initialize seq_cond and atom_cond masks.
            batch = initialize_sampling_masks(batch)
                        
            # Score samples.
            sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
            id_to_aux = model.score_samples(batch, sampling_inputs=sampling_inputs)

            # Store results.
            for example_id, aux in id_to_aux.items():
                outputs["example_id"].append(example_id)
                chain_info = non_rcsb.initialize_chain_info_from_atom_array(aux["atom_array"])
                outputs["seq"].append(
                    ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                )
                outputs["U"].append(aux["U"])
                outputs["U_i"].append(aux["U_i"])

            pbar.update(B)
    pbar.close()

    return outputs


def score_samples_ensemble(
    *,
    model: SeqDenoiser,
    data_cfg: DictConfig,
    sampling_cfg: DictConfig,
    pdb_to_conformers: dict[str, list[str]],  # maps from a given pdb name to its conformer pdb files
    device: str,
) -> dict[str, Any]:
    """
    Score sequences using Potts parameters computed from an ensemble of input backbones.
    """
    outputs = defaultdict(list)

    # Process PDBs in parallel.
    parallel_context = Parallel(n_jobs=sampling_cfg.num_workers) if sampling_cfg.num_workers > 1 else nullcontext()
    with parallel_context as parallel_pool:
        for pdb_name, pdb_paths in tqdm(pdb_to_conformers.items(), desc=f"Scoring {len(pdb_to_conformers)} PDBs..."):
            # Create tied_sampling_ids by tying all samples together.
            batch = get_sd_batch(pdb_paths, device=device, data_cfg=data_cfg, parallel_pool=parallel_pool)
            batch["tied_sampling_ids"] = torch.zeros(len(pdb_paths), device=device, dtype=torch.long)

            # Ensure that all entries in the batch have the same residue and chain index so that they're aligned.
            if not sampling_cfg["ensemble_ignore_res_idx_mismatch"]:
                _validate_ensemble_alignment(batch)

            # Initialize seq_cond and atom_cond masks.
            batch = initialize_sampling_masks(batch)

            # Score samples.
            sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
            id_to_aux = model.score_samples(batch, sampling_inputs=sampling_inputs)

            # Store results.
            for example_id, aux in id_to_aux.items():
                outputs["example_id"].append(example_id)
                chain_info = non_rcsb.initialize_chain_info_from_atom_array(aux["atom_array"])
                outputs["seq"].append(
                    ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                )
                outputs["U"].append(aux["U"])
                outputs["U_i"].append(aux["U_i"])

    return outputs


def get_sd_batch(
    pdb_paths: list[str], *, 
    featurizer_cfg: DictConfig | None, 
    device: str, parallel_pool: Parallel | None, 
    load_from_cache: bool = True,
    metadata: pd.DataFrame = None,
) -> dict[str, Any]:
    """
    Given a list of pdb file paths, return a batch of sequence design model features.

    If data_cfg is None, use default cif parser args.
    """
    if parallel_pool is None:
        # Load PDBs sequentially.
        batch_examples = [get_sd_example(pdb_path, featurizer_cfg, load_from_cache=load_from_cache, metadata=metadata) for pdb_path in pdb_paths]
    else:
        # Load PDBs in parallel.
        batch_examples = parallel_pool(delayed(get_sd_example)(pdb_path, featurizer_cfg, load_from_cache=load_from_cache, metadata=metadata) for pdb_path in pdb_paths)

    # Collate examples.
    batch = sd_collator(batch_examples)
    batch = to(batch, device)

    return batch


def get_sd_example(pdb_path: str = None,
                   featurizer_cfg: DictConfig = None, 
                   load_from_cache: bool = True, 
                   metadata: pd.DataFrame = None,
                   data_cfg: DictConfig = None,
                   preprocess_transform_cfg: DictConfig = None,
                   ) -> dict[str, Any]:
    """
    Given a pdb file path, return a dictionary of sequence design model features.

    If data_cfg is None, use default cif parser args.
    """
    
    # Load the cache example if it exists.
    if load_from_cache:
        # Todo: Right now, only one ligand is supported. Need to modify this to support
        # Todo: i) Single protein chain with multiple ligands at different pocket sites
        # Todo: ii) Single protein chain with multiple ligands at the same pocket site
        example = load_cached_example(pdb_path)
        pdb_id = Path(pdb_path).stem
    else:
        example = preprocess_pdb(pdb_path, data_cfg = data_cfg, preprocess_transform_cfg = preprocess_transform_cfg)
        pdb_id = (Path(pdb_path).stem).split("_")[0]
        
    metadata_example = metadata[metadata["pdb_id"] == pdb_id].reset_index(drop=True)
    
    #! Replace information in the example with the information from the metadata
    #! Since when caching, it doesn't really care about the metadata as it's just caching the whole structure        
    example['example_id'] = metadata_example["example_id"].iloc[0]
    query_pn_unit_iids = metadata_example["q_pn_unit_iid_1"].tolist() + metadata_example["q_pn_unit_iid_2"].tolist()
    example["query_pn_unit_iids"] = query_pn_unit_iids
    
    example['extra_info'] = {} #! delete all the information preexisting in the example
    row_dict = metadata_example.iloc[0].to_dict() # To series to ignore the index
    example['extra_info'] = row_dict
                                                                
    # Featurize the example.
    featurizer = sd_featurizer(**featurizer_cfg)
    example = featurizer(example)

    return example

def load_cached_example(pdb_path: str) -> dict[str, torch.Tensor]:
    cached_example_path = f"{pdb_path}"
    return torch.load(cached_example_path, map_location="cpu", weights_only=False)

def preprocess_pdb(pdb_path: str, data_cfg: DictConfig | None, 
                   preprocess_transform_cfg: DictConfig | None) -> dict[str, Any]:
    """
    Preprocess a PDB file using the preprocessing pipeline.
    """
    # Set up arguments for parsing cifs with AtomWorks.
    if data_cfg is None:
        default_cif_parser_args = {
            "add_missing_atoms": True,
            "remove_waters": True,
            "remove_ccds": [],
            "fix_ligands_at_symmetry_centers": True,
            "fix_arginines": True,
            "convert_mse_to_met": True,
            "hydrogen_policy": "remove",
            "extra_fields": "all",
        }
        cif_parser_args = default_cif_parser_args
    else:
        cif_parser_args = OmegaConf.to_container(data_cfg.cif_parser_args, resolve=True)

    # Read in the CIF data.
    transformation_id = "1"  # Leep only the first assembly.
    cif_parser_args["build_assembly"] = [transformation_id]
    input_data = aw_parse(pdb_path, **cif_parser_args)
    atom_array_from_cif = input_data["assemblies"][transformation_id][0]  # (1, num_atoms) -> (num_atoms)

    # Run the preprocessing pipeline on the CIF data.
    pipeline = preprocess_transform(**dict(preprocess_transform_cfg))
    return pipeline(
        data={
            "example_id": Path(pdb_path).stem,
            "atom_array": atom_array_from_cif,
            "chain_info": input_data["chain_info"],
        }
    )


def initialize_sampling_masks(batch: dict[str, TensorType["b ..."]]) -> dict[str, torch.Tensor]:
    """
    Initialize the sampling masks for the batch. Modifies batch in place and returns it.
    """
    # Initialize sequence mask: always condition on non-protein or non-standard residues.
    standard_prot_mask = batch["chain_is_protein"] & ~batch["is_atomized"]
    
    batch["seq_cond_mask"] = torch.zeros_like(batch["token_pad_mask"])
    batch["seq_cond_mask"] = torch.where(
        standard_prot_mask, torch.zeros_like(batch["seq_cond_mask"]), batch["token_resolved_mask"]
    )
    batch["seq_cond_mask"] *= batch["token_pad_mask"]

    # Initialize atom mask: 
    # masks for protein backbone atoms
    atomwise_chain_is_protein = batch["chain_is_protein"].gather(dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"] # re-mask out pad atoms
    prot_bb_atom_mask = batch["prot_bb_atom_mask"] * batch["atom_resolved_mask"] * atomwise_chain_is_protein 

    # atom_cond_mask
    batch["atom_cond_mask"] = torch.where(atomwise_chain_is_protein.bool(), prot_bb_atom_mask.bool(), batch["atom_resolved_mask"].bool())
    batch["atom_cond_mask"] = batch["atom_cond_mask"] & batch["atom_pad_mask"].bool()

    return batch

def initialize_pocket_mask(batch: dict[str, TensorType["b ..."]],
                                  ligand_pocket_dist_cutoff: float = None,
                                  small_molecule_only: bool = True) -> dict[str, torch.Tensor]:
        
    if ligand_pocket_dist_cutoff is None:
        ligand_pocket_dist_cutoff = 5.0
        
    B, N, _ = batch["coords"].shape
    coords = batch["coords"] * batch["atom_resolved_mask"].unsqueeze(-1) * batch["atom_pad_mask"].unsqueeze(-1)
    atom_mask = batch["atom_resolved_mask"] * batch["atom_pad_mask"]
    
    # Compute protein coords
    protein_token_mask = batch["chain_is_protein"] * batch["is_protein"] * batch["token_resolved_mask"] * batch["token_pad_mask"] # [B, N_tokens]
    protein_atom_mask = torch.gather(protein_token_mask, dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"]                
    protein_coords = coords * protein_atom_mask.unsqueeze(-1)
    
    # Compute ligand coords
    non_protein_token_mask = ~batch["chain_is_protein"] * batch["token_resolved_mask"] * batch["token_pad_mask"]
    non_protein_atom_mask = torch.gather(non_protein_token_mask, dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"]
    non_protein_coords = coords * non_protein_atom_mask.unsqueeze(-1)
    dist_mat_mask = protein_atom_mask[:, :, None] * non_protein_atom_mask[:, None, :] 
    
    pocket_atom_mask = torch.cdist(protein_coords, non_protein_coords) 
    pocket_atom_mask = torch.where(dist_mat_mask.bool(), pocket_atom_mask, torch.ones_like(pocket_atom_mask, device=coords.device) * torch.inf)
    pocket_atom_mask = pocket_atom_mask < ligand_pocket_dist_cutoff # [B, N_atoms, N_atoms]
    
    pocket_atom_mask = torch.any(pocket_atom_mask, dim=-1) # [B, N_atoms]
    pocket_token_mask = torch.zeros_like(batch["token_pad_mask"], device=coords.device, dtype=torch.bool) # [B, N_tokens]
    pocket_token_mask.scatter_(dim=-1, index=batch["atom_to_token_map"], src=pocket_atom_mask.bool()) # [B, N_tokens]        
    pocket_token_mask = pocket_token_mask * batch["token_pad_mask"].bool() * batch["token_resolved_mask"].bool()
    
    batch["pocket_token_mask"] = pocket_token_mask
    
    return batch


def parse_fixed_pos_info(
    batch: dict[str, TensorType["b ..."]], pos_constraint_df: pd.DataFrame | None, verbose: bool = False
) -> dict[str, torch.Tensor]:
    """
    Given a pos_constraint_df containing fixed positions for each PDB, return a batch updated with:
    - a mask for seq-level and atom-level conditioning
    - possibly overridden "res_type"

    The pos_constraint_df should have the following format:
    index: PDB name (not including extension)
    columns: ["fixed_pos_seq", "fixed_pos_scn"]
    where each entry is a comma-separated string of positions in the format "A1-100,B1-100", "A1-10,A15-20", or np.nan.
    """

    seq_cond_mask, atom_cond_mask = batch["seq_cond_mask"].clone(), batch["atom_cond_mask"].clone()

    if pos_constraint_df is None:
        if verbose:
            print("No fixed positions specified, redesigning all positions.")
        return batch

    for i, example_id in enumerate(batch["example_id"]):
        if verbose:
            print(f"\n======================== {example_id} ========================")

        if example_id not in pos_constraint_df.index:
            if verbose:
                print(f"No fixed positions found for {example_id}")
            continue

        ### Get fixed positions from df ###
        row = pos_constraint_df.loc[example_id]
        fixed_pos_seq, fixed_pos_scn = (
            row.get("fixed_pos_seq", np.nan),
            row.get("fixed_pos_scn", np.nan),
        )  # get fixed positions for this PDB

        # Set up example
        example = {k: v[i] for k, v in batch.items()}

        ### Override sequence at specified positions and condition on them ###
        fixed_pos_override_seq = row.get("fixed_pos_override_seq", np.nan)
        if not pd.isna(fixed_pos_override_seq):
            if verbose:
                print(f"{example_id}: Overriding sequence at positions {fixed_pos_override_seq}")

            # parse the override string into a list of positions and aatypes
            pdb_pos, override_abs_pos, override_aatypes = parse_fixed_pos_override_seq_str(
                fixed_pos_override_seq, example["atom_array"]
            )
            for abs_pos_i, aa in zip(override_abs_pos, override_aatypes):
                # Update restype in batch.
                batch["restype"][i, abs_pos_i] = F.one_hot(
                    torch.tensor(const.AF3_ENCODING.encode_aa_seq(aa), device=batch["restype"].device),
                    num_classes=const.AF3_ENCODING.n_tokens,
                )

            # Update atom array annotations.
            token_pad_mask = batch["token_pad_mask"][
                i
            ].bool()  # we need to get rid of padding since atom_arrays are not padded
            resnames = const.AF3_ENCODING.idx_to_token[batch["restype"][i][token_pad_mask].argmax(dim=-1).cpu().numpy()]
            atomwise_resnames = spread_token_wise(batch["atom_array"][i], resnames)
            batch["atom_array"][i].set_annotation("res_name", atomwise_resnames)

            # add to fixed_pos_seq
            fixed_pos_seq = f"{fixed_pos_seq}," if not pd.isna(fixed_pos_seq) else ""
            fixed_pos_seq += ",".join(pdb_pos)  # add the positions to the fixed_pos_seq to condition on them

        ### Create override masks based on fixed sequence and sidechain positions ###
        if not pd.isna(fixed_pos_seq):
            # sequence override
            if verbose:
                print(f"{example_id}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_pos_str(fixed_pos_seq, example["atom_array"])
            seq_cond_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            if verbose:
                print("Fixed sequence:")
                visualize_conditioning_sequences(
                    example["atom_array"],
                    seq_cond_mask[i][example["token_pad_mask"].bool()],
                    example["asym_id"][example["token_pad_mask"].bool()],
                    example["feat_metadata"]["asym_name"],
                )
        else:
            if verbose:
                print(f"{example_id}: No fixed sequence positions specified.")

        if not pd.isna(fixed_pos_scn):
            # sidechain override
            if verbose:
                print(f"{example_id}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_pos_str(fixed_pos_scn, example["atom_array"])
            scn_atom_mask = torch.isin(
                example["atom_to_token_map"],
                torch.tensor(abs_fixed_pos_scn, device=example["atom_to_token_map"].device),
            )
            atom_cond_mask[i] = torch.where(scn_atom_mask, example["atom_resolved_mask"], atom_cond_mask[i])

            # ensure that we're not fixing sidechains when we override the PDB sequence
            scn_cond_num_atoms = apply_token_wise(example["atom_array"], scn_atom_mask.cpu().numpy(), np.sum)
            if not pd.isna(fixed_pos_override_seq):
                assert (
                    scn_cond_num_atoms[override_abs_pos] == 0
                ).all(), "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

            # print fixed sidechains
            if verbose:
                print("Fixed sidechains:")
                visualize_conditioning_sequences(
                    example["atom_array"],
                    torch.tensor(scn_cond_num_atoms > 0),
                    example["asym_id"][example["token_pad_mask"].bool()].cpu(),
                    example["feat_metadata"]["asym_name"],
                )
        else:
            if verbose:
                print(f"{example_id}: No fixed sidechain positions specified.")

    # Update batch
    batch["seq_cond_mask"] = seq_cond_mask
    batch["atom_cond_mask"] = atom_cond_mask
    return batch


def parse_pos_restrict_aatype_info(
    batch: dict[str, TensorType["b ..."]], pos_constraint_df: pd.DataFrame | None, verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Given a pos_constraint_df containing position restrictions for each PDB, return:
    - a mask indicating which positions have restricted amino acid sampling
    - a mask indicating which amino acids are allowed at each position

    The pos_constraint_df should have the following format:
    index: PDB name (not including extension)
    columns: ["pos_restrict_aatype"]
    where each entry is a comma-separated string of positions in the format "A1:AVG,B10:ILMV", or None.
    """
    B, N = batch["token_pad_mask"].shape
    K = const.AF3_ENCODING.n_tokens

    if pos_constraint_df is None:
        if verbose:
            print("No amino acid restrictions specified, allowing all amino acids at all positions.")
        return None

    # Initialize masks for the entire batch
    restrict_pos_mask = torch.zeros((B, N), dtype=torch.float32, device=batch["token_pad_mask"].device)
    allowed_aatype_mask = torch.ones((B, N, K), dtype=torch.float32, device=batch["token_pad_mask"].device)

    if verbose:
        print("\n************** Position-wise amino acid restrictions **************")

    for i, pdb_key in enumerate(batch["example_id"]):
        if pdb_key not in pos_constraint_df.index:
            if verbose:
                print(f"{pdb_key}: No amino acid restrictions specified.")
            continue

        # Get position restrictions from df
        row = pos_constraint_df.loc[pdb_key]
        pos_restrict_aatype = row.get("pos_restrict_aatype", np.nan)

        if pd.isna(pos_restrict_aatype):
            if verbose:
                print(f"{pdb_key}: No position-wise amino acid restrictions specified.")
            continue

        # Set up example
        example = {k: v[i] for k, v in batch.items()}

        if verbose:
            print(f"{pdb_key}: Restricting amino acid sampling at positions {pos_restrict_aatype}")

        # Parse the restriction string into lists of positions and allowed amino acids
        pdb_pos, abs_pos, allowed_aatypes = parse_pos_restrict_aatype_str(pos_restrict_aatype, example["atom_array"])

        # Mark positions with restrictions
        restrict_pos_mask[i, abs_pos] = 1.0

        # Apply restrictions for each position
        for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
            # First, disallow all amino acids at this position
            allowed_aatype_mask[i, pos_idx, :] = 0.0

            # Then allow only the specified amino acids
            for aa in allowed_aa:
                if aa in const.PROT_LETTER_TO_TOKEN:
                    allowed_aatype_mask[i, pos_idx, const.AF3_ENCODING.encode_aa(aa)] = 1.0
                else:
                    print(
                        f"Warning: Unknown amino acid '{aa}' in restriction for {pdb_key} "
                        f"at position {pdb_pos[abs_pos.index(pos_idx)]}"
                    )

        if verbose:
            # Print a summary of the restrictions
            for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
                pos_str = pdb_pos[abs_pos.index(pos_idx)]
                print(f" * Position {pos_str}: Restricted to {allowed_aa}")

    if verbose:
        print("\n********************************************************\n")

    return restrict_pos_mask, allowed_aatype_mask


def parse_fixed_pos_str(fixed_pos_str: str, atom_array: AtomArray) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A", "B1", "C10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A,B1,C10-25").
        atom_array (AtomArray): AtomArray object containing the atom array.

    Returns:
        TensorType["k", int]: List of absolute indices to set to 1 in the masks.
    """
    chain_annotation = "chain_id"  # we use chain_id for fixing positions
    residue_index = atom_array.res_id[get_token_starts(atom_array)]
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices  # no positions specified

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match_with_residues = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        # Match pattern for just a chain ID, e.g., "A"
        match_chain_only = re.match(r"([A-Za-z])$", pos)

        if match_with_residues:
            chain_letter = match_with_residues.group(1)
            start_residue = int(match_with_residues.group(2))
            end_residue = int(match_with_residues.group(3)) if match_with_residues.group(3) else start_residue

            if chain_letter not in atom_array.get_annotation(chain_annotation):
                raise ValueError(
                    f"Chain ID {chain_letter} not found in chain annotation: {np.unique(atom_array.get_annotation(chain_annotation))}."
                )

            # For the given chain, create a mask for all residues in the desired range
            atomwise_range_mask = (
                (atom_array.get_annotation(chain_annotation) == chain_letter)
                & (atom_array.res_id >= start_residue)
                & (atom_array.res_id <= end_residue)
            )
            range_mask = apply_token_wise(atom_array, atomwise_range_mask, np.any)  # get per-token mask
            matching_indices = np.where(range_mask)[0]

            # Check that each residue in the requested range; warn if not found
            found_residues = set(residue_index[matching_indices].tolist())

            for r in range(start_residue, end_residue + 1):
                if r not in found_residues:
                    print(f"Warning: Requested position {chain_letter}{r} not found in structure.")

            # Extend our fixed indices with whatever we did find
            fixed_indices.extend(matching_indices.tolist())
        elif match_chain_only:
            chain_letter = match_chain_only.group(1)

            if chain_letter not in atom_array.get_annotation(chain_annotation):
                raise ValueError(
                    f"Chain ID {chain_letter} not found in chain annotation: {np.unique(atom_array.get_annotation(chain_annotation))}."
                )

            # For the given chain, create a mask for all residues
            atomwise_chain_mask = atom_array.get_annotation(chain_annotation) == chain_letter
            chain_mask = apply_token_wise(atom_array, atomwise_chain_mask, np.any)
            matching_indices = np.where(chain_mask)[0]
            fixed_indices.extend(matching_indices.tolist())
        else:
            raise ValueError(f"Invalid position format: {pos}")

    return fixed_indices


def parse_fixed_pos_override_seq_str(
    override_str: str, atom_array: AtomArray
) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a fixed position sequence override string in the format "A26:A,A27:L" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and override amino acids (e.g., ["A", "L"]).

    Args:
        override_str (str): Comma-separated string of position overrides
                           in the format "<chain+residue>:<desired aatype>"
        atom_array (AtomArray): AtomArray object containing the atom array.

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

        if len(aatype) != 1 or aatype not in const.PROT_LETTER_TO_TOKEN:
            raise ValueError(f"Invalid aatype: {aatype} in {override}. Expected single letter amino acid code.")

        pdb_pos.append(pos)
        override_aatypes.append(aatype)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array)

    return pdb_pos, abs_pos, override_aatypes


def parse_pos_restrict_aatype_str(
    pos_restrict_str: str, atom_array: AtomArray
) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a position restriction string in the format "A26:AVG,A27:VG" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and allowed aatypes (e.g., ["AVG", "VG"]).

    Args:
        pos_restrict_str (str): Comma-separated string of position restrictions
                               in the format "<chain+residue>:<allowed aatypes>"
        atom_array (AtomArray): AtomArray object containing the atom array.

    Returns:
        tuple: (pdb_pos, abs_pos, allowed_aatypes) - lists with corresponding entries
    """
    if not pos_restrict_str or pos_restrict_str.strip() == "":
        return [], [], []

    pdb_pos = []
    allowed_aatypes = []

    # Split by comma and process each restriction.
    restrictions = [r.strip() for r in pos_restrict_str.split(",") if r.strip()]

    for restriction in restrictions:
        # Split by colon to get position and allowed aatypes.
        parts = restriction.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid restriction format: {restriction}. Expected format: 'A26:AVG'")

        pos, aatypes = parts[0].strip(), parts[1].strip()
        pdb_pos.append(pos)
        allowed_aatypes.append(aatypes)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array)

    return pdb_pos, abs_pos, allowed_aatypes


def visualize_conditioning_sequences(
    atom_array: AtomArray,
    cond_mask: TensorType["n", int],
    asym_id: TensorType["n", int],
    asym_names: list[str],
) -> str:
    """
    Visualize the conditioning sequence for a given atom array.
    """
    chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_array)
    sequences = {}

    # Map from chain_name to asym_id.
    chain_names = [x.split("_")[0] for x in asym_names]  # for now, ignore transforms
    chain_name_to_asym_id = {chain_name: i for i, chain_name in enumerate(chain_names)}

    for chain_name, info in chain_info.items():
        sequence = info["processed_entity_canonical_sequence"]
        # Replace with "-" where cond_mask is 0
        chain_cond_mask = cond_mask[asym_id == chain_name_to_asym_id[chain_name]]
        sequence = "".join([x if chain_cond_mask[i] else "-" for i, x in enumerate(sequence)])
        sequences[chain_name] = sequence

    for chain_name, sequence in sequences.items():
        print(f"Chain {chain_name}: {sequence}")


def _validate_ensemble_alignment(batch: dict[str, TensorType["b ..."]]):
    """
    Validate that the alignment of the batch is correct.
    """
    if not (batch["residue_index"] == batch["residue_index"][0]).all().item():
        raise ValueError(
            "Residue index mismatch between decoys. If positions are not aligned, aggregation of potts "
            "parameters will be incorrect and will yield nonsensical results. If this was intentional, "
            "set ensemble_ignore_res_idx_mismatch=True."
        )
    if not (batch["asym_id"] == batch["asym_id"][0]).all().item():
        raise ValueError(
            "Chain ID mismatch between decoys. If positions are not aligned, aggregation of potts "
            "parameters will be incorrect and will yield nonsensical results. If this was intentional, "
            "set ensemble_ignore_res_idx_mismatch=True."
        )
        
def convert_stem(stem: str) -> str:
    # {example_id}{1a28}{1}{['A_1', 'C_1']}_sample0
    m = re.search(r"\{[^}]*\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}_sample(\d+)", stem)
    if not m:
        return stem  # no match

    id1, id2, list_str, idx = m.groups()  # '1a28', '1', "['A_1', 'C_1']", '0'

    # extract elements from the list: 'A_1', 'C_1' → ["A_1", "C_1"]
    items = re.findall(r"'([^']+)'", list_str)

    # remove underscore from each element: "A_1" → "A1"
    items = [x.replace("_", "") for x in items]

    # recombine to the desired format
    return f"{id1}_{id2}_{'_'.join(items)}_sample{idx}"
        
    
def _fix_cif_formal_charge(cif_path: str | Path) -> None:
    """
    Fix pdbx_formal_charge format in CIF files for OpenStructure compatibility.
    
    OpenStructure expects integer values (e.g., "1", "-1") for pdbx_formal_charge,
    but biotite may write signed format (e.g., "+1"). This function converts
    "+N" to "N" in the CIF file.
    
    Args:
        cif_path: Path to the CIF file to fix.
    """
    cif_path = Path(cif_path)
    if not cif_path.exists():
        return
    
    with open(cif_path, 'r') as f:
        content = f.read()
    
    # Replace +N with N in the pdbx_formal_charge field
    # This regex matches space/tab followed by +digit(s) followed by space/newline
    # Only replace positive charges (+1, +2, etc.) since -1, -2 are already valid
    fixed_content = re.sub(r'(\s)\+(\d+)(\s)', r'\1\2\3', content)
    
    if content != fixed_content:
        with open(cif_path, 'w') as f:
            f.write(fixed_content)
    