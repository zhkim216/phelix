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
from atomworks.io.utils.io_utils import to_cif_string, to_cif_file, load_any
from atomworks.ml.utils.token import apply_token_wise, get_token_starts, spread_token_wise
import atomworks.enums as aw_enums
from biotite.structure import AtomArray, AtomArrayStack, get_residue_starts
from biotite.structure.filter import filter_amino_acids
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
from allatom_design.data.const import AF3_ENCODING
from allatom_design.checkpoint_utils import get_cfg_from_ckpt
from allatom_design.data.data import to
from allatom_design.data.datasets.atomworks_sd_dataset import sd_collator
from allatom_design.data.transform.preprocess import preprocess_transform, preprocess_transform_for_designs_from_other_methods
from allatom_design.data.transform.sd_featurizer import (sd_featurizer, 
                                                         sd_featurizer_with_load_any, 
                                                         sd_featurizer_for_af3_prediction, 
                                                         sd_featurizer_for_designs_from_other_methods)
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.data.transform.pad import pad_dim


def compute_sequence_recovery(
    sampled_atom_array: AtomArray,
    orig_res_types: torch.Tensor,
    seq_mask: torch.Tensor,
    pocket_mask: torch.Tensor = None,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Compute sequence recovery metrics for a sampled atom array.
    
    Args:
        sampled_atom_array: AtomArray with sampled sequence
        orig_res_types: Original residue types as tensor (token-level, argmax of one-hot)
        seq_mask: Mask for valid sequence positions (1 = compute recovery, 0 = ignore)
        pocket_mask: Optional mask for pocket positions (1 = pocket, 0 = non-pocket)
        device: Device for tensor operations
        
    Returns:
        dict with keys:
            - seq_recovery: Overall sequence recovery
            - pocket_seq_recovery: Pocket sequence recovery (if pocket_mask provided)
    """
    # Get sampled residue types from atom array
    token_starts = get_token_starts(sampled_atom_array)
    samp_res_names = sampled_atom_array.res_name[token_starts]
    samp_res_types = AF3_ENCODING.encode(samp_res_names)
    samp_res_types = torch.tensor(samp_res_types, device=device)
    samp_res_types = pad_dim(samp_res_types, 0, len(orig_res_types) - len(samp_res_types))
    
    results = {}
    
    # Compute overall sequence recovery
    seq_recovery = (samp_res_types == orig_res_types).float() * seq_mask
    seq_recovery = seq_recovery.sum() / seq_mask.sum().clamp(min=1e-8)
    results["seq_recovery"] = seq_recovery
    
    # Compute pocket sequence recovery (optional)
    if pocket_mask is not None:
        pocket_seq_recovery = (samp_res_types == orig_res_types).float() * pocket_mask
        pocket_seq_recovery = pocket_seq_recovery.sum() / pocket_mask.sum().clamp(min=1e-8)
        results["pocket_seq_recovery"] = pocket_seq_recovery
    
    return results


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


def extract_ligand_from_structure(
    atom_array: AtomArray, 
    ligand_pn_unit_iids: str | list[str] = None
) -> AtomArray:
    """
    Extract ligand atoms from an atom array based on pn_unit_iid(s).
    
    Some ligands may consist of canonical amino acids, so filtering by
    pn_unit_iid is more reliable than filtering by residue type.
    
    Args:
        atom_array: AtomArray containing protein and ligand atoms
        ligand_pn_unit_iids: pn_unit_iid(s) of the ligand(s) to extract.
                             Can be a single pn_unit_iid string (e.g., "B_1") or
                             a list of pn_unit_iids (e.g., ["B_1", "C_1"]).
                             If None, falls back to extracting non-amino acid atoms.
        
    Returns:
        AtomArray containing only ligand atoms from specified pn_unit_iid(s)
    """
    if ligand_pn_unit_iids is None:
        # Fallback: extract non-amino acid atoms
        protein_mask = filter_amino_acids(atom_array)
        ligand_mask = ~protein_mask
    else:
        # Extract by pn_unit_iid(s)
        if isinstance(ligand_pn_unit_iids, str):
            ligand_pn_unit_iids = [ligand_pn_unit_iids]
        
        # Use pn_unit_iid annotation directly
        ligand_mask = np.isin(atom_array.pn_unit_iid, ligand_pn_unit_iids)
    
    if not ligand_mask.any():
        return None
    
    return atom_array[ligand_mask]


def run_lc_seq_des(
    *,
    model: SeqDenoiser = None,
    data_cfg: DictConfig = None,
    transform_cfg: DictConfig = None,    
    sampling_cfg: DictConfig = None,
    metadata: pd.DataFrame = None,
    pdb_paths: list[str] = None,
    device: str = None,
    out_dir: str = None,
    pos_constraint_df: pd.DataFrame | None = None,
    protein_only: bool = False,
    fix_pocket_seq: bool = False,
    pocket_distance: float = 8.0,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """
    Given a list of processed structure files, run sequence design on them.

    If out_dir is not None, PDBs with sampled sequences will be saved to the provided directory. In this case, run_aux
    will be a dictionary with the following keys:
        - "out_pdb": list of output PDB paths
        - "pred_seqs": list of predicted sequences as a string for each sample
        
    Args:
        ...
        ligand_chain_ids: Chain ID(s) of the ligand(s) to extract when protein_only=True.
                          Can be a single chain ID string (e.g., "B") or a list of chain IDs.
                          If None, falls back to extracting non-amino acid atoms.
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
                                    
            batch = get_sd_batch(batch_pdb_paths,  data_cfg=data_cfg, transform_cfg=transform_cfg, 
                                 device=device, parallel_pool=parallel_pool, metadata=metadata,
                                 protein_only=protein_only)
                                 
                                 

            # Initialize seq_cond and atom_cond masks.
            batch = initialize_sampling_masks(batch)
            
            #! Revert, directly use AnnotateLigandPockets transform instead of initialize_pocket_mask
            # Initialize ligand pocket mask if ligand conditioning is enabled.
            # if sampling_cfg.ligand_conditioning:
            #     batch = initialize_pocket_mask(batch, ligand_pocket_dist_cutoff=sampling_cfg.ligand_pocket_dist_cutoff,
            #                                           small_molecule_only=sampling_cfg.small_molecule_only)

            # If fix_pocket_seq is enabled, create pos_constraint_df from ligand pocket
            if fix_pocket_seq:
                pos_constraint_df = None
                pos_constraint_df = create_pos_constraint_from_ligand_pocket(batch)
                print(f"Created pos_constraint_df from ligand pocket: {pos_constraint_df}")
                                        
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
            
            # Get designed_cif_save_args from data_cfg_for_design
            cif_save_args = OmegaConf.to_container(data_cfg.cif_save_args, resolve=True) if data_cfg and data_cfg.get("cif_save_args") else {}
            
            # If protein_only, load original structures from cached .pt files to extract ligands                        
            original_ligands = {}
            if protein_only:
                for pdb_path in batch_pdb_paths:
                    try:
                        # Load from cached .pt file
                        cached_example = torch.load(str(pdb_path), map_location="cpu", weights_only=False)
                        orig_array = cached_example.get("atom_array")                        
                        
                        # Get pdb_id and find ligand pn_unit_iid from metadata
                        pdb_id = Path(pdb_path).stem.split("_")[0]
                        
                        # Determine ligand pn_unit_iid from metadata
                        current_ligand_pn_unit_iids = None  
                
                        metadata_row = metadata[metadata["pdb_id"] == pdb_id]
                        assert metadata_row is not None, f"Metadata row is None for {pdb_id}"
                            
                        # Find non-protein (ligand) pn_unit_iid
                        # q_pn_unit_is_protein_1/2 indicates if that unit is protein
                        ligand_pn_unit_iid_list = []
                        for suffix_num in ["1", "2"]:
                            is_protein_col = f"q_pn_unit_is_protein_{suffix_num}"
                            pn_unit_iid_col = f"q_pn_unit_iid_{suffix_num}"
                            if is_protein_col in metadata_row.columns and pn_unit_iid_col in metadata_row.columns:
                                is_protein = metadata_row[is_protein_col].iloc[0]
                                if not is_protein:  # This is ligand
                                    ligand_pn_unit_iid = metadata_row[pn_unit_iid_col].iloc[0]                                    
                                    assert ligand_pn_unit_iid is not None, f"Ligand pn_unit_iid is None for {pdb_id}"                                        
                                    ligand_pn_unit_iid_list.append(ligand_pn_unit_iid)
                            if ligand_pn_unit_iid_list:
                                # Use pn_unit_iid directly (e.g., "B_1") - no suffix removal needed
                                current_ligand_pn_unit_iids = ligand_pn_unit_iid_list
                                print(f"Detected ligand pn_unit_iid(s) for {pdb_id}: {current_ligand_pn_unit_iids}")
                        
                        ligand_array = extract_ligand_from_structure(orig_array, ligand_pn_unit_iids=current_ligand_pn_unit_iids)
                        original_ligands[pdb_id] = ligand_array
                    except Exception as e:
                        print(f"Warning: Failed to extract ligand from {pdb_path}: {e}")
                        
            for si, (example_id, atom_arrays) in enumerate(id_to_atom_arrays.items()):
                aux = id_to_aux[example_id]
                
                # Add _protein_only suffix if protein_only mode (after convert_stem to avoid regex issues)
                suffix = "_protein_only" if protein_only else ""
                sample_stems = [convert_stem(f"{example_id}_sample{si}") + suffix for si in range(len(atom_arrays))]
                
                samp_bb_ligand_atom_arrays = []                                
                sample_seq_recovery_sum = 0.0
                sample_sp_seq_recovery_sum = 0.0                
                # Save outputs, calculate sequence recovery metrics for each sample.
                for ai, sample_stem in enumerate(sample_stems):
                    outputs["example_id"].append(example_id)
                    outputs["U"].append(aux[ai]["U"])
                    
                    samp_atom_array = atom_arrays[ai]
                    
                    # Save atom_array and sequence
                    outputs["atom_array"].append(samp_atom_array)                                        
                    chain_info = non_rcsb.initialize_chain_info_from_atom_array(samp_atom_array)
                    outputs["seq"].append(
                        ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                    )
                    
                    # Process atom arrays for further task (e.g., sequence recovery metrics, save cif files for af3 template conditioning)
                    samp_prot_atom_array = samp_atom_array[samp_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
                    samp_prot_bb_atom_array = samp_prot_atom_array[samp_prot_atom_array.is_backbone_atom]    
                    
                    if not protein_only:
                        samp_ligand_atom_array = samp_atom_array[samp_atom_array.chain_type != aw_enums.ChainType.POLYPEPTIDE_L]
                    else:
                        # For protein_only mode, only save with_ligand file (skip protein-only file)
                        pdb_id = sample_stem.split("_")[0]
                        samp_ligand_atom_array = original_ligands.get(pdb_id)                                          
                    
                    if samp_ligand_atom_array is None:
                        print(f"Warning: Ligand array is None for {pdb_id}")
                        continue
                    
                    # Combine protein backbone and ligand atoms
                    samp_prot_bb_atom_array = samp_prot_atom_array[samp_prot_atom_array.is_backbone_atom]                                                
                    samp_atom_array_no_sidechain = samp_prot_bb_atom_array + samp_ligand_atom_array #! biotite use + operator to concatenate atom arrays                                       
                    
                    # Renumber atom_id sequentially (1-indexed)
                    samp_atom_array_no_sidechain.atom_id = np.arange(1, len(samp_atom_array_no_sidechain) + 1)
                    
                    # Save samp_atom_array_no_sidechain to outputs
                    outputs["bb_ligand_atom_array"].append(samp_atom_array_no_sidechain)
                    
                    # Append samp_atom_array_no_sidechain to bb_ligand_atom_arrays to calculate sequence recovery metrics
                    samp_bb_ligand_atom_arrays.append(samp_atom_array_no_sidechain)
                    
                    # atom_array with gaps for af3 template conditioning
                    samp_prot_bb_atom_array_with_gaps = samp_prot_bb_atom_array.copy()
                                                                                                                                                                
                    # Insert UNK atoms for gaps in protein backbone atom array
                    samp_atom_array_no_sidechain_with_gaps = insert_unk_residues_for_gaps_in_atom_array(samp_prot_bb_atom_array_with_gaps)
                    samp_atom_array_no_sidechain_with_gaps = samp_atom_array_no_sidechain_with_gaps + samp_ligand_atom_array
                    
                    # Renumber atom_id sequentially (1-indexed)
                    samp_atom_array_no_sidechain_with_gaps.atom_id = np.arange(1, len(samp_atom_array_no_sidechain_with_gaps) + 1)
                    
                    if not protein_only:
                        out_file = f"{sample_out_dir}/{sample_stem}.cif"                        
                        out_file = to_cif_file(samp_atom_array_no_sidechain, out_file, file_type="cif", fill_gaps_in_poly_records=False, **cif_save_args)                        
                        _fix_cif_formal_charge(out_file)                        
                        outputs["out_pdb"].append(out_file)                        
                        
                        out_file_for_af3_tc = f"{sample_out_dir}/{sample_stem}_for_af3_tc.cif"
                        out_file_for_af3_tc = to_cif_file(samp_atom_array_no_sidechain_with_gaps, out_file_for_af3_tc, file_type="cif", fill_gaps_in_poly_records=False, **cif_save_args)
                        _fix_cif_formal_charge(out_file_for_af3_tc)
                        outputs["out_pdb_for_af3_tc"].append(out_file_for_af3_tc)         
                    else:                                                                                            
                        # Save with _protein_only_with_ligand suffix only                    
                        with_ligand_stem = sample_stem.replace("_protein_only", "_protein_only_with_ligand")
                        out_file_with_ligand = f"{sample_out_dir}/{with_ligand_stem}.cif"                         
                        
                        out_file_with_ligand = to_cif_file(samp_atom_array_no_sidechain,
                                                        out_file_with_ligand,
                                                        file_type="cif",
                                                        fill_gaps_in_poly_records=False,
                                                        **cif_save_args)
                                                                        
                        _fix_cif_formal_charge(out_file_with_ligand)                        
                        outputs["out_pdb"].append(out_file_with_ligand)
                        print(f"Saved protein+ligand sample: {out_file_with_ligand}")                                                    
                        
                        out_file_with_ligand_for_tc = f"{sample_out_dir}/{with_ligand_stem}_for_af3_tc.cif"
                        out_file_with_ligand_for_tc = to_cif_file(samp_atom_array_no_sidechain_with_gaps, out_file_with_ligand_for_tc, 
                                                                  file_type="cif", fill_gaps_in_poly_records=False, **cif_save_args)
                        _fix_cif_formal_charge(out_file_with_ligand_for_tc)
                        outputs["out_pdb_for_af3_tc"].append(out_file_with_ligand_for_tc)                                                      
                                    
                    # Compute sequence recovery metrics                            
                    orig_res_types = batch["restype"][si].argmax(dim=-1)          
                    seq_mask = (1 - batch["seq_cond_mask"][si]) * batch["token_pad_mask"][si] * batch["token_resolved_mask"][si]
                    if all_native:
                        seq_mask = batch["token_resolved_mask"][si]
                        lp_seq_mask = batch["pocket_token_mask"][si] * batch["token_pad_mask"][si]
                    else:
                        if (not protein_only) and (not fix_pocket_seq):
                            lp_seq_mask = (1 - batch["seq_cond_mask"][si]) * batch["token_pad_mask"][si] * batch["pocket_token_mask"][si]
                        elif (not protein_only) and fix_pocket_seq:
                            lp_seq_mask = batch["pocket_token_mask"][si] * batch["token_pad_mask"][si]
                        else:
                            native_atom_array = batch["atom_array"][si]
                            native_prot_atom_array = native_atom_array[native_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
                            native_prot_ca_atom_array = native_prot_atom_array[native_prot_atom_array.atom_name == "CA"]                                                                        
                
                    # Calculate sequence recovery metrics for each sample                                                
                    if not protein_only:
                        recovery_metrics = compute_sequence_recovery(
                            sampled_atom_array=atom_arrays[ai],
                            orig_res_types=orig_res_types,
                            seq_mask=seq_mask,
                            pocket_mask=lp_seq_mask,
                            device=device,
                        )
                        seq_recovery = recovery_metrics["seq_recovery"]
                        sp_seq_recovery = recovery_metrics["pocket_seq_recovery"]
                    else:
                        #! Assume we're not conditioning on any sequences, for now.                                                
                        samp_bb_ligand_atom_array = samp_bb_ligand_atom_arrays[ai]
                        receptor_chain = samp_bb_ligand_atom_array[samp_bb_ligand_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L].chain_id[0]                                                                                                      
                        ligand_chain = samp_bb_ligand_atom_array[samp_bb_ligand_atom_array.chain_type != aw_enums.ChainType.POLYPEPTIDE_L].chain_id[0]
                                                        
                        samp_bb_ligand_atom_array = annotate_ligand_pockets(atom_array = samp_bb_ligand_atom_array, 
                                                                                            pocket_distance = pocket_distance, 
                                                                                            receptor_chain = receptor_chain, 
                                                                                            ligand_chain = ligand_chain)
                            
                        samp_prot_atom_array = samp_bb_ligand_atom_array[samp_bb_ligand_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
                        
                        samp_prot_ca_atom_array = samp_prot_atom_array[samp_prot_atom_array.atom_name == "CA"]
                        valid_mask = (samp_prot_ca_atom_array.is_backbone_atom) & ~(samp_prot_ca_atom_array.res_name == "UNK") & (samp_prot_ca_atom_array.occupancy > 0.0) & (samp_prot_ca_atom_array.hetero == False)
                        valid_lp_mask = (samp_prot_ca_atom_array.is_ligand_pocket) & valid_mask
                        
                        samp_seq = samp_prot_ca_atom_array[valid_mask].res_name
                        _res_ids_designed = samp_prot_ca_atom_array[valid_mask].res_id
                        
                        samp_pocket_seq = samp_prot_ca_atom_array[valid_lp_mask].res_name
                        _res_ids_designed_in_pocket = samp_prot_ca_atom_array[valid_lp_mask].res_id
                        
                        native_seq = native_prot_ca_atom_array[np.isin(native_prot_ca_atom_array.res_id, _res_ids_designed)].res_name                            
                        native_pocket_seq = native_prot_ca_atom_array[np.isin(native_prot_ca_atom_array.res_id, _res_ids_designed_in_pocket)].res_name
                        try:
                            seq_recovery = ((native_seq == samp_seq).sum() / len(samp_seq))
                            sp_seq_recovery = ((native_pocket_seq == samp_pocket_seq).sum() / len(samp_pocket_seq))
                        except Exception as e:
                            print(f"Error calculating sequence recovery: {e}")
                            seq_recovery = 0.0
                            sp_seq_recovery = 0.0
                                                                                    
                    # Save sequence recovery metrics
                    outputs["sample_seq_recovery"].append(seq_recovery)
                    sample_seq_recovery_sum += seq_recovery
                                                                        
                    outputs["sample_sp_seq_recovery"].append(sp_seq_recovery)
                    sample_sp_seq_recovery_sum += sp_seq_recovery                        
                    # Todo: Compute mp and np sequence recovery
                    
                    print (f"sample {ai} of {example_id}: seq recovery: {seq_recovery}, sp seq recovery: {sp_seq_recovery}")                                        
                                        
                sample_avg_seq_recovery = sample_seq_recovery_sum / len(atom_arrays)
                sample_avg_sp_seq_recovery = sample_sp_seq_recovery_sum / len(atom_arrays)
                outputs["sample_avg_seq_recovery"].append(sample_avg_seq_recovery)
                outputs["sample_avg_sp_seq_recovery"].append(sample_avg_sp_seq_recovery)
                                
                total_avg_seq_recovery += sample_avg_seq_recovery
                total_avg_sp_seq_recovery += sample_avg_sp_seq_recovery

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

    # Save sample_metadata.pt for later use (e.g., skip_sampling mode)
    sample_metadata = {}
    for idx in range(len(outputs["out_pdb"])):
        sample_stem = Path(outputs["out_pdb"][idx]).stem
        sample_metadata[sample_stem] = {
            "example_id": outputs["example_id"][idx],
            "out_pdb": outputs["out_pdb"][idx],
            "out_pdb_for_af3_tc": outputs["out_pdb_for_af3_tc"][idx],
            "U": outputs["U"][idx],
            "sample_seq_recovery": outputs["sample_seq_recovery"][idx] if "sample_seq_recovery" in outputs else None,
            "sample_sp_seq_recovery": outputs["sample_sp_seq_recovery"][idx] if "sample_sp_seq_recovery" in outputs else None,
        }
    torch.save(sample_metadata, f"{sample_out_dir}/sample_metadata.pt")
    print(f"Saved sample_metadata.pt with {len(sample_metadata)} samples to {sample_out_dir}")

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
    data_cfg: DictConfig = None,    
    transform_cfg: DictConfig | None = None, 
    device: str = None, 
    parallel_pool: Parallel | None = None, 
    metadata: pd.DataFrame = None,
    protein_only: bool = False,
) -> dict[str, Any]:
    """
    Given a list of pdb file paths, return a batch of sequence design model features.

    If data_cfg is None, use default cif parser args.
    """
    if parallel_pool is None:
        # Load PDBs sequentially.
        batch_examples = [get_sd_example(pdb_path = pdb_path, data_cfg=data_cfg,
                                         transform_cfg = transform_cfg,  
                                         metadata=metadata,
                                         protein_only=protein_only) for pdb_path in pdb_paths]                                        
                                         
                                         
                                         
    else:
        # Load PDBs in parallel.
        batch_examples = parallel_pool(delayed(get_sd_example)(pdb_path = pdb_path, data_cfg=data_cfg, transform_cfg = transform_cfg,                                         
                                                               metadata=metadata,
                                                               protein_only=protein_only) for pdb_path in pdb_paths)

    # Collate examples.
    batch = sd_collator(batch_examples)
    batch = to(batch, device)

    return batch

#!########## Get SD example functions ##########
def get_sd_example(pdb_path: str = None,
                   data_cfg: DictConfig = None,
                   transform_cfg: DictConfig = None, 
                   metadata: pd.DataFrame = None,       
                   load_from_cache: bool = False,  
                   use_load_any: bool = False,         
                   protein_only: bool = False,
                   ) -> dict[str, Any]:
    """
    Given a pdb file path, return a dictionary of sequence design model features.

    If data_cfg is None, use default cif parser args.
    
    Args:
        load_from_cache: If True, load from cached .pt file
        use_load_any: If True, use load_any to load atom_array from cif file (preserves pn_unit_iid)
        load_from_pdb: If both load_from_cache and use_load_any are False, use preprocess_pdb
    """
    if transform_cfg is not None:
        featurizer_cfg = transform_cfg.featurizer_cfg
        preprocess_cfg = transform_cfg.preprocess_cfg
    if data_cfg is not None:
        load_from_cache = data_cfg.load_from_cache        
        use_load_any = data_cfg.use_load_any
    
    # Load the cache example if it exists.
    if load_from_cache:
        # Todo: Right now, only one ligand is supported. Need to modify this to support
        # Todo: i) Single protein chain with multiple ligands at different pocket sites
        # Todo: ii) Single protein chain with multiple ligands at the same pocket site
        example = load_cached_example(pdb_path)
        pdb_id = Path(pdb_path).stem
    elif use_load_any:
        # Load atom_array from cif using load_any (preserves pn_unit_iid annotation)
        example = load_example_with_load_any(pdb_path)
        pdb_id = (Path(pdb_path).stem).split("_")[0]
    else:
        # load_from_pdb: Use preprocess_pdb pipeline
        example = preprocess_pdb(pdb_path, data_cfg = data_cfg, preprocess_transform_cfg = preprocess_cfg)
        pdb_id = (Path(pdb_path).stem).split("_")[0]
    
    metadata_example = metadata[metadata["pdb_id"] == pdb_id].reset_index(drop=True)
    
    #! Replace information in the example with the information from the metadata
    #! Since when caching, it doesn't really care about the metadata as it's just caching the whole structure        
        
    example['example_id'] = metadata_example["example_id"].iloc[0]
    if protein_only:
        print(f"Protein only mode: {pdb_id}")
        protein_chain_cols = ["q_pn_unit_is_protein_1", "q_pn_unit_is_protein_2"]
        suffix = ""
        for col in protein_chain_cols:
            if metadata_example[col].iloc[0]:
                suffix = col.split("_")[-1]
                break

        other_suffixes = ["_1", "_2"]
        other_suffixes.remove(f"_{suffix}")
        new_metadata = {}
        for col in metadata_example.columns:            
            if any(col.endswith(s) for s in other_suffixes):
                continue
                        
            if col.endswith(f"_{suffix}"):
                new_col = col[:-len(f"_{suffix}")]
            else:
                new_col = col
            
            new_metadata[new_col] = metadata_example[col].iloc[0]
        
        example["query_pn_unit_iids"] = new_metadata["q_pn_unit_iid"]
        example['extra_info'] = new_metadata
        
    else:
        query_pn_unit_iids = metadata_example["q_pn_unit_iid_1"].tolist() + metadata_example["q_pn_unit_iid_2"].tolist()
        example["query_pn_unit_iids"] = query_pn_unit_iids
        
        example['extra_info'] = {} #! delete all the information preexisting in the example
        row_dict = metadata_example.iloc[0].to_dict() # To series to ignore the index
        example['extra_info'] = row_dict
    
    # Featurize the example.
    if not use_load_any:                
        featurizer = sd_featurizer(**featurizer_cfg)                                                                
    else:
        featurizer = sd_featurizer_with_load_any()
        
    example = featurizer(example)

    return example    

def get_sd_example_from_af3_prediction(pdb_path: str = None,
                   data_cfg: DictConfig = None,
                   transform_cfg: DictConfig = None, 
                   metadata: pd.DataFrame = None,
                   ) -> dict[str, Any]:
    """
    Given a pdb file path from AF3 prediction, return a dictionary of sequence design model features.     
    Args:
        data_cfg: Configuration for loading the cif file
        transform_cfg: Configuration for transforming the cif file
        metadata: Metadata for the pdb file
    """
    
    preprocess_transform_cfg = transform_cfg.preprocess_cfg
    featurizer_cfg = transform_cfg.featurizer_cfg
            
    # load_from_pdb: Use preprocess_pdb pipeline
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
    featurizer = sd_featurizer_for_af3_prediction(**featurizer_cfg)                                                                            
    example = featurizer(example)

    return example    

def get_sd_example_from_designs_from_other_methods(pdb_path: str = None,
                                                   data_cfg: DictConfig = None,
                                                   transform_cfg: DictConfig = None,
                                                   ) -> dict[str, Any]:
    """
    Given a pdb file path from designed samples from other methods, return a dictionary of sequence design model features.
    """
    preprocess_transform_cfg = transform_cfg.preprocess_cfg
    featurizer_cfg = transform_cfg.featurizer_cfg
    
    # load_from_pdb: Use preprocess_pdb pipeline
    example = preprocess_designs_from_other_methods(pdb_path = pdb_path, 
                                                    data_cfg = data_cfg, 
                                                    preprocess_transform_cfg = preprocess_transform_cfg)
    
    featurizer = sd_featurizer_for_designs_from_other_methods(**featurizer_cfg)
    example = featurizer(example)
    
    return example
    
    #! Replace information in the example with the information from the metadata

#!########################################################################################################
#!###### Preprocessing functions #######

def preprocess_pdb(pdb_path: str | None, data_cfg: DictConfig | None, 
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
    
def preprocess_designs_from_other_methods(pdb_path: str | None,
                                          data_cfg: DictConfig | None,
                                          preprocess_transform_cfg: DictConfig | None) -> dict[str, Any]:
    """
    Preprocess a PDB file using the preprocessing pipeline for designed samples from non-atomworks frameworks.
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
    pipeline = preprocess_transform_for_designs_from_other_methods(**dict(preprocess_transform_cfg))
    return pipeline(
        data={
            "example_id": Path(pdb_path).stem,
            "atom_array": atom_array_from_cif,
            "chain_info": input_data["chain_info"],
        }
    )

#!########################################################################################################

def load_cached_example(pdb_path: str) -> dict[str, torch.Tensor]:
    cached_example_path = f"{pdb_path}"
    return torch.load(cached_example_path, map_location="cpu", weights_only=False)

def load_example_with_load_any(pdb_path: str) -> dict[str, Any]:
    """
    Load atom_array from cif file using load_any.
    Designed samples are already preprocessed, so we just need to fix annotation types.
    """
    from biotite.structure import AtomArrayStack
    
    # Load with all extra_fields
    atom_array = load_any(pdb_path, extra_fields="all")
    # load_any may return AtomArrayStack, extract first array if so
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    
    # Fix annotation types (CIF stores everything as strings)
    atom_array = _fix_cif_annotation_types(atom_array)
    
    # Create example dict with atom_array
    example = {"atom_array": atom_array}
    return example

def _fix_cif_annotation_types(atom_array) -> "AtomArray":
    """
    Fix annotation types for atom_array loaded from CIF.
    CIF format stores all values as strings, so we need to convert back to proper types.
    Uses del_annotation + set_annotation pattern from atomworks examples.
    """
    # Boolean annotations
    bool_annotations = ['atomize', 'is_polymer', 'is_aromatic', 'is_covalent_modification', 
                        'is_backbone_atom', 'hetero', 'is_leaving_atom', 'is_n_terminal_atom', 'is_c_terminal_atom']
    for ann in bool_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ('U', 'S', 'O'):  # String types
                new_val = (val == "True")
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)
    
    # Integer annotations
    int_annotations = ['chain_type', 'atomic_number', 'within_chain_res_idx', 'within_poly_res_idx', 
                       'chain_entity', 'molecule_entity', 'pn_unit_entity', 'token_id', 'transformation_id',
                       'pdbx_PDB_model_num', 'label_entity_id', 'label_seq_id', 'auth_seq_id', 'molecule_id',
                       'molecule_iid', 'charge', 'pdbx_formal_charge']
    for ann in int_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ('U', 'S', 'O'):  # String types
                # Handle '?' or empty values
                new_val = np.array([int(v) if str(v).lstrip('-').isdigit() else 0 for v in val])
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)
    
    # Float annotations
    float_annotations = ['B_iso_or_equiv', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'b_factor']
    for ann in float_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ('U', 'S', 'O'):  # String types
                new_val = np.array([float(v) if v not in ('?', '.', '') else np.nan for v in val])
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)
    
    return atom_array

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

def insert_unk_residues_for_gaps_in_atom_array(atom_array: AtomArray) -> AtomArray:
    """
    Insert UNK CA atoms at gap positions (where res_id is not consecutive).
    Designed for protein backbone atom array
    
    Args:
        atom_array: Atom array
        
    Returns:
        AtomArray with UNK CA atoms inserted at gap positions
    """
    
    annotations = atom_array.get_annotation_categories()
    annot_categories_to_include = ["res_id", "res_name", "atom_name", "alt_atom_id",\
        "atom_id", "element", "hetero", "occupancy", "b_factor", "stereo",\
        "is_aromatic", "is_backbone_atom", "is_polymer", "charge", "atomic_number", \
        "atomize", "is_covalent_modification", "uses_alt_atom_id", "ins_code", "chain_id",\
        "pn_unit_id", "molecule_id", "chain_entity", "pn_unit_entity", "molecule_entity", \
        "transformation_id", "chain_iid", "pn_unit_iid", "molecule_iid", "chain_type"
    ]
    
    annot_categories_to_copy = ["chain_id", "pn_unit_id", "molecule_id", \
                                "chain_entity", "pn_unit_entity", "molecule_entity", \
                                "transformation_id", "chain_iid", "pn_unit_iid", "molecule_iid", \
                                "chain_type"]
    
    annot_categories_to_not_copy = [x for x in annot_categories_to_include if x not in annot_categories_to_copy]
    
    # Delete annotations that are not in annotations_to_include
    for annot in annotations:
        if annot not in annot_categories_to_include:
            atom_array.del_annotation(annot)
            
    # Get unique residues (first atom of each residue)
    res_starts = get_residue_starts(atom_array)
    res_ids = atom_array.res_id[res_starts]
    chain_ids = atom_array.chain_id[res_starts]
        
    # Find gaps: positions where res_id difference > 1 AND same chain
    res_id_diff = np.diff(res_ids)
    same_chain = chain_ids[:-1] == chain_ids[1:]
    gap_indices = np.where((res_id_diff > 1) & same_chain)[0]
    
    if len(gap_indices) == 0:
        print(f"No gaps found in the atom array")
        return atom_array  # No gaps, return as is
    
    # Collect UNK atoms to insert
    unk_atoms_list = []
    
    for gap_idx in gap_indices:
        start_res_id = res_ids[gap_idx]
        end_res_id = res_ids[gap_idx + 1]
        
        # Get template atom for annotations (use atom at gap_idx)
        template_atom_idx = res_starts[gap_idx]
        
        # Create UNK atoms for each missing res_id
        for missing_res_id in range(start_res_id + 1, end_res_id):
            # Create single atom array for UNK CA
            unk_atom = AtomArray(1)
            
            # Set coordinates to [0, 0, 0]
            unk_atom.coord[0] = [0.0, 0.0, 0.0]
                                                                            
            # Copy annotations from template atom
            for annot in annot_categories_to_include:                
                if annot == "res_id":
                    unk_atom.set_annotation(annot, np.array([missing_res_id]))
                elif annot == "res_name":
                    unk_atom.set_annotation(annot, np.array(["UNK"]))
                elif annot in ("atom_name", "alt_atom_id"):
                    unk_atom.set_annotation(annot, np.array(["CA"]))
                elif annot == "atom_id":
                    unk_atom.set_annotation(annot, np.array([0]))  # Will be renumbered later
                elif annot == "element":
                    unk_atom.set_annotation(annot, np.array(["C"]))
                elif annot == "hetero":
                    unk_atom.set_annotation(annot, np.array([False]))
                elif annot == "occupancy":
                    unk_atom.set_annotation(annot, np.array([0.0]))
                elif annot == "b_factor":
                    unk_atom.set_annotation(annot, np.array([0.0]))
                elif annot == "stereo":
                    unk_atom.set_annotation(annot, np.array(["S"]))                
                elif annot == "is_aromatic":
                    unk_atom.set_annotation(annot, np.array([False]))
                elif annot == "is_backbone_atom":
                    unk_atom.set_annotation(annot, np.array([True]))
                elif annot == "is_polymer":
                    unk_atom.set_annotation(annot, np.array([True]))
                elif annot == "charge":
                    unk_atom.set_annotation(annot, np.array([0]))
                elif annot == "atomic_number":
                    unk_atom.set_annotation(annot, np.array([6]))
                elif annot == "atomize":
                    unk_atom.set_annotation(annot, np.array([True]))
                elif annot == "is_covalent_modification":
                    unk_atom.set_annotation(annot, np.array([False]))
                elif annot == "uses_alt_atom_id":
                    unk_atom.set_annotation(annot, np.array([False]))
                elif annot == "ins_code":
                    unk_atom.set_annotation(annot, np.array([""]))  # Empty insertion code
                elif annot in annot_categories_to_copy:
                    template_val = getattr(atom_array, annot)[template_atom_idx]
                    unk_atom.set_annotation(annot, np.array([template_val]))                          
            
            unk_atoms_list.append(unk_atom)
            
    # Concatenate all UNK atoms
    if len(unk_atoms_list) == 0:
        print(f"No UNK atoms to insert (gaps detected but no missing residues)")
        return atom_array
    
    all_unk_atoms = unk_atoms_list[0]
    for unk_atom in unk_atoms_list[1:]:
        all_unk_atoms = all_unk_atoms + unk_atom
    
    # Combine with original and sort by res_id
    combined = atom_array + all_unk_atoms
    
    # Sort by chain_id and res_id to maintain proper order
    sort_indices = np.lexsort((combined.res_id, combined.chain_id))
    combined = combined[sort_indices]
    
    # Renumber atom_id sequentially (1-indexed)
    combined.atom_id = np.arange(1, len(combined) + 1)
    
    return combined

def create_pos_constraint_from_ligand_pocket(batch: dict) -> pd.DataFrame | None:
    """
    Create a pos_constraint_df from the ligand pocket information in the batch.
    """
    rows = []
    
    for i, example_id in enumerate(batch["example_id"]):
        atom_array = batch["atom_array"][i]
                
        pocket_token_mask = batch["pocket_token_mask"][i].cpu().numpy().astype(bool)  
        token_pad_mask = batch["token_pad_mask"][i].cpu().numpy().astype(bool)
        
        pocket_indices = np.where(pocket_token_mask & token_pad_mask)[0]
        if len(pocket_indices) == 0:
            continue
        
        # token_starts로 chain_id, res_id 가져오기
        token_starts = get_token_starts(atom_array)
        pocket_chain_ids = atom_array.chain_id[token_starts[pocket_indices]]
        pocket_res_ids = atom_array.res_id[token_starts[pocket_indices]]
        
        # "A1-10,A15-20,B5-8" 형식으로 변환
        fixed_pos_str = _indices_to_pos_string(pocket_chain_ids, pocket_res_ids)
        
        rows.append({
            "pdb_key": example_id,
            "fixed_pos_seq": fixed_pos_str,
            "fixed_pos_scn": np.nan,
        })
    
    if not rows:
        return None
    
    return pd.DataFrame(rows).set_index("pdb_key")


def _indices_to_pos_string(chain_ids: np.ndarray, res_ids: np.ndarray) -> str:
    """Convert (chain_id, res_id) array to "A1-10,B5-8" format"""
    chain_to_res = {}
    for chain_id, res_id in zip(chain_ids, res_ids):
        chain_to_res.setdefault(chain_id, []).append(res_id)
    
    pos_parts = []
    for chain_id in sorted(chain_to_res.keys()):
        res_list = sorted(set(chain_to_res[chain_id]))
        if not res_list:
            continue
        
        # Group consecutive res_id into ranges
        ranges = []
        start = end = res_list[0]
        for r in res_list[1:]:
            if r == end + 1:
                end = r
            else:
                ranges.append((start, end))
                start = end = r
        ranges.append((start, end))
        
        for s, e in ranges:
            pos_parts.append(f"{chain_id}{s}" if s == e else f"{chain_id}{s}-{e}")
    
    return ",".join(pos_parts)