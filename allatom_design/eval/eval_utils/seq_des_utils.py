"""
Utils for sampling from sequence design models.
"""

import copy
import re
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightning as L
from atomworks.io.utils import non_rcsb
from atomworks.io.utils.io_utils import to_cif_string, to_cif_file
from atomworks.ml.utils.token import apply_token_wise, get_token_starts, spread_token_wise
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise

import atomworks.enums as aw_enums
from biotite.structure import AtomArray, get_residue_starts
from biotite.structure.filter import filter_amino_acids
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
from allatom_design.data.const import AF3_ENCODING

from allatom_design.eval.eval_utils.sd_data_utils import (
    get_sd_batch,
    prepare_designed_sample,
    preprocess_input,
)
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets, annotate_ligand_pockets_pseudocb
from atomworks.ml.transforms.filters import remove_unresolved_tokens

from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.data.transform.pad import pad_dim
from allatom_design.eval.eval_utils.eval_setup_utils import get_training_checkpoints, get_pdb_files, get_seq_des_model
from allatom_design.utils.sample_io_utils import save_cif_file
from allatom_design.utils.atom_array_utils import clean_up_and_renumber_atom_array, insert_unk_residues_for_gaps_in_atom_array
from allatom_design.eval.eval_utils.eval_metrics import calculate_sequence_recovery

###########################################################
# Sequence Design / Sampling Functions
###########################################################

def redesign_with_lcaliby(seed: int = 0,
                        input_sample_is_designed: bool = False,
                        sample_dict: dict = None,
                        pdb_cfg: DictConfig | None = None,
                        seq_des_cfg: DictConfig = None,
                        cif_parse_cfg: DictConfig = None,
                        preprocess_cfg: DictConfig = None,
                        featurizer_cfg: DictConfig = None,
                        cif_save_cfg: DictConfig = None,                          
                        sampling_inputs_df: pd.DataFrame = None,
                        log_dir: Path = None,
                        pos_constraint_df: pd.DataFrame = None,
                        protein_only: bool = False,
                        pocket_only: bool = False,
                        pocket_featurizer_cfg: dict | None = None,
                        pocket_distances_for_seq_recovery: list[float] = None,
                        csv_suffix: str = "") -> list[tuple[dict, Path, dict]]:
    """
    Redesign pocket sequence using lcaliby model for each checkpoint.
    
    Args:
        seed: Random seed.
        sample_dict: Dictionary of sample information.
        pdb_cfg: Optional pdb file discovery config; forwarded to downstream sampling utilities.
        seq_des_cfg: Sequence design configuration.
        cif_parse_cfg: CIF parser configuration for loading samples.
        preprocess_cfg: Preprocessing configuration.
        featurizer_cfg: Featurizer configuration.
        cif_save_cfg: CIF save configuration.
        sampling_inputs_df: Sampling inputs DataFrame.
        log_dir: Log directory.
        pos_constraint_df: Positional constraints DataFrame.
        protein_only: If True, condition only on protein atoms (exclude ligands from atom_cond_mask).
        pocket_only: If True, use pocket-only featurizer to crop to ligand pocket.
        pocket_featurizer_cfg: Config dict for the pocket-only featurizer.
    Returns:
        List of tuples: (sample_dict, output_dir, ckpt_info) for each checkpoint.
    """
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sd_ckpts, pattern = get_training_checkpoints(
        seq_des_cfg.denoiser_train_dir, "seq_denoiser",
        seq_des_cfg.ckpt_cfg.eval_every_n_ckpts, 
        seq_des_cfg.ckpt_cfg.start_step, 
        seq_des_cfg.ckpt_cfg.end_step,
        seq_des_cfg.ckpt_cfg.use_ema,
        seq_des_cfg.ckpt_cfg.eval_last_ckpt
    )
    
    input_sample_paths = [sample_dict[sid]['input_sample_path'] for sid in sample_dict.keys()]
            
    results = []    
    for sd_ckpt in tqdm(sd_ckpts, desc="Redesigning sequence using lcaliby"):
        match = pattern.search(Path(sd_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))
        
        log_dir_per_ckpt = log_dir / f"step_{global_step}_epoch_{epoch}"
        log_dir_per_ckpt.mkdir(parents=True, exist_ok=True)
        
        ckpt_info = {"global_step": global_step, "epoch": epoch, "ckpt_path": sd_ckpt}
        
        L.seed_everything(seed)
        
        seq_des_cfg.atom_mpnn.ckpt_path = sd_ckpt
        seq_des_model = get_seq_des_model(cfg = seq_des_cfg, device=device)
        
        outputs = run_lc_seq_des(
            model=seq_des_model["model"], 
            input_sample_is_designed = input_sample_is_designed,
            cif_parse_cfg=cif_parse_cfg,
            preprocess_cfg=preprocess_cfg,
            featurizer_cfg=featurizer_cfg,
            cif_save_cfg=cif_save_cfg,                                     
            sampling_cfg=seq_des_model["sampling_cfg"],                          
            sampling_inputs_df=sampling_inputs_df,
            pdb_paths=input_sample_paths,             
            device=device,             
            out_dir=str(log_dir_per_ckpt),
            pos_constraint_df=pos_constraint_df,
            protein_only=protein_only,
            pocket_only=pocket_only,
            pocket_featurizer_cfg=pocket_featurizer_cfg,
            pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
        )
                
        sample_dict_per_ckpt = copy.deepcopy(sample_dict)
        
        # Save sequence recovery metrics into .csv file (one row per sample)
        seq_recovery_metrics_list = []
        for example_id, output in outputs.items():
            sample_ids = output["designed_sample_id"]
            seq_recovery_metrics = output["seq_recovery_metrics"]
            
            for i in range(len(sample_ids)):
                seq_recovery_metrics_list.append({
                    "example_id": example_id,
                    "designed_sample_id": sample_ids[i],
                    **seq_recovery_metrics[i]
                })
                
                metrics_to_print = ", ".join([f"{k}: {v:.3f}" for k, v in seq_recovery_metrics[i].items()])
                print(f"sample {i} of {example_id}: {metrics_to_print}")
                
        seq_recovery_metrics_df = pd.DataFrame(seq_recovery_metrics_list)
        seq_recovery_metrics_df.to_csv(Path(log_dir_per_ckpt, f"seq_recovery_metrics{csv_suffix}.csv"), index=False)
        
        # Store outputs in sample_dict_per_ckpt
        for example_id, output in outputs.items():  
            sample_dict_per_ckpt[example_id]['designed_sample_id'] = output["designed_sample_id"]
            sample_dict_per_ckpt[example_id]['designed_sample_atom_array'] = output["designed_sample_atom_array"]
            sample_dict_per_ckpt[example_id]['designed_sample_seq'] = output["designed_sample_seq"]
            sample_dict_per_ckpt[example_id]['designed_sample_path'] = output["designed_sample_path"]
            sample_dict_per_ckpt[example_id]['designed_sample_path_for_af3_tc'] = output["designed_sample_path_for_af3_tc"]                                            
                        
        # Extract pdb_chain_info from outputs for af3 prediction
        for example_id in sample_dict_per_ckpt.keys():
            pdb_chain_info = defaultdict(list)
            designed_sample_atom_array = sample_dict_per_ckpt[example_id]["designed_sample_atom_array"][0]
            desgiend_sample_prot_atom_array = designed_sample_atom_array[designed_sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
            desgiend_sample_ligand_atom_array = designed_sample_atom_array[np.isin(designed_sample_atom_array.chain_type, list(aw_enums.ChainTypeInfo.NON_POLYMERS))]
            protein_pn_unit_iids = [str(pn_unit_iid) for pn_unit_iid in np.unique(desgiend_sample_prot_atom_array.pn_unit_iid)]
            ligand_pn_unit_iids = [str(pn_unit_iid) for pn_unit_iid in np.unique(desgiend_sample_ligand_atom_array.pn_unit_iid)]
            ligand_ccd_codes = [str(desgiend_sample_ligand_atom_array[desgiend_sample_ligand_atom_array.pn_unit_iid == pn_unit_iid].res_name[0]) for pn_unit_iid in ligand_pn_unit_iids]
            ligand_pn_unit_iids_ccd_codes = list(zip(ligand_pn_unit_iids, ligand_ccd_codes))

            for pn_unit_iid in protein_pn_unit_iids:
                pdb_chain_info["protein_pn_unit_iids"].append(str(pn_unit_iid))

            for pn_unit_iid, ccd_code in ligand_pn_unit_iids_ccd_codes:
                pdb_chain_info["ligand_pn_unit_iids"].append(str(pn_unit_iid))
                pdb_chain_info["ligand_ccd_codes"].append(str(ccd_code))            

            sample_dict_per_ckpt[example_id]["pdb_chain_info"] = pdb_chain_info  
                    
        results.append((sample_dict_per_ckpt, log_dir_per_ckpt, ckpt_info))
    
    return results

def run_lc_seq_des(
    *,
    model: SeqDenoiser = None,
    input_sample_is_designed: bool = False,
    cif_parse_cfg: DictConfig = None,
    preprocess_cfg: DictConfig = None,
    featurizer_cfg: DictConfig = None,
    cif_save_cfg: DictConfig = None,
    sampling_cfg: DictConfig = None,
    sampling_inputs_df: pd.DataFrame = None,
    pdb_paths: list[str] = None,
    pdb_cfg: DictConfig | None = None,
    device: str = None,
    out_dir: str = None,
    pos_constraint_df: pd.DataFrame | None = None,
    protein_only: bool = False,
    pocket_only: bool = False,
    pocket_featurizer_cfg: dict | None = None,
    pocket_distances_for_seq_recovery: list[float] = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """
    Given a list of processed structure files, run sequence design on them.

    If out_dir is not None, PDBs with sampled sequences will be saved to the provided directory. In this case, run_aux
    will be a dictionary with the following keys:
        - "out_pdb": list of output PDB paths
        - "pred_seqs": list of predicted sequences as a string for each sample
        
    Args:
        ...
        protein_only: If True, condition only on protein atoms (exclude ligands from atom_cond_mask).
        pocket_only: If True, use pocket-only featurizer to crop to ligand pocket.
        pocket_featurizer_cfg: Config dict for the pocket-only featurizer.
    """
    # Set up outputs.
    outputs = {}
    
    # directory for output PDBs
    sample_out_dir = f"{out_dir}/samples"
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    sample_out_dir_for_af3_tc = f"{out_dir}/samples_for_af3_tc"
    Path(sample_out_dir_for_af3_tc).mkdir(parents=True, exist_ok=True)
    if pocket_only:
        sample_out_dir_pocket_only_full = f"{out_dir}/samples_pocket_only_full"
        Path(sample_out_dir_pocket_only_full).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df.
    if pos_constraint_df is not None:
        valid_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # Columns in input df must be a subset of valid columns.
            print(f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. Found: {pos_constraint_df.columns}")            
            cols_to_keep = [c for c in valid_columns if c in pos_constraint_df.columns]
            print(f"Keeping columns: {cols_to_keep}")
            pos_constraint_df = pos_constraint_df[cols_to_keep]            
            
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
    
    with parallel_context as parallel_pool:
        for bi in range(0, len(pdb_paths), sampling_cfg.batch_size):
            batch_pdb_paths = pdb_paths[bi : bi + sampling_cfg.batch_size]
            B = len(batch_pdb_paths)                            
                                     
            batch = get_sd_batch(pdb_paths = batch_pdb_paths, 
                                 sample_is_designed = input_sample_is_designed,
                                 cif_parse_cfg = cif_parse_cfg,
                                 preprocess_cfg = preprocess_cfg,
                                 featurizer_cfg = featurizer_cfg, 
                                 device=device, 
                                 parallel_pool=parallel_pool, 
                                 sampling_inputs_df=sampling_inputs_df,
                                 pocket_only=pocket_only,
                                 pocket_featurizer_cfg=pocket_featurizer_cfg)                                                
                                                                                                   
            # Initialize seq_cond and atom_cond masks.
            batch = initialize_sampling_masks(batch, protein_only=protein_only)                        
                                        
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
                                                            
            for sample_idx, (example_id, atom_arrays) in enumerate(id_to_atom_arrays.items()):
                if example_id not in outputs:
                    outputs[example_id] = defaultdict(list)
                aux = id_to_aux[example_id]
                                                                                
                for ai, designed_atom_array in enumerate(atom_arrays):                            
                    designed_sample_id = f"{example_id}_sample{ai}"
                    outputs[example_id]["designed_sample_id"].append(designed_sample_id)
                    outputs[example_id]["U"].append(aux[ai]["U"])
                    
                    # Save atom_array and sequence                                        
                    chain_info = non_rcsb.initialize_chain_info_from_atom_array(designed_atom_array)
                    outputs[example_id]["designed_sample_seq"].append(
                        ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                    )
                    
                    # Clean up designed atom array for saving
                    designed_atom_array = clean_up_and_renumber_atom_array(designed_atom_array)
                    
                    # Save samp_atom_array_no_sidechain to outputs
                    outputs[example_id]["designed_sample_atom_array"].append(designed_atom_array)
                    
                    # atom_array with gaps for af3 template conditioning
                    designed_atom_array_with_gaps = designed_atom_array.copy()
                                                                                                                                                                
                    # Insert UNK atoms for gaps in protein backbone atom array
                    designed_atom_array_with_gaps = insert_unk_residues_for_gaps_in_atom_array(designed_atom_array_with_gaps)
                                                                            
                    # Save designed atom array to cif file
                    out_file = f"{sample_out_dir}/{designed_sample_id}.cif"                        
                    save_cif_file(designed_atom_array, out_file, cif_save_cfg=cif_save_cfg)
                    outputs[example_id]["designed_sample_path"].append(out_file)      
                    
                    # Save designed atom array with gaps for af3 template conditioning                                                               
                    out_file_for_af3_tc = f"{sample_out_dir_for_af3_tc}/{designed_sample_id}.cif"
                    save_cif_file(designed_atom_array_with_gaps, out_file_for_af3_tc, cif_save_cfg=cif_save_cfg)                    
                    outputs[example_id]["designed_sample_path_for_af3_tc"].append(out_file_for_af3_tc)  
                                                                                
                    input_atom_array = batch["atom_array"][example_id_to_batch_idx[example_id]]                    
                    
                    if pocket_only:
                        source_atom_array = batch['crop_info'][example_id_to_batch_idx[example_id]]['atom_array']
                        source_atom_array = clean_up_and_renumber_atom_array(source_atom_array)
                                                
                        # designed from residue unit mapping
                        des_res_starts = get_residue_starts(designed_atom_array)
                        designed_res_name_map = dict(zip(
                            zip(designed_atom_array.chain_id[des_res_starts],
                                designed_atom_array.res_id[des_res_starts]),
                            designed_atom_array.res_name[des_res_starts]
                        ))

                        # replace source res_name with designed res_name
                        new_res_names = source_atom_array.res_name.copy()
                        for (chain_id, res_id), res_name in designed_res_name_map.items():
                            mask = (source_atom_array.chain_id == chain_id) & (source_atom_array.res_id == res_id)
                            new_res_names[mask] = res_name
                        source_atom_array.res_name = new_res_names
                        source_atom_array.atom_id = np.arange(1, len(source_atom_array) + 1)
                        
                        out_file_pocket_only_full = f"{sample_out_dir_pocket_only_full}/{designed_sample_id}.cif"
                        save_cif_file(source_atom_array, out_file_pocket_only_full, cif_save_cfg=cif_save_cfg)                        
                                                                        
                    
                    # Calculate sequence recovery metrics
                    seq_recovery_metrics = calculate_sequence_recovery(input_atom_array, designed_atom_array,
                                                                       pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery)                    
                    outputs[example_id]["seq_recovery_metrics"].append(seq_recovery_metrics)                                                                                                                                                                                                                                                                        
            pbar.update(B)
    pbar.close()
    
        
    # Convert tensors to CPU values
    for example_id, example_outputs in outputs.items():
        for k, v in example_outputs.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                example_outputs[k] = [t.detach().cpu().item() for t in v]
            elif isinstance(v, torch.Tensor):
                example_outputs[k] = v.detach().cpu().item()                    

          
    # Save sample_metadata.pt for later use 
    sample_metadata = {}
    for example_id, example_outputs in outputs.items():
        for idx in range(len(example_outputs["designed_sample_id"])):
            designed_sample_id = example_outputs["designed_sample_id"][idx]
            sample_metadata[designed_sample_id] = {
                "example_id": example_id,
                "designed_sample_id": designed_sample_id,
                "designed_sample_path": example_outputs["designed_sample_path"][idx],
                "designed_sample_seq": example_outputs["designed_sample_seq"][idx],
                "U": example_outputs["U"][idx],
            }
    torch.save(sample_metadata, f"{sample_out_dir}/sample_metadata.pt")
    print(f"Saved sample_metadata.pt with {len(sample_metadata)} samples to {sample_out_dir}")

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


###########################################################
# Ligand Utilities
###########################################################

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



###########################################################
# Scoring Functions
###########################################################

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


###########################################################
# Sampling Mask Initialization
###########################################################

def initialize_sampling_masks(batch: dict[str, TensorType["b ..."]], protein_only: bool = False) -> dict[str, torch.Tensor]:
    """
    Initialize the sampling masks for the batch. Modifies batch in place and returns it.
    """        
    # Initialize sequence mask: always condition on non-protein or non-standard residues.
    seq_cond_mask = torch.zeros_like(batch["token_pad_mask"])
    standard_aa_prot_token_mask = batch["token_is_protein_chain"] * (~batch["is_atomized"]) * batch["token_resolved_mask"] * batch["token_pad_mask"]
    
    seq_cond_mask = torch.where(standard_aa_prot_token_mask.bool(),
                                    seq_cond_mask,
                                    batch["token_resolved_mask"])
    
    batch["seq_cond_mask"] = seq_cond_mask * batch["token_pad_mask"] * batch["token_resolved_mask"]

    # Initialize atom mask: condition on backbone atoms of standard amino acids in protein chains or all atoms in non-standard residues and non-protein chains            
    standard_aa_prot_atom_mask = batch["atom_is_protein_chain"] * (1 - batch["atom_is_atomized"]) * batch["atom_resolved_mask"] * batch["atom_pad_mask"]    
    standard_aa_prot_bb_atom_mask = standard_aa_prot_atom_mask * batch["prot_bb_atom_mask"]
    
    batch["atom_cond_mask"] = torch.where(standard_aa_prot_atom_mask.bool(),
                                          standard_aa_prot_bb_atom_mask,
                                          batch["atom_resolved_mask"])
    
    if protein_only:
        batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_is_protein_chain"]
        batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_is_protein_chain"]
    
    # Ensure that all atoms in atom_cond_mask are resolved and atom_cond_mask is masked out the padding atoms
    batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_pad_mask"] * batch["atom_resolved_mask"]
                                          
    return batch
    

###########################################################
# Position Constraint Parsing
###########################################################

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


###########################################################
# Misc Utilities
###########################################################

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
        
# def convert_stem(stem: str) -> str:
#     # {example_id}{1a28}{1}{['A_1', 'C_1']}_sample0
#     m = re.search(r"\{[^}]*\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}_sample(\d+)", stem)
#     if not m:
#         return stem  # no match

#     id1, id2, list_str, idx = m.groups()  # '1a28', '1', "['A_1', 'C_1']", '0'

#     # extract elements from the list: 'A_1', 'C_1' → ["A_1", "C_1"]
#     items = re.findall(r"'([^']+)'", list_str)

#     # remove underscore from each element: "A_1" → "A1"
#     items = [x.replace("_", "") for x in items]

#     # recombine to the desired format
#     return f"{id1}_{id2}_{'_'.join(items)}_sample{idx}"
        
    
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


def create_pos_constraint_dict_from_atom_array(
    pdb_key: str = None,
    atom_array: AtomArray = None) -> dict:
    """
    Create a pos_constraint_dict from an atom array.
    Assume atom_array is already filtered to only include residues to be constrained.
    """
    pos_constraint_dict = {}
    res_starts = get_residue_starts(atom_array)
    res_ids = atom_array.res_id[res_starts]
    chain_ids = atom_array.chain_id[res_starts]
    fixed_pos_str = _indices_to_pos_string(chain_ids, res_ids)
    
    pos_constraint_dict['pdb_key'] = pdb_key
    pos_constraint_dict['fixed_pos_seq'] = fixed_pos_str
    pos_constraint_dict['fixed_pos_scn'] = np.nan
    
    return pos_constraint_dict

###########################################################
# Redesign Functions
###########################################################

def _replace_pocket_sequence_with_native(native_atom_array: AtomArray = None,
                                        sample_atom_array: AtomArray = None,
                                        pocket_distance: float = 5.0,
                                        protein_pn_unit_iids: list[str] = None,
                                        ligand_pn_unit_iids: list[str] = None) -> tuple[AtomArray, int]:
    """
    Replace pocket sequence in sample with native pocket sequence.

    Args:
        native_atom_array: Native atom array.
        sample_atom_array: Sample atom array.
        pocket_distance: Pocket distance.
        protein_pn_unit_iids: Protein pn_unit_iids.
        ligand_pn_unit_iids: Ligand pn_unit_iids.
    Returns:
        tuple: (modified sample_atom_array, number of replaced pocket residues)
    """
    native_atom_array = annotate_ligand_pockets(atom_array=native_atom_array,
                                                pocket_distance=pocket_distance,
                                                receptor_pn_unit_iids=protein_pn_unit_iids,
                                                ligand_pn_unit_iids=ligand_pn_unit_iids)
    
    # Spread residue-wise: if any atom in a residue is in pocket, mark all atoms in that residue as pocket
    residue_wise_pocket_mask = apply_and_spread_residue_wise(native_atom_array, native_atom_array.get_annotation("is_ligand_pocket"), function=np.any)
    
    # Get native backbone atom array in pocket
    prot_bb_mask = (native_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L) & (native_atom_array.is_backbone_atom)    
    pocket_bb_atom_array = native_atom_array[prot_bb_mask & residue_wise_pocket_mask]
            
    # Get pocket residue starts
    pocket_res_starts = get_residue_starts(pocket_bb_atom_array)
    
    # Get pocket chain ids, residue ids, and residue names
    pocket_chain_ids = pocket_bb_atom_array.chain_id[pocket_res_starts]
    pocket_res_ids = pocket_bb_atom_array.res_id[pocket_res_starts]
    pocket_res_names = pocket_bb_atom_array.res_name[pocket_res_starts]
    pocket_residue_to_name = {
        (chain, res): name
        for chain, res, name in zip(pocket_chain_ids, pocket_res_ids, pocket_res_names)
    }
    
    # Replace pocket sequence with native pocket sequence
    replaced_residue_keys = set()
    for i in range(len(sample_atom_array)):
        key = (sample_atom_array.chain_id[i], sample_atom_array.res_id[i])
        if key in pocket_residue_to_name:
            sample_atom_array.res_name[i] = pocket_residue_to_name[key]
            replaced_residue_keys.add(key)
    
    num_replaced_pocket_residues = len(replaced_residue_keys)
    return sample_atom_array, num_replaced_pocket_residues

def load_samples_for_native_redesign(sample_dict: dict,
                                     cfg: DictConfig) -> dict:
    """
    Load designed sample atom arrays and native atom arrays for native sequence redesign.
    Populates sample_dict with keys required by redesign_with_native:
        - sample_atom_array: designed sample atom array
        - native_atom_array: native atom array
        - pdb_chain_info: dict with protein_pn_unit_iids, ligand_pn_unit_iids, ligand_ccd_codes
    
    Args:
        sample_dict: Dictionary of sample information (from prepare_sample_dict).
        cfg: Configuration object.
        
    Returns:
        sample_dict: Updated dictionary with loaded atom arrays and chain info.
    """
    native_cif_dir = Path(cfg.redesign_cfg.native_cif_dir)
    
    # Determine parse/preprocess configs for designed samples
    if not cfg.input_sample_is_designed:
        designed_cif_parse_cfg = cfg.cif_cfg.parse.native
        designed_preprocess_cfg = cfg.preprocess_cfg.native
    else:
        designed_cif_parse_cfg = cfg.cif_cfg.parse.designed_samples
        designed_preprocess_cfg = cfg.preprocess_cfg.designed_samples
    
    sample_ids = list(sample_dict.keys())
    for sample_id in tqdm(sample_ids, desc="Loading samples for native redesign"):
        # 1. Load designed sample atom array
        designed_example = prepare_designed_sample(
            pdb_path=sample_dict[sample_id]['input_sample_path'],            
            preprocess_cfg=designed_preprocess_cfg,
            featurizer_cfg=cfg.featurizer_cfg.prepare_designed_samples,
        )
        sample_atom_array = designed_example["atom_array"]
        sample_dict[sample_id]["sample_atom_array"] = sample_atom_array
        
        # 2. Extract pdb_id from sample_id (e.g., "1bzc" from "1bzc_sample0")
        pdb_id = sample_id.rsplit("_sample", 1)[0]
        sample_dict[sample_id]["pdb_id"] = pdb_id
        
        # 3. Load native atom array
        native_cif_path = native_cif_dir / f"{pdb_id}.cif"
        if not native_cif_path.exists():
            print(f"Warning: Native CIF not found for {pdb_id} at {native_cif_path}, skipping.")
            continue
            
        native_example = preprocess_input(
            pdb_path=str(native_cif_path),
            cif_parse_cfg=cfg.cif_cfg.parse.native,
            preprocess_cfg=cfg.preprocess_cfg.native,
            sample_is_designed=False
        )
        sample_dict[sample_id]["native_atom_array"] = native_example["atom_array"]
        
        # 4. Extract pdb_chain_info from designed sample atom array
        pdb_chain_info = {
            "protein_pn_unit_iids": [],
            "ligand_pn_unit_iids": [],
            "ligand_ccd_codes": []
        }

        prot_atom_array = sample_atom_array[sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
        ligand_atom_array = sample_atom_array[np.isin(sample_atom_array.chain_type, list(aw_enums.ChainTypeInfo.NON_POLYMERS))]

        protein_pn_unit_iids = [str(pn_unit_iid) for pn_unit_iid in np.unique(prot_atom_array.pn_unit_iid)]
        ligand_pn_unit_iids = [str(pn_unit_iid) for pn_unit_iid in np.unique(ligand_atom_array.pn_unit_iid)]
        ligand_ccd_codes = [str(ligand_atom_array[ligand_atom_array.pn_unit_iid == pn_unit_iid].res_name[0])
                           for pn_unit_iid in ligand_pn_unit_iids]

        for pn_unit_iid in protein_pn_unit_iids:
            pdb_chain_info["protein_pn_unit_iids"].append(str(pn_unit_iid))
        for pn_unit_iid, ccd_code in zip(ligand_pn_unit_iids, ligand_ccd_codes):
            pdb_chain_info["ligand_pn_unit_iids"].append(str(pn_unit_iid))
            pdb_chain_info["ligand_ccd_codes"].append(str(ccd_code))
        
        sample_dict[sample_id]["pdb_chain_info"] = pdb_chain_info
    
    # Remove samples that failed to load native atom arrays
    failed_ids = [sid for sid in sample_dict if "native_atom_array" not in sample_dict[sid]]
    for sid in failed_ids:
        print(f"Removing {sid} from sample_dict (failed to load native atom array)")
        del sample_dict[sid]
    
    return sample_dict


def redesign_with_native(sample_dict: dict, 
                         cfg: DictConfig, 
                         out_dir: Path) -> dict:
    """
    Replace pocket sequence with native pocket sequence.
    
    Args:
        sample_dict: Dictionary of sample information.
            Required keys per sample: native_atom_array, pdb_chain_info, sample_atom_array
        cfg: Configuration object.
        out_dir: Output directory for redesigned samples.
        
    Returns:
        sample_dict: Updated dictionary with redesigned sequences.
            - designed_sample_id: list of designed sample IDs (consistent with lcaliby output)
            - designed_sample_atom_array: list of designed atom arrays
            - designed_sample_path: list of paths to saved CIF files
            - designed_sample_path_for_af3_tc: list of paths for AF3 template-conditioned (same as designed_sample_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pocket_distance = cfg.redesign_cfg.pocket_distance
    
    sample_ids = list(sample_dict.keys())
    for sample_id in tqdm(sample_ids, desc="Replacing pocket sequence with native"):
        native_atom_array = sample_dict[sample_id]["native_atom_array"]
        protein_pn_unit_iids = sample_dict[sample_id]["pdb_chain_info"]["protein_pn_unit_iids"]
        ligand_pn_unit_iids = sample_dict[sample_id]["pdb_chain_info"]["ligand_pn_unit_iids"]
        sample_atom_array = sample_dict[sample_id]["sample_atom_array"]

        redesigned_sample_atom_array, num_redesigned_pocket_residues = _replace_pocket_sequence_with_native(
            native_atom_array=native_atom_array,
            sample_atom_array=sample_atom_array,
            pocket_distance=pocket_distance,
            protein_pn_unit_iids=protein_pn_unit_iids,
            ligand_pn_unit_iids=ligand_pn_unit_iids
        )
        
        # Create designed_sample_id (consistent with lcaliby output format)
        designed_sample_id = f"{sample_id}_rwn_{pocket_distance}"
        cif_path = Path(out_dir, f"{designed_sample_id}.cif")
        to_cif_file(redesigned_sample_atom_array, str(cif_path))
        
        # Store as lists (consistent with lcaliby output structure)
        sample_dict[sample_id]["designed_sample_id"] = [designed_sample_id]
        sample_dict[sample_id]["designed_sample_atom_array"] = [redesigned_sample_atom_array]
        sample_dict[sample_id]["designed_sample_path"] = [str(cif_path)]
        sample_dict[sample_id]["designed_sample_path_for_af3_tc"] = [str(cif_path)]
        sample_dict[sample_id]["num_redesigned_pocket_residues"] = num_redesigned_pocket_residues   
        
        print(f"Redesigned {num_redesigned_pocket_residues} pocket residues for {sample_id} with native pocket sequence, within {pocket_distance} Å of the ligand")
    
    # Save redesign summary CSV
    redesign_metrics_list = []
    for sample_id in sample_ids:
        redesign_metrics_list.append({
            "sample_id": sample_id,
            "pdb_id": sample_dict[sample_id].get("pdb_id", ""),
            "designed_sample_id": sample_dict[sample_id]["designed_sample_id"][0],
            "num_redesigned_pocket_residues": sample_dict[sample_id]["num_redesigned_pocket_residues"],
            "pocket_distance": pocket_distance,
            "designed_sample_path": sample_dict[sample_id]["designed_sample_path"][0],
        })
    redesign_metrics_df = pd.DataFrame(redesign_metrics_list)
    csv_path = Path(out_dir, "redesign_with_native_summary.csv")
    redesign_metrics_df.to_csv(csv_path, index=False)
    print(f"Saved redesign summary to {csv_path}")
    
    return sample_dict


######################################
# Utils for making pos_constraint_df
######################################

def create_pos_constraint_dict_from_pocket(
    pdb_key: str,
    atom_array: AtomArray,
    pocket_distance: float = 5.0,
    constraint_type: str = "pocket",  # "pocket" or "scaffold"
    receptor_pn_unit_iids: list[str] = None,
    ligand_pn_unit_iids: list[str] = None,
    use_pseudocb_for_pocket_annotation: bool = False,
    sample_path: str = None,
    return_ligand_mpnn_format: bool = False,
) -> dict:
    """
    Create a pos_constraint_dict from an atom array based on ligand pocket annotation.

    Args:
        pdb_key: Identifier for the PDB entry
        atom_array: AtomArray containing protein and ligand atoms
        pocket_distance: Distance threshold for pocket identification (Angstroms)
        constraint_type: "pocket" to constrain pocket residues, "scaffold" to constrain non-pocket residues
        receptor_pn_unit_iids: List of receptor (protein) pn_unit_iids
        ligand_pn_unit_iids: List of ligand pn_unit_iids
        sample_path: Path to the CIF file (required if return_ligand_mpnn_format=True)
        return_ligand_mpnn_format: If True, also include LigandMPNN CSV fields (pdb_path, chains, fixed_residues)

    Returns:
        Dictionary with pdb_key, fixed_pos_seq, fixed_pos_scn, and metadata.
        If return_ligand_mpnn_format=True, also includes pdb_path, chains, fixed_residues for LigandMPNN.
    """
    # Annotate ligand pockets
    if use_pseudocb_for_pocket_annotation:
        annotated_atom_array = annotate_ligand_pockets_pseudocb(
            atom_array=atom_array,
            pocket_distance=pocket_distance,
            n_min_ligand_atoms=1,
            annotation_name="is_ligand_pocket"
        )
    else:
        annotated_atom_array = annotate_ligand_pockets(
            atom_array=atom_array,
            pocket_distance=pocket_distance,
            receptor_pn_unit_iids=receptor_pn_unit_iids,
            ligand_pn_unit_iids=ligand_pn_unit_iids,
            annotation_name="is_ligand_pocket"
        )
    
    # Spread residue-wise: if any atom in a residue is in pocket, mark all atoms in that residue as pocket
    residue_wise_pocket_mask = apply_and_spread_residue_wise(annotated_atom_array, annotated_atom_array.get_annotation("is_ligand_pocket"), function=np.any)
    protein_mask = annotated_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L

    if constraint_type == "pocket":
        constrained_mask = protein_mask & residue_wise_pocket_mask
    elif constraint_type == "scaffold":
        constrained_mask = protein_mask & ~residue_wise_pocket_mask        
    else: 
        raise ValueError(f"Invalid constraint type: {constraint_type}")

    # Get constrained atom array     
    constrained_atom_array = annotated_atom_array[constrained_mask]
    
    # Early return if no constrained residues
    if len(constrained_atom_array) == 0:
        return {
            'pdb_key': pdb_key,
            'fixed_pos_seq': "",
            'fixed_pos_scn': np.nan,
            'pocket_distance': pocket_distance,
            'constraint_type': constraint_type,
            'num_constrained_residues': 0,
        }, {}
    
    # Get residue starts
    res_starts = get_residue_starts(constrained_atom_array)
    chain_ids = constrained_atom_array.chain_id[res_starts]
    res_ids = constrained_atom_array.res_id[res_starts]
    
    result = {
        'pdb_key': pdb_key,
        'fixed_pos_seq': _indices_to_pos_string(chain_ids, res_ids),
        'fixed_pos_scn': np.nan,
        'pocket_distance': pocket_distance,
        'constraint_type': constraint_type,
        'num_constrained_residues': len(res_starts),
    }
    
    # Add LigandMPNN format fields if requested
    results_for_ligand_mpnn = {}
    if return_ligand_mpnn_format:                               
        results_for_ligand_mpnn['pdb_path'] = sample_path if sample_path else ""
        fixed_residues_list = [f"{cid}{rid}" for cid, rid in zip(chain_ids, res_ids)]
        results_for_ligand_mpnn['fixed_residues'] = " ".join(fixed_residues_list)
        # Get chains to parse (protein + ligand)
        protein_chain_ids = list({pn_unit_iid.split("_")[0] for pn_unit_iid in receptor_pn_unit_iids})
        ligand_chain_ids = list({pn_unit_iid.split("_")[0] for pn_unit_iid in ligand_pn_unit_iids})
        results_for_ligand_mpnn['chains'] = ",".join(protein_chain_ids + ligand_chain_ids)                    
                                    
    return result, results_for_ligand_mpnn

def _make_single_pos_constraint_dict(
    sample_path: str=None,
    sampling_inputs_df: pd.DataFrame=None,
    cif_parse_cfg: DictConfig=None,
    preprocess_cfg: DictConfig=None,
    sample_is_designed: bool=False,
    pocket_distance: float=6.0,
    constraint_type: str="pocket",
    
    save_ligand_mpnn_csv: bool=True,
    use_pseudocb_for_pocket_annotation: bool=False,
) -> dict:
    """Process a single CIF file and return constraint info.

    Returns:
        Dict with keys: "status" ("ok", "no_ligand", "error"),
        "pdb_key", "pos_constraint_dict", "ligand_mpnn_dict", "error_msg"
    """
    pdb_key = Path(sample_path).stem
    try:
        example = preprocess_input(
            pdb_path=str(sample_path),
            cif_parse_cfg=cif_parse_cfg,
            preprocess_cfg=preprocess_cfg,
            sample_is_designed=sample_is_designed,
        )
        atom_array = example["atom_array"]
        atom_array = remove_unresolved_tokens(atom_array)
        
        # Take only the query PN units
        if sampling_inputs_df is not None:
            sampling_input = sampling_inputs_df[sampling_inputs_df["pdb_id"] == pdb_key]
            query_pn_unit_iids = ast.literal_eval(sampling_input["query_pn_unit_iids"].iloc[0])
            atom_array = atom_array[np.isin(atom_array.pn_unit_iid, query_pn_unit_iids)]

        protein_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
        non_protein_mask = ~protein_mask
        receptor_pn_unit_iids = list(np.unique(atom_array.pn_unit_iid[protein_mask]))
        ligand_pn_unit_iids = list(np.unique(atom_array.pn_unit_iid[non_protein_mask]))

        if len(ligand_pn_unit_iids) == 0:
            return {"status": "no_ligand", "pdb_key": pdb_key}

        pos_constraint_dict, ligand_mpnn_dict = create_pos_constraint_dict_from_pocket(
            pdb_key=pdb_key,
            atom_array=atom_array,
            pocket_distance=pocket_distance,
            constraint_type=constraint_type,
            receptor_pn_unit_iids=receptor_pn_unit_iids,
            ligand_pn_unit_iids=ligand_pn_unit_iids,
            use_pseudocb_for_pocket_annotation=use_pseudocb_for_pocket_annotation,
            sample_path=sample_path,
            return_ligand_mpnn_format=save_ligand_mpnn_csv,
        )
        return {
            "status": "ok",
            "pdb_key": pdb_key,
            "pos_constraint_dict": pos_constraint_dict,
            "ligand_mpnn_dict": ligand_mpnn_dict,
        }
    except Exception as e:
        return {"status": "error", "pdb_key": pdb_key, "error_msg": str(e)}

def make_pos_constraint_df(
    pdb_cfg: DictConfig = None,
    sampling_inputs_df: pd.DataFrame = None,
    output_path: str = None,
    pocket_distance: float = 5.0,
    constraint_type: str = "pocket",  # "pocket" or "scaffold"
    cif_parse_cfg: DictConfig = None,
    preprocess_cfg: DictConfig = None,
    sample_is_designed: bool = False,    
    debug: bool = False,
    num_debug_samples: int = 5,
    save_ligand_mpnn_csv: bool = True,
    use_pseudocb_for_pocket_annotation: bool = False,
    num_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a positional constraint DataFrame for multiple CIF files.

    Args:
        cif_dir: Directory containing CIF files
        pdb_list_file: Text file with list of CIF filenames (one per line). If None, use all CIFs in cif_dir.
        output_path: Path to save the output parquet file
        pocket_distance: Distance threshold for pocket identification
        constraint_type: "pocket" or "scaffold"
        data_cfg: Configuration for CIF parsing
        transform_cfg: Configuration for preprocessing and featurization
        pdb_chain_info_dict: Dictionary mapping pdb_id to pdb_chain_info
        debug: If True, only process num_debug_samples samples
        num_debug_samples: Number of samples to process in debug mode
        save_ligand_mpnn_csv: If True, also save LigandMPNN input CSV

    Returns:
        Tuple of (positional constraint DataFrame, LigandMPNN input DataFrame)
    """
    # Get list of CIF files to process    
    sample_paths = get_pdb_files(**pdb_cfg)
    pdb_ids = [Path(sample_path).stem for sample_path in sample_paths]
    if sampling_inputs_df is not None:
        pdb_id_set = set(sampling_inputs_df["pdb_id"].values)
        valid_indices = [i for i, pdb_id in enumerate(pdb_ids) if pdb_id in pdb_id_set]
        sample_paths = [sample_paths[i] for i in valid_indices]
        pdb_ids = [pdb_ids[i] for i in valid_indices]

    # Debug mode: limit number of samples
    if debug:
        sample_paths = sample_paths[:num_debug_samples]
        pdb_ids = pdb_ids[:num_debug_samples]
        print(f"[DEBUG MODE] Processing only {len(sample_paths)} samples")

    print(f"Found {len(sample_paths)} samples to process")

    rows = []
    failed_pdbs = []
    results_for_ligand_mpnn = []

    if num_workers > 1:
        print(f"Using {num_workers} workers for parallel processing")
        results = Parallel(n_jobs=num_workers, backend="loky")(
            delayed(_make_single_pos_constraint_dict)(
                sample_path=sample_path,
                sampling_inputs_df=sampling_inputs_df,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                sample_is_designed=sample_is_designed,
                pocket_distance=pocket_distance,
                constraint_type=constraint_type,
                save_ligand_mpnn_csv=save_ligand_mpnn_csv,
                use_pseudocb_for_pocket_annotation=use_pseudocb_for_pocket_annotation,
            )
            for sample_path in tqdm(sample_paths, desc=f"Dispatching samples ({constraint_type})")
        )
        for result in results:
            if result["status"] == "ok":
                rows.append(result["pos_constraint_dict"])
                if result["ligand_mpnn_dict"]:
                    results_for_ligand_mpnn.append(result["ligand_mpnn_dict"])
            elif result["status"] == "no_ligand":
                print(f"Warning: No ligand found in {result['pdb_key']}, skipping...")
                failed_pdbs.append(result["pdb_key"])
            else:
                print(f"Error processing {result['pdb_key']}: {result.get('error_msg', 'unknown')}")
                failed_pdbs.append(result["pdb_key"])
    else:
        for sample_path in tqdm(sample_paths, desc=f"Processing samples ({constraint_type})"):
            result = _make_single_pos_constraint_dict(
                sample_path=sample_path,
                sampling_inputs_df=sampling_inputs_df,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                sample_is_designed=sample_is_designed,
                pocket_distance=pocket_distance,
                constraint_type=constraint_type,
                save_ligand_mpnn_csv=save_ligand_mpnn_csv,
                use_pseudocb_for_pocket_annotation=use_pseudocb_for_pocket_annotation,
            )
            if result["status"] == "ok":
                rows.append(result["pos_constraint_dict"])
                if result["ligand_mpnn_dict"]:
                    results_for_ligand_mpnn.append(result["ligand_mpnn_dict"])
            elif result["status"] == "no_ligand":
                print(f"Warning: No ligand found in {result['pdb_key']}, skipping...")
                failed_pdbs.append(result["pdb_key"])
            else:
                print(f"Error processing {result['pdb_key']}: {result.get('error_msg', 'unknown')}")
                failed_pdbs.append(result["pdb_key"])
    
    # Create DataFrame
    df = pd.DataFrame(rows)
            
    # Create LigandMPNN DataFrame
    ligand_mpnn_input_df = pd.DataFrame(results_for_ligand_mpnn) if results_for_ligand_mpnn else pd.DataFrame()
    
    print(f"\nSuccessfully processed {len(df)} CIF files")
    print(f"Failed: {len(failed_pdbs)} CIF files")
    
    if failed_pdbs:
        print(f"Failed PDBs: {failed_pdbs[:10]}{'...' if len(failed_pdbs) > 10 else ''}")
    
    # Save to file if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Drop metadata columns before saving (for minimal version)
        cols_to_drop = ["pocket_distance", "constraint_type", "num_constrained_residues"]
        df_to_save = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        if output_path.suffix == ".parquet":
            df_to_save.to_parquet(output_path)
        elif output_path.suffix == ".csv":
            df_to_save.to_csv(output_path)
        else:
            # Default to csv
            df_to_save.to_csv(output_path)
        
        print(f"Saved positional constraint DataFrame to {output_path}")
        
        # Also save full version with metadata
        full_output_path = output_path.parent / (output_path.stem + "_full" + output_path.suffix)
        if full_output_path.suffix == ".parquet":
            df.to_parquet(full_output_path)
        elif full_output_path.suffix == ".csv":
            df.to_csv(full_output_path)
        else:
            df.to_parquet(full_output_path)
        
        print(f"Saved full positional constraint DataFrame to {full_output_path}")
        
        # Save LigandMPNN input CSV
        if save_ligand_mpnn_csv and len(ligand_mpnn_input_df) > 0:
            ligand_mpnn_csv_path = output_path.parent / (output_path.stem + "_for_ligandmpnn.csv")
            # Select only required columns for LigandMPNN: pdb_path, chains, fixed_residues
            ligand_mpnn_df_to_save = ligand_mpnn_input_df[['pdb_path', 'chains', 'fixed_residues']]
            ligand_mpnn_df_to_save.to_csv(ligand_mpnn_csv_path, index=False)
            print(f"Saved LigandMPNN input CSV to {ligand_mpnn_csv_path}")
    
    return df, ligand_mpnn_input_df
