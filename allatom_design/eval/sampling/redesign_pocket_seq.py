from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_cached_example_files, get_pdb_files, get_training_checkpoints, wandb_setup)

from biotite.structure import AtomArray
import biotite.structure as struc
from biotite.structure.residues import get_residue_starts
from biotite.structure.filter import filter_amino_acids
import numpy as np

from omegaconf import OmegaConf, DictConfig
import hydra
import pandas as pd
from pathlib import Path
import torch
import json
import yaml
import wandb
from tqdm import tqdm
import copy

import atomworks.enums as aw_enums
from atomworks.io.utils.sequence import aa_chem_comp_3to1
from atomworks.io.utils.io_utils import to_cif_file

from allatom_design.eval.eval_utils.seq_des_utils import (get_sd_example, 
                                                          load_example_with_load_any,                                                          
                                                          get_sd_example_from_af3_prediction, 
                                                          get_sd_example_from_designs_from_other_methods, 
                                                          run_lc_seq_des,
                                                          get_seq_des_model,
                                                          _indices_to_pos_string)
from allatom_design.eval.eval_utils.folding_utils import (run_af3_single_sequence,
                                                          find_pred_sample_path_af3,
                                                          make_af3_json)
from allatom_design.eval.eval_utils.eval_metrics import _compute_self_consistency_metrics_atomarray, _compute_docking_metrics_atomarray
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets

import lightning as L


###########################################################
# Utility Functions
###########################################################

def create_sample_dict(sample_paths: list[str] = None, 
                       sample_ids: list[str] = None, 
                       pdb_ids: list[str] = None) -> dict:
    """
    Create a dictionary of sample information.
    """
    if sample_ids is None:
        sample_ids = [Path(sample_path).stem for sample_path in sample_paths]
    if pdb_ids is None:
        pdb_ids = [Path(sample_path).stem.split("_")[0] for sample_path in sample_paths]
    
    sample_dict = {}
    for i, sample_id in enumerate(sample_ids):
        sample_dict[sample_id] = {}
        sample_dict[sample_id]['sample_path'] = sample_paths[i]
        sample_dict[sample_id]['pdb_id'] = pdb_ids[i]
    return sample_dict


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


def replace_pocket_sequence_with_native(native_atom_array: AtomArray = None,
                                        sample_atom_array: AtomArray = None,
                                        pocket_distance: float = 5.0,
                                        protein_chains: list[str] = None,
                                        ligand_chains: list[str] = None) -> AtomArray:
    """
    Args:
        native_atom_array: Native atom array.
        sample_atom_array: Sample atom array.
        pocket_distance: Pocket distance.
        protein_chains: Protein chains.
        ligand_chains: Ligand chains.
    Returns:
        AtomArray: Atom array with pocket sequence replaced with native pocket sequence.
    
    """
    native_atom_array = annotate_ligand_pockets(atom_array=native_atom_array, 
                                                pocket_distance=pocket_distance, 
                                                receptor_chains=protein_chains,
                                                ligand_chains=ligand_chains)
    
    # Get native protein atom array and backbone atom array
    native_prot_atom_array = native_atom_array[native_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
    native_bb_atom_array = native_prot_atom_array[native_prot_atom_array.is_backbone_atom]
    pocket_bb_atom_array = native_bb_atom_array[native_bb_atom_array.is_ligand_pocket]
    
    
    
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
    num_replaced_pocket_residues = 0
    for i in range(len(sample_atom_array)):
        key = (sample_atom_array.chain_id[i], sample_atom_array.res_id[i])
        if key in pocket_residue_to_name:
            sample_atom_array.res_name[i] = pocket_residue_to_name[key]
            num_replaced_pocket_residues += 1
            
    return sample_atom_array, num_replaced_pocket_residues

def create_pos_constraint_dict_from_atom_array(
    pdb_key: str = None,
    atom_array: AtomArray = None) -> pd.DataFrame:
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
# Phase 1: Prepare Samples (Common)
###########################################################

def prepare_samples(cfg: DictConfig, metadata: pd.DataFrame) -> tuple[dict, Path]:
    """
    Prepare sample_dict with ligand extraction and designed structure loading.
    
    Args:
        cfg: Configuration object.
        metadata: Metadata DataFrame.
        
    Returns:
        sample_dict: Dictionary containing sample information.
        processed_original_samples_dir: Directory where original samples are saved.
    """
    # Get PDB files
    sample_paths = get_pdb_files(**cfg.pdb_cfg)
    
    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]
        
    # Initialize dictionary for storing sample information
    sample_dict = create_sample_dict(sample_paths=sample_paths)
    
    # Extract native atom array and ligand atom array from cached examples
    sample_dict = _extract_ligand_atom_array_from_cached_examples(
        sample_dict=sample_dict, 
        cached_example_path=cfg.cached_example_path, 
        metadata=metadata
    )
    
    return sample_dict


def _extract_ligand_atom_array_from_cached_examples(sample_dict: dict = None, 
                                                   cached_example_path: str = None, 
                                                   metadata: pd.DataFrame = None) -> dict:
    """    
    Extract native atom array and ligand atom array from cached examples.
    Args:
        sample_dict: Dictionary of sample information. Required keys: 'sample_path', 'pdb_id'.
        cached_example_path: Path to cached examples.
        metadata: Metadata DataFrame.
    Returns:
        Dictionary of sample information with native atom array and ligand atom array.
    """
    # Extract ligand atom array from cached examples
    sample_ids = list(sample_dict.keys())
    
    print("Extracting ligand atoms from cached examples...")
    for sample_id in tqdm(sample_ids, desc="Extracting ligands"):
        pdb_id = sample_dict[sample_id]['pdb_id']
        sample_dict[sample_id]["native_ligand_atom_array"] = None  # Initialize as None        
        cached_example = torch.load(f"{cached_example_path}/{pdb_id}.pt", map_location="cpu", weights_only=False)
        native_atom_array = cached_example.get("atom_array")           
        metadata_row = metadata[metadata["pdb_id"] == pdb_id]
        
        # Determine ligand pn_unit_iid from metadata
        current_ligand_pn_unit_iids = None  
        
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
                
        ligand_array = extract_ligand_from_structure(native_atom_array, ligand_pn_unit_iids=current_ligand_pn_unit_iids)
        sample_dict[sample_id]["native_ligand_atom_array"] = ligand_array
        sample_dict[sample_id]["native_atom_array"] = native_atom_array
    
    return sample_dict


def load_designed_structures(sample_dict: dict,
                             cfg: DictConfig,
                             save_dir: Path) -> dict:
    """
    Load designed structures from caliby codebase and combine with ligand.
    
    Args:
        sample_dict: Dictionary of sample information.
        cfg: Configuration object.
        save_dir: Directory to save the combined structures.
        
    Returns:
        sample_dict: Updated dictionary with designed structures.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = list(sample_dict.keys())
            
    for sample_id in tqdm(sample_ids, desc="Loading Caliby designed structures"):        
        
        sample_dict[sample_id]["sample_atom_array"] = None        
        pdb_chain_info = {
            "protein_chains": [],
            "ligand_chains": [],
            "ligand_ccd_codes": []
        }
                  
        # Load a designed structure and featurize it
        example = get_sd_example_from_designs_from_other_methods(
            pdb_path=sample_dict[sample_id]['sample_path'], 
            data_cfg=cfg.data_cfg_for_designed_samples, 
            transform_cfg=cfg.transform_cfg_for_designed_samples
        )
        
        sample_atom_array = example["atom_array"]
        sample_bb_atom_array = sample_atom_array[sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
        sample_bb_atom_array = sample_bb_atom_array[sample_bb_atom_array.is_backbone_atom]
        ligand_atom_array = sample_dict[sample_id]["native_ligand_atom_array"]
        
        # Combine backbone and ligand
        sample_bb_ligand_atom_array = struc.concatenate([sample_bb_atom_array, ligand_atom_array])
        # Renumber atom_id sequentially (1-indexed)
        sample_bb_ligand_atom_array.atom_id = np.arange(1, len(sample_bb_ligand_atom_array) + 1)            
        sample_dict[sample_id]["sample_atom_array_with_ligand"] = sample_bb_ligand_atom_array                    
        
        # Save the original atom array with ligand
        cif_path = Path(save_dir, f"{sample_id}_with_ligand.cif")
        to_cif_file(sample_bb_ligand_atom_array, str(cif_path))      
        sample_dict[sample_id]["sample_atom_array_with_ligand_path"] = cif_path
            
        protein_chains = [str(chain_id) for chain_id in np.unique(sample_bb_atom_array.chain_id)]    
        ligand_chains = [str(chain_id) for chain_id in np.unique(ligand_atom_array.chain_id)]
        ccd_codes = [str(ligand_atom_array[ligand_atom_array.chain_id == chain].res_name[0]) for chain in ligand_chains]
        ligand_chains_ccd_codes = list(zip(ligand_chains, ccd_codes))
        
        for chain in protein_chains:
            pdb_chain_info["protein_chains"].append(str(chain))
        
        for chain, ccd_code in ligand_chains_ccd_codes:
            pdb_chain_info["ligand_chains"].append(str(chain))
            pdb_chain_info["ligand_ccd_codes"].append(str(ccd_code))            
            
        sample_dict[sample_id]["pdb_chain_info"] = pdb_chain_info
    
    return sample_dict


###########################################################
# Phase 2a: Redesign with Native Sequence
###########################################################

def redesign_with_native(sample_dict: dict, 
                         cfg: DictConfig, 
                         out_dir: Path) -> dict:
    """
    Replace pocket sequence with native pocket sequence.
    
    Args:
        sample_dict: Dictionary of sample information.
        cfg: Configuration object.
        out_dir: Output directory for redesigned samples.
        
    Returns:
        sample_dict: Updated dictionary with redesigned sequences.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pocket_distance = cfg.redesign_cfg.pocket_distance
    
    sample_ids = list(sample_dict.keys())
    for sample_id in tqdm(sample_ids, desc="Replacing pocket sequence with native"):
        native_atom_array = sample_dict[sample_id]["native_atom_array"]
        protein_chains = sample_dict[sample_id]["pdb_chain_info"]["protein_chains"]
        ligand_chains = sample_dict[sample_id]["pdb_chain_info"]["ligand_chains"]
        sample_atom_array_with_ligand = sample_dict[sample_id]["sample_atom_array_with_ligand"]
        
        redesigned_sample_atom_array, num_redesigned_pocket_residues = replace_pocket_sequence_with_native(
            native_atom_array=native_atom_array,
            sample_atom_array=sample_atom_array_with_ligand,
            pocket_distance=pocket_distance,
            protein_chains=protein_chains,
            ligand_chains=ligand_chains
        )
        
        # Update sample_dict
        sample_dict[sample_id]["redesigned_sample_atom_array"] = redesigned_sample_atom_array
        sample_dict[sample_id]["num_redesigned_pocket_residues"] = num_redesigned_pocket_residues   
        print(f"Redesigned {num_redesigned_pocket_residues} pocket residues for {sample_id} with native pocket sequence, within {pocket_distance} Å of the ligand")
        
        # Save the replaced sample atom array
        cif_path = Path(out_dir, f"{sample_id}_rwn_{pocket_distance}.cif")
        to_cif_file(redesigned_sample_atom_array, str(cif_path))
    
    return sample_dict


###########################################################
# Phase 2b: Redesign with Lcaliby
###########################################################

def redesign_with_lcaliby(sample_dict: dict,
                          cfg: DictConfig,
                          metadata: pd.DataFrame,
                          log_dir: Path) -> list[tuple[dict, Path, dict]]:
    """
    Redesign pocket sequence using lcaliby model for each checkpoint.
    
    Args:
        sample_dict: Dictionary of sample information.
        cfg: Configuration object.
        metadata: Metadata DataFrame.
        base_out_dir: Base output directory.
        
    Returns:
        List of tuples: (sample_dict, output_dir, ckpt_info) for each checkpoint.
    """
    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get checkpoints
    sd_ckpts, pattern = get_training_checkpoints(
        cfg.denoiser_train_dir, "seq_denoiser",
        cfg.eval_every_n_ckpts, cfg.start_step, cfg.end_step, cfg.use_ema,
        cfg.get("eval_last_ckpt", True)
    )
    
    # Get sample paths
    sample_paths = [sample_dict[sid]['sample_path'] for sid in sample_dict.keys()]
    sample_with_ligand_paths = [sample_dict[sid]['sample_atom_array_with_ligand_path'] for sid in sample_dict.keys()]
        
    # Create pos_constraint_df for the scaffold part of sample_with_ligand
    rows = []
    for sample_id in sample_dict.keys():
        sample_atom_array_with_ligand = sample_dict[sample_id]["sample_atom_array_with_ligand"]
        sample_atom_array_with_ligand = annotate_ligand_pockets(atom_array=sample_atom_array_with_ligand, 
                                                                pocket_distance=cfg.redesign_cfg.pocket_distance, 
                                                                annotate_scaffold=True,
                                                                annotation_name="is_scaffold")
        
        scaffold_atom_array = sample_atom_array_with_ligand[sample_atom_array_with_ligand.is_scaffold]
        
        pos_constraint_dict = create_pos_constraint_dict_from_atom_array(
            pdb_key=sample_id,
            atom_array=scaffold_atom_array
        )
        
        rows.append(pos_constraint_dict)
    
    scaffold_pos_constraint_df = pd.DataFrame(rows).set_index("pdb_key")                     
    
    # Run lcaliby sequence design for each checkpoint
    results = []    
    for sd_ckpt in tqdm(sd_ckpts, desc="Redesigning pocket sequence using lcaliby"):
        match = pattern.search(Path(sd_ckpt).name)
        global_step, epoch = int(match.group(1)), int(match.group(2))
        
        # Create checkpoint-specific output directory
        log_dir_per_ckpt = log_dir / f"step_{global_step}_epoch_{epoch}"
        log_dir_per_ckpt.mkdir(parents=True, exist_ok=True)
        
        ckpt_info = {"global_step": global_step, "epoch": epoch, "ckpt_path": sd_ckpt}
        
        # Reset seed per checkpoint
        L.seed_everything(cfg.seed)
        
        # Load sequence design model
        cfg.seq_des_cfg.atom_mpnn.ckpt_path = sd_ckpt
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
        
        # Run lcaliby sequence design
        outputs = run_lc_seq_des(
            model=seq_des_model["model"], 
            data_cfg=cfg.data_cfg_for_design,
            transform_cfg=cfg.transform_cfg_for_design,                                     
            sampling_cfg=seq_des_model["sampling_cfg"],                          
            metadata=metadata,
            pdb_paths=sample_with_ligand_paths, 
            device=device,             
            out_dir=str(log_dir_per_ckpt),
            protein_only=cfg.get("protein_only", False),
            fix_pocket_seq=cfg.get("fix_pocket_seq", False),
            pocket_distance=cfg.pocket_distance,
            redesign_pocket_seq=True,
            pos_constraint_df=scaffold_pos_constraint_df,
        )
        
        # Deep copy and update sample_dict with lcaliby outputs
        # Note: run_lc_seq_des saves CIF files, we need to update sample_dict accordingly
        ckpt_sample_dict = copy.deepcopy(sample_dict)
        
        # Update sample_atom_array from outputs if available
        if "bb_ligand_atom_array" in outputs:
            for idx, example_id in enumerate(outputs["example_id"]):
                if example_id in ckpt_sample_dict:
                    ckpt_sample_dict[example_id]["sample_atom_array"] = outputs["bb_ligand_atom_array"][idx]
        
        results.append((ckpt_sample_dict, log_dir_per_ckpt, ckpt_info))
    
    return results


###########################################################
# Phase 3: AF3 Evaluation
###########################################################

def evaluate_af3_consistency(sample_id_list: list[str] = None,
                             pdb_id_list: list[str] = None,
                             sample_atom_array_list: list[AtomArray] = None,
                             pdb_chain_info: list[dict] = None,                             
                             num_redesigned_pocket_residue_list: list[int] = None,
                             out_dir: Path = None,
                             cfg: DictConfig = None,
                             ckpt_info: dict = None) -> None:
    """
    Run AF3 self-consistency and docking evaluation.
    
    Args:
        sample_dict: Dictionary of sample information.
        out_dir: Output directory.
        cfg: Configuration object.
        ckpt_info: Checkpoint info (optional, for wandb logging).
    """
    struct_pred_cfg = cfg.struct_pred_cfg
    
    # Make json input directory
    af3_ss_input_dir = Path(out_dir, "af3_ss_inputs")
    af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Make a directory for af3 single-sequence prediction outputs
    af3_ss_pred_dir = Path(out_dir, "af3_ss_preds")
    af3_ss_pred_dir.mkdir(parents=True, exist_ok=True)
                    
    print("Creating AF3 JSON input files...")
    
    af3_ss_json_paths, _, pdb_chain_info = make_af3_json(
        af3_ss_input_dir=af3_ss_input_dir,
        af3_tc_input_dir=None,
        sample_id_list=sample_id_list,
        pdb_id_list=pdb_id_list,
        sample_atom_array_list=sample_atom_array_list,
        template_pdb_path_list=None,
        pdb_chain_info=pdb_chain_info,
        metadata=None,
        json_config=struct_pred_cfg.af3.json_config
    )
    
    print(f"Created {len(sample_id_list)} AF3 JSON input files in {af3_ss_input_dir}")
        
    # Run AF3 self-consistency and docking evaluation
    af3_runner_path = struct_pred_cfg.af3.runner_path
    af3_inference_config = struct_pred_cfg.af3.inference_config
    
    sample_id_to_per_pred_sc_metrics = {}
    sample_id_to_per_pred_docking_metrics = {}
    
    print("\n" + "="*80)
    print("Running AF3 Self-Consistency Evaluation")
    print("="*80 + "\n")        
    
    for i in tqdm(range(len(sample_id_list)), desc="AF3 predictions"):
        sample_id = sample_id_list[i]
        pdb_id = pdb_id_list[i]                         
        ss_json_path = af3_ss_json_paths[i]       
        sample_atom_array = sample_atom_array_list[i]
        
        try:
            run_af3_single_sequence(str(ss_json_path), str(af3_ss_pred_dir), 
                                    runner_path=af3_runner_path, 
                                    inference_config=af3_inference_config)
        except Exception as e:
            print(f"AF3 single sequence prediction failed for {pdb_id}: {e}")
            continue
            
        _, pred_ss_sample_paths = find_pred_sample_path_af3(out_dir=str(af3_ss_pred_dir), 
                                                            job_name=sample_id)
        
        if len(pred_ss_sample_paths) == 0:
            print(f"No AF3 predicted structure found for {pdb_id}")
            continue
        
        else:   
            sample_id_to_per_pred_sc_metrics[sample_id] = {}      
            sample_id_to_per_pred_docking_metrics[sample_id] = {}                                          
            for pred_idx, pred_ss_sample_path in enumerate(pred_ss_sample_paths):
                try:
                    pred_example = get_sd_example_from_af3_prediction(
                        pdb_path=pred_ss_sample_path,
                        data_cfg=cfg.data_cfg_for_af3_prediction,
                        transform_cfg=cfg.transform_cfg_for_af3_prediction
                    )
                                                                                    
                    pred_atom_array = pred_example["atom_array"]
                    per_pred_sc_metrics = _compute_self_consistency_metrics_atomarray(
                        pred_atom_array=pred_atom_array,
                        sample_atom_array=sample_atom_array,
                        pred_sample_path=pred_ss_sample_path,
                        return_aligned_atom_array=False
                    )                                                                                            
            
                except Exception as e:
                    print(f"Self-consistency metrics computation failed for {sample_id, pred_idx}: {e}")
                    continue
                else:            
                    # Store self-consistency metrics in sample_dict
                    sample_id_to_per_pred_sc_metrics[sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics
                
                try: 
                    per_pred_docking_metrics = _compute_docking_metrics_atomarray(
                        pred_atom_array=pred_atom_array,
                        sample_atom_array=sample_atom_array,
                        pred_sample_path=pred_ss_sample_path,
                        return_aligned_atom_array=False,
                        pocket_distance_for_metrics=cfg.docking_metrics_cfg.pocket_distance_for_metrics,
                        receptor_chain=pdb_chain_info[sample_id]["protein_chains"][0],
                        ligand_chain=pdb_chain_info[sample_id]["ligand_chains"][0]
                    )
                
                except Exception as e:
                    print(f"Docking metrics computation failed for {sample_id, pred_idx}: {e}")
                    continue
                else:
                    # Store docking metrics in sample_dict
                    sample_id_to_per_pred_docking_metrics[sample_id][f"diffusion_{pred_idx}"] = per_pred_docking_metrics
    
    # Aggregate best metrics
    sample_id_best_sc_metrics = _aggregate_best_sc_metrics(sample_id_to_per_pred_sc_metrics)
    sample_id_best_docking_metrics = _aggregate_best_docking_metrics(sample_id_to_per_pred_docking_metrics)
    
    # Add num_replaced_pocket_residues if available
    if num_redesigned_pocket_residue_list is not None:        
        for sample_id, num_replaced_pocket_residues in zip(sample_id_list, num_redesigned_pocket_residue_list):
            if sample_id in sample_id_best_docking_metrics:
                sample_id_best_docking_metrics[sample_id]["num_replaced_pocket_residues"] = num_replaced_pocket_residues
            
    # Save results
    _save_metrics_results(
        out_dir=out_dir,
        sample_id_to_per_pred_sc_metrics=sample_id_to_per_pred_sc_metrics,
        sample_id_to_per_pred_docking_metrics=sample_id_to_per_pred_docking_metrics,
        sample_id_best_sc_metrics=sample_id_best_sc_metrics,
        sample_id_best_docking_metrics=sample_id_best_docking_metrics,
        cfg=cfg,
        ckpt_info=ckpt_info
    )
    
    print("\n" + "="*80)
    print("AF3 Self-Consistency and Docking Evaluation Complete")
    print(f"Results saved to {out_dir}")
    print("="*80 + "\n")


def _aggregate_best_sc_metrics(sample_id_to_per_pred_sc_metrics: dict) -> dict:
    """Aggregate best self-consistency metrics (by max avg_ca_plddt)."""
    sample_id_best_sc_metrics = {}        
    for sample_id, per_pred_sc_metrics in sample_id_to_per_pred_sc_metrics.items():
        if not per_pred_sc_metrics:
            continue
        # Find the prediction with max avg_ca_plddt
        best_pred = max(per_pred_sc_metrics.values(), key=lambda x: x["avg_ca_plddt"])
        sample_id_best_sc_metrics[sample_id] = {
            "avg_ca_plddt": best_pred["avg_ca_plddt"],
            "sc_ca_rmsd": best_pred["sc_ca_rmsd"]
        }
    return sample_id_best_sc_metrics


def _aggregate_best_docking_metrics(sample_id_to_per_pred_docking_metrics: dict) -> dict:
    """Aggregate best docking metrics (by max ligand_plddt)."""
    sample_id_best_docking_metrics = {}
    for sample_id, per_pred_docking_metrics in sample_id_to_per_pred_docking_metrics.items():
        if not per_pred_docking_metrics:
            continue
        # Find the prediction with max ligand_plddt
        best_pred = max(per_pred_docking_metrics.values(), key=lambda x: x["ligand_plddt"])
        sample_id_best_docking_metrics[sample_id] = {
            "ligand_rmsd": best_pred["ligand_rmsd"],
            "binding_site_rmsd": best_pred["binding_site_rmsd"],
            "ligand_plddt": best_pred["ligand_plddt"],
            "binding_site_plddt": best_pred["binding_site_plddt"],
            "iptm": best_pred["iptm"],
            "interface_min_pae": best_pred["interface_min_pae"],
        }
    return sample_id_best_docking_metrics


def _save_metrics_results(out_dir: Path,
                          sample_id_to_per_pred_sc_metrics: dict,
                          sample_id_to_per_pred_docking_metrics: dict,
                          sample_id_best_sc_metrics: dict,
                          sample_id_best_docking_metrics: dict,
                          cfg: DictConfig,
                          ckpt_info: dict = None) -> None:
    """Save metrics results to CSV and log to wandb."""
    
    # Self-consistency metrics
    all_sc_metrics_df = pd.DataFrame.from_dict(sample_id_to_per_pred_sc_metrics, orient='index')
    all_sc_metrics_df = all_sc_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
    all_sc_metrics_df.to_csv(Path(out_dir, "all_sc_metrics_results.csv"), index=False)
    
    # Docking metrics
    all_docking_metrics_df = pd.DataFrame.from_dict(sample_id_to_per_pred_docking_metrics, orient='index')
    all_docking_metrics_df = all_docking_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
    all_docking_metrics_df.to_csv(Path(out_dir, "all_docking_metrics_results.csv"), index=False)
    
    # Best self-consistency metrics
    best_sc_metrics_df = pd.DataFrame.from_dict(sample_id_best_sc_metrics, orient='index')
    best_sc_metrics_df = best_sc_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
    best_sc_metrics_df.to_csv(Path(out_dir, "best_sc_metrics_results.csv"), index=False)
    
    # Best docking metrics
    best_docking_metrics_df = pd.DataFrame.from_dict(sample_id_best_docking_metrics, orient='index')
    best_docking_metrics_df = best_docking_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
    best_docking_metrics_df.to_csv(Path(out_dir, "best_docking_metrics_results.csv"), index=False)
    
    # Log summary metrics to wandb
    if sample_id_best_sc_metrics:
        best_sc_ca_rmsds = [m["sc_ca_rmsd"] for m in sample_id_best_sc_metrics.values()]
        best_avg_ca_plddts = [m["avg_ca_plddt"] for m in sample_id_best_sc_metrics.values()]
        
        wandb_metrics = {                
            "eval/median/sc_ca_rmsd": np.median(best_sc_ca_rmsds),
            "eval/median/avg_ca_plddt": np.median(best_avg_ca_plddts),            
        }
        
        if ckpt_info:
            wandb_metrics["trainer/global_step"] = ckpt_info["global_step"]
            wandb_metrics["trainer/epoch"] = ckpt_info["epoch"]
        
        if not cfg.wandb.no_wandb:
            wandb.log(wandb_metrics, commit=True)
            print(f"Logged metrics to wandb: {wandb_metrics}")
    
    if sample_id_best_docking_metrics:
        best_ligand_rmsd = [m["ligand_rmsd"] for m in sample_id_best_docking_metrics.values()]
        best_binding_site_rmsd = [m["binding_site_rmsd"] for m in sample_id_best_docking_metrics.values()]
        best_ligand_plddt = [m["ligand_plddt"] for m in sample_id_best_docking_metrics.values()]
        best_binding_site_plddt = [m["binding_site_plddt"] for m in sample_id_best_docking_metrics.values()]
        best_iptm = [m["iptm"] for m in sample_id_best_docking_metrics.values()]
        best_interface_min_pae = [m["interface_min_pae"] for m in sample_id_best_docking_metrics.values()]
        
        wandb_metrics = {                
            "eval/median/ligand_rmsd": np.median(best_ligand_rmsd),
            "eval/median/binding_site_rmsd": np.median(best_binding_site_rmsd),
            "eval/median/ligand_plddt": np.median(best_ligand_plddt),
            "eval/median/binding_site_plddt": np.median(best_binding_site_plddt),
            "eval/median/iptm": np.median(best_iptm),
            "eval/median/interface_min_pae": np.median(best_interface_min_pae),
        }
        
        if ckpt_info:
            wandb_metrics["trainer/global_step"] = ckpt_info["global_step"]
            wandb_metrics["trainer/epoch"] = ckpt_info["epoch"]
        
        if not cfg.wandb.no_wandb:
            wandb.log(wandb_metrics, commit=True)
            print(f"Logged metrics to wandb: {wandb_metrics}")


###########################################################
# Main
###########################################################

@hydra.main(config_path="../../configs_local/eval/sampling", config_name="redesign_pocket_seq", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Redesign pocket sequence using native sequence or lcaliby.
    Assume samples are already designed by caliby codebase.
    """
    ###########################################################
    # Phase 0: Basic setup
    ###########################################################
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Debug mode adjustments
    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"
    
    # Setup wandb logging
    log_dir = Path(wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb))
    
    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)
    
    # Load metadata
    metadata = pd.read_parquet(cfg.metadata_path)
    
    ###########################################################
    # Phase 1: Prepare samples (common)
    ###########################################################
    print("\n" + "="*80)
    print("Phase 1: Preparing samples")
    print("="*80 + "\n")
    
    sample_dict = prepare_samples(cfg, metadata)
    
    # Make a directory for saving original structures with ligand
    processed_original_samples_dir = log_dir / "original_samples_with_ligand"
    
    # Load designed structures and combine with ligand
    print("Loading designed structures and combining with ligand...")
    sample_dict = load_designed_structures(
        sample_dict=sample_dict, 
        cfg=cfg, 
        save_dir=processed_original_samples_dir
    )
    
    ###########################################################
    # Phase 2: Redesign pocket sequence
    ###########################################################
    print("\n" + "="*80)
    print("Phase 2: Redesigning pocket sequence")
    print("="*80 + "\n")
    
    if cfg.redesign_cfg.redesign_pocket_seq:
        if cfg.redesign_cfg.use_native_pocket_seq:
            # Native replace mode
            redesign_out_dir = log_dir / "redesigned_samples"
            sample_dict = redesign_with_native(sample_dict, cfg, redesign_out_dir)
            
            sample_id_list = list(sample_dict.keys())
            pdb_id_list = [sample_dict[sid]['pdb_id'] for sid in sample_id_list]
            sample_atom_array_list = [sample_dict[sid]['redesigned_sample_atom_array'] for sid in sample_id_list]
            pdb_chain_info = {sid: sample_dict[sid]['pdb_chain_info'] for sid in sample_id_list}
            num_redesigned_pocket_residue_list = [sample_dict[sid]['num_redesigned_pocket_residues'] for sid in sample_id_list]
            
            # Evaluate if needed
            if cfg.evaluate_self_consistency:
                print("\n" + "="*80)
                print("Phase 3: AF3 Evaluation")
                print("="*80 + "\n")
                evaluate_af3_consistency(sample_id_list=sample_id_list, 
                                         pdb_id_list=pdb_id_list, 
                                         sample_atom_array_list=sample_atom_array_list, 
                                         pdb_chain_info=pdb_chain_info, 
                                         num_redesigned_pocket_residue_list=num_redesigned_pocket_residue_list, 
                                         out_dir=log_dir, 
                                         cfg=cfg,
                                         ckpt_info=None)
        else:
            # Lcaliby mode - iterate over checkpoints        
            results = redesign_with_lcaliby(sample_dict, cfg, metadata, log_dir)
            
            # Evaluate each checkpoint
            if cfg.evaluate_self_consistency:
                print("\n" + "="*80)
                print("Phase 3: AF3 Evaluation (per checkpoint)")
                print("="*80 + "\n")
                for ckpt_sample_dict, ckpt_out_dir, ckpt_info in results:
                    print(f"\nEvaluating checkpoint: step_{ckpt_info['global_step']}_epoch_{ckpt_info['epoch']}")
                    evaluate_af3_consistency(ckpt_sample_dict, ckpt_out_dir, cfg, ckpt_info)
    else:
        redesign_out_dir = log_dir / "samples"
        sample_id_list = list(sample_dict.keys())
        pdb_id_list = [sample_dict[sid]['pdb_id'] for sid in sample_id_list]
        sample_atom_array_list = [sample_dict[sid]['sample_atom_array_with_ligand'] for sid in sample_id_list]
        pdb_chain_info = {sid: sample_dict[sid]['pdb_chain_info'] for sid in sample_id_list}
        # No redesign, just evaluate original
        if cfg.evaluate_self_consistency:
            print("\n" + "="*80)
            print("Phase 3: AF3 Evaluation (no redesign)")
            print("="*80 + "\n")
            evaluate_af3_consistency(sample_id_list=sample_id_list, 
                                         pdb_id_list=pdb_id_list, 
                                         sample_atom_array_list=sample_atom_array_list, 
                                         pdb_chain_info=pdb_chain_info, 
                                         num_redesigned_pocket_residue_list=None, 
                                         out_dir=log_dir, 
                                         cfg=cfg,
                                         ckpt_info=None)
    
    print("\n" + "="*80)
    print("All phases complete!")
    print(f"Results saved to {log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
