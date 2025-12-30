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

import atomworks.enums as aw_enums
from atomworks.io.utils.sequence import aa_chem_comp_3to1
from atomworks.io.utils.io_utils import to_cif_file

from allatom_design.eval.eval_utils.seq_des_utils import (get_sd_example, 
                                                          load_example_with_load_any,                                                          
                                                          get_sd_example_from_af3_prediction, 
                                                          get_sd_example_from_designs_from_other_methods)
from allatom_design.eval.eval_utils.folding_utils import (run_af3_single_sequence,
                                                          find_pred_sample_path_af3,
                                                          make_af3_json)
from allatom_design.eval.eval_utils.eval_metrics import _compute_self_consistency_metrics_atomarray, _compute_docking_metrics_atomarray
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets

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

def extract_ligand_atom_array_from_cached_examples(sample_dict: dict = None, 
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

def load_designed_structures_from_caliby(sample_dict: dict = None,
                                    data_cfg: DictConfig = None,
                                    transform_cfg: DictConfig = None,
                                    return_combined_atom_array: bool = False,
                                    save_combined_atom_array: bool = False,
                                    save_dir = None,
                                    ) -> dict:
    """
    Load designed structures from caliby codebase.
    args:
        sample_dict: Dictionary of sample information. Required keys: 'sample_path', 'pdb_id'.
        data_cfg: Data configuration for loading designed structures.
        transform_cfg: Transform configuration for featurizing designed structures.
        return_combined_atom_array: If True, return the designed structures with ligand from the native structure.
        save_combined_atom_array: If True, save the designed structures with ligand from the native structure.
        save_dir: Directory to save the designed structures with ligand from the native structure.
    returns:
        sample_dict: Dictionary of sample information with designed structures with ligand from the native structure.
        pdb_chain_info: Dictionary of pdb chain information.
    """
    # Get sample ids
    sample_ids = list(sample_dict.keys())
            
    # Check if save_dir is provided if return_combined_atom_array and save_combined_atom_array are True
    if return_combined_atom_array and save_combined_atom_array:
        assert save_dir is not None, "save_dir is required if return_combined_atom_array and save_combined_atom_array are True"
    
    for sample_id in tqdm(sample_ids, desc="Loading Caliby designed structures"):        
        
        sample_dict[sample_id]["sample_atom_array"] = None        
        pdb_chain_info = {
            "protein_chains": [],
            "ligand_chains": [],
            "ligand_ccd_codes": []
        }
                  
        # Load a designed structure and featurize it
        example = get_sd_example_from_designs_from_other_methods(pdb_path = sample_dict[sample_id]['sample_path'], 
                                                                data_cfg = data_cfg, 
                                                                transform_cfg = transform_cfg)
        
        sample_atom_array = example["atom_array"]
        sample_bb_atom_array = sample_atom_array[sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
        sample_bb_atom_array = sample_bb_atom_array[sample_bb_atom_array.is_backbone_atom]
        ligand_atom_array = sample_dict[sample_id]["native_ligand_atom_array"]
        
        if return_combined_atom_array:
            sample_bb_ligand_atom_array = struc.concatenate([sample_bb_atom_array, ligand_atom_array])
            # Renumber atom_id sequentially (1-indexed)
            sample_bb_ligand_atom_array.atom_id = np.arange(1, len(sample_bb_ligand_atom_array) + 1)            
            sample_dict[sample_id]["sample_atom_array"] = sample_bb_ligand_atom_array                    
            if save_combined_atom_array:                
                # Save the combined atom array, can be used later if pocket sequence is not redesigned
                cif_path = Path(save_dir, f"{sample_id}.cif")
                to_cif_file(sample_bb_ligand_atom_array, str(cif_path))
                
                # Save the original atom array with ligand, can be used later for checking the original pocket sequence if pocket sequence is redesigned
                backup_cif_path = Path(save_dir, f"{sample_id}_original_with_ligand.cif")
                to_cif_file(sample_bb_ligand_atom_array, str(backup_cif_path))        
            
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


@hydra.main(config_path="../../configs/eval/sampling", config_name="redesign_pocket_seq", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Redesign pocket sequence using lcaliby.
    Assume samples are already designed by caliby codebase.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Debug mode adjustments
    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"
    
    # Setup wandb logging
    log_dir = wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb)
    
    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)
    
    # Load metadata
    metadata = pd.read_parquet(cfg.metadata_path)
    
    # Setup output directory (use log_dir instead of base_out_dir for organized outputs)
    base_out_dir = Path(log_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get config sections
    cached_example_path = cfg.cached_example_path
    pdb_cfg = cfg.pdb_cfg
    data_cfg_for_af3_prediction = cfg.data_cfg_for_af3_prediction
    transform_cfg_for_af3_prediction = cfg.transform_cfg_for_af3_prediction
    struct_pred_cfg = cfg.struct_pred_cfg
    
    # Make a directory for saving aligned structures
    processed_samples_dir = Path(base_out_dir, "samples")
    processed_samples_dir.mkdir(parents=True, exist_ok=True)
        
    # Get PDB files
    sample_paths = get_pdb_files(**pdb_cfg)
    
    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]
        
    # Initialize dictionary for storing sample information
    sample_dict = create_sample_dict(sample_paths=sample_paths)
            
    # Extract native atom array and ligand atom array from cached examples
    sample_dict = extract_ligand_atom_array_from_cached_examples(sample_dict=sample_dict, 
                                                                 cached_example_path=cached_example_path, 
                                                                 metadata=metadata)
        
        
    # Load designed structures from caliby and combine with ligand
    print("Loading designed structures and combining with ligand...")
    sample_dict = load_designed_structures_from_caliby(sample_dict=sample_dict, 
                                                       data_cfg=cfg.data_cfg_for_design, 
                                                       transform_cfg=cfg.transform_cfg_for_design, 
                                                       return_combined_atom_array=True, 
                                                       save_combined_atom_array=True, 
                                                       save_dir=processed_samples_dir)
        
    
    if cfg.redesign_cfg.redesign_pocket_seq:
        if cfg.redesign_cfg.use_native_pocket_seq:            
            sample_ids = list(sample_dict.keys())
            for sample_id in tqdm(sample_ids, desc="Replacing pocket sequence with native pocket sequence"):
                native_atom_array = sample_dict[sample_id]["native_atom_array"]
                protein_chains = sample_dict[sample_id]["pdb_chain_info"]["protein_chains"]
                ligand_chains = sample_dict[sample_id]["pdb_chain_info"]["ligand_chains"]
                sample_atom_array = sample_dict[sample_id]["sample_atom_array"]
                
                
                sample_atom_array, num_replaced_pocket_residues = replace_pocket_sequence_with_native(native_atom_array=native_atom_array,
                                                                       sample_atom_array=sample_atom_array,
                                                                       pocket_distance=cfg.redesign_cfg.pocket_distance,
                                                                       protein_chains=protein_chains,
                                                                       ligand_chains=ligand_chains)
                
                # Overwrite the sample atom array in sample_dict with the replaced one
                sample_dict[sample_id]["sample_atom_array"] = sample_atom_array
                sample_dict[sample_id]["num_replaced_pocket_residues"] = num_replaced_pocket_residues   
                print(f"Replaced {num_replaced_pocket_residues} pocket residues for {sample_id}")
                
                # Save the replaced sample atom array. It'll overwrite the original atom array saved in the load_designed_structures_from_caliby function
                cif_path = Path(processed_samples_dir, f"{sample_id}.cif")
                to_cif_file(sample_atom_array, str(cif_path))
                                
        else:
            print("Need to implement redesigning pocket sequence using lcaliby")
                                                     
        
    ###########################################################
    # Make AF3 JSON input files for single-sequence prediction
    ###########################################################        
    
    if cfg.evaluate_self_consistency:
        # Make json input directory
        af3_ss_input_dir = Path(base_out_dir, "af3_ss_inputs")
        af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Make a directory for af3 single-sequence prediction outputs
        af3_ss_pred_dir = Path(base_out_dir, "af3_ss_preds")
        af3_ss_pred_dir.mkdir(parents=True, exist_ok=True)
                        
        print("Creating AF3 JSON input files...")
        
        sample_ids = list(sample_dict.keys())
        pdb_ids = [sample_dict[sid]['pdb_id'] for sid in sample_ids]
        sample_atom_arrays = [sample_dict[sid]['sample_atom_array'] for sid in sample_ids]
        pdb_chain_info = {sid: sample_dict[sid]['pdb_chain_info'] for sid in sample_ids}
        
        af3_ss_json_paths, _, pdb_chain_info = make_af3_json(af3_ss_input_dir=af3_ss_input_dir,
                                                                af3_tc_input_dir=None,
                                                                sample_id_list=sample_ids,
                                                                pdb_id_list=pdb_ids,
                                                                sample_atom_array_list=sample_atom_arrays,
                                                                template_pdb_path_list=None,
                                                                pdb_chain_info=pdb_chain_info,
                                                                metadata=None,
                                                                json_config=struct_pred_cfg.af3.json_config)
        
        print(f"Created {len(sample_ids)} AF3 JSON input files in {af3_ss_input_dir}")
            
        ###########################################################
        # Run AF3 self-consistency and docking evaluation
        ###########################################################        
        af3_runner_path = struct_pred_cfg.af3.runner_path
        af3_inference_config = struct_pred_cfg.af3.inference_config
        
        sample_id_to_per_pred_sc_metrics = {}
        sample_id_to_per_pred_docking_metrics = {}
        
        print("\n" + "="*80)
        print("Running AF3 Self-Consistency Evaluation")
        print("="*80 + "\n")        
        
        for i in tqdm(range(len(sample_ids)), desc="AF3 predictions"):
            sample_id = sample_ids[i]
            pdb_id = pdb_ids[i]                        
            ss_json_path = af3_ss_json_paths[i]       
            sample_atom_array = sample_atom_arrays[i]
            
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
                        pred_example = get_sd_example_from_af3_prediction(pdb_path=pred_ss_sample_path,
                                                                        data_cfg=data_cfg_for_af3_prediction,
                                                                        transform_cfg=transform_cfg_for_af3_prediction)
                                                                                        
                        pred_atom_array = pred_example["atom_array"]
                        per_pred_sc_metrics = _compute_self_consistency_metrics_atomarray(
                            pred_atom_array=pred_atom_array,
                            sample_atom_array=sample_atom_array,
                            pred_sample_path=pred_ss_sample_path,
                            return_aligned_atom_array=False)                                                                                            
                
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
                        receptor_chain = pdb_chain_info[sample_id]["protein_chains"][0],
                        ligand_chain = pdb_chain_info[sample_id]["ligand_chains"][0])
                    
                    except Exception as e:
                        print(f"Docking metrics computation failed for {sample_id, pred_idx}: {e}")
                        continue
                    else:
                        # Store docking metrics in sample_dict
                        sample_id_to_per_pred_docking_metrics[sample_id][f"diffusion_{pred_idx}"] = per_pred_docking_metrics
        
        
        
        sample_id_best_sc_metrics = {}        
        for sample_id, per_pred_sc_metrics in sample_id_to_per_pred_sc_metrics.items():
            best_sc_metrics = {}
            # Find the prediction with max avg_ca_plddt
            best_pred = max(per_pred_sc_metrics.values(), key=lambda x: x["avg_ca_plddt"])
            best_sc_metrics["avg_ca_plddt"] = best_pred["avg_ca_plddt"]
            best_sc_metrics["sc_ca_rmsd"] = best_pred["sc_ca_rmsd"]
            sample_id_best_sc_metrics[sample_id] = best_sc_metrics
            
        sample_id_best_docking_metrics = {}
        for sample_id, per_pred_docking_metrics in sample_id_to_per_pred_docking_metrics.items():
            # Find the prediction with max ligand_plddt
            best_pred = max(per_pred_docking_metrics.values(), key=lambda x: x["ligand_plddt"])
            best_docking_metrics = {
                "ligand_rmsd": best_pred["ligand_rmsd"],
                "binding_site_rmsd": best_pred["binding_site_rmsd"],
                "ligand_plddt": best_pred["ligand_plddt"],
                "binding_site_plddt": best_pred["binding_site_plddt"],
                "iptm": best_pred["iptm"],
                "interface_min_pae": best_pred["interface_min_pae"],
            }
            sample_id_best_docking_metrics[sample_id] = best_docking_metrics  
        
        if "num_replaced_pocket_residues" in sample_dict.keys():
            for sample_id, num_replaced_pocket_residues in sample_dict.items():
                sample_id_best_docking_metrics[sample_id]["num_replaced_pocket_residues"] = num_replaced_pocket_residues            
                
        ### Save all results                
        # Self-consistency metrics
        all_sc_metrics_df = pd.DataFrame.from_dict(sample_id_to_per_pred_sc_metrics, orient='index')
        all_sc_metrics_df = all_sc_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
        all_sc_metrics_df.to_csv(Path(base_out_dir, "all_sc_metrics_results.csv"), index=False)
        
        # Docking metrics
        all_docking_metrics_df = pd.DataFrame.from_dict(sample_id_to_per_pred_docking_metrics, orient='index')
        all_docking_metrics_df = all_docking_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
        all_docking_metrics_df.to_csv(Path(base_out_dir, "all_docking_metrics_results.csv"), index=False)
        
        ### Save best results        
        # Self-consistency metrics
        best_sc_metrics_df = pd.DataFrame.from_dict(sample_id_best_sc_metrics, orient='index')
        best_sc_metrics_df = best_sc_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
        best_sc_metrics_df.to_csv(Path(base_out_dir, "best_sc_metrics_results.csv"), index=False)
        
        # Docking metrics
        best_docking_metrics_df = pd.DataFrame.from_dict(sample_id_best_docking_metrics, orient='index')
        best_docking_metrics_df = best_docking_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
        best_docking_metrics_df.to_csv(Path(base_out_dir, "best_docking_metrics_results.csv"), index=False)
        
        # Log summary metrics to wandb
        if sample_id_best_sc_metrics:
            best_sc_ca_rmsds = [m["sc_ca_rmsd"] for m in sample_id_best_sc_metrics.values()]
            best_avg_ca_plddts = [m["avg_ca_plddt"] for m in sample_id_best_sc_metrics.values()]
            
            wandb_metrics = {                
                "eval/median/sc_ca_rmsd": np.median(best_sc_ca_rmsds),
                "eval/median/avg_ca_plddt": np.median(best_avg_ca_plddts),            
            }
            
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
            
            if not cfg.wandb.no_wandb:
                wandb.log(wandb_metrics, commit=True)
                print(f"Logged metrics to wandb: {wandb_metrics}")
                                        
        print("\n" + "="*80)
        print("AF3 Self-Consistency and Docking Evaluation Complete")
        print(f"Results saved to {base_out_dir}")
        print("="*80 + "\n")        


if __name__ == "__main__":
    main()
