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
                                                          find_pred_sample_path_af3)
from allatom_design.eval.eval_utils.eval_metrics import _compute_self_consistency_metrics_atomarray


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


@hydra.main(config_path="../configs/eval", config_name="eval_pmpnn_lmpnnval_af3", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Run AF3 self-consistency evaluation on designed samples.
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
    
    # Get PDB files
    sample_paths = get_pdb_files(**pdb_cfg)
    
    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]
    
    sample_ids = [Path(sample_path).stem for sample_path in sample_paths]
    pdb_ids = [Path(sample_path).stem.split("_")[0] for sample_path in sample_paths]
    
    # Define protein and ligand chain types
    protein_chain_type = aw_enums.ChainType.POLYPEPTIDE_L
    ligand_chain_types = [chain_type for chain_type in aw_enums.ChainTypeInfo.NON_POLYMERS]
    
    # Initialize dictionary for storing sample information
    sample_dict = {}
    for i, sample_id in enumerate(sample_ids):
        sample_dict[sample_id] = {}
        sample_dict[sample_id]['sample_path'] = sample_paths[i]
        sample_dict[sample_id]['pdb_id'] = pdb_ids[i]        
    
    # Load designed structures and combine with ligand
    print("Loading designed structures...")
    for sample_id in tqdm(sample_ids, desc="Loading structures"):
        pdb_id = sample_dict[sample_id]['pdb_id']
        sample_dict[sample_id]["sample_atom_array"] = None
        pdb_chain_info = {}
        pdb_chain_info["protein_chains"] = []
        pdb_chain_info["ligand_chains_ccd_codes"] = []
        
        # Load a designed structure and featurize it
        example = get_sd_example_from_designs_from_other_methods(pdb_path = sample_dict[sample_id]['sample_path'], 
                                                                data_cfg = cfg.data_cfg_for_design, 
                                                                transform_cfg = cfg.transform_cfg_for_design)
                
        
        sample_atom_array = example["atom_array"]
        
        
        protein_chains = [str(chain_id) for chain_id in np.unique(sample_atom_array[sample_atom_array.chain_type == protein_chain_type].chain_id)]    
        ligand_chains = [str(chain_id) for chain_id in np.unique(sample_atom_array[np.isin(sample_atom_array.chain_type, ligand_chain_types)].chain_id)]
        ccd_codes = [str(sample_atom_array[sample_atom_array.chain_id == chain].res_name[0]) for chain in ligand_chains]
        ligand_chains_ccd_codes = list(zip(ligand_chains, ccd_codes))
        
        for chain in protein_chains:
            pdb_chain_info["protein_chains"].append(str(chain))
        
        for chain, ccd_code in ligand_chains_ccd_codes:
            pdb_chain_info["ligand_chains_ccd_codes"].append((str(chain), str(ccd_code)))
            
        sample_dict[sample_id]["sample_atom_array"] = sample_atom_array
        sample_dict[sample_id]["pdb_chain_info"] = pdb_chain_info
        
    ###########################################################
    # Make AF3 JSON input files for single-sequence prediction
    ###########################################################        
            
    # Make json input directory
    af3_ss_input_dir = Path(base_out_dir, "af3_ss_inputs")
    af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Make a directory for af3 single-sequence prediction outputs
    af3_ss_pred_dir = Path(base_out_dir, "af3_ss_preds")
    af3_ss_pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Model seeds
    model_seeds = list(struct_pred_cfg.af3.json_config.model_seeds)
    version = int(struct_pred_cfg.af3.json_config.version)
            
    print("Creating AF3 JSON input files...")
    for sample_id in tqdm(sample_ids, desc="Creating AF3 JSONs"):        
        pdb_id = sample_dict[sample_id]['pdb_id']
        sample_atom_array = sample_dict[sample_id]["sample_atom_array"]
        protein_chains = sample_dict[sample_id]["pdb_chain_info"]["protein_chains"]
        ligand_chains_ccd_codes = sample_dict[sample_id]["pdb_chain_info"]["ligand_chains_ccd_codes"]                    
            
        ss_sequences = []
        for protein_chain in protein_chains:
            _res_starts = get_residue_starts(sample_atom_array[sample_atom_array.chain_id == protein_chain])
            _res_ids = sample_atom_array[sample_atom_array.chain_id == protein_chain].res_id[_res_starts]
            _res_ids_0based = _res_ids - np.min(_res_ids)
            
            full_length = np.max(_res_ids) - np.min(_res_ids) + 1
            chain_seq_with_gaps = np.full(full_length, "UNK")
            
            chain_seq = sample_atom_array[sample_atom_array.chain_id == protein_chain].res_name[_res_starts]
            chain_seq_with_gaps[_res_ids_0based] = chain_seq
            processed_entity_canonical_sequence_with_gaps = "".join(aa_chem_comp_3to1(standard_only=False).get(res_name, "X") for res_name in chain_seq_with_gaps)
            
            ss_sequences.append({
                "protein": {
                    "id": protein_chain,
                    "sequence": processed_entity_canonical_sequence_with_gaps,
                    "unpairedMsa": "",
                    "pairedMsa": ""
                }
            })            
        
        for ligand_chain, ccd_code in ligand_chains_ccd_codes:
            ss_sequences.append({
                "ligand": {
                    "id": ligand_chain,
                    "ccdCodes": [ccd_code],
                }
            })                        
        
        sample_af3_ss_json = {
            "name": sample_id,
            "sequences": ss_sequences,
            "modelSeeds": model_seeds,
            "dialect": "alphafold3",
            "version": version,
        }
        
        json_path = Path(af3_ss_input_dir, f"{sample_id}.json")        
        with open(json_path, "w") as f:
            json.dump(sample_af3_ss_json, f)
    
    print(f"Created {len(sample_ids)} AF3 JSON input files in {af3_ss_input_dir}")
    
    ###########################################################
    # Run AF3 self-consistency evaluation
    ###########################################################
    
    if cfg.evaluate_self_consistency:
        af3_runner_path = struct_pred_cfg.af3.runner_path
        af3_inference_config = struct_pred_cfg.af3.inference_config
        
        sample_id_to_per_pred_sc_metrics = {}
        
        print("\n" + "="*80)
        print("Running AF3 Self-Consistency Evaluation")
        print("="*80 + "\n")
        
        pbar = tqdm(sample_ids, desc="AF3 predictions")
        for sample_id in pbar:
            pdb_id = sample_dict[sample_id]['pdb_id']
            pbar.set_postfix_str(f"PDB: {pdb_id}")
            
            ss_json_path = Path(af3_ss_input_dir, f"{sample_id}.json")           
            sample_atom_array = sample_dict[sample_id]["sample_atom_array"]
            
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
                try:                    
                    for pred_idx, pred_ss_sample_path in enumerate(pred_ss_sample_paths):
                        pred_example = get_sd_example_from_af3_prediction(pdb_path=pred_ss_sample_path,
                                                                        data_cfg=data_cfg_for_af3_prediction,
                                                                        transform_cfg=transform_cfg_for_af3_prediction,
                                                                        metadata=metadata)
                        pred_atom_array = pred_example["atom_array"]
                        per_pred_sc_metrics = _compute_self_consistency_metrics_atomarray(
                            pred_atom_array=pred_atom_array,
                            sample_atom_array=sample_atom_array,
                            pred_sample_path=pred_ss_sample_path,
                            return_aligned_atom_array=False)
                                                                        
                        # Store metrics in sample_dict
                        sample_id_to_per_pred_sc_metrics[sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics
                
                except Exception as e:
                    print(f"Self-consistency metrics computation failed for {pdb_id}: {e}")
                    continue
        
        sample_id_best_sc_metrics = {}
        for sample_id, per_pred_sc_metrics in sample_id_to_per_pred_sc_metrics.items():
            best_sc_metrics = {}
            best_sc_metrics["sc_ca_rmsd"] = min(per_pred_sc_metrics.values(), key=lambda x: x["sc_ca_rmsd"])["sc_ca_rmsd"]
            best_sc_metrics["avg_ca_plddt"] = max(per_pred_sc_metrics.values(), key=lambda x: x["avg_ca_plddt"])["avg_ca_plddt"]
            sample_id_best_sc_metrics[sample_id] = best_sc_metrics
                
        # Save all results        
        all_sc_metrics_df = pd.DataFrame.from_dict(sample_id_to_per_pred_sc_metrics, orient='index')
        all_sc_metrics_df = all_sc_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
        all_sc_metrics_df.to_csv(Path(base_out_dir, "all_sc_metrics_results.csv"), index=False)
        
        # Save best results
        best_sc_metrics_df = pd.DataFrame.from_dict(sample_id_best_sc_metrics, orient='index')
        best_sc_metrics_df = best_sc_metrics_df.reset_index().rename(columns={'index': 'sample_id'})
        best_sc_metrics_df.to_csv(Path(base_out_dir, "best_sc_metrics_results.csv"), index=False)
        
        # Log summary metrics to wandb
        if sample_id_best_sc_metrics:
            best_sc_ca_rmsds = [m["sc_ca_rmsd"] for m in sample_id_best_sc_metrics.values()]
            best_avg_ca_plddts = [m["avg_ca_plddt"] for m in sample_id_best_sc_metrics.values()]
            
            wandb_metrics = {                
                "eval/median/best/sc_ca_rmsd": np.median(best_sc_ca_rmsds),
                "eval/median/best/avg_ca_plddt": np.median(best_avg_ca_plddts),            
            }
            
            if not cfg.wandb.no_wandb:
                wandb.log(wandb_metrics, commit=True)
                print(f"Logged metrics to wandb: {wandb_metrics}")
        
        print("\n" + "="*80)
        print("AF3 Self-Consistency Evaluation Complete")
        print(f"Results saved to {base_out_dir}")
        print("="*80 + "\n")        


if __name__ == "__main__":
    main()
