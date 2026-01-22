import json
import subprocess
import sys
import os
import importlib.util
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import wandb

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.data import data
from allatom_design.data.residue_constants import STANDARD_ATOM_MASK
from atomworks.io.utils.atom_array_plus import AtomArray
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import get_1_from_3_letter_code
from atomworks.enums import ChainType
from atomworks.constants import (AF3_EXCLUDED_LIGANDS, STANDARD_AA,
                                    STANDARD_DNA, STANDARD_RNA)

# ============================================================================
# AF3 In-Process Runner Utils (Internal)
# ============================================================================

# Global cache for AF3 runner module
_AF3_RUNNER_MOD = None
_AF3_FLAGS_INITIALIZED = False


def _load_af3_runner(runner_path: str):
    """
    Load run_alphafold.py as a module dynamically.
    Cached after first load.
    
    Args:
        runner_path: Path to the AF3 runner script (e.g., run_alphafold_debug_local.py)
    
    Returns:
        The loaded module
    """
    global _AF3_RUNNER_MOD
    if _AF3_RUNNER_MOD is not None:
        return _AF3_RUNNER_MOD
    
    spec = importlib.util.spec_from_file_location("af3_runner", runner_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _AF3_RUNNER_MOD = mod
    return mod


def _run_af3_inprocess(
    json_path: str,
    out_dir: str,
    runner_path: str,
    inference_config: dict,
    mode: str = "ss",  # "ss" for single-sequence, "tc" for template-conditioned
) -> None:
    """
    Run AF3 in-process without subprocess.
    This avoids GPU exclusive mode issues on HPC clusters.
    
    Args:
        json_path: Path to the input JSON file
        out_dir: Output directory for predictions
        runner_path: Path to the AF3 runner script
        inference_config: Configuration dict containing base, ss, tc settings
        mode: "ss" for single-sequence, "tc" for template-conditioned
    """
    from absl import flags
    
    global _AF3_FLAGS_INITIALIZED
    
    # Check if prediction already exists
    sample_dir = Path(out_dir) / Path(json_path).stem
    sample_cif_files = list(sample_dir.rglob("*.cif"))
    if sample_cif_files:
        print(f"AF3 prediction already exists for {Path(json_path).stem}")
        return
    
    # Clear PyTorch GPU memory before running JAX
    torch.cuda.empty_cache()
    
    # Load AF3 runner module
    runner = _load_af3_runner(runner_path)
    FLAGS = flags.FLAGS
    
    # Get mode-specific config
    mode_config = inference_config.get(mode, {})
    base_config = inference_config.get('base', {})
    
    # Build argv for AF3
    argv = [
        "run_af3",  # program name (placeholder)
        f"--json_path={json_path}",
        f"--output_dir={out_dir}",
        f"--model_dir={base_config.get('model_dir', '')}",
        "--run_data_pipeline=True",
        "--run_inference=True",
        f"--db_dir={base_config.get('db_dir', '')}",
        f"--flash_attention_implementation={base_config.get('flash_attention_implementation', 'triton')}",
        f"--num_recycles={mode_config.get('num_recycles', 3)}",
        f"--num_diffusion_samples={mode_config.get('num_diffusion_samples', 5)}",
        f"--max_templates={mode_config.get('max_templates', 0)}",
        f"--ligand_protein_template_conditioning_mode={mode_config.get('ligand_protein_template_conditioning_mode', 0)}",
        f"--mask_template_sidechains={mode_config.get('mask_template_sidechains', True)}",
        f"--mask_template_sequence={mode_config.get('mask_template_sequence', True)}",
        "--force_output_dir=True",
    ]
    
    # Add max_template_date for template-conditioned mode
    if mode == "tc" and 'max_template_date' in mode_config:
        argv.append(f"--max_template_date={mode_config['max_template_date']}")
    
    # Reset flags for multiple calls
    # Use mark_as_parsed to allow re-parsing
    try:
        FLAGS.unparse_flags()
    except Exception:
        # If unparse_flags fails, try alternative approach
        pass
    
    # Parse the new argv
    try:
        FLAGS(argv)
    except flags.Error as e:
        # Flags already defined - need to just update values
        print(f"Warning: Flag parsing issue (likely already parsed): {e}")
        # Try to set values directly
        FLAGS.json_path = json_path
        FLAGS.output_dir = out_dir
        FLAGS.model_dir = base_config.get('model_dir', '')
        FLAGS.run_data_pipeline = True
        FLAGS.run_inference = True
        FLAGS.db_dir = [base_config.get('db_dir', '')]
        FLAGS.flash_attention_implementation = base_config.get('flash_attention_implementation', 'triton')
        FLAGS.num_recycles = mode_config.get('num_recycles', 3)
        FLAGS.num_diffusion_samples = mode_config.get('num_diffusion_samples', 5)
        FLAGS.max_templates = mode_config.get('max_templates', 0)
        FLAGS.ligand_protein_template_conditioning_mode = mode_config.get('ligand_protein_template_conditioning_mode', 0)
        FLAGS.mask_template_sidechains = mode_config.get('mask_template_sidechains', True)
        FLAGS.mask_template_sequence = mode_config.get('mask_template_sequence', True)
        FLAGS.force_output_dir = True
        if mode == "tc" and 'max_template_date' in mode_config:
            FLAGS.max_template_date = mode_config['max_template_date']
    
    _AF3_FLAGS_INITIALIZED = True
    
    # Run AF3 main function
    try:
        runner.main(None)
    except SystemExit as e:
        # Catch sys.exit() calls from AF3 and don't let them kill our process
        if e.code != 0 and e.code is not None:
            raise RuntimeError(f"AF3 main() exited with code {e.code}")
        # Exit code 0 or None is fine

# ============================================================================
# AF3 JSON Input Creation
# ============================================================================

def _chain_letters(n: int) -> list[str]:
    """Generate chain letters like A, B, ..., Z, AA, BA, CA, ..."""
    letters = []
    base = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    if n <= 26:
        return base[:n]
    # Extend like A, B, ..., Z, AA, BA, CA, ... (reverse spreadsheet style used in AF3 docs)
    letters.extend(base)
    idx = 0
    while len(letters) < n:
        letters.extend([f"{base[i]}{base[idx]}" for i in range(26)])
        idx += 1
    return letters[:n]


def make_af3_json(af3_ss_input_dir: str = None,
                    af3_tc_input_dir: str = None,           
                    sample_dict: dict = None,                             
                    metadata: pd.DataFrame = None,                                   
                    json_config: dict = None,
                    make_tc_input: bool = False,
                    ) -> None:
    """
    Create AF3 JSON input files for single-sequence and template-conditioned inference.
    
    Args:
        af3_ss_input_dir: Directory to save AF3 single-sequence input JSON files
        af3_tc_input_dir: Directory to save AF3 template-conditioned input JSON files
        sample_id_list: List of sample IDs
        pdb_id_list: List of PDB IDs
        sample_atom_array_list: List of sample atom arrays
        template_pdb_path_list: List of template PDB paths
        pdb_chain_info: PDB chain info dictionary
        metadata: Metadata DataFrame        
        json_config: Configuration dict containing AF3 model_seeds and version
    
    Note:
        Either of metadata or pdb_chain_info must be provided.
        All lists must have the same length and be aligned by index
        (i.e., sample_id_list[i] corresponds to pdb_id_list[i], sample_atom_array_list[i], and template_pdb_path_list[i])    
    """                           
    model_seeds = list(json_config.get('model_seeds', [42]))
    version = int(json_config.get('version', 2))
                    
    use_metadata = False
    if metadata is not None and pdb_chain_info is None:        
        protein_columns = ['q_pn_unit_is_protein']
        nonpolymer_ligand_columns = ['q_pn_unit_is_small_molecule', 'q_pn_unit_is_metal', 'q_pn_unit_non_polymer_res_names']
        polymer_ligand_columns = ['q_pn_unit_is_peptide', 'q_pn_unit_is_nuc_ligand', 'q_pn_unit_is_nuc_polymer']

        pdb_chain_info = {}
        
        expanded_protein_columns = []
        expanded_nonpolymer_ligand_columns = []
        expanded_polymer_ligand_columns = []
        for column in protein_columns:
            expanded_protein_columns.extend([f'{column}_{i}' for i in [1,2]])
        for column in nonpolymer_ligand_columns:
            expanded_nonpolymer_ligand_columns.extend([f'{column}_{i}' for i in [1,2]])
        for column in polymer_ligand_columns:
            expanded_polymer_ligand_columns.extend([f'{column}_{i}' for i in [1,2]])
                    
        for _, row in metadata.iterrows():
            pdb_key = row["pdb_id"]            
            pdb_chain_info[pdb_key] = {}
            pdb_chain_info[pdb_key]['protein_chain_iids'] = []
            pdb_chain_info[pdb_key]['ligand_chain_iids'] = []       
            pdb_chain_info[pdb_key]['ligand_ccd_codes'] = []
            
            for column in expanded_protein_columns:
                if row[column]:
                    suffix = column.split("_")[-1]
                    protein_chain_iid = row[f'q_pn_unit_iid_{suffix}']
                    pdb_chain_info[pdb_key]['protein_chain_iids'].append(protein_chain_iid)
                    
            for column in expanded_nonpolymer_ligand_columns:
                if row[column]:
                    suffix = column.split("_")[-1]
                    ligand_chain_iid = row[f'q_pn_unit_iid_{suffix}']
                    ligand_ccd_code = row[f'q_pn_unit_non_polymer_res_names_{suffix}']
                    pdb_chain_info[pdb_key]['ligand_chain_iids'].append(ligand_chain_iid)
                    pdb_chain_info[pdb_key]['ligand_ccd_codes'].append(ligand_ccd_code)                                               
        
        use_metadata = True        
                                       
    for input_sample_id in tqdm(sample_dict.keys(), desc="Creating AF3 JSONs"):        
        sample_dict[input_sample_id]['af3_ss_json_paths'] = []
        if make_tc_input:
            sample_dict[input_sample_id]['af3_tc_json_paths'] = []
        subsample_dict = sample_dict[input_sample_id]
        for dsidx, designed_sample_id in enumerate(subsample_dict['designed_sample_id']):
            
            designed_sample_atom_array = subsample_dict['designed_sample_atom_array'][dsidx]
            pdb_chain_info = subsample_dict['pdb_chain_info']
            
            if make_tc_input:
                template_sample_path = subsample_dict['designed_sample_path_for_af3_tc'][dsidx]
            else:
                template_sample_path = None
            
            job_name = designed_sample_id                                                                                
                                                            
            protein_chain_iids = pdb_chain_info['protein_chain_iids']        
            ligand_chain_iids = pdb_chain_info['ligand_chain_iids']
            ligand_ccd_codes = pdb_chain_info['ligand_ccd_codes']
            
            ss_sequences = []
            tc_sequences = []
            for protein_chain_iid in protein_chain_iids:
                chain_mask = (designed_sample_atom_array.chain_iid == protein_chain_iid)
                _res_starts = get_residue_starts(designed_sample_atom_array[chain_mask])
                _res_ids = designed_sample_atom_array[chain_mask].res_id[_res_starts]
                _res_ids_0based = _res_ids - np.min(_res_ids)
                
                # Make full sequence with UNK for missing residues, to properly address missing residues in the sequence, in af3 prediction
                # This method only fills in the gaps between the actual residues, not the gaps at the beginning or end of the chain
                full_length = np.max(_res_ids) - np.min(_res_ids) + 1
                chain_seq_with_gaps = np.full(full_length, "UNK")
                
                # Replace residues with actual sequence
                chain_seq = designed_sample_atom_array[chain_mask].res_name[_res_starts]            
                chain_seq_with_gaps[_res_ids_0based] = chain_seq
                
                # Detect modified residues
                # Get hetero flag
                chain_hetero = designed_sample_atom_array[chain_mask].hetero[_res_starts]
                hetero_flags_with_gaps = np.full(full_length, False)
                hetero_flags_with_gaps[_res_ids_0based] = chain_hetero
                
                # Make a list of modified residues and a list of sequence letters
                modifications = []
                sequence_letters = []                
                for idx, (res_name, is_hetero) in enumerate(zip(chain_seq_with_gaps, hetero_flags_with_gaps)):
                    one_letter = get_1_from_3_letter_code(
                        res_name, 
                        chain_type=ChainType.POLYPEPTIDE_L,
                        use_closest_canonical=False
                    )
                    sequence_letters.append(one_letter)
                    
                    if is_hetero and res_name not in STANDARD_AA and res_name != "UNK":
                        modifications.append({
                            "ptmType": res_name, # CCD code
                            "ptmPosition": idx + 1 # 1-based index
                        })
                                                    
                
                sequence_with_gaps = "".join(sequence_letters)                
                
                if make_tc_input:
                    # Make template indices for the actual sequence. 0-based
                    query_indices = template_indices = [int(x) for x in list(_res_ids_0based)]
                                    
                ss_sequences.append({
                    "protein": {
                        "id": protein_chain_iid.split("_")[0], 
                        "sequence": sequence_with_gaps,
                        "modifications": modifications if modifications else [],
                        "unpairedMsa": "",
                        "pairedMsa": "",                    
                        }
                    }                
                )
                
                if make_tc_input:
                    tc_sequences.append({
                        "protein": {
                            "id": protein_chain_iid.split("_")[0],
                            "sequence": sequence_with_gaps, 
                            "modifications": modifications if modifications else [],
                            "unpairedMsa": "",
                            "pairedMsa": "",
                            "templates": [
                                {
                                    "mmcifPath": template_sample_path,
                                    "queryIndices": query_indices,
                                    "templateIndices": template_indices,
                                    "templateChainId": protein_chain_iid.split("_")[0],
                                }
                            ]
                        }
                    })                
            
            
            for ligand_chain_iid, ligand_ccd_code in zip(ligand_chain_iids, ligand_ccd_codes):                    
                ss_sequences.append({
                    "ligand": {
                        "id": ligand_chain_iid.split("_")[0],
                        "ccdCodes": [ligand_ccd_code]
                    }
                })
                
                if make_tc_input:
                    tc_sequences.append({
                        "ligand": {
                            "id": ligand_chain_iid.split("_")[0],
                            "ccdCodes": [ligand_ccd_code]
                        }
                    })
            
            
            sample_af3_ss_json = {
                "name": job_name,
                "sequences": ss_sequences,
                "modelSeeds": model_seeds,
                "dialect": "alphafold3",
                "version": version,
            }
            
            if make_tc_input:
                sample_af3_tc_json = {
                    "name": job_name,
                    "sequences": tc_sequences,
                    "modelSeeds": model_seeds,
                    "dialect": "alphafold3",
                    "version": version,
                }
            
            # input json paths and save json files            
            json_path_ss = Path(af3_ss_input_dir, f"{job_name}.json")
            with open(json_path_ss, "w") as f:
                json.dump(sample_af3_ss_json, f)
            
            sample_dict[input_sample_id]['af3_ss_json_paths'].append(json_path_ss)
                                
            if make_tc_input:
                json_path_tc = Path(af3_tc_input_dir, f"{job_name}.json")
                with open(json_path_tc, "w") as f:
                    json.dump(sample_af3_tc_json, f)
                sample_dict[input_sample_id]['af3_tc_json_paths'].append(json_path_tc)                          

    return sample_dict


# ============================================================================
# AF3 Inference Functions
# ============================================================================

def run_af3_single_sequence(json_path: str,
                            out_dir: str,
                            runner_path: str,                            
                            inference_config: dict = None,
                            use_subprocess: bool = False,
                            ) -> None:
    """Run AF3 single-sequence inference.
    
    Args:
        json_path: Path to the input JSON file
        out_dir: Output directory for predictions
        runner_path: Path to the AF3 runner script
        inference_config: Configuration dict containing base, ss, tc settings
        use_subprocess: If True, use subprocess (old behavior). 
                       If False, run in-process (avoids GPU exclusive mode issues)
    """
    if use_subprocess:
        # Legacy subprocess approach (may fail on GPU exclusive mode)
        sample_dir = out_dir + "/" + Path(json_path).stem
        sample_cif_files = list(Path(sample_dir).rglob("*.cif"))
        if sample_cif_files:
            print(f"AF3 prediction already exists for {Path(json_path).stem}")
            return
        else:    
            cmd = [
                sys.executable,  # Use current Python interpreter
                runner_path,
                f"--json_path={json_path}",
                f"--output_dir={out_dir}",
                f"--model_dir={inference_config.base.get('model_dir', None)}",
                "--run_data_pipeline=True",
                "--run_inference=True",
                f"--db_dir={inference_config.base.get('db_dir', None)}",
                f"--flash_attention_implementation={inference_config.base.get('flash_attention_implementation', 'triton')}",
                f"--num_recycles={inference_config.ss.get('num_recycles', 3)}",
                f"--num_diffusion_samples={inference_config.ss.get('num_diffusion_samples', 5)}",
                f"--max_templates={inference_config.ss.get('max_templates', 0)}",
                f"--ligand_protein_template_conditioning_mode={inference_config.ss.get('ligand_protein_template_conditioning_mode', 0)}",
            ]    
            env = os.environ.copy()
            subprocess.run(cmd, check=True, env=env)
    else:
        # In-process approach (avoids GPU exclusive mode issues)
        _run_af3_inprocess(
            json_path=json_path,
            out_dir=out_dir,
            runner_path=runner_path,
            inference_config=inference_config,
            mode="ss",
        )  


def run_af3_template_conditioned(json_path: str,
                            out_dir: str,
                            runner_path: str,
                            inference_config: dict = None,
                            use_subprocess: bool = False,
                            ) -> None:
    """Run AF3 template-conditioned inference.
    
    Args:
        json_path: Path to the input JSON file
        out_dir: Output directory for predictions
        runner_path: Path to the AF3 runner script
        inference_config: Configuration dict containing base, ss, tc settings
        use_subprocess: If True, use subprocess (old behavior). 
                       If False, run in-process (avoids GPU exclusive mode issues)
    """
    if use_subprocess:
        # Legacy subprocess approach (may fail on GPU exclusive mode)
        sample_dir = out_dir + "/" + Path(json_path).stem
        sample_cif_files = list(Path(sample_dir).rglob("*.cif"))
        if sample_cif_files:
            print(f"AF3 prediction already exists for {Path(json_path).stem}")
            return
        else:    
            cmd = [
                sys.executable,  # Use current Python interpreter
                runner_path,
                f"--json_path={json_path}",
                f"--output_dir={out_dir}",
                f"--model_dir={inference_config.base.get('model_dir', None)}",
                "--run_data_pipeline=True",
                "--run_inference=True",
                f"--db_dir={inference_config.base.get('db_dir', None)}",
                f"--flash_attention_implementation={inference_config.base.get('flash_attention_implementation', 'triton')}",
                f"--num_recycles={inference_config.tc.get('num_recycles', 3)}",
                f"--num_diffusion_samples={inference_config.tc.get('num_diffusion_samples', 5)}",
                f"--max_templates={inference_config.tc.get('max_templates', 1)}",
                f"--ligand_protein_template_conditioning_mode={inference_config.tc.get('ligand_protein_template_conditioning_mode', 1)}",
                f"--max_template_date={inference_config.tc.get('max_template_date', '2025-11-21')}",  # Dummy date to run template-conditioning AF3
            ]    
            env = os.environ.copy()       
            subprocess.run(cmd, check=True, env=env)
    else:
        # In-process approach (avoids GPU exclusive mode issues)
        _run_af3_inprocess(
            json_path=json_path,
            out_dir=out_dir,
            runner_path=runner_path,
            inference_config=inference_config,
            mode="tc",
        ) 


# ============================================================================
# AF3 Prediction Path Utils
# ============================================================================

def find_pred_sample_path_af3(out_dir: str = None,
                           job_name: str = None) -> tuple[list[Path], list[Path]]:
    """Find AF3 prediction sample paths for a given job name."""
    dir = Path(out_dir, job_name)
    sample_dirs = []
    sample_cif_paths = []
    for d in dir.iterdir():
        if d.is_dir():
            sample_dirs.append(d)
            cif_path = [p for p in d.glob("*.cif") if p.stem.endswith("model")][0]
            sample_cif_paths.append(cif_path)
            
    return sample_dirs, sample_cif_paths


# ============================================================================
# AF3 Evaluation Functions
# ============================================================================

def evaluate_af3_self_consistency(sample_dict: dict = None,
                                  num_redesigned_pocket_residue_list: list[int] = None,
                                  out_dir: Path = None,
                                  struct_pred_cfg: DictConfig = None,
                                  cif_parse_cfg: DictConfig = None,
                                  preprocess_cfg: DictConfig = None,
                                  featurizer_cfg: DictConfig = None,                             
                                  docking_metrics_cfg: DictConfig = None,
                                  ckpt_info: dict = None,
                                  no_wandb: bool = False,
                                  calculate_metrics_only: bool = False) -> None:
    """
    Run AF3 self-consistency and docking evaluation.
    
    Args:
        sample_id_list: List of sample IDs.
        pdb_id_list: List of PDB IDs.
        sample_atom_array_list: List of sample atom arrays.
        pdb_chain_info: PDB chain info dictionary.
        num_redesigned_pocket_residue_list: List of number of redesigned pocket residues.
        out_dir: Output directory.
        cfg: Configuration object.
        ckpt_info: Checkpoint info (optional, for wandb logging).
    """
    # Import here to avoid circular imports
    from allatom_design.eval.eval_utils.seq_des_utils import prepare_af3_prediction
    from allatom_design.eval.eval_utils.eval_metrics import (
        _compute_self_consistency_metrics_atomarray, 
        _compute_docking_metrics_atomarray
    )
            
    # Make json input directory
    af3_ss_input_dir = Path(out_dir, "af3_ss_inputs")
    af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Make a directory for af3 single-sequence prediction outputs
    af3_ss_pred_dir = Path(out_dir, "af3_ss_preds")
    af3_ss_pred_dir.mkdir(parents=True, exist_ok=True)
                    
    print("Creating AF3 JSON input files...")
    
    sample_dict = make_af3_json(
        af3_ss_input_dir=af3_ss_input_dir,
        af3_tc_input_dir=None,
        sample_dict=sample_dict,        
        metadata=None,
        json_config=struct_pred_cfg.af3.json_config
    )
    

    # Run AF3 self-consistency and docking evaluation
    af3_runner_path = struct_pred_cfg.af3.runner_path
    af3_inference_config = struct_pred_cfg.af3.inference_config
    
    designed_sample_id_to_per_pred_sc_metrics = {}
    designed_sample_id_to_per_pred_docking_metrics = {}
    
    print("\n" + "="*80)
    print("Running AF3 Self-Consistency Evaluation")
    print("="*80 + "\n")        
    
    for input_sample_id in tqdm(sample_dict.keys(), desc="AF3 predictions"):
        subsample_dict = sample_dict[input_sample_id]
                
        for dsidx, designed_sample_id in enumerate(subsample_dict['designed_sample_id']):
            # Initialize metrics dict for this designed_sample_id (with input_sample_id for reverse lookup)
            designed_sample_id_to_per_pred_sc_metrics[designed_sample_id] = {"input_sample_id": input_sample_id}
            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id] = {"input_sample_id": input_sample_id}
            
            designed_sample_atom_array = subsample_dict['designed_sample_atom_array'][dsidx]
            pdb_chain_info = subsample_dict['pdb_chain_info']
            ss_json_path = subsample_dict['af3_ss_json_paths'][dsidx]            
        
            # Get protein and ligand chain ids, because AF3 expects chain ids, not chain iids
            protein_chain_iids = pdb_chain_info['protein_chain_iids']
            ligand_chain_iids = pdb_chain_info['ligand_chain_iids']
            
            if not calculate_metrics_only:
                # Run AF3 single-sequence prediction
                try:
                    run_af3_single_sequence(str(ss_json_path), str(af3_ss_pred_dir), 
                                            runner_path=af3_runner_path, 
                                            inference_config=af3_inference_config)
                except Exception as e:
                    print(f"AF3 single sequence prediction failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}: {e}")
                    continue
                
            _, pred_ss_sample_paths = find_pred_sample_path_af3(out_dir=str(af3_ss_pred_dir), 
                                                                job_name=designed_sample_id)
            
            if len(pred_ss_sample_paths) == 0:
                print(f"No AF3 predicted structure found for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}")
                continue
        
            else:                                                              
                for pred_idx, pred_ss_sample_path in enumerate(pred_ss_sample_paths):
                    try:
                        pred_example = prepare_af3_prediction(
                            pdb_path=pred_ss_sample_path,
                            cif_parse_cfg=cif_parse_cfg,
                            preprocess_cfg=preprocess_cfg,
                            featurizer_cfg=featurizer_cfg,  
                        )
                                                                                    
                        pred_atom_array = pred_example["atom_array"]
                        per_pred_sc_metrics = _compute_self_consistency_metrics_atomarray(
                            pred_atom_array=pred_atom_array,
                            sample_atom_array=designed_sample_atom_array,
                            pred_sample_path=pred_ss_sample_path,
                            return_aligned_atom_array=False
                    )                                                                                            
            
                    except Exception as e:
                        print(f"Self-consistency metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                        continue
                    else:            
                        # Store self-consistency metrics
                        designed_sample_id_to_per_pred_sc_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics
                
                    # Only compute docking metrics if ligand exists
                    if ligand_chain_iids:
                        try: 
                            per_pred_docking_metrics = _compute_docking_metrics_atomarray(
                                pred_atom_array=pred_atom_array,
                                sample_atom_array=designed_sample_atom_array,
                                pred_sample_path=pred_ss_sample_path,
                                return_aligned_atom_array=False,
                                pocket_distance_for_metrics=docking_metrics_cfg.pocket_distance_for_metrics,
                                receptor_chain_iid=protein_chain_iids[0], #! FIXME
                                ligand_chain_iid=ligand_chain_iids[0] #! FIXME
                        )
                
                        except Exception as e:
                            print(f"Docking metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                            continue
                        else:
                            # Store docking metrics
                            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_docking_metrics
    
    # Aggregate best metrics per designed_sample_id (best diffusion sample)
    designed_sample_id_best_sc_metrics = _aggregate_best_sc_metrics_per_designed_sample(designed_sample_id_to_per_pred_sc_metrics)
    designed_sample_id_best_docking_metrics = _aggregate_best_docking_metrics_per_designed_sample(designed_sample_id_to_per_pred_docking_metrics)
    
    # Aggregate best metrics per input_sample_id (best designed sample)
    input_sample_id_best_sc_metrics = _aggregate_best_sc_metrics_per_input_sample(designed_sample_id_best_sc_metrics)
    input_sample_id_best_docking_metrics = _aggregate_best_docking_metrics_per_input_sample(designed_sample_id_best_docking_metrics)
            
    # Save results
    _save_metrics_results(
        out_dir=out_dir,
        designed_sample_id_to_per_pred_sc_metrics=designed_sample_id_to_per_pred_sc_metrics,
        designed_sample_id_to_per_pred_docking_metrics=designed_sample_id_to_per_pred_docking_metrics,
        designed_sample_id_best_sc_metrics=designed_sample_id_best_sc_metrics,
        designed_sample_id_best_docking_metrics=designed_sample_id_best_docking_metrics,
        input_sample_id_best_sc_metrics=input_sample_id_best_sc_metrics,
        input_sample_id_best_docking_metrics=input_sample_id_best_docking_metrics,
        no_wandb=no_wandb,
        ckpt_info=ckpt_info
    )
    
    print("\n" + "="*80)
    print("AF3 Self-Consistency and Docking Evaluation Complete")
    print(f"Results saved to {out_dir}")
    print("="*80 + "\n")


def _aggregate_best_sc_metrics_per_designed_sample(designed_sample_id_to_per_pred_sc_metrics: dict) -> dict:
    """
    Aggregate best self-consistency metrics per designed_sample_id (by max avg_ca_plddt across diffusion samples).
    
    Returns:
        dict: {designed_sample_id: {"input_sample_id": ..., "avg_ca_plddt": ..., "sc_ca_rmsd": ...}}
    """
    designed_sample_id_best_sc_metrics = {}        
    for designed_sample_id, per_pred_sc_metrics in designed_sample_id_to_per_pred_sc_metrics.items():
        input_sample_id = per_pred_sc_metrics.get("input_sample_id")
        
        # Filter only diffusion predictions (exclude metadata keys like "input_sample_id")
        diffusion_preds = {k: v for k, v in per_pred_sc_metrics.items() if k.startswith("diffusion_")}
        
        if not diffusion_preds:
            continue
            
        # Find the prediction with max avg_ca_plddt
        best_pred = max(diffusion_preds.values(), key=lambda x: x["avg_ca_plddt"])
        designed_sample_id_best_sc_metrics[designed_sample_id] = {
            "input_sample_id": input_sample_id,
            "avg_ca_plddt": best_pred["avg_ca_plddt"],
            "sc_ca_rmsd": best_pred["sc_ca_rmsd"]
        }
    return designed_sample_id_best_sc_metrics


def _aggregate_best_docking_metrics_per_designed_sample(designed_sample_id_to_per_pred_docking_metrics: dict) -> dict:
    """
    Aggregate best docking metrics per designed_sample_id (by max ligand_plddt across diffusion samples).
    
    Returns:
        dict: {designed_sample_id: {"input_sample_id": ..., "ligand_rmsd": ..., ...}}
    """
    designed_sample_id_best_docking_metrics = {}
    for designed_sample_id, per_pred_docking_metrics in designed_sample_id_to_per_pred_docking_metrics.items():
        input_sample_id = per_pred_docking_metrics.get("input_sample_id")
        
        # Filter only diffusion predictions with valid ligand_plddt
        diffusion_preds = {
            k: v for k, v in per_pred_docking_metrics.items() 
            if k.startswith("diffusion_") and "ligand_plddt" in v and v["ligand_plddt"] is not None
        }
        
        if not diffusion_preds:
            continue
            
        # Find the prediction with max ligand_plddt
        best_pred = max(diffusion_preds.values(), key=lambda x: x["ligand_plddt"])
        designed_sample_id_best_docking_metrics[designed_sample_id] = {
            "input_sample_id": input_sample_id,
            "ligand_rmsd": best_pred["ligand_rmsd"],
            "binding_site_rmsd": best_pred["binding_site_rmsd"],
            "ligand_plddt": best_pred["ligand_plddt"],
            "binding_site_plddt": best_pred["binding_site_plddt"],
            "iptm": best_pred["iptm"],
            "interface_min_pae": best_pred["interface_min_pae"],
        }
    return designed_sample_id_best_docking_metrics


def _aggregate_best_sc_metrics_per_input_sample(designed_sample_id_best_sc_metrics: dict) -> dict:
    """
    Aggregate best self-consistency metrics per input_sample_id (by max avg_ca_plddt across designed samples).
    
    Returns:
        dict: {input_sample_id: {"best_designed_sample_id": ..., "avg_ca_plddt": ..., "sc_ca_rmsd": ...}}
    """
    # Group by input_sample_id
    input_sample_id_to_designed_samples = defaultdict(list)
    for designed_sample_id, metrics in designed_sample_id_best_sc_metrics.items():
        input_sample_id = metrics["input_sample_id"]
        input_sample_id_to_designed_samples[input_sample_id].append((designed_sample_id, metrics))
    
    # Find best designed_sample_id per input_sample_id
    input_sample_id_best_sc_metrics = {}
    for input_sample_id, designed_samples in input_sample_id_to_designed_samples.items():
        best_designed_sample_id, best_metrics = max(designed_samples, key=lambda x: x[1]["avg_ca_plddt"])
        input_sample_id_best_sc_metrics[input_sample_id] = {
            "best_designed_sample_id": best_designed_sample_id,
            "avg_ca_plddt": best_metrics["avg_ca_plddt"],
            "sc_ca_rmsd": best_metrics["sc_ca_rmsd"]
        }
    return input_sample_id_best_sc_metrics


def _aggregate_best_docking_metrics_per_input_sample(designed_sample_id_best_docking_metrics: dict) -> dict:
    """
    Aggregate best docking metrics per input_sample_id (by max ligand_plddt across designed samples).
    
    Returns:
        dict: {input_sample_id: {"best_designed_sample_id": ..., "ligand_rmsd": ..., ...}}
    """
    # Group by input_sample_id
    input_sample_id_to_designed_samples = defaultdict(list)
    for designed_sample_id, metrics in designed_sample_id_best_docking_metrics.items():
        input_sample_id = metrics["input_sample_id"]
        input_sample_id_to_designed_samples[input_sample_id].append((designed_sample_id, metrics))
    
    # Find best designed_sample_id per input_sample_id
    input_sample_id_best_docking_metrics = {}
    for input_sample_id, designed_samples in input_sample_id_to_designed_samples.items():
        best_designed_sample_id, best_metrics = max(designed_samples, key=lambda x: x[1]["ligand_plddt"])
        input_sample_id_best_docking_metrics[input_sample_id] = {
            "best_designed_sample_id": best_designed_sample_id,
            "ligand_rmsd": best_metrics["ligand_rmsd"],
            "binding_site_rmsd": best_metrics["binding_site_rmsd"],
            "ligand_plddt": best_metrics["ligand_plddt"],
            "binding_site_plddt": best_metrics["binding_site_plddt"],
            "iptm": best_metrics["iptm"],
            "interface_min_pae": best_metrics["interface_min_pae"],
        }
    return input_sample_id_best_docking_metrics


def _save_metrics_results(out_dir: Path = None,
                          designed_sample_id_to_per_pred_sc_metrics: dict = None,
                          designed_sample_id_to_per_pred_docking_metrics: dict = None,
                          designed_sample_id_best_sc_metrics: dict = None,
                          designed_sample_id_best_docking_metrics: dict = None,
                          input_sample_id_best_sc_metrics: dict = None,
                          input_sample_id_best_docking_metrics: dict = None,
                          no_wandb: bool = False,
                          ckpt_info: dict = None) -> None:
    """Save metrics results to CSV and log to wandb."""
    
    # All self-consistency metrics per designed_sample_id (with all diffusion samples)
    all_sc_metrics_df = pd.DataFrame.from_dict(designed_sample_id_to_per_pred_sc_metrics, orient='index')
    all_sc_metrics_df = all_sc_metrics_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    all_sc_metrics_df.to_csv(Path(out_dir, "all_sc_metrics_per_designed_sample.csv"), index=False)
    
    # All docking metrics per designed_sample_id (with all diffusion samples)
    all_docking_metrics_df = pd.DataFrame.from_dict(designed_sample_id_to_per_pred_docking_metrics, orient='index')
    all_docking_metrics_df = all_docking_metrics_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    all_docking_metrics_df.to_csv(Path(out_dir, "all_docking_metrics_per_designed_sample.csv"), index=False)
    
    # Best self-consistency metrics per designed_sample_id (best diffusion sample)
    best_sc_per_designed_df = pd.DataFrame.from_dict(designed_sample_id_best_sc_metrics, orient='index')
    best_sc_per_designed_df = best_sc_per_designed_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    best_sc_per_designed_df.to_csv(Path(out_dir, "best_sc_metrics_per_designed_sample.csv"), index=False)
    
    # Best docking metrics per designed_sample_id (best diffusion sample)
    best_docking_per_designed_df = pd.DataFrame.from_dict(designed_sample_id_best_docking_metrics, orient='index')
    best_docking_per_designed_df = best_docking_per_designed_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    best_docking_per_designed_df.to_csv(Path(out_dir, "best_docking_metrics_per_designed_sample.csv"), index=False)
    
    # Best self-consistency metrics per input_sample_id (best designed sample)
    best_sc_per_input_df = pd.DataFrame.from_dict(input_sample_id_best_sc_metrics, orient='index')
    best_sc_per_input_df = best_sc_per_input_df.reset_index().rename(columns={'index': 'input_sample_id'})
    best_sc_per_input_df.to_csv(Path(out_dir, "best_sc_metrics_per_input_sample.csv"), index=False)
    
    # Best docking metrics per input_sample_id (best designed sample)
    best_docking_per_input_df = pd.DataFrame.from_dict(input_sample_id_best_docking_metrics, orient='index')
    best_docking_per_input_df = best_docking_per_input_df.reset_index().rename(columns={'index': 'input_sample_id'})
    best_docking_per_input_df.to_csv(Path(out_dir, "best_docking_metrics_per_input_sample.csv"), index=False)
    
    # Log summary metrics to wandb (using input_sample_id level for final reporting)
    if input_sample_id_best_sc_metrics:
        best_sc_ca_rmsds = [m["sc_ca_rmsd"] for m in input_sample_id_best_sc_metrics.values()]
        best_avg_ca_plddts = [m["avg_ca_plddt"] for m in input_sample_id_best_sc_metrics.values()]
        
        wandb_metrics = {                
            "eval/median/sc_ca_rmsd": np.median(best_sc_ca_rmsds),
            "eval/median/avg_ca_plddt": np.median(best_avg_ca_plddts),            
        }
        
        if ckpt_info:
            wandb_metrics["trainer/global_step"] = ckpt_info["global_step"]
            wandb_metrics["trainer/epoch"] = ckpt_info["epoch"]
        
        if not no_wandb:
            wandb.log(wandb_metrics, commit=True)
            print(f"Logged metrics to wandb: {wandb_metrics}")
    
    if input_sample_id_best_docking_metrics:
        best_ligand_rmsd = [m["ligand_rmsd"] for m in input_sample_id_best_docking_metrics.values()]
        best_binding_site_rmsd = [m["binding_site_rmsd"] for m in input_sample_id_best_docking_metrics.values()]
        best_ligand_plddt = [m["ligand_plddt"] for m in input_sample_id_best_docking_metrics.values()]
        best_binding_site_plddt = [m["binding_site_plddt"] for m in input_sample_id_best_docking_metrics.values()]
        best_iptm = [m["iptm"] for m in input_sample_id_best_docking_metrics.values()]
        best_interface_min_pae = [m["interface_min_pae"] for m in input_sample_id_best_docking_metrics.values()]
        
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
        
        if not no_wandb:
            wandb.log(wandb_metrics, commit=True)
            print(f"Logged metrics to wandb: {wandb_metrics}")


# ============================================================================
# ESMFold Utils
# ============================================================================

def create_batched_seq_dataset(all_sequences: list[str],
                               all_residue_indices: list[TensorType["n_s", int]],
                               all_chain_indices: list[TensorType["n_s", int]],
                               max_tokens_per_batch: int = 1024,
                               ) -> Generator[dict, None, None]:
    """
    Create a batched dataset of sequences for ESMFold, sorting by sequence length and limiting batch size.

    Loosely based on https://github.com/facebookresearch/esm/blob/c9c7d4f0fec964ce10c3e11dccec6c16edaa5144/scripts/fold.py#L66
    """
    # Sort by sequence length
    B = len(all_sequences)
    examples = [(seq, residx, chain_idx, id) for seq, residx, chain_idx, id in zip(all_sequences, all_residue_indices, all_chain_indices, range(B))]
    examples = sorted(examples, key=lambda x: len(x[0]))

    # Define collator
    def collate_fn(examples: list[tuple[str, TensorType["n", int], int]]) -> dict[str, List]:
        """
        Given a list of examples, collate them into a batch with keys:
        - sequence: (b) sequence
        - residue_index: (b n) residue index
        - id: (b) unique identifier for each sequence
        """
        batch = {"sequence": [], "residue_index": [], "id": [], "chain_index": []}

        N = max(len(seq) for seq, _, _, _ in examples)
        for seq, residx, chain_idx, id in examples:
            batch["sequence"].append(seq)
            batch["residue_index"].append(data.make_fixed_size_1d(residx, fixed_size=N, start_idx=None))
            batch["chain_index"].append(data.make_fixed_size_1d(chain_idx, fixed_size=N, start_idx=None))
            batch["id"].append(id)

        batch["residue_index"] = torch.stack(batch["residue_index"], dim=0).to(torch.long)
        batch["chain_index"] = torch.stack(batch["chain_index"], dim=0).to(torch.long)
        return batch

    # Yield batches
    batch_examples, num_tokens = [], 0

    total_tokens = sum(len(seq) for seq in all_sequences)
    pbar = tqdm(total=total_tokens, desc="Number of ESMFold tokens processed", leave=False)

    for seq, residx, chain_idx, id in examples:
        # If adding this sequence would exceed the token limit, yield the current batch
        if num_tokens + len(seq) > max_tokens_per_batch and num_tokens > 0:
            yield collate_fn(batch_examples)
            batch_examples, num_tokens = [], 0

        # Add this sequence to the current batch
        batch_examples.append((seq, residx, chain_idx, id))
        num_tokens += len(seq)
        pbar.update(len(seq))

    yield collate_fn(batch_examples)


# ============================================================================
# Legacy Code (AF2/ESMFold - commented out)
# ============================================================================

# def run_af2(sequences_list: list[str],
#             residue_index_list: list[TensorType["n_s", int]],
#             chain_index_list: list[TensorType["n_s", int]],
#             pdbs: list[str],  # used for extracting residue index. TODO remove dependence on pdb file
#             af_model: "mk_af_model",
#             out_dir: str,
#             num_models: int,
#             sample_models: bool,
#             num_recycles: int,
#             save_best: bool = True,
#             rm_template_interchain: bool = False,
#             chains: str | None = None,
#             **kwargs) -> tuple[dict[str, torch.Tensor], list[str]]:
#     """
#     Predict sequences with AlphaFold2.

#     Return a tuple (dictionary of outputs, output filenames).
#     """
#     Path(out_dir).mkdir(exist_ok=True, parents=True)
#     output_files = []

#     # Predict structures
#     for _, (seq, pdb, residue_index, chain_index) in enumerate(zip(sequences_list, pdbs, residue_index_list, chain_index_list)):
#         output_pdb = f"{out_dir}/af2_{Path(pdb).stem}.pdb"
#         assert len(chain_index_list[0].unique()) == 1, "Multi-chain prediction not supported yet"
#         # af_model.prep_inputs(pdb, chains, ignore_missing=False)
#         _prep_struct_pred(af_model, residue_index)

#         af_model.restart()
#         af_model.set_opt("template", rm_ic=rm_template_interchain)
#         af_model.predict(seq=seq,
#                          num_models=num_models,
#                          sample_models=sample_models,
#                          num_recycles=num_recycles,
#                          verbose=False)

#         af_model._save_results(save_best=save_best, best_metric="plddt", verbose=False)

#         if save_best:
#             save_best_model(af_model, output_pdb)
#         else:
#             af_model.save_current_pdb(output_pdb)

#         output_files.append(output_pdb)

#     preds = [data.load_feats_from_pdb(pdb) for pdb in output_files]

#     # Preprocess plddt-CA
#     plddt = [pred["b_factors"] for pred in preds]
#     ca_plddt = [pred["b_factors"][:, 1] for pred in preds]
#     avg_ca_plddt = [torch.mean(ca_plddt, dim=0, keepdim=True) for ca_plddt in ca_plddt]  # keep sequence dim for consistency

#     # Prepare AF2 outputs
#     af2_outputs = {
#         "pred_coords": [pred["all_atom_positions"] for pred in preds],
#         "plddt": plddt,
#         "ca_plddt": ca_plddt,
#         "seq_mask": [pred["seq_mask"] for pred in preds],
#         "aatype": [pred["aatype"] for pred in preds],
#         "residue_index": [pred["residue_index"].long() for pred in preds],
#         "avg_ca_plddt": avg_ca_plddt,
#         "atom_mask": [pred["all_atom_mask"] for pred in preds],
#     }

#     return af2_outputs, output_files


# def get_esmfold_model(device: str):
#     # Set up ESMFold
#     esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
#     esmfold.esm = esmfold.esm.half()
#     esmfold = esmfold.to(device)
#     tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
#     return esmfold, tokenizer


# def get_struct_pred_model(cfg: DictConfig,
#                           device: str) -> dict[str, Any]:
#     """
#     Get structure prediction model components as a dictionary based on config.

#     Example config:
#     struct_pred_cfg:
#         model_name: "esmfold"  # ["esmfold", "boltz1"]
#         boltz1:
#         esmfold:
#             max_tokens_per_batch: 1024
#         af2_interface:
#             data_dir: # directory containing "params/" with af2 model params
#             num_models: 1
#             num_recycles: 3
#             use_multimer: false
#     """
#     model_name = cfg.model_name
#     base_cfg = OmegaConf.load(cfg.base_cfg)
#     cfg = OmegaConf.merge(base_cfg, cfg)

#     struct_pred_model = {"model_name": model_name, "cfg": cfg, "device": device}
#     if model_name == "boltz1":
#         struct_pred_model["boltz1"] = get_boltz_model(cfg.boltz1, device=device)
#         struct_pred_model["trainer_fn"] = partial(make_boltz_trainer,
#                                                   num_workers=cfg.boltz1.num_workers)
#         struct_pred_model["data_cfg"] = hydra.utils.instantiate(cfg.boltz1.data_cfg)

#     elif model_name == "esmfold":
#         esmfold, tokenizer = get_esmfold_model(device=device)
#         struct_pred_model["esmfold"] = esmfold
#         struct_pred_model["tokenizer"] = tokenizer
#         struct_pred_model["data_cfg"] = hydra.utils.instantiate(cfg.boltz1.data_cfg)  # useful to have boltz tokenizer/featurizer
#     elif model_name == "af2_interface":
#         clear_mem()
#         af2_cfg = cfg.af2_interface

#         # get AF2 model for predicting complex
#         if af2_cfg.hard_target:
#             complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=af2_cfg.num_recycles, data_dir=af2_cfg.data_dir,
#                                                          use_multimer=False, use_initial_guess=True, use_initial_atom_pos=False)
#         else:
#             complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=af2_cfg.num_recycles, data_dir=af2_cfg.data_dir,
#                                                          use_multimer=False, use_initial_guess=False, use_initial_atom_pos=False)

#         # get AF2 model for predicting binder in isolation
#         af_model = mk_af_model(use_multimer=False,
#                                use_templates=False,
#                                best_metric="ptm",
#                                data_dir=af2_cfg.data_dir)

#         struct_pred_model["af_model_complex"] = complex_prediction_model
#         struct_pred_model["af_model_binder"] = af_model
#     else:
#         raise ValueError(f"Invalid model name: {model_name}")

#     return struct_pred_model
