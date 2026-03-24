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

# Global caches for AF3 runner module, ModelRunner, and DataPipelineConfig
_AF3_RUNNER_MOD = None
_AF3_MODEL_RUNNER = None
_AF3_DATA_PIPELINE_CONFIG = None
_AF3_MAX_TEMPLATE_DATE = None
_AF3_BUCKETS = None


def _load_af3_runner(runner_path: str):
    """
    Load run_alphafold.py as a module dynamically.
    Cached after first load.
    """
    global _AF3_RUNNER_MOD
    if _AF3_RUNNER_MOD is not None:
        return _AF3_RUNNER_MOD
    
    spec = importlib.util.spec_from_file_location("af3_runner", runner_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _AF3_RUNNER_MOD = mod
    return mod


def _get_af3_model_runner_and_config(
    runner_path: str,
    inference_config: dict,
    mode: str = "ss",
):
    """
    Get or create cached AF3 ModelRunner and DataPipelineConfig.
    ModelRunner and DataPipelineConfig are created once and reused across calls.
    """
    import datetime as dt
    import pathlib
    import jax
    
    global _AF3_MODEL_RUNNER, _AF3_DATA_PIPELINE_CONFIG
    global _AF3_MAX_TEMPLATE_DATE, _AF3_BUCKETS
    
    runner = _load_af3_runner(runner_path)
    base_config = inference_config.get('base', {})
    mode_config = inference_config.get(mode, {})
    
    if _AF3_MODEL_RUNNER is None:
        torch.cuda.empty_cache()
        
        import shutil
        from alphafold3.jax.attention import attention
        from alphafold3.data import pipeline
        import typing
        
        flash_attn = base_config.get('flash_attention_implementation', 'triton')
        
        devices = jax.local_devices(backend='gpu')
        print(f'[AF3 init] Found devices: {devices}, using device 0: {devices[0]}')
        
        model_config = runner.make_model_config(
            flash_attention_implementation=typing.cast(attention.Implementation, flash_attn),
            num_diffusion_samples=mode_config.get('num_diffusion_samples', 5),
            num_recycles=mode_config.get('num_recycles', 3),
            return_embeddings=False,
            return_distogram=False,
            ligand_protein_template_conditioning_mode=mode_config.get('ligand_protein_template_conditioning_mode', 0),
            mask_template_sidechains=mode_config.get('mask_template_sidechains', True),
            mask_template_sequence=mode_config.get('mask_template_sequence', True),
        )
        
        _AF3_MODEL_RUNNER = runner.ModelRunner(
            config=model_config,
            device=devices[0],
            model_dir=pathlib.Path(base_config.get('model_dir', '')),
        )
        print('[AF3 init] Loading model parameters...')
        _ = _AF3_MODEL_RUNNER.model_params
        print('[AF3 init] Model parameters loaded and cached.')
        
        max_template_date_str = mode_config.get('max_template_date', '2021-09-30')
        _AF3_MAX_TEMPLATE_DATE = dt.date.fromisoformat(max_template_date_str)
        
        buckets_list = [256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]
        _AF3_BUCKETS = tuple(buckets_list)
        
        db_dir = base_config.get('db_dir', '')
        expand_path = lambda x: runner.replace_db_dir(x, [db_dir])
        _AF3_DATA_PIPELINE_CONFIG = pipeline.DataPipelineConfig(
            jackhmmer_binary_path=shutil.which('jackhmmer'),
            nhmmer_binary_path=shutil.which('nhmmer'),
            hmmalign_binary_path=shutil.which('hmmalign'),
            hmmsearch_binary_path=shutil.which('hmmsearch'),
            hmmbuild_binary_path=shutil.which('hmmbuild'),
            small_bfd_database_path=expand_path('${DB_DIR}/bfd-first_non_consensus_sequences.fasta'),
            mgnify_database_path=expand_path('${DB_DIR}/mgy_clusters_2022_05.fa'),
            uniprot_cluster_annot_database_path=expand_path('${DB_DIR}/uniprot_all_2021_04.fa'),
            uniref90_database_path=expand_path('${DB_DIR}/uniref90_2022_05.fa'),
            ntrna_database_path=expand_path('${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta'),
            rfam_database_path=expand_path('${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta'),
            rna_central_database_path=expand_path('${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta'),
            pdb_database_path=expand_path('${DB_DIR}/mmcif_files'),
            seqres_database_path=expand_path('${DB_DIR}/pdb_seqres_2022_09_28.fasta'),
            max_template_date=_AF3_MAX_TEMPLATE_DATE,
        )
        print('[AF3 init] DataPipelineConfig created and cached.')
    
    return runner, _AF3_MODEL_RUNNER, _AF3_DATA_PIPELINE_CONFIG


def _run_af3_inprocess(
    json_path: str,
    out_dir: str,
    runner_path: str,
    inference_config: dict,
    mode: str = "ss",
) -> None:
    """
    Run AF3 in-process without subprocess, reusing a cached ModelRunner.
    This avoids GPU exclusive mode issues and prevents GPU memory accumulation
    from repeated model loading.
    """
    import pathlib
    from alphafold3.common import folding_input
    
    sample_dir = Path(out_dir) / Path(json_path).stem
    sample_cif_files = list(sample_dir.rglob("*.cif"))
    if sample_cif_files:
        print(f"AF3 prediction already exists for {Path(json_path).stem}")
        return
    
    runner, model_runner, data_pipeline_config = _get_af3_model_runner_and_config(
        runner_path=runner_path,
        inference_config=inference_config,
        mode=mode,
    )
    
    mode_config = inference_config.get(mode, {})
    
    fold_inputs = folding_input.load_fold_inputs_from_path(pathlib.Path(json_path))
    
    for fold_input_item in fold_inputs:
        output_dir = os.path.join(out_dir, fold_input_item.sanitised_name())
        try:
            runner.process_fold_input(
                fold_input=fold_input_item,
                data_pipeline_config=data_pipeline_config,
                model_runner=model_runner,
                output_dir=output_dir,
                buckets=_AF3_BUCKETS,
                ref_max_modified_date=_AF3_MAX_TEMPLATE_DATE,
                conformer_max_iterations=None,
                resolve_msa_overlaps=True,
                max_templates=mode_config.get('max_templates', 0),
                ligand_protein_template_conditioning_mode=mode_config.get('ligand_protein_template_conditioning_mode', 0),
                force_output_dir=True,
            )
        except SystemExit as e:
            if e.code != 0 and e.code is not None:
                raise RuntimeError(f"AF3 process_fold_input exited with code {e.code}")
        except Exception as e:
            print(f"AF3 prediction failed for {Path(json_path).stem}: {e}")
            raise

# ============================================================================
# AF3 JSON Input Creation
# ============================================================================

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

    Todo: need to split pn_unit_iids into separate chain iids to make af3 inputs for multi-chain ligands later
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
            pdb_chain_info[pdb_key]['protein_pn_unit_iids'] = []
            pdb_chain_info[pdb_key]['ligand_pn_unit_iids'] = []       
            pdb_chain_info[pdb_key]['ligand_ccd_codes'] = []
            
            for column in expanded_protein_columns:
                if row[column]:
                    suffix = column.split("_")[-1]
                    protein_pn_unit_iid = row[f'q_pn_unit_iid_{suffix}']
                    pdb_chain_info[pdb_key]['protein_pn_unit_iids'].append(protein_pn_unit_iid)
                    
            for column in expanded_nonpolymer_ligand_columns:
                if row[column]:
                    suffix = column.split("_")[-1]
                    ligand_pn_unit_iid = row[f'q_pn_unit_iid_{suffix}']
                    ligand_ccd_code = row[f'q_pn_unit_non_polymer_res_names_{suffix}']
                    pdb_chain_info[pdb_key]['ligand_pn_unit_iids'].append(ligand_pn_unit_iid)
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
                                                            
            protein_pn_unit_iids = pdb_chain_info['protein_pn_unit_iids']        
            ligand_pn_unit_iids = pdb_chain_info['ligand_pn_unit_iids']
            ligand_ccd_codes = pdb_chain_info['ligand_ccd_codes']
            
            ss_sequences = []
            tc_sequences = []
            for protein_pn_unit_iid in protein_pn_unit_iids:
                chain_mask = (designed_sample_atom_array.pn_unit_iid == protein_pn_unit_iid)
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
                        "id": protein_pn_unit_iid.split("_")[0], 
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
                            "id": protein_pn_unit_iid.split("_")[0],
                            "sequence": sequence_with_gaps, 
                            "modifications": modifications if modifications else [],
                            "unpairedMsa": "",
                            "pairedMsa": "",
                            "templates": [
                                {
                                    "mmcifPath": template_sample_path,
                                    "queryIndices": query_indices,
                                    "templateIndices": template_indices,
                                    "templateChainId": protein_pn_unit_iid.split("_")[0],
                                }
                            ]
                        }
                    })                
            
            
            for ligand_pn_unit_iid, ligand_ccd_code in zip(ligand_pn_unit_iids, ligand_ccd_codes):                    
                ss_sequences.append({
                    "ligand": {
                        "id": ligand_pn_unit_iid.split("_")[0],
                        "ccdCodes": [ligand_ccd_code]
                    }
                })
                
                if make_tc_input:
                    tc_sequences.append({
                        "ligand": {
                            "id": ligand_pn_unit_iid.split("_")[0],
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
                f"--mask_template_sidechains={inference_config.tc.get('mask_template_sidechains', True)}",
                f"--mask_template_sequence={inference_config.tc.get('mask_template_sequence', True)}",
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
                                  out_dir: Path = None,
                                  struct_pred_cfg: DictConfig = None,
                                  cif_parse_cfg: DictConfig = None,
                                  preprocess_cfg: DictConfig = None,
                                  featurizer_cfg: DictConfig = None,
                                  pocket_cfg: DictConfig = None,
                                  ckpt_info: dict = None,
                                  no_wandb: bool = False,
                                  calculate_metrics_only: bool = False,
                                  csv_suffix: str = "",
                                  input_sample_is_designed: bool = True) -> None:
    """
    Run AF3 self-consistency and docking evaluation.
    
    Args:
        sample_id_list: List of sample IDs.
        pdb_id_list: List of PDB IDs.
        sample_atom_array_list: List of sample atom arrays.
        pdb_chain_info: PDB chain info dictionary.
        out_dir: Output directory.
        cfg: Configuration object.
        ckpt_info: Checkpoint info (optional, for wandb logging).
    """
    # Import here to avoid circular imports
    from allatom_design.eval.eval_utils.sd_data_utils import prepare_af3_prediction
    from allatom_design.eval.eval_utils.eval_metrics import (
        compute_self_consistency_metrics_atomarray, 
        compute_docking_metrics_atomarray
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
            protein_pn_unit_iids = pdb_chain_info['protein_pn_unit_iids']
            ligand_pn_unit_iids = pdb_chain_info['ligand_pn_unit_iids']
            
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

            for pred_idx, pred_ss_sample_path in enumerate(pred_ss_sample_paths):
                try:
                    pred_example = prepare_af3_prediction(
                        pdb_path=pred_ss_sample_path,  
                        cif_parse_cfg=cif_parse_cfg,                      
                        preprocess_cfg=preprocess_cfg,
                        featurizer_cfg=featurizer_cfg,  
                    )
                                                                                
                    pred_atom_array = pred_example["atom_array"]
                    per_pred_sc_metrics = compute_self_consistency_metrics_atomarray(
                        pred_atom_array=pred_atom_array,
                        sample_atom_array=designed_sample_atom_array,
                        pred_sample_path=pred_ss_sample_path,                        
                    )                                                                                            
        
                except Exception as e:
                    print(f"Self-consistency metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                    continue
                else:            
                    designed_sample_id_to_per_pred_sc_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics
            
                if ligand_pn_unit_iids:
                    try: 
                        per_pred_docking_metrics = compute_docking_metrics_atomarray(
                            pred_atom_array=pred_atom_array,
                            sample_atom_array=designed_sample_atom_array,
                            pred_sample_path=pred_ss_sample_path,                            
                            pocket_distance_for_docking_metrics=pocket_cfg.pocket_distance_for_docking_metrics,
                            receptor_pn_unit_iids=protein_pn_unit_iids,
                            ligand_pn_unit_iids=ligand_pn_unit_iids,
                            ref_sample_is_designed=input_sample_is_designed,
                        )

                    except Exception as e:
                        print(f"Docking metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                        continue
                    else:
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
        ckpt_info=ckpt_info,
        csv_suffix=csv_suffix
    )
    
    print("\n" + "="*80)
    print("AF3 Self-Consistency and Docking Evaluation Complete")
    print(f"Results saved to {out_dir}")
    print("="*80 + "\n")


def evaluate_af3_docking_consistency(sample_dict: dict = None,
                                     out_dir: Path = None,
                                     struct_pred_cfg: DictConfig = None,
                                     cif_parse_cfg: DictConfig = None,
                                     preprocess_cfg: DictConfig = None,
                                     featurizer_cfg: DictConfig = None,
                                     pocket_cfg: DictConfig = None,
                                     ckpt_info: dict = None,
                                     no_wandb: bool = False,
                                     calculate_metrics_only: bool = False,
                                     csv_suffix: str = "",
                                     input_sample_is_designed: bool = True) -> None:
    """
    Run AF3 template-conditioned docking consistency evaluation.
    
    Uses template-conditioned AF3 predictions (designed backbone as template)
    to evaluate both self-consistency and docking metrics against the designed sample.
    
    Args:
        sample_dict: Dictionary of sample data (must contain 'designed_sample_path_for_af3_tc').
        out_dir: Output directory.
        struct_pred_cfg: Structure prediction configuration.
        cif_parse_cfg: CIF parsing configuration for AF3 predictions.
        preprocess_cfg: Preprocessing configuration for AF3 predictions.
        featurizer_cfg: Featurizer configuration for AF3 predictions.
        pocket_cfg: Pocket configuration for docking metrics.
        ckpt_info: Checkpoint info (optional, for wandb logging).
        no_wandb: If True, disable wandb logging.
        calculate_metrics_only: If True, skip AF3 prediction and only compute metrics.
        csv_suffix: Optional suffix for CSV filenames (e.g. "_array_0" for array jobs).
    """
    # Import here to avoid circular imports
    from allatom_design.eval.eval_utils.sd_data_utils import prepare_af3_prediction
    from allatom_design.eval.eval_utils.eval_metrics import (
        _compute_self_consistency_metrics_atomarray, 
        _compute_docking_metrics_atomarray
    )
            
    # Make JSON input directories
    af3_ss_input_dir = Path(out_dir, "af3_ss_inputs")  # needed by make_af3_json (always creates SS JSONs)
    af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
    af3_tc_input_dir = Path(out_dir, "af3_tc_inputs")
    af3_tc_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Make a directory for af3 template-conditioned prediction outputs
    af3_tc_pred_dir = Path(out_dir, "af3_tc_preds")
    af3_tc_pred_dir.mkdir(parents=True, exist_ok=True)
                    
    print("Creating AF3 JSON input files (template-conditioned)...")
    
    sample_dict = make_af3_json(
        af3_ss_input_dir=af3_ss_input_dir,
        af3_tc_input_dir=af3_tc_input_dir,
        sample_dict=sample_dict,        
        metadata=None,
        json_config=struct_pred_cfg.af3.json_config,
        make_tc_input=True,
    )
    

    # Run AF3 template-conditioned docking evaluation
    af3_runner_path = struct_pred_cfg.af3.runner_path
    af3_inference_config = struct_pred_cfg.af3.inference_config
    
    designed_sample_id_to_per_pred_sc_metrics = {}
    designed_sample_id_to_per_pred_docking_metrics = {}
    
    print("\n" + "="*80)
    print("Running AF3 Docking Consistency Evaluation (Template-Conditioned)")
    print("="*80 + "\n")        
    
    for input_sample_id in tqdm(sample_dict.keys(), desc="AF3 TC predictions"):
        subsample_dict = sample_dict[input_sample_id]
                
        for dsidx, designed_sample_id in enumerate(subsample_dict['designed_sample_id']):
            # Initialize metrics dict for this designed_sample_id (with input_sample_id for reverse lookup)
            designed_sample_id_to_per_pred_sc_metrics[designed_sample_id] = {"input_sample_id": input_sample_id}
            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id] = {"input_sample_id": input_sample_id}
            
            designed_sample_atom_array = subsample_dict['designed_sample_atom_array'][dsidx]
            pdb_chain_info = subsample_dict['pdb_chain_info']
            tc_json_path = subsample_dict['af3_tc_json_paths'][dsidx]            
        
            # Get protein and ligand chain ids
            protein_pn_unit_iids = pdb_chain_info['protein_pn_unit_iids']
            ligand_pn_unit_iids = pdb_chain_info['ligand_pn_unit_iids']
            
            if not calculate_metrics_only:
                # Run AF3 template-conditioned prediction
                try:
                    run_af3_template_conditioned(str(tc_json_path), str(af3_tc_pred_dir), 
                                            runner_path=af3_runner_path, 
                                            inference_config=af3_inference_config)
                except Exception as e:
                    print(f"AF3 template-conditioned prediction failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}: {e}")
                    continue
                
            _, pred_tc_sample_paths = find_pred_sample_path_af3(out_dir=str(af3_tc_pred_dir), 
                                                                job_name=designed_sample_id)
            
            if len(pred_tc_sample_paths) == 0:
                print(f"No AF3 TC predicted structure found for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}")
                continue
        
            else:                                                              
                for pred_idx, pred_tc_sample_path in enumerate(pred_tc_sample_paths):
                    try:
                        pred_example = prepare_af3_prediction(
                            pdb_path=pred_tc_sample_path,                            
                            preprocess_cfg=preprocess_cfg,
                            featurizer_cfg=featurizer_cfg,  
                        )
                                                                                    
                        pred_atom_array = pred_example["atom_array"]
                        per_pred_sc_metrics = _compute_self_consistency_metrics_atomarray(
                            pred_atom_array=pred_atom_array,
                            sample_atom_array=designed_sample_atom_array,
                            pred_sample_path=pred_tc_sample_path,
                            return_aligned_atom_array=False
                    )                                                                                            
            
                    except Exception as e:
                        print(f"Self-consistency metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                        continue
                    else:            
                        # Store self-consistency metrics
                        designed_sample_id_to_per_pred_sc_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics
                
                    # Only compute docking metrics if ligand exists
                    if ligand_pn_unit_iids:
                        try: 
                            per_pred_docking_metrics = _compute_docking_metrics_atomarray(
                                pred_atom_array=pred_atom_array,
                                sample_atom_array=designed_sample_atom_array,
                                pred_sample_path=pred_tc_sample_path,
                                return_aligned_atom_array=False,
                                pocket_distance_for_docking_metrics=pocket_cfg.pocket_distance_for_docking_metrics,
                                receptor_pn_unit_iids=protein_pn_unit_iids,
                                ligand_pn_unit_iids=ligand_pn_unit_iids,
                                ref_sample_is_designed=input_sample_is_designed,
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
            
    # Save results with "tc_" prefix to distinguish from self-consistency (SS) results
    _save_metrics_results(
        out_dir=out_dir,
        designed_sample_id_to_per_pred_sc_metrics=designed_sample_id_to_per_pred_sc_metrics,
        designed_sample_id_to_per_pred_docking_metrics=designed_sample_id_to_per_pred_docking_metrics,
        designed_sample_id_best_sc_metrics=designed_sample_id_best_sc_metrics,
        designed_sample_id_best_docking_metrics=designed_sample_id_best_docking_metrics,
        input_sample_id_best_sc_metrics=input_sample_id_best_sc_metrics,
        input_sample_id_best_docking_metrics=input_sample_id_best_docking_metrics,
        no_wandb=no_wandb,
        ckpt_info=ckpt_info,
        csv_suffix=csv_suffix,
        mode_prefix="tc_"
    )
    
    print("\n" + "="*80)
    print("AF3 Docking Consistency Evaluation (Template-Conditioned) Complete")
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
                          ckpt_info: dict = None,
                          csv_suffix: str = "",
                          mode_prefix: str = "") -> None:
    """Save metrics results to CSV and log to wandb.
    
    Args:
        csv_suffix: Optional suffix for CSV filenames (e.g. "_array_0" for array jobs).
        mode_prefix: Optional prefix for CSV filenames and wandb keys to distinguish
                     between SS and TC results (e.g. "tc_" for template-conditioned).
    """
    
    # All self-consistency metrics per designed_sample_id (with all diffusion samples)
    all_sc_metrics_df = pd.DataFrame.from_dict(designed_sample_id_to_per_pred_sc_metrics, orient='index')
    all_sc_metrics_df = all_sc_metrics_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    all_sc_metrics_df.to_csv(Path(out_dir, f"{mode_prefix}all_sc_metrics_per_designed_sample{csv_suffix}.csv"), index=False)
    
    # All docking metrics per designed_sample_id (with all diffusion samples)
    all_docking_metrics_df = pd.DataFrame.from_dict(designed_sample_id_to_per_pred_docking_metrics, orient='index')
    all_docking_metrics_df = all_docking_metrics_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    all_docking_metrics_df.to_csv(Path(out_dir, f"{mode_prefix}all_docking_metrics_per_designed_sample{csv_suffix}.csv"), index=False)
    
    # Best self-consistency metrics per designed_sample_id (best diffusion sample)
    best_sc_per_designed_df = pd.DataFrame.from_dict(designed_sample_id_best_sc_metrics, orient='index')
    best_sc_per_designed_df = best_sc_per_designed_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    best_sc_per_designed_df.to_csv(Path(out_dir, f"{mode_prefix}best_sc_metrics_per_designed_sample{csv_suffix}.csv"), index=False)
    
    # Best docking metrics per designed_sample_id (best diffusion sample)
    best_docking_per_designed_df = pd.DataFrame.from_dict(designed_sample_id_best_docking_metrics, orient='index')
    best_docking_per_designed_df = best_docking_per_designed_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    best_docking_per_designed_df.to_csv(Path(out_dir, f"{mode_prefix}best_docking_metrics_per_designed_sample{csv_suffix}.csv"), index=False)
    
    # Best self-consistency metrics per input_sample_id (best designed sample)
    best_sc_per_input_df = pd.DataFrame.from_dict(input_sample_id_best_sc_metrics, orient='index')
    best_sc_per_input_df = best_sc_per_input_df.reset_index().rename(columns={'index': 'input_sample_id'})
    best_sc_per_input_df.to_csv(Path(out_dir, f"{mode_prefix}best_sc_metrics_per_input_sample{csv_suffix}.csv"), index=False)
    
    # Best docking metrics per input_sample_id (best designed sample)
    best_docking_per_input_df = pd.DataFrame.from_dict(input_sample_id_best_docking_metrics, orient='index')
    best_docking_per_input_df = best_docking_per_input_df.reset_index().rename(columns={'index': 'input_sample_id'})
    best_docking_per_input_df.to_csv(Path(out_dir, f"{mode_prefix}best_docking_metrics_per_input_sample{csv_suffix}.csv"), index=False)
    
    # Log summary metrics to wandb (using input_sample_id level for final reporting)
    if input_sample_id_best_sc_metrics:
        best_sc_ca_rmsds = [m["sc_ca_rmsd"] for m in input_sample_id_best_sc_metrics.values()]
        best_avg_ca_plddts = [m["avg_ca_plddt"] for m in input_sample_id_best_sc_metrics.values()]
        
        wandb_metrics = {                
            f"eval/median/{mode_prefix}sc_ca_rmsd": np.median(best_sc_ca_rmsds),
            f"eval/median/{mode_prefix}avg_ca_plddt": np.median(best_avg_ca_plddts),            
        }
        
        if ckpt_info:
            wandb_metrics["trainer/global_step"] = ckpt_info["global_step"]
            wandb_metrics["trainer/epoch"] = ckpt_info["epoch"]
        
        if not no_wandb:
            wandb.log(wandb_metrics, commit=True)
            print(f"Logged metrics to wandb: {wandb_metrics}")
    
    if input_sample_id_best_docking_metrics:
        best_ligand_rmsd = [m["ligand_rmsd"] for m in input_sample_id_best_docking_metrics.values() if m["ligand_rmsd"] is not None]
        best_binding_site_rmsd = [m["binding_site_rmsd"] for m in input_sample_id_best_docking_metrics.values() if m["binding_site_rmsd"] is not None]
        best_ligand_plddt = [m["ligand_plddt"] for m in input_sample_id_best_docking_metrics.values() if m["ligand_plddt"] is not None]
        best_binding_site_plddt = [m["binding_site_plddt"] for m in input_sample_id_best_docking_metrics.values() if m["binding_site_plddt"] is not None]
        best_iptm = [m["iptm"] for m in input_sample_id_best_docking_metrics.values() if m["iptm"] is not None]
        best_interface_min_pae = [m["interface_min_pae"] for m in input_sample_id_best_docking_metrics.values() if m["interface_min_pae"] is not None]
        
        wandb_metrics = {                
            f"eval/median/{mode_prefix}ligand_rmsd": np.median(best_ligand_rmsd) if best_ligand_rmsd else None,
            f"eval/median/{mode_prefix}binding_site_rmsd": np.median(best_binding_site_rmsd) if best_binding_site_rmsd else None,
            f"eval/median/{mode_prefix}ligand_plddt": np.median(best_ligand_plddt) if best_ligand_plddt else None,
            f"eval/median/{mode_prefix}binding_site_plddt": np.median(best_binding_site_plddt) if best_binding_site_plddt else None,
            f"eval/median/{mode_prefix}iptm": np.median(best_iptm) if best_iptm else None,
            f"eval/median/{mode_prefix}interface_min_pae": np.median(best_interface_min_pae) if best_interface_min_pae else None,
        }
        
        if ckpt_info:
            wandb_metrics["trainer/global_step"] = ckpt_info["global_step"]
            wandb_metrics["trainer/epoch"] = ckpt_info["epoch"]
        
        if not no_wandb:
            wandb.log(wandb_metrics, commit=True)
            print(f"Logged metrics to wandb: {wandb_metrics}")


