from pathlib import Path

from biotite.structure import AtomArray
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import torch
from tqdm import tqdm
import wandb
import yaml

from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_cached_example_files, get_pdb_files, get_training_checkpoints, wandb_setup)
from allatom_design.eval.eval_utils.seq_des_utils import (
    get_sd_example, 
    load_example_with_load_any,                                                          
    prepare_af3_prediction, 
    prepare_designed_sample, 
    run_lc_seq_des,
    get_seq_des_model,
    _indices_to_pos_string,
    # Sample Dict / Data Preparation
    create_sample_dict,
    prepare_samples,
    _extract_ligand_atom_array_from_cached_examples,
    load_designed_samples,
    # Pocket Sequence Utilities
    extract_ligand_from_structure,
    _replace_pocket_sequence_with_native,
    create_pos_constraint_dict_from_atom_array,
    # Redesign Functions
    redesign_with_native,
    redesign_with_lcaliby,
)
from allatom_design.eval.eval_utils.folding_utils import (
    run_af3_single_sequence,
    find_pred_sample_path_af3,
    make_af3_json,
    evaluate_af3_self_consistency,
)




###########################################################
# Main
###########################################################

@hydra.main(config_path="../../configs/eval/sampling", config_name="lc_seq_des", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Redesign sequence using native sequence or lcaliby.
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
    if cfg.metadata_path is not None:
        metadata = pd.read_parquet(cfg.metadata_path)
    else:
        metadata = None
        
    # Load pos_constraint_df
    if cfg.pos_constraint_csv is not None:
        pos_constraint_df = pd.read_csv(cfg.pos_constraint_csv)
    else:
        pos_constraint_df = None
    
    ###########################################################
    # Phase 1: Prepare samples (common)
    ###########################################################
    print("\n" + "="*80)
    print("Phase 1: Preparing samples")
    print("="*80 + "\n")
    
    sample_dict = prepare_samples(cfg = cfg, 
                                  metadata = metadata)
    
    # Make a directory for saving original structures with ligand
    processed_original_samples_dir = log_dir / "original_samples"
    
    # Load designed structures and combine with ligand if needed
    print("Loading designed structures and combining with ligand...")
    
    sample_dict = load_designed_samples(
        sample_dict=sample_dict, 
        cif_parse_cfg=cfg.cif_cfg.parse.designed_samples,
        preprocess_cfg=cfg.preprocess_cfg.designed_samples,
        featurizer_cfg=cfg.featurizer_cfg.prepare_designed_samples,
        add_ligands_to_designed_samples=cfg.ligand_source_cfg.get("add_ligands_to_designed_samples", False),
        is_all_atom_sample=cfg.get("is_all_atom_sample", False),
        save_dir=processed_original_samples_dir
    )
    
    ###########################################################
    # Phase 2: Design sequence (redesign partial sequence or full design)
    ###########################################################
    print("\n" + "="*80)
    print("Phase 2: Redesigning pocket sequence")
    print("="*80 + "\n")
        
    if cfg.redesign_cfg.get("use_native_seq", False):
        # Native replace mode
        redesign_out_dir = log_dir / "redesigned_samples"
        sample_dict = redesign_with_native(sample_dict, cfg, redesign_out_dir)
        
        # Evaluate if needed
        if cfg.struct_pred_cfg.evaluate_self_consistency:
            print("\n" + "="*80)
            print("Phase 3: AF3 Evaluation")
            print("="*80 + "\n")
            evaluate_af3_self_consistency(
                sample_dict=sample_dict,
                num_redesigned_pocket_residue_list=None, 
                out_dir=log_dir, 
                struct_pred_cfg=cfg.struct_pred_cfg,
                cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
                preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
                featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
                docking_metrics_cfg=cfg.docking_metrics_cfg,
                no_wandb=cfg.wandb.no_wandb,
                ckpt_info=None,
                calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only
            )
    else:
        # Lcaliby mode - iterate over checkpoints     
        if not cfg.sample_is_designed: #! For redesigning native proteins
            cif_parse_cfg_lcaliby = cfg.cif_cfg.parse.native
            preprocess_cfg_lcaliby = cfg.preprocess_cfg.native                
        else: #! For redesigning designed proteins
            cif_parse_cfg_lcaliby = cfg.cif_cfg.parse.designed_samples
            preprocess_cfg_lcaliby = cfg.preprocess_cfg.designed_samples
                                    
        results = redesign_with_lcaliby(seed = cfg.seed,
                                        sample_is_designed = cfg.sample_is_designed,
                                        sample_dict = sample_dict,
                                        seq_des_cfg = cfg.seq_des_cfg,
                                        cif_parse_cfg = cif_parse_cfg_lcaliby,
                                        preprocess_cfg = preprocess_cfg_lcaliby,
                                        featurizer_cfg = cfg.featurizer_cfg.design,
                                        cif_save_cfg = cfg.cif_cfg.save,                                            
                                        metadata = metadata,
                                        log_dir = log_dir,
                                        pos_constraint_df = pos_constraint_df)
        
        # Evaluate each checkpoint
        if cfg.struct_pred_cfg.evaluate_self_consistency:
            print("\n" + "="*80)
            print("Phase 3: AF3 Evaluation (per checkpoint)")
            print("="*80 + "\n")
            for sample_dict_per_ckpt, log_dir_per_ckpt, ckpt_info in results:                                                            
                print(f"\nEvaluating checkpoint: step_{ckpt_info['global_step']}_epoch_{ckpt_info['epoch']}")
                
                evaluate_af3_self_consistency(
                    sample_dict = sample_dict_per_ckpt,                         
                    num_redesigned_pocket_residue_list=None, 
                    out_dir=log_dir_per_ckpt, 
                    struct_pred_cfg=cfg.struct_pred_cfg,
                    cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
                    preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
                    featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
                    docking_metrics_cfg=cfg.docking_metrics_cfg,
                    no_wandb=cfg.wandb.no_wandb,
                    ckpt_info=ckpt_info,
                    calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only
                )    
    print("\n" + "="*80)
    print("All phases complete!")
    print(f"Results saved to {log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
