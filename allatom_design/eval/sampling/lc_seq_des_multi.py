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
    get_pdb_files, get_training_checkpoints, wandb_setup)
from allatom_design.eval.eval_utils.seq_des_utils import (
    get_sd_example, 
    load_example_with_load_any,                                                          
    prepare_af3_prediction, 
    prepare_designed_sample, 
    run_lc_seq_des,
    get_seq_des_model,
    _indices_to_pos_string,
    # Sample Dict / Data Preparation    
    prepare_sample_dict,
    _extract_ligand_atom_array_from_cached_examples,
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

@hydra.main(config_path="../../configs_local/eval/sampling", config_name="lc_seq_des_multi", version_base="1.3.2")
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
    
    # Load metadata for input samples
    if cfg.sampling_inputs_csv is not None:
        sampling_inputs_df = pd.read_csv(cfg.sampling_inputs_csv)
    else:
        sampling_inputs_df = None
        
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
    
    sample_dict = prepare_sample_dict(cfg = cfg, 
                                  sampling_inputs_df = sampling_inputs_df)
            
    # Todo: implement adding ligands to input samples if needed later
    
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
                pocket_cfg=cfg.pocket_cfg,
                no_wandb=cfg.wandb.no_wandb,
                ckpt_info=None,
                calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only
            )
    else:
        # Lcaliby mode - iterate over checkpoints     
        if not cfg.input_sample_is_designed: #! For redesigning native proteins
            cif_parse_cfg_lcaliby = cfg.cif_cfg.parse.native
            preprocess_cfg_lcaliby = cfg.preprocess_cfg.native                
        else: #! For redesigning designed proteins
            cif_parse_cfg_lcaliby = cfg.cif_cfg.parse.designed_samples
            preprocess_cfg_lcaliby = cfg.preprocess_cfg.designed_samples
                                    
        results = redesign_with_lcaliby(seed = cfg.seed,
                                        input_sample_is_designed = cfg.input_sample_is_designed,
                                        sample_dict = sample_dict,
                                        seq_des_cfg = cfg.seq_des_cfg,
                                        cif_parse_cfg = cif_parse_cfg_lcaliby,
                                        preprocess_cfg = preprocess_cfg_lcaliby,
                                        featurizer_cfg = cfg.featurizer_cfg.design,
                                        cif_save_cfg = cfg.cif_cfg.save,                                            
                                        sampling_inputs_df = sampling_inputs_df,
                                        log_dir = log_dir,
                                        pos_constraint_df = pos_constraint_df,
                                        protein_only = cfg.get("protein_only", False))
        
        
                    
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
                    pocket_cfg=cfg.pocket_cfg,
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
