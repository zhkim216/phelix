from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_pdb_files, wandb_setup)

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
                                                          get_sd_example_from_designs_from_other_methods,
                                                          prepare_samples,
                                                          load_designed_samples,                                                        
                                                          )
from allatom_design.eval.eval_utils.folding_utils import (run_af3_single_sequence,
                                                          find_pred_sample_path_af3,
                                                          make_af3_json,
                                                          evaluate_af3_consistency)
from allatom_design.eval.eval_utils.eval_metrics import (_compute_self_consistency_metrics_atomarray,
                                                         _compute_docking_metrics_atomarray)
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets






@hydra.main(config_path="../configs/eval", config_name="run_sc_eval_af3", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Run AF3 self-consistency evaluation.
    Assume samples are already designed by other methods.
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
    
    ###########################################################
    # Phase 1: Prepare samples (common)
    ###########################################################
    print("\n" + "="*80)
    print("Phase 1: Preparing samples")
    print("="*80 + "\n")
    
    sample_dict = prepare_samples(cfg = cfg, metadata = metadata)
    
    # Make a directory for saving original structures with ligand    
    processed_original_samples_dir = log_dir / "original_samples"
    
    # Load designed structures and combine with ligand
    print("Loading designed structures and combining with ligand...")
    sample_dict = load_designed_samples(
        sample_dict=sample_dict, 
        data_cfg_for_designed_samples=cfg.data_cfg_for_designed_samples,
        transform_cfg_for_designed_samples=cfg.transform_cfg_for_designed_samples,
        add_ligands_to_designed_samples=cfg.ligand_source_cfg.get("add_ligands_to_designed_samples", False),
        is_all_atom_sample=cfg.get("is_all_atom_sample", False),
        save_dir=processed_original_samples_dir
    )
                                                                                         
    ###########################################################
    # Run AF3 prediction and compute metrics
    ###########################################################            
    sample_id_list = list(sample_dict.keys())
    pdb_id_list = [sample_dict[sid]["pdb_id"] for sid in sample_id_list]
    sample_atom_array_list = [sample_dict[sid]["sample_atom_array"] for sid in sample_id_list]
    pdb_chain_info = {sid: sample_dict[sid]["pdb_chain_info"] for sid in sample_id_list}
                
    evaluate_af3_consistency(sample_id_list=sample_id_list, 
                                    pdb_id_list=pdb_id_list, 
                                    sample_atom_array_list=sample_atom_array_list, 
                                    pdb_chain_info=pdb_chain_info, 
                                    num_redesigned_pocket_residue_list=None, 
                                    out_dir=log_dir, 
                                    cfg=cfg,
                                    ckpt_info=None,
                                    calculate_metrics_only=cfg.calculate_metrics_only)
        

if __name__ == "__main__":
    main()
