from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import yaml
from tqdm import tqdm

import atomworks.enums as aw_enums

from allatom_design.eval.eval_utils.eval_setup_utils import (
    get_pdb_files, wandb_setup)
from allatom_design.eval.eval_utils.seq_des_utils import (
    prepare_designed_sample,
)
from allatom_design.eval.eval_utils.folding_utils import (
    evaluate_af3_self_consistency,
)


def extract_pdb_chain_info(atom_array) -> dict:
    """
    Extract protein and ligand chain info from an atom array.
    Reuses the logic from redesign_with_lcaliby (seq_des_utils.py lines 2627-2644).
    """
    pdb_chain_info = defaultdict(list)
    
    prot_atom_array = atom_array[atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
    ligand_atom_array = atom_array[np.isin(atom_array.chain_type, list(aw_enums.ChainTypeInfo.NON_POLYMERS))]
    
    protein_chain_iids = [str(chain_iid) for chain_iid in np.unique(prot_atom_array.chain_iid)]
    ligand_chain_iids = [str(chain_iid) for chain_iid in np.unique(ligand_atom_array.chain_iid)]
    ligand_ccd_codes = [
        str(ligand_atom_array[ligand_atom_array.chain_iid == chain_iid].res_name[0])
        for chain_iid in ligand_chain_iids
    ]
    
    for chain_iid in protein_chain_iids:
        pdb_chain_info["protein_chain_iids"].append(str(chain_iid))
    
    for chain_iid, ccd_code in zip(ligand_chain_iids, ligand_ccd_codes):
        pdb_chain_info["ligand_chain_iids"].append(str(chain_iid))
        pdb_chain_info["ligand_ccd_codes"].append(str(ccd_code))
    
    return pdb_chain_info


@hydra.main(config_path="../configs/eval", config_name="run_sc_eval_af3", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Run AF3 self-consistency evaluation on pre-designed samples.
    Assumes samples are already designed (e.g. by caliby) and contain ligands.
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
    
    # Compute CSV suffix for array jobs
    array_id = cfg.pdb_cfg.get("array_id", None)
    csv_suffix = f"_array_{array_id}" if array_id is not None else ""
    
    ###########################################################
    # Phase 1: Load designed samples and extract chain info
    ###########################################################
    print("\n" + "="*80)
    print("Phase 1: Loading designed samples")
    print("="*80 + "\n")
    
    # Get CIF file paths
    sample_paths = get_pdb_files(**cfg.pdb_cfg)
    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]        
    
    # Build sample_dict in evaluate_af3_self_consistency format
    sample_dict = {}
    for sample_path in tqdm(sample_paths, desc="Loading designed samples"):
        sample_id = Path(sample_path).stem
        
        try:
            example = prepare_designed_sample(
                pdb_path=sample_path,
                cif_parse_cfg=cfg.cif_cfg.parse.designed_samples,
                preprocess_cfg=cfg.preprocess_cfg.designed_samples,
                featurizer_cfg=cfg.featurizer_cfg.prepare_designed_samples,
            )
        except Exception as e:
            print(f"Failed to load {sample_id}: {e}")
            continue
        
        atom_array = example["atom_array"]
        
        # Extract chain info
        pdb_chain_info = extract_pdb_chain_info(atom_array)
        
        if not pdb_chain_info["protein_chain_iids"]:
            print(f"Warning: No protein chains found in {sample_id}, skipping")
            continue
        
        # Build entry in evaluate_af3_self_consistency format
        # Each designed sample is both the "input" and the "designed" sample
        sample_dict[sample_id] = {
            "input_sample_path": sample_path,
            "input_sample_id": sample_id,
            "designed_sample_id": [sample_id],
            "designed_sample_atom_array": [atom_array],
            "pdb_chain_info": pdb_chain_info,
        }
    
    print(f"\nSuccessfully loaded {len(sample_dict)} samples")
    
    ###########################################################
    # Phase 2: AF3 Evaluation
    ###########################################################
    if cfg.struct_pred_cfg.evaluate_self_consistency:
        print("\n" + "="*80)
        print("Phase 2: AF3 Self-Consistency Evaluation")
        print("="*80 + "\n")
        
        evaluate_af3_self_consistency(
            sample_dict=sample_dict,
            out_dir=log_dir,
            struct_pred_cfg=cfg.struct_pred_cfg,
            cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
            preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
            featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
            pocket_cfg=cfg.pocket_cfg,
            no_wandb=cfg.wandb.no_wandb,
            ckpt_info=None,
            calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only,
            csv_suffix=csv_suffix,
        )
    
    print("\n" + "="*80)
    print("All phases complete!")
    print(f"Results saved to {log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
