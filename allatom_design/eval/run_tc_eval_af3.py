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
from allatom_design.eval.eval_utils.sd_data_utils import prepare_designed_sample
from allatom_design.eval.eval_utils.folding_utils import (
    evaluate_af3_docking_consistency,
)
from allatom_design.utils.atom_array_utils import insert_unk_residues_for_gaps_in_atom_array
from allatom_design.utils.sample_io_utils import _fix_cif_formal_charge
from atomworks.io.utils.io_utils import to_cif_file


def extract_pdb_chain_info(atom_array) -> dict:
    """
    Extract protein and ligand chain info from an atom array.
    Reuses the logic from redesign_with_lcaliby (seq_des_utils.py lines 2627-2644).
    """
    pdb_chain_info = defaultdict(list)
    
    prot_atom_array = atom_array[atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
    ligand_atom_array = atom_array[np.isin(atom_array.chain_type, list(aw_enums.ChainTypeInfo.NON_POLYMERS))]
    
    protein_pn_unit_iids = [str(pn_unit_iid) for pn_unit_iid in np.unique(prot_atom_array.pn_unit_iid)]
    ligand_pn_unit_iids = [str(pn_unit_iid) for pn_unit_iid in np.unique(ligand_atom_array.pn_unit_iid)]
    ligand_ccd_codes = [
        str(ligand_atom_array[ligand_atom_array.pn_unit_iid == pn_unit_iid].res_name[0])
        for pn_unit_iid in ligand_pn_unit_iids
    ]

    for pn_unit_iid in protein_pn_unit_iids:
        pdb_chain_info["protein_pn_unit_iids"].append(str(pn_unit_iid))

    for pn_unit_iid, ccd_code in zip(ligand_pn_unit_iids, ligand_ccd_codes):
        pdb_chain_info["ligand_pn_unit_iids"].append(str(pn_unit_iid))
        pdb_chain_info["ligand_ccd_codes"].append(str(ccd_code))
    
    return pdb_chain_info


def prepare_tc_template_cif(atom_array, 
                            out_path: str, 
                            cif_save_args: dict) -> str:
    """
    Prepare a template CIF file for AF3 template-conditioned prediction.
    
    1. Separate protein and ligand atom arrays
    2. Insert UNK CA atoms for gaps in protein backbone
    3. Add dummy b_factor if missing (AF3 requires _atom_site.B_iso_or_equiv)
    4. Save to CIF and fix formal charges
    
    Args:
        atom_array: Full atom array (protein + ligand)
        out_path: Output CIF file path
        cif_save_args: CIF save arguments (from cif_cfg.save)
    
    Returns:
        Path to saved CIF file
    """
    # Separate protein and ligand
    prot_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    ligand_mask = np.isin(atom_array.chain_type, list(aw_enums.ChainTypeInfo.NON_POLYMERS))
    
    prot_atom_array = atom_array[prot_mask]
    ligand_atom_array = atom_array[ligand_mask]
    
    # Insert UNK residues for gaps in protein backbone
    prot_atom_array_with_gaps = insert_unk_residues_for_gaps_in_atom_array(prot_atom_array)
    
    # Combine protein (with gaps) + ligand
    tc_atom_array = prot_atom_array_with_gaps + ligand_atom_array
    
    # Renumber atom_id sequentially (1-indexed)
    tc_atom_array.atom_id = np.arange(1, len(tc_atom_array) + 1)
    
    # Ensure b_factor annotation exists (AF3 requires _atom_site.B_iso_or_equiv)
    if "b_factor" not in tc_atom_array.get_annotation_categories():
        tc_atom_array.set_annotation("b_factor", np.zeros(len(tc_atom_array)))
    
    # Save to CIF
    out_path = to_cif_file(
        tc_atom_array, out_path, 
        file_type="cif", 
        **cif_save_args
    )
    _fix_cif_formal_charge(out_path)
    
    return out_path


@hydra.main(config_path="../configs_local/eval", config_name="run_tc_eval_af3", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Run AF3 template-conditioned docking consistency evaluation on pre-designed samples.
    Assumes samples are already designed (e.g. by caliby/lcaliby) and contain ligands.
    
    Workflow:
        1. Load designed sample CIF files
        2. Prepare TC template CIFs (insert UNK gaps + b_factor)
        3. Run evaluate_af3_docking_consistency (TC prediction + metrics)
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
    # Phase 1: Load designed samples and prepare TC templates
    ###########################################################
    print("\n" + "="*80)
    print("Phase 1: Loading designed samples and preparing TC templates")
    print("="*80 + "\n")
    
    # Get CIF file paths
    sample_paths = get_pdb_files(**cfg.pdb_cfg)
    
    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]
    
    # Create TC template CIF output directory
    tc_template_dir = Path(log_dir, "samples_for_af3_tc")
    tc_template_dir.mkdir(parents=True, exist_ok=True)
    
    # Get CIF save args
    cif_save_args = OmegaConf.to_container(cfg.cif_cfg.save, resolve=True) if cfg.cif_cfg.get("save") else {}
    
    # Build sample_dict in evaluate_af3_docking_consistency format
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
        
        if not pdb_chain_info["protein_pn_unit_iids"]:
            print(f"Warning: No protein chains found in {sample_id}, skipping")
            continue
        
        # Prepare TC template CIF (insert gaps + b_factor)
        tc_template_path = str(tc_template_dir / f"{sample_id}.cif")
        try:
            tc_template_path = prepare_tc_template_cif(
                atom_array=atom_array,
                out_path=tc_template_path,
                cif_save_args=cif_save_args,
            )
        except Exception as e:
            print(f"Failed to prepare TC template for {sample_id}: {e}")
            continue
        
        # Build entry in evaluate_af3_docking_consistency format
        sample_dict[sample_id] = {
            "input_sample_path": sample_path,
            "input_sample_id": sample_id,
            "designed_sample_id": [sample_id],
            "designed_sample_atom_array": [atom_array],
            "designed_sample_path_for_af3_tc": [tc_template_path],
            "pdb_chain_info": pdb_chain_info,
        }
    
    print(f"\nSuccessfully loaded {len(sample_dict)} samples")
    
    ###########################################################
    # Phase 2: AF3 Docking Consistency Evaluation
    ###########################################################
    if cfg.struct_pred_cfg.evaluate_docking_consistency:
        print("\n" + "="*80)
        print("Phase 2: AF3 Docking Consistency Evaluation (Template-Conditioned)")
        print("="*80 + "\n")
        
        evaluate_af3_docking_consistency(
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
