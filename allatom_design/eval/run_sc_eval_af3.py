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
from allatom_design.eval.eval_utils.sd_data_utils import (
    preprocess_input,
    resolve_query_pn_unit_iids,
)
from allatom_design.eval.eval_utils.folding_utils import (
    evaluate_af3_self_consistency,
)
from allatom_design.utils.sample_io_utils import load_example_with_parse
from allatom_design.data.transform.sd_featurizer import featurizer_designed_samples


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


@hydra.main(config_path="../configs_local/eval", config_name="run_sc_eval_af3", version_base="1.3.2")
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

    # Load sampling_inputs_csv if provided (filters samples & chains)
    sampling_inputs_df = None
    if cfg.get("sampling_inputs_csv", None) is not None:
        sampling_inputs_df = pd.read_csv(cfg.sampling_inputs_csv)
        print(f"Loaded sampling inputs: {len(sampling_inputs_df)} entries from {cfg.sampling_inputs_csv}")

        # Filter sample_paths to only include PDB IDs present in the CSV
        pdb_id_set = set(sampling_inputs_df["pdb_id"].astype(str).str.lower().values)
        sample_paths = [
            p for p in sample_paths
            if Path(p).stem.split("_")[0].lower() in pdb_id_set
        ]
        print(f"Filtered to {len(sample_paths)} samples matching sampling inputs")

    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]

    # Select parsing/preprocessing config based on input type
    if cfg.input_sample_is_designed:
        cif_parse_cfg = cfg.cif_cfg.parse.designed_samples
        preprocess_cfg_input = cfg.preprocess_cfg.designed_samples
    else:
        cif_parse_cfg = cfg.cif_cfg.parse.native_samples
        preprocess_cfg_input = cfg.preprocess_cfg.native_samples

    # Build sample_dict in evaluate_af3_self_consistency format
    sample_dict = {}
    desc = "Loading designed samples" if cfg.input_sample_is_designed else "Loading native samples"
    for sample_path in tqdm(sample_paths, desc=desc):
        sample_id = Path(sample_path).stem

        try:
            example = load_example_with_parse(sample_path, cif_parse_cfg)
            example = preprocess_input(
                example=example,
                preprocess_cfg=preprocess_cfg_input,
                sample_is_designed=cfg.input_sample_is_designed,
            )
            featurizer_cfg_dict = OmegaConf.to_container(
                cfg.featurizer_cfg.prepare_designed_samples, resolve=True
            )
            featurizer = featurizer_designed_samples(**featurizer_cfg_dict)
            example = featurizer(example)
        except Exception as e:
            print(f"Failed to load {sample_id}: {e}")
            continue
        
        atom_array = example["atom_array"]
        
        # Extract chain info
        pdb_chain_info = extract_pdb_chain_info(atom_array)

        # Filter chains by query_pn_unit_iids from sampling_inputs_csv
        if sampling_inputs_df is not None:
            pdb_id = Path(sample_path).stem.split("_")[0]
            query_pn_unit_iids = resolve_query_pn_unit_iids(
                atom_array=atom_array,
                sampling_inputs_df=sampling_inputs_df,
                pdb_id=pdb_id,
            )
            query_set = set(query_pn_unit_iids)
            filtered_info = defaultdict(list)
            for pn_unit_iid in pdb_chain_info["protein_pn_unit_iids"]:
                if pn_unit_iid in query_set:
                    filtered_info["protein_pn_unit_iids"].append(pn_unit_iid)
            for pn_unit_iid, ccd_code in zip(
                pdb_chain_info["ligand_pn_unit_iids"],
                pdb_chain_info["ligand_ccd_codes"],
            ):
                if pn_unit_iid in query_set:
                    filtered_info["ligand_pn_unit_iids"].append(pn_unit_iid)
                    filtered_info["ligand_ccd_codes"].append(ccd_code)
            pdb_chain_info = filtered_info

            # Filter atom_array to only include atoms from query chains
            # so that reference structure matches AF3 prediction scope
            atom_array = atom_array[np.isin(atom_array.pn_unit_iid, list(query_set))]

        if not pdb_chain_info["protein_pn_unit_iids"]:
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
            input_sample_is_designed=cfg.input_sample_is_designed,
        )
    
    print("\n" + "="*80)
    print("All phases complete!")
    print(f"Results saved to {log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
