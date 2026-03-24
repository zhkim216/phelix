"""
Calculate AF3 metrics only from pre-existing designed samples and AF3 predictions.

Usage:
    python -m allatom_design.eval.sampling.calculate_af3_metrics_only

Expects data_dir to contain:
    - samples/           : designed sample CIF files ({input_sample_id}_sample{idx}.cif)
    - af3_ss_preds/      : AF3 prediction subdirectories (one per designed_sample_id)

Note:
    Currently assumes input_sample_is_designed=True, meaning docking metrics use
    annotate_ligand_pockets_pseudocb() for pocket identification (pseudo-CB based).

    TODO: For native sample recalculation, input_sample_is_designed should be set to False
    so that docking metrics use annotate_ligand_pockets() (all-atom based) instead.
"""

import re
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import yaml
from tqdm import tqdm

import atomworks.enums as aw_enums

from allatom_design.utils.sample_io_utils import load_example_with_parse
from allatom_design.eval.eval_utils.sd_data_utils import preprocess_input
from allatom_design.data.transform.sd_featurizer import featurizer_designed_samples
from allatom_design.eval.eval_utils.folding_utils import evaluate_af3_self_consistency


def extract_pdb_chain_info(atom_array) -> dict:
    """
    Extract protein and ligand chain info from an atom array.
    Reuses the logic from run_sc_eval_af3.py.
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


@hydra.main(config_path="../../configs_local/eval/sampling", config_name="calculate_af3_metrics_only", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Calculate AF3 metrics only from pre-existing designed samples and AF3 predictions.
    """
    ###########################################################
    # Phase 0: Setup
    ###########################################################
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    data_dir = Path(cfg.data_dir)
    samples_dir = data_dir / "samples"

    # Save config
    config_out_path = data_dir / "config_calculate_af3_metrics_only.yaml"
    with open(config_out_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Compute CSV suffix: array suffix + user-defined suffix
    array_id = cfg.get("array_id", None)
    output_csv_suffix = cfg.get("output_csv_suffix", "")
    array_suffix = f"_array_{array_id}" if array_id is not None else ""
    csv_suffix = f"{array_suffix}{output_csv_suffix}"

    ###########################################################
    # Phase 1: Load designed samples and group by input_sample_id
    ###########################################################
    print("\n" + "="*80)
    print("Phase 1: Loading designed samples")
    print(f"Samples directory: {samples_dir}")
    print("="*80 + "\n")

    af3_preds_dir = data_dir / "af3_ss_preds"

    sample_cif_paths = sorted(samples_dir.glob("*.cif"))

    if cfg.get("debug", False):
        num_debug_samples = cfg.get("num_debug_samples", 2)
        sample_cif_paths = sample_cif_paths[:num_debug_samples]

    if not sample_cif_paths:
        print(f"No CIF files found in {samples_dir}")
        return

    print(f"Found {len(sample_cif_paths)} designed sample CIF files")

    # Filter: only keep designed samples that have AF3 predictions
    filtered_cif_paths = []
    num_skipped = 0
    for cif_path in sample_cif_paths:
        pred_dir = af3_preds_dir / cif_path.stem
        if pred_dir.is_dir():
            filtered_cif_paths.append(cif_path)
        else:
            num_skipped += 1

    if num_skipped > 0:
        print(f"Skipped {num_skipped} designed samples without AF3 predictions")
    print(f"Proceeding with {len(filtered_cif_paths)} designed samples that have AF3 predictions")

    if not filtered_cif_paths:
        print("No designed samples with AF3 predictions found. Exiting.")
        return

    # Group by input_sample_id (strip _sample\d+ suffix)
    input_to_designed_paths = defaultdict(list)
    for cif_path in filtered_cif_paths:
        designed_sample_id = cif_path.stem
        input_sample_id = re.sub(r'_sample\d+$', '', designed_sample_id)
        input_to_designed_paths[input_sample_id].append(cif_path)

    print(f"Grouped into {len(input_to_designed_paths)} input samples")

    # Build sample_dict
    sample_dict = {}
    for input_sample_id, designed_paths in tqdm(input_to_designed_paths.items(), desc="Loading designed samples"):
        entry = {
            "input_sample_id": input_sample_id,
            "designed_sample_id": [],
            "designed_sample_atom_array": [],
            "designed_sample_path": [],
        }

        pdb_chain_info = None
        for designed_path in designed_paths:
            designed_sample_id = designed_path.stem
            try:
                example = load_example_with_parse(str(designed_path), cfg.cif_cfg.parse.designed_samples)
                example = preprocess_input(
                    example=example,
                    preprocess_cfg=cfg.preprocess_cfg.designed_samples,
                    sample_is_designed=True,
                )
                feat_cfg = OmegaConf.to_container(cfg.featurizer_cfg.prepare_designed_samples, resolve=True)
                featurizer = featurizer_designed_samples(**feat_cfg)
                example = featurizer(example)
            except Exception as e:
                print(f"Failed to load {designed_sample_id}: {e}")
                continue

            atom_array = example["atom_array"]
            entry["designed_sample_id"].append(designed_sample_id)
            entry["designed_sample_atom_array"].append(atom_array)
            entry["designed_sample_path"].append(str(designed_path))

            if pdb_chain_info is None:
                pdb_chain_info = extract_pdb_chain_info(atom_array)

        if not entry["designed_sample_id"]:
            print(f"Warning: No valid designed samples for {input_sample_id}, skipping")
            continue

        entry["pdb_chain_info"] = pdb_chain_info
        sample_dict[input_sample_id] = entry

    print(f"\nSuccessfully loaded {len(sample_dict)} input samples "
          f"({sum(len(v['designed_sample_id']) for v in sample_dict.values())} designed samples total)")

    ###########################################################
    # Phase 2: Compute metrics from existing AF3 predictions
    ###########################################################
    if cfg.struct_pred_cfg.evaluate_self_consistency:
        print("\n" + "="*80)
        print("Phase 2: Computing AF3 Self-Consistency & Docking Metrics")
        print(f"AF3 predictions directory: {data_dir / 'af3_ss_preds'}")
        print(f"Output CSV suffix: '{csv_suffix}'")
        print("="*80 + "\n")

        evaluate_af3_self_consistency(
            sample_dict=sample_dict,
            out_dir=data_dir,
            struct_pred_cfg=cfg.struct_pred_cfg,
            cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
            preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
            featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
            pocket_cfg=cfg.pocket_cfg,
            no_wandb=True,
            calculate_metrics_only=True,
            csv_suffix=csv_suffix,
            input_sample_is_designed=cfg.input_sample_is_designed,
        )

    print("\n" + "="*80)
    print("Metrics calculation complete!")
    print(f"Results saved to {data_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
