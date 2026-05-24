"""
Calculate sequence recovery from folders of native and designed CIF files.

Usage:
    python -m allatom_design.eval.eval_utils.sequence_recovery \
        --native_cif_dir /path/to/native_cifs \
        --designed_sample_dir /path/to/samples \
        --sampling_inputs_csv /path/to/sampling_inputs.csv \
        --output_csv /path/to/output.csv
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from omegaconf import OmegaConf
from tqdm import tqdm

from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise

from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets
from allatom_design.utils.atom_array_utils import get_valid_standard_aa_residue_mask
from allatom_design.utils.sample_io_utils import load_example_with_parse

# CIF parse configs from lc_seq_des_multi.yaml
NATIVE_CIF_PARSE_CFG = {
    "add_missing_atoms": True,
    "remove_waters": True,
    "remove_ccds": [],
    "fix_ligands_at_symmetry_centers": True,
    "fix_arginines": True,
    "convert_mse_to_met": True,
    "hydrogen_policy": "remove",
    "extra_fields": "all",
}

DESIGNED_CIF_PARSE_CFG = {
    "add_missing_atoms": False,
    "remove_waters": False,
    "remove_ccds": [],
    "fix_ligands_at_symmetry_centers": False,
    "fix_arginines": False,
    "convert_mse_to_met": True,
    "hydrogen_policy": "remove",
    "extra_fields": None,
}


def calculate_sequence_recovery(input_atom_array: AtomArray, designed_atom_array: AtomArray,
                                pocket_distances_for_seq_recovery: list[float] = [4.0, 5.0, 6.0],
                                pocket_distance_bins: list[tuple[float, float]] | None = None,
                                n_min_ligand_atoms: int = 5) -> dict[str, float]:
    """
    Calculate sequence recovery and pocket sequence recovery between input and designed atom arrays.
    """
    seq_recovery_metrics = {}

    input_valid_residue_mask = get_valid_standard_aa_residue_mask(input_atom_array)

    input_seq_mask = input_valid_residue_mask & (input_atom_array.atom_name == "CA")
    input_res_ids = input_atom_array[input_seq_mask].res_id
    input_res_names = input_atom_array[input_seq_mask].res_name

    designed_valid_residue_mask = get_valid_standard_aa_residue_mask(designed_atom_array)
    designed_seq_mask = designed_valid_residue_mask & np.isin(designed_atom_array.res_id, input_res_ids) & (designed_atom_array.atom_name == "CA")
    designed_res_names = designed_atom_array[designed_seq_mask].res_name

    seq_recovery_metrics["seq_recovery_ratio"] = (input_res_names == designed_res_names).mean()

    edge_to_residue_mask: dict[float, np.ndarray] = {}

    for pocket_distance in pocket_distances_for_seq_recovery:
        input_atom_array = annotate_ligand_pockets(
            input_atom_array,
            pocket_distance=pocket_distance,
            n_min_ligand_atoms=n_min_ligand_atoms,
            annotation_name=f"is_ligand_pocket_{pocket_distance}",
        )
        input_pocket_residue_mask = apply_and_spread_residue_wise(
            input_atom_array,
            input_atom_array.get_annotation(f"is_ligand_pocket_{pocket_distance}"),
            function=np.any,
        )
        edge_to_residue_mask[float(pocket_distance)] = input_pocket_residue_mask
        input_pocket_seq_mask = input_seq_mask & input_pocket_residue_mask

        input_pocket_res_ids = input_atom_array[input_pocket_seq_mask].res_id
        input_pocket_res_names = input_atom_array[input_pocket_seq_mask].res_name

        designed_pocket_seq_mask = np.isin(designed_atom_array.res_id, input_pocket_res_ids) & (designed_atom_array.atom_name == "CA")
        designed_pocket_res_names = designed_atom_array[designed_pocket_seq_mask].res_name

        seq_recovery_metrics[f"pocket_recovery_ratio_{pocket_distance}"] = (input_pocket_res_names == designed_pocket_res_names).mean()
        seq_recovery_metrics[f"pocket_n_residues_{pocket_distance}"] = int(len(input_pocket_res_names))

    if pocket_distance_bins:
        for lo, hi in pocket_distance_bins:
            lo_f, hi_f = float(lo), float(hi)
            for d in (lo_f, hi_f):
                if d == 0.0:
                    continue
                if d not in edge_to_residue_mask:
                    ann = f"is_ligand_pocket_{d}"
                    input_atom_array = annotate_ligand_pockets(
                        input_atom_array,
                        pocket_distance=d,
                        n_min_ligand_atoms=n_min_ligand_atoms,
                        annotation_name=ann,
                    )
                    edge_to_residue_mask[d] = apply_and_spread_residue_wise(
                        input_atom_array,
                        input_atom_array.get_annotation(ann),
                        function=np.any,
                    )

            if lo_f == 0.0:
                bin_residue_mask = edge_to_residue_mask[hi_f]
            else:
                bin_residue_mask = edge_to_residue_mask[hi_f] & ~edge_to_residue_mask[lo_f]

            input_bin_seq_mask = input_seq_mask & bin_residue_mask
            input_bin_res_ids = input_atom_array[input_bin_seq_mask].res_id
            input_bin_res_names = input_atom_array[input_bin_seq_mask].res_name

            designed_bin_seq_mask = np.isin(designed_atom_array.res_id, input_bin_res_ids) & (designed_atom_array.atom_name == "CA")
            designed_bin_res_names = designed_atom_array[designed_bin_seq_mask].res_name

            key = f"pocket_recovery_bin_{lo_f}_to_{hi_f}"
            n_key = f"pocket_n_residues_bin_{lo_f}_to_{hi_f}"
            if len(input_bin_res_names) == 0:
                seq_recovery_metrics[key] = float("nan")
            else:
                seq_recovery_metrics[key] = float((input_bin_res_names == designed_bin_res_names).mean())
            seq_recovery_metrics[n_key] = int(len(input_bin_res_names))

    return seq_recovery_metrics


def _compute_recovery(native_aa, designed_aa, pocket_distances):
    """Compute sequence recovery matching by (chain_id, res_id)."""
    metrics = {}

    # Valid standard AA CA atoms
    n_mask = get_valid_standard_aa_residue_mask(native_aa) & (native_aa.atom_name == "CA")
    d_mask = get_valid_standard_aa_residue_mask(designed_aa) & (designed_aa.atom_name == "CA")
    n_ca = native_aa[n_mask]
    d_ca = designed_aa[d_mask]

    # Native lookup: (chain_id, res_id) → res_name
    native_lookup = {
        (n_ca.chain_id[i], int(n_ca.res_id[i])): n_ca.res_name[i]
        for i in range(len(n_ca))
    }

    # Overall sequence recovery
    matched = []
    for i in range(len(d_ca)):
        key = (d_ca.chain_id[i], int(d_ca.res_id[i]))
        if key in native_lookup:
            matched.append(native_lookup[key] == d_ca.res_name[i])
    metrics["seq_recovery_ratio"] = float(np.mean(matched)) if matched else 0.0

    # Pocket recovery at each distance
    receptor_pn_unit_iids = list(np.unique(designed_aa[designed_aa.is_polymer].pn_unit_iid))
    ligand_pn_unit_iids = list(np.unique(designed_aa[~designed_aa.is_polymer].pn_unit_iid))

    for pocket_distance in pocket_distances:
        native_aa = annotate_ligand_pockets(
            native_aa, pocket_distance=pocket_distance,
            annotation_name=f"is_ligand_pocket_{pocket_distance}",
            receptor_pn_unit_iids=receptor_pn_unit_iids,
            ligand_pn_unit_iids=ligand_pn_unit_iids,
        )
        pocket_residue_mask = apply_and_spread_residue_wise(
            native_aa, native_aa.get_annotation(f"is_ligand_pocket_{pocket_distance}"), function=np.any,
        )
        pocket_ca_mask = n_mask & pocket_residue_mask
        pocket_ca = native_aa[pocket_ca_mask]

        # Pocket native lookup
        pocket_lookup = {
            (pocket_ca.chain_id[i], int(pocket_ca.res_id[i])): pocket_ca.res_name[i]
            for i in range(len(pocket_ca))
        }

        pocket_matched = []
        for i in range(len(d_ca)):
            key = (d_ca.chain_id[i], int(d_ca.res_id[i]))
            if key in pocket_lookup:
                pocket_matched.append(pocket_lookup[key] == d_ca.res_name[i])
        metrics[f"pocket_recovery_ratio_{pocket_distance}"] = float(np.mean(pocket_matched)) if pocket_matched else float("nan")

    return metrics


def calculate_sequence_recovery_from_folders(
    native_cif_dir: str | Path,
    designed_sample_dir: str | Path,
    sampling_inputs_csv: str | Path,
    output_csv: str | Path | None = None,
    pocket_distances: list[float] = [4.0, 5.0, 6.0],
    native_cif_parse_cfg: dict = NATIVE_CIF_PARSE_CFG,
    designed_cif_parse_cfg: dict = DESIGNED_CIF_PARSE_CFG,
) -> pd.DataFrame:
    """Calculate sequence recovery for designed samples against native reference structures."""
    native_cif_dir = Path(native_cif_dir)
    designed_sample_dir = Path(designed_sample_dir)
    pd.read_csv(sampling_inputs_csv)  # validate CSV exists

    native_cfg = OmegaConf.create(native_cif_parse_cfg)
    designed_cfg = OmegaConf.create(designed_cif_parse_cfg)

    # Group designed samples by pdb_id
    pdb_id_to_samples: dict[str, list[Path]] = defaultdict(list)
    for cif_path in sorted(designed_sample_dir.glob("*.cif")):
        pdb_id_to_samples[cif_path.stem.split("_")[0]].append(cif_path)

    results = []
    native_cache = {}

    for pdb_id, sample_paths in tqdm(pdb_id_to_samples.items(), desc="Calculating sequence recovery"):
        native_cif_path = native_cif_dir / f"{pdb_id}.cif"
        if not native_cif_path.exists():
            print(f"Warning: native CIF not found for {pdb_id}, skipping")
            continue

        if pdb_id not in native_cache:
            try:
                native_cache[pdb_id] = load_example_with_parse(str(native_cif_path), native_cfg)["atom_array"]
            except Exception as e:
                print(f"Warning: failed to parse native CIF for {pdb_id}: {e}")
                continue

        native_aa = native_cache[pdb_id]

        for sample_path in sample_paths:
            try:
                designed_aa = load_example_with_parse(str(sample_path), designed_cfg)["atom_array"]
            except Exception as e:
                print(f"Warning: failed to parse {sample_path.name}: {e}")
                continue

            metrics = _compute_recovery(native_aa, designed_aa, pocket_distances)
            results.append({"pdb_id": pdb_id, "designed_sample_id": sample_path.stem, **metrics})

    results_df = pd.DataFrame(results)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

    if len(results_df) > 0:
        print(f"\n--- Summary ({len(results_df)} samples, {results_df['pdb_id'].nunique()} pdb_ids) ---")
        for col in results_df.select_dtypes(include="number").columns:
            print(f"  {col}: mean={results_df[col].mean():.4f}, std={results_df[col].std():.4f}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate sequence recovery from CIF folders")
    parser.add_argument("--native_cif_dir", type=str, required=True)
    parser.add_argument("--designed_sample_dir", type=str, required=True)
    parser.add_argument("--sampling_inputs_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--pocket_distances", nargs="+", type=float, default=[4.0, 5.0, 6.0])
    args = parser.parse_args()

    calculate_sequence_recovery_from_folders(
        native_cif_dir=args.native_cif_dir,
        designed_sample_dir=args.designed_sample_dir,
        sampling_inputs_csv=args.sampling_inputs_csv,
        output_csv=args.output_csv,
        pocket_distances=args.pocket_distances,
    )
