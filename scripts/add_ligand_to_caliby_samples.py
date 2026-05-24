#!/usr/bin/env python
"""Add ligands from denovo_val_cifs to caliby-designed samples.

For each caliby sample CIF (backbone-only, no ligand), this script:
1. Finds the corresponding denovo reference CIF (contains ligand)
2. Extracts the ligand atoms from the denovo reference
3. Concatenates the caliby protein + denovo ligand
4. Saves the combined structure as a new CIF file

Filename mapping: {CCD}_len_{L}_{IDX}_model_{M}_sample{N}.cif -> {CCD}_len_{L}_{IDX}_model_{M}.cif
"""

import argparse
import re
from pathlib import Path

import biotite.structure as struc
from biotite.structure import AtomArray
from biotite.structure.filter import filter_amino_acids
from omegaconf import OmegaConf
from tqdm import tqdm

from allatom_design.utils.sample_io_utils import load_example_with_parse, save_cif_file


DENOVO_PARSE_CFG = OmegaConf.create({
    "add_missing_atoms": False,
    "remove_waters": True,
    "remove_ccds": [],
    "fix_ligands_at_symmetry_centers": True,
    "fix_arginines": True,
    "convert_mse_to_met": True,
    "hydrogen_policy": "remove",
    "extra_fields": "all",
})

CALIBY_PARSE_CFG = OmegaConf.create({
    "add_missing_atoms": False,
    "remove_waters": True,
    "remove_ccds": [],
    "fix_ligands_at_symmetry_centers": False,
    "fix_arginines": False,
    "convert_mse_to_met": False,
    "hydrogen_policy": "remove",
    "extra_fields": "all",
})


def extract_ligand(atom_array: AtomArray) -> AtomArray | None:
    """Extract non-amino-acid (ligand) atoms from an atom array."""
    protein_mask = filter_amino_acids(atom_array)
    ligand_mask = ~protein_mask
    if not ligand_mask.any():
        return None
    return atom_array[ligand_mask]


def get_denovo_stem(caliby_stem: str) -> str:
    """Strip _sample{N} suffix to get denovo reference base name."""
    return re.sub(r'_sample\d+$', '', caliby_stem)


def main():
    parser = argparse.ArgumentParser(description="Add ligands from denovo CIFs to caliby samples")
    parser.add_argument("--caliby_dir", type=str,
                        default="/home/possu/jinho/datasets/val_cifs/caliby_samples/samples")
    parser.add_argument("--denovo_dir", type=str,
                        default="/home/possu/jinho/datasets/val_cifs/denovo_val_cifs")
    parser.add_argument("--output_dir", type=str,
                        default="/home/possu/jinho/datasets/val_cifs/caliby_samples/samples_with_ligand")
    parser.add_argument("--sample_lengths", type=int, nargs="+", default=None,
                        help="Filter by sample length (e.g., --sample_lengths 150)")
    parser.add_argument("--ccd_codes", type=str, nargs="+", default=None,
                        help="Filter by CCD codes (e.g., --ccd_codes 0H7 0NU)")
    args = parser.parse_args()

    caliby_dir = Path(args.caliby_dir)
    denovo_dir = Path(args.denovo_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # List caliby CIF files
    caliby_cifs = sorted(caliby_dir.glob("*.cif"))

    # Optional filtering
    if args.sample_lengths:
        length_strs = [f"_len_{l}_" for l in args.sample_lengths]
        caliby_cifs = [c for c in caliby_cifs if any(ls in c.name for ls in length_strs)]

    if args.ccd_codes:
        ccd_set = set(args.ccd_codes)
        caliby_cifs = [c for c in caliby_cifs if c.name.split("_")[0] in ccd_set]

    print(f"Processing {len(caliby_cifs)} caliby samples...")

    # Cache: denovo base name -> ligand AtomArray
    ligand_cache: dict[str, AtomArray | None] = {}

    success_count = 0
    skip_count = 0
    error_count = 0

    for caliby_cif in tqdm(caliby_cifs, desc="Adding ligands"):
        caliby_stem = caliby_cif.stem
        denovo_stem = get_denovo_stem(caliby_stem)
        denovo_cif = denovo_dir / f"{denovo_stem}.cif"

        if not denovo_cif.exists():
            print(f"Warning: No denovo reference for {caliby_stem}")
            skip_count += 1
            continue

        # Extract ligand from denovo (cached)
        if denovo_stem not in ligand_cache:
            try:
                denovo_example = load_example_with_parse(str(denovo_cif), DENOVO_PARSE_CFG)
                ligand_cache[denovo_stem] = extract_ligand(denovo_example["atom_array"])
            except Exception as e:
                print(f"Error parsing denovo {denovo_stem}: {e}")
                ligand_cache[denovo_stem] = None
                error_count += 1

        ligand_array = ligand_cache[denovo_stem]
        if ligand_array is None:
            skip_count += 1
            continue

        # Load caliby CIF
        try:
            caliby_example = load_example_with_parse(str(caliby_cif), CALIBY_PARSE_CFG)
            caliby_array = caliby_example["atom_array"]
        except Exception as e:
            print(f"Error parsing caliby {caliby_stem}: {e}")
            error_count += 1
            continue

        # Concatenate protein + ligand
        combined = struc.concatenate([caliby_array, ligand_array])

        # Save
        output_path = output_dir / caliby_cif.name
        try:
            save_cif_file(combined, str(output_path))
            success_count += 1
        except Exception as e:
            print(f"Error saving {caliby_stem}: {e}")
            error_count += 1

    print(f"\nDone! Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
