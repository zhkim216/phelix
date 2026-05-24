#!/usr/bin/env python3
"""
Properly fix PDB files with quoted atom names.

Problem: ProDy's writePDB adds quotes around atom names with apostrophes.
  Original: HETATM  903 "C1'" PNT C 246
  Wrong fix (just remove quotes): HETATM  903 C1' PNT C 246  <- columns shift!

Correct fix: Replace the quoted atom name with properly aligned atom name.
  Correct: HETATM  903  C1' PNT C 246  <- 4-character atom name field preserved

Usage (Sherlock):
    python scripts/sherlock_scripts/jinho/eval_seq_des_training/fix_pdb_quotes_proper.py
"""

import os
import glob
import re
from pathlib import Path

# Sherlock paths
BASE_OUT_DIR = "/scratch/users/zhkim216/out_dir/eval_ligand_seq_des"

EXP_NAMES = [
    "eval_pmpnn_lmpnnval_af3",
    "eval_lmpnn005_lmpnnval_af3",
    "eval_lmpnn010_lmpnnval_af3",
    "eval_lmpnn020_lmpnnval_af3",
    "eval_lmpnn030_lmpnnval_af3",
]

# Standard amino acids for HETATM -> ATOM conversion
STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
               'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}


def fix_pdb_line(line):
    """Fix a single PDB line: remove quotes and convert HETATM standard AA to ATOM."""
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return line

    # Convert HETATM with standard amino acids to ATOM
    if line.startswith("HETATM"):
        resname = line[17:20].strip()
        if resname in STANDARD_AA:
            line = "ATOM  " + line[6:]

    # Check if line has any quotes (could be anywhere in atom name field)
    if '"' not in line:
        return line

    # Find all quoted sections in the line
    # The atom name is in columns 13-16 (1-indexed), but quoted names extend beyond
    quote_start = line.find('"')
    if quote_start == -1:
        return line

    quote_end = line.find('"', quote_start + 1)
    if quote_end == -1:
        return line

    # Extract the quoted atom name
    atom_name = line[quote_start + 1:quote_end]  # e.g., "C1'" -> C1', "H5'1" -> H5'1

    # Determine proper alignment:
    # - 4-char names (like H5'1, HO3') start at column 13 (no leading space)
    # - 3-char or less names start at column 14 (with leading space)
    if len(atom_name) >= 4:
        # 4+ character atom names: left-aligned, truncate to 4
        fixed_atom_name = atom_name[:4]
    else:
        # 3 or less: right-aligned in 4-char field (space prefix)
        fixed_atom_name = f" {atom_name}".ljust(4)[:4]

    # Rebuild line: before quote + fixed atom name + after second quote
    original_len = len(line)

    # The quoted section starts at quote_start, we need to preserve column 12 as the starting point
    # for atom name if quote_start == 12
    if quote_start >= 12:
        # Replace the quoted section with properly formatted atom name
        fixed_line = line[:12] + fixed_atom_name + line[quote_end + 1:]
    else:
        # Unusual case, just remove quotes
        fixed_line = line[:quote_start] + fixed_atom_name + line[quote_end + 1:]

    # Pad to original length to maintain PDB fixed-width format
    if len(fixed_line) < original_len:
        has_newline = line.endswith('\n')
        content = fixed_line.rstrip('\n')
        padding_needed = original_len - len(fixed_line)
        if has_newline:
            padding_needed += 1  # Account for newline in original_len
        fixed_line = content + ' ' * padding_needed
        if has_newline:
            fixed_line = fixed_line + '\n'

    return fixed_line


def fix_pdb_file(pdb_path):
    """Fix all lines in a PDB file."""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = [fix_pdb_line(line) for line in lines]

    # Check if anything changed
    if lines != fixed_lines:
        with open(pdb_path, 'w') as f:
            f.writelines(fixed_lines)
        return True
    return False


def main():
    print("=" * 80)
    print("Fixing PDB quotes with proper column alignment")
    print("=" * 80)

    for exp_name in EXP_NAMES:
        pdb_dir = Path(BASE_OUT_DIR) / exp_name / "samples" / "backbones"

        if not pdb_dir.exists():
            print(f"\n[SKIP] {exp_name}: Directory not found")
            continue

        pdb_files = sorted(glob.glob(str(pdb_dir / "*.pdb")))
        print(f"\n[{exp_name}] Processing {len(pdb_files)} PDB files...")

        fixed_count = 0
        for pdb_path in pdb_files:
            if fix_pdb_file(pdb_path):
                fixed_count += 1

        print(f"  ✓ Fixed {fixed_count} files")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
