"""
This script is used for load structure and metadata using atomworks framework.
Written by Jinho Kim, 251019
"""

import argparse
from pathlib import Path

from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor
from allatom_design.data.transform.preprocess import preprocess_transform

MMCIF_DIR = "/home/possu/jinho/datasets/pdb_mirror"
CIF_PARSE_KWARGS = {
    "add_missing_atoms": True,
    "remove_waters": True,
    "remove_ccds": [],
    "fix_ligands_at_symmetry_centers": True,
    "fix_arginines": True,
    "convert_mse_to_met": True,
    "hydrogen_policy": "remove",
    "add_bond_types_from_struct_conn": ["covale", "metalc"],
}
DATA_PREPROCESSOR_KWARGS = {
    "from_rcsb": True,
    "build_assembly": "first",    
    "close_distance": 30.0,
    "contact_distance": 5,
    "clash_distance": 1.0,
    "ignore_residues": [],
    "polymer_pn_unit_limit": 500,
    **CIF_PARSE_KWARGS,
}

UNDESIRED_RES_NAMES = []

def load_full_with_metadata(pdb_id: str):
    dp = DataPreprocessor(**DATA_PREPROCESSOR_KWARGS)
    cif_path = Path(MMCIF_DIR, f"{pdb_id[1:3]}/{pdb_id}.cif.gz")
    records, filtered_atom_arrays = dp.get_rows(cif_path, return_filtered_atom_array=True)
    atom_array = filtered_atom_arrays[0]  # pn_unit_id / pn_unit_iid already verbose (e.g., "A", "A_1")

    # Apply preprocess transform
    pipeline = preprocess_transform(undesired_res_names=UNDESIRED_RES_NAMES)
    out = pipeline({"atom_array": atom_array})
    atom_array = out["atom_array"]

    return atom_array, records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_id", type=str, default="9icq")
    args = parser.parse_args()
    atom_array, records = load_full_with_metadata(args.pdb_id)
    print(atom_array)
    print(records)