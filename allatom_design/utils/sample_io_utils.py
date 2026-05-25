from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from omegaconf import DictConfig, OmegaConf

from atomworks.io.parser import parse as aw_parse
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, get_residue_starts

###########################################################
# Functions for loading examples
###########################################################

def load_example_with_parse(
    pdb_path: str,
    cif_parse_cfg: DictConfig | None = None,
) -> dict[str, Any]:
    """
    Load an example dictionary from a structure file using atomworks parse.
    """
    if cif_parse_cfg is None:
        cif_parse_cfg = {
            "add_missing_atoms": True,
            "remove_waters": True,
            "remove_ccds": [],
            "fix_ligands_at_symmetry_centers": True,
            "fix_arginines": True,
            "convert_mse_to_met": True,
            "hydrogen_policy": "remove",
            "extra_fields": "all",
        }
    else:
        cif_parse_cfg = OmegaConf.to_container(cif_parse_cfg, resolve=True)

    transformation_id = "1"
    cif_parse_cfg["build_assembly"] = [transformation_id]
    input_data = aw_parse(pdb_path, **cif_parse_cfg)
    atom_array = input_data["assemblies"][transformation_id][0]
    
    # Fix annotation types for atom_array loaded from CIF.
    atom_array = fix_cif_annotation_types_atom_array(atom_array)
    
    chain_info = input_data["chain_info"]

    return {"example_id": Path(pdb_path).stem, "atom_array": atom_array, "chain_info": chain_info}

def fix_cif_annotation_types_atom_array(atom_array: AtomArray) -> AtomArray:
    """
    Fix annotation types for atom_array loaded from CIF.
    CIF format stores values as strings, so convert back to expected numeric/bool types where possible.
    """
    bool_annotations = [
        "atomize",
        "is_polymer",
        "is_aromatic",
        "is_covalent_modification",
        "is_backbone_atom",
        "hetero",
        "is_leaving_atom",
        "is_n_terminal_atom",
        "is_c_terminal_atom",
    ]
    
    for ann in bool_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ("U", "S", "O"):
                new_val = val == "True"
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)

    int_annotations = [
        "chain_type",
        "atomic_number",
        "within_chain_res_idx",
        "within_poly_res_idx",
        "chain_entity",
        "molecule_entity",
        "pn_unit_entity",
        "token_id",
        "transformation_id",
        "pdbx_PDB_model_num",
        "label_entity_id",
        "label_seq_id",
        "auth_seq_id",
        "molecule_id",
        "molecule_iid",
        "charge",
        "pdbx_formal_charge",
    ]
    for ann in int_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ("U", "S", "O"):                
                new_val = np.array([int(v) if str(v).lstrip("-").isdigit() else 0 for v in val])                
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)

    float_annotations = ["B_iso_or_equiv", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "b_factor"]
    for ann in float_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ("U", "S", "O"):
                new_val = np.array([float(v) if v not in ("?", ".", "") else np.nan for v in val])
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)

    return atom_array

### Functions for saving cif files ###

def save_cif_file(
    atom_array: AtomArray,
    cif_path: str | Path,
    cif_save_cfg: dict[str, Any] | None = None,
) -> None:    
    """
    Save an atom array to a cif file.
    """
    if cif_save_cfg is None:
        cif_save_cfg = {
            "file_type": "cif",
            "date": "1959-01-07",
            "include_entity_poly": True,
            "include_entity_nonpoly": True,
            "include_nan_coords": False,
            "include_bonds": True,
            "extra_fields": [],
            "exclude_field_keys": ["token_id", "is_ligand_pocket"],
            "extra_categories": {
                "pdbx_audit_revision_history": {
                    "ordinal": [1],
                    "revision_id": [1],
                    "revision_date": ["1959-01-07"],
                    "major_revision": [1],
                    "minor_revision": [0],
                    "revision_description": ["Dummy date for template-conditioning AF3"],
                },
            },
        }
    else:
        cif_save_cfg = OmegaConf.to_container(cif_save_cfg, resolve=True)
    
    # Ensure b_factor annotation exists in atom array for AF3 template conditioning
    # AF3 requires _atom_site.B_iso_or_equiv in template CIF files
    if "b_factor" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("b_factor", np.zeros(len(atom_array)))
    
    try:
        to_cif_file(atom_array, cif_path, **cif_save_cfg)
    except AttributeError as exc:
        if cif_save_cfg.get("include_bonds", True) and "convert_bond_type" in str(exc):
            retry_cfg = dict(cif_save_cfg)
            retry_cfg["include_bonds"] = False
            print(
                f"Warning: failed to write bonds for {cif_path}; "
                "retrying with include_bonds=False"
            )
            to_cif_file(atom_array, cif_path, **retry_cfg)
        else:
            raise
    fix_cif_formal_charge_format(cif_path)
    return cif_path


def fix_cif_formal_charge_format(cif_path: str | Path) -> None:
    """
    Fix pdbx_formal_charge format in CIF files for OpenStructure compatibility.
    Convert +N -> N while preserving negatives.
    """
    cif_path = Path(cif_path)
    if not cif_path.exists():
        return

    with open(cif_path, "r") as f:
        content = f.read()

    fixed_content = re.sub(r"(\s)\+(\d+)(\s)", r"\1\2\3", content)

    if content != fixed_content:
        with open(cif_path, "w") as f:
            f.write(fixed_content)
