from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray, get_residue_starts
import atomworks.enums as aw_enums
from atomworks.constants import STANDARD_AA
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise


def get_valid_standard_aa_residue_mask(atom_array: AtomArray) -> np.ndarray:
    """
    Get a boolean mask for atoms belonging to valid standard amino acid residues.
    A residue is valid if it is:
      1. A standard amino acid in a polypeptide chain (not hetero)
      2. Has all backbone atoms (N, CA, C, O) resolved (occupancy > 0)
    """
    standard_aa_prot_mask = (
        (atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
        & np.isin(atom_array.res_name, STANDARD_AA)
        & ~atom_array.hetero
    )
    is_ncaco_resolved = (
        np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])
        & (atom_array.occupancy > 0)
    )
    has_all_backbone = apply_and_spread_residue_wise(
        atom_array, is_ncaco_resolved, lambda x: np.sum(x) == 4
    )
    return standard_aa_prot_mask & has_all_backbone


def insert_unk_residues_for_gaps_in_atom_array(atom_array: AtomArray) -> AtomArray:
    """
    Insert UNK/CA atoms at residue index gaps (non-consecutive res_id within a chain) in atom array of protein chains.    
    """
    annotations = atom_array.get_annotation_categories()
    annot_categories_to_include = [
        "res_id",
        "res_name",
        "atom_name",
        "alt_atom_id",
        "atom_id",
        "element",
        "hetero",
        "occupancy",
        "b_factor",
        "stereo",
        "is_aromatic",
        "is_backbone_atom",
        "is_polymer",
        "charge",
        "atomic_number",
        "atomize",
        "is_covalent_modification",
        "uses_alt_atom_id",
        "ins_code",
        "chain_id",
        "pn_unit_id",
        "molecule_id",
        "chain_entity",
        "pn_unit_entity",
        "molecule_entity",
        "transformation_id",
        "chain_iid",
        "pn_unit_iid",
        "molecule_iid",
        "chain_type",
    ]

    annot_categories_to_copy = [
        "chain_id",
        "pn_unit_id",
        "molecule_id",
        "chain_entity",
        "pn_unit_entity",
        "molecule_entity",
        "transformation_id",
        "chain_iid",
        "pn_unit_iid",
        "molecule_iid",
        "chain_type",
    ]
    
    protein_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    protein_atom_array = atom_array[protein_mask]
    non_protein_mask = ~protein_mask
    non_protein_atom_array = atom_array[non_protein_mask]

    # Delete annotations that are not in the include list.
    for annot in annotations:
        if annot not in annot_categories_to_include:
            protein_atom_array.del_annotation(annot)

    # Get residue starts and detect gaps by chain.
    res_starts = get_residue_starts(protein_atom_array)
    res_ids = protein_atom_array.res_id[res_starts]
    chain_ids = protein_atom_array.chain_id[res_starts]

    res_id_diff = np.diff(res_ids)
    same_chain = chain_ids[:-1] == chain_ids[1:]
    gap_indices = np.where((res_id_diff > 1) & same_chain)[0]

    if len(gap_indices) == 0:
        print("No gaps found in the atom array")        
        return protein_atom_array + non_protein_atom_array

    unk_atoms_list: list[AtomArray] = []

    for gap_idx in gap_indices:
        start_res_id = res_ids[gap_idx]
        end_res_id = res_ids[gap_idx + 1]
        template_atom_idx = res_starts[gap_idx]

        for missing_res_id in range(start_res_id + 1, end_res_id):
            unk_atom = AtomArray(1)
            unk_atom.coord[0] = [0.0, 0.0, 0.0]

            for annot in annot_categories_to_include:
                if annot == "res_id":
                    unk_atom.set_annotation(annot, np.array([missing_res_id]))
                elif annot == "res_name":
                    unk_atom.set_annotation(annot, np.array(["UNK"]))
                elif annot in ("atom_name", "alt_atom_id"):
                    unk_atom.set_annotation(annot, np.array(["CA"]))
                elif annot == "atom_id":
                    unk_atom.set_annotation(annot, np.array([0]))
                elif annot == "element":
                    unk_atom.set_annotation(annot, np.array(["C"]))
                elif annot == "hetero":
                    unk_atom.set_annotation(annot, np.array([False]))
                elif annot == "occupancy":
                    unk_atom.set_annotation(annot, np.array([0.0]))
                elif annot == "b_factor":
                    unk_atom.set_annotation(annot, np.array([0.0]))
                elif annot == "stereo":
                    unk_atom.set_annotation(annot, np.array(["S"]))
                elif annot == "is_aromatic":
                    unk_atom.set_annotation(annot, np.array([False]))
                elif annot == "is_backbone_atom":
                    unk_atom.set_annotation(annot, np.array([True]))
                elif annot == "is_polymer":
                    unk_atom.set_annotation(annot, np.array([True]))
                elif annot == "charge":
                    unk_atom.set_annotation(annot, np.array([0]))
                elif annot == "atomic_number":
                    unk_atom.set_annotation(annot, np.array([6]))
                elif annot == "atomize":
                    unk_atom.set_annotation(annot, np.array([True]))
                elif annot == "is_covalent_modification":
                    unk_atom.set_annotation(annot, np.array([False]))
                elif annot == "uses_alt_atom_id":
                    unk_atom.set_annotation(annot, np.array([False]))
                elif annot == "ins_code":
                    unk_atom.set_annotation(annot, np.array([""]))
                elif annot in annot_categories_to_copy:
                    template_val = getattr(atom_array, annot)[template_atom_idx]
                    unk_atom.set_annotation(annot, np.array([template_val]))

            unk_atoms_list.append(unk_atom)

    if len(unk_atoms_list) == 0:
        print("No UNK atoms to insert (gaps detected but no missing residues)")
        return atom_array

    all_unk_atoms = unk_atoms_list[0]
    for unk_atom in unk_atoms_list[1:]:
        all_unk_atoms = all_unk_atoms + unk_atom

    atom_array_with_gaps = protein_atom_array + all_unk_atoms + non_protein_atom_array
    sort_indices = np.lexsort((atom_array_with_gaps.res_id, atom_array_with_gaps.chain_id))
    atom_array_with_gaps = atom_array_with_gaps[sort_indices]
    atom_array_with_gaps.atom_id = np.arange(1, len(atom_array_with_gaps) + 1)
    
    return atom_array_with_gaps


def clean_up_and_renumber_atom_array(atom_array: AtomArray) -> AtomArray:
    """
    Clean up and renumber an atom array.
    """
            
    valid_coords_mask = ~np.isnan(atom_array.coord).any(axis=1)

    #! mask out atoms with NaN coordiantes (sidechain atoms). But atomized residues (hetero, covalent, etc.) could be included
    atom_array = atom_array[valid_coords_mask]                    

    # Renumber atom_id sequentially (1-indexed)
    atom_array.atom_id = np.arange(1, len(atom_array) + 1)
    return atom_array