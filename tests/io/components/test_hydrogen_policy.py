from typing import Any

import numpy as np
import pytest

from atomworks.io.parser import parse
from atomworks.io.transforms.atom_array import add_hydrogen_atom_positions
from atomworks.io.utils.testing import has_ambiguous_annotation_set
from tests.io.conftest import get_pdb_path

TEST_CASES = [{"pdb_id": "1jj8", "count": 705}, {"pdb_id": "3kz8", "count": 6258}, {"pdb_id": "2r5z", "count": 1632}]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_add_hydrogen_atom_positions(test_case: dict[str, Any]):
    path = get_pdb_path(test_case["pdb_id"])

    # Load anf infer hydrogens...
    result_added_hydrogens = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="infer",
    )

    # ...assert that there are hydrogens
    atom_array_added_hydrogens = result_added_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model
    has_resolved_coordinates = ~np.isnan(atom_array_added_hydrogens.coord).any(axis=-1)
    non_nan_array = atom_array_added_hydrogens[has_resolved_coordinates]
    final_h_count = np.sum(non_nan_array.atomic_number == 1)
    assert final_h_count == test_case["count"]


def test_add_hydrogens_with_nan_coords():
    # Parse the AtomArray from CIF without adding hydrogens -- this results in some NaN coordinates
    result_without_adding_hydrogens = parse(
        filename=get_pdb_path("1jj8"),
        build_assembly="all",
        hydrogen_policy="keep",
    )
    atom_array_without_adding_hydrogens = result_without_adding_hydrogens["assemblies"]["1"][0]

    # Check for ambiguous annotations in the original AtomArray
    unique_atom_labels = ["chain_id", "res_id", "res_name", "atom_name"]
    assert not has_ambiguous_annotation_set(atom_array_without_adding_hydrogens, unique_atom_labels)

    # Get nan information for the original AtomArray
    original_nan_coords_mask = np.any(np.isnan(atom_array_without_adding_hydrogens.coord), axis=1)
    original_hydrogens_mask = atom_array_without_adding_hydrogens.element == "H"

    # Add hydrogens
    atom_array_added_hydrogens = add_hydrogen_atom_positions(atom_array_without_adding_hydrogens)

    # Check for ambiguous annotations in the output AtomArray
    assert not has_ambiguous_annotation_set(atom_array_added_hydrogens, unique_atom_labels)

    # Get nan information for the output AtomArray
    result_nan_coords_mask = np.any(np.isnan(atom_array_added_hydrogens.coord), axis=1)
    heavy_atom_nan_idces = np.where(result_nan_coords_mask & ~(atom_array_added_hydrogens.element == "H"))[0]

    # Check that the number of heavy atoms with nan coordinates is unchanged
    assert len(heavy_atom_nan_idces) == sum(original_nan_coords_mask & ~original_hydrogens_mask)

    # Check the bonded hydrogens for each heavy atom with nan coordinates
    for idx in heavy_atom_nan_idces:
        bonded_atoms = atom_array_added_hydrogens.bonds.get_bonds(idx)[0]
        bonded_h_atoms = bonded_atoms[atom_array_added_hydrogens[bonded_atoms].element == "H"]
        for h_atom in bonded_h_atoms:
            # Find the corresponding hydrogen atom in the original array, if present
            h_atom_obj = atom_array_added_hydrogens[h_atom]
            matching_mask = original_hydrogens_mask
            for annot in unique_atom_labels:
                annot_mask = getattr(atom_array_without_adding_hydrogens, annot) == getattr(h_atom_obj, annot)
                matching_mask &= annot_mask

            # Newly-added hydrogens should have NaN coordinates
            if not np.any(matching_mask):
                assert np.all(np.isnan(h_atom_obj.coord))

            # Previously-existing hydrogens should have the same coordinates as before
            else:
                original_atom = atom_array_without_adding_hydrogens[matching_mask]
                assert len(original_atom) == 1
                assert np.allclose(h_atom_obj.coord, original_atom.coord, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
