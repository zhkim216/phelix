from typing import Any

import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.ml.transforms.sasa import CalculateSASA, calculate_atomwise_rasa
from atomworks.ml.utils.testing import cached_parse

# Define test cases
# (all values for radii and SASA are from "WhatIF")
SASA_TEST_CASES = [
    {
        "pdb_id": "1fu2",  # (multi-chain protein)
        "probe_radius": 1.4,  # default radius for water as a solvent
        "atom_radii": "ProtOr",
        "point_number": 100,
        "spot_checks": [
            {"atom_name": "H"},  # should be nan
            {"atom_name": "ZN"},  # should be nan
            {"atom_name": "N"},  # should be not nan
        ],
    },
    {
        "pdb_id": "3p42",  # testing protein with certain NaN coordinates
        "probe_radius": 1.4,  # default radius for water as a solvent
        "atom_radii": "ProtOr",
        "point_number": 100,
        "spot_checks": [
            {"atom_name": "H"},  # should be nan
            {"atom_name": "N"},  # should be not nan
        ],
    },
]


def _define_residue_keys(
    atom_array: AtomArray,
):
    """
    Defines a key for each residue in the atom array.
    """
    res_ids = atom_array.res_id
    res_names = atom_array.res_name
    chain_ids = atom_array.chain_id
    transformation_ids = atom_array.transformation_id

    # Create unique keys for each residue using chain_id, res_id, and res_name
    res_keys = [
        f"{chain}_{transform}:{name}_{res_id}"
        for chain, transform, name, res_id in zip(chain_ids, transformation_ids, res_names, res_ids, strict=False)
    ]
    return res_keys


def _count_atoms_in_each_residue(atom_array: AtomArray) -> dict[int, int]:
    """Count the number of atoms in each residue in the atom array."""
    # Get the residue keys
    res_keys = _define_residue_keys(atom_array)
    # Count atoms per unique residue
    unique_keys, counts = np.unique(res_keys, return_counts=True)
    return dict(zip(unique_keys, counts, strict=False))


def _get_indices_of_singleton_residues(atom_array: AtomArray) -> np.ndarray:
    """Get the indices of residues with only one atom in the atom array."""
    res_count_mapping = _count_atoms_in_each_residue(atom_array)

    res_keys = _define_residue_keys(atom_array)
    singleton_residue_keys = [key for key, count in res_count_mapping.items() if count == 1]
    singleton_residue_indices = np.isin(res_keys, singleton_residue_keys)
    return singleton_residue_indices


@pytest.mark.parametrize("test_case", SASA_TEST_CASES)
def test_calculate_sasa(test_case: dict[str, Any]):
    """
    Test the CalculateSASA transform using a multi-chain protein.
    Checks:
    - The SASA of atoms that should not have SASA calculated are NaN
    - The SASA of heavy atoms that should have SASA are >=0
    """

    # Load the atom array
    data = cached_parse(test_case["pdb_id"])

    # Apply the transform
    transform = CalculateSASA(
        probe_radius=test_case["probe_radius"],
        atom_radii=test_case["atom_radii"],
        point_number=test_case["point_number"],
    )
    data = transform(data)

    # Check SASA values of specific atoms

    for spot_check in test_case["spot_checks"]:
        atom_mask = data["atom_array"].atom_name == spot_check["atom_name"]
        if data["atom_array"][atom_mask].atom_name[0] in (["ZN", "NA", "H"]):
            assert np.isnan(data["atom_array"][atom_mask].sasa).all()
        else:
            valid_mask = ~np.isnan(data["atom_array"][atom_mask].coord).all(axis=1)
            assert np.all(data["atom_array"][atom_mask][valid_mask].sasa >= 0)


@pytest.mark.parametrize("test_case", SASA_TEST_CASES)
def test_calculate_rasa(test_case: dict[str, Any]):
    """
    Test the CalculateRASA transform using a multi-chain protein.
    Checks:
    - The RASA of atoms that should not have RASA calculated are NaN
    - The RASA of heavy atoms that should have RASA are >=0
    """

    # Load the atom array
    data = cached_parse(test_case["pdb_id"])

    # Apply the transform
    atom_array = data["atom_array"]
    rasa = calculate_atomwise_rasa(
        atom_array,
        probe_radius=test_case["probe_radius"],
        atom_radii=test_case["atom_radii"],
        point_number=test_case["point_number"],
    )
    data["atom_array"].set_annotation("rasa", rasa)

    singleton_residue_indices = _get_indices_of_singleton_residues(atom_array)
    # convert indices to boolean mask
    is_singleton_residues = np.zeros(len(atom_array), dtype=bool)
    if len(singleton_residue_indices) > 0:
        is_singleton_residues[singleton_residue_indices] = True

    # Check RASA values of specific atoms
    atom_mask = data["atom_array"].element == "H"
    atom_mask |= data["atom_array"].occupancy == 0
    atom_mask |= is_singleton_residues
    assert np.isnan(data["atom_array"][atom_mask].rasa).all()
    assert np.all(data["atom_array"][~atom_mask].rasa >= 0), "RASA should be >= 0 for heavy atoms"
    assert np.all(data["atom_array"][~atom_mask].rasa <= 1), "RASA should be <= 1 for heavy atoms"


def test_calculate_rasa_failure():
    data = cached_parse("7eeu")
    atom_array = data["atom_array"]
    rasa = calculate_atomwise_rasa(
        atom_array,
    )
    assert np.isnan(rasa).all(), "RASA should be NaN for all atoms in this case"
