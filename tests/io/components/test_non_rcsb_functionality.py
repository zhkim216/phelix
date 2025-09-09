import numpy as np
import pytest

from atomworks.enums import ChainType
from atomworks.io.parser import parse
from atomworks.io.utils.non_rcsb import initialize_chain_info_from_atom_array
from tests.io.conftest import CHAIN_TYPE_TEST_CASES, TEST_DATA_IO, get_pdb_path

CIF_PATHS = [TEST_DATA_IO / "example_distillation_output.cif"]


@pytest.mark.parametrize("path", CIF_PATHS)
def test_load_with_all_resolved(path: str):
    result = parse(
        filename=path,
        add_missing_atoms=True,
        remove_ccds=[],
        hydrogen_policy="remove",
    )
    # Check if processing runs through
    assert result is not None

    # Check if the extra metadata is present (from the custom `_extra_metadata` CIFCategory)
    assert result["metadata"]["extra_metadata"] is not None


def test_af2_predicted_pdb_example():
    result = parse(
        filename=TEST_DATA_IO / "UniRef50_A0A0S8JQ92_AF2_predicted.pdb",
        remove_waters=True,
        remove_ccds=[],
    )
    # Check if processing runs through
    assert result is not None


def test_af3_cond_design_cif_example():
    result = parse(
        filename=TEST_DATA_IO / "example_conditional_generation_output.cif",
        add_missing_atoms=False,
        extra_fields="all",
    )
    # Check if processing runs through
    assert result is not None

    atom_array = result["assemblies"]["1"][0]
    assert atom_array.condition_motif_index is not None
    assert atom_array.condition_coords_gt_x is not None

    result = parse(
        filename=TEST_DATA_IO / "example_conditional_generation_output.cif",
        add_missing_atoms=False,
        extra_fields=["condition_coords_gt_x"],
    )
    atom_array = result["assemblies"]["1"][0]
    assert atom_array.condition_coords_gt_x is not None
    assert "condition_coords_gt_y" not in atom_array._annot


def test_bcif_example():
    result = parse(
        filename=TEST_DATA_IO / "6lyz.bcif",
    )
    # Check if processing runs through
    assert result is not None


def test_pdb_with_same_chain_poly_non_poly():
    result = parse(
        filename=TEST_DATA_IO / "1qfe.pdb",
        hydrogen_policy="remove",
    )
    # Check if processing runs through
    assert result is not None

    # Check that we don't have any chains with polymeric and non-polymeric residues
    atom_array = result["assemblies"]["1"][0]
    polymer_chain_ids = np.unique(atom_array.chain_id[atom_array.is_polymer])
    non_polymer_chain_ids = np.unique(atom_array.chain_id[~atom_array.is_polymer])
    assert len(set(polymer_chain_ids).intersection(non_polymer_chain_ids)) == 0

    # Assert that all residues have coordinates
    if not np.all(np.isfinite(atom_array.coord)):
        culprits = atom_array[(~np.isfinite(atom_array.coord)).any(axis=1)]
        raise RuntimeError(f"Some residues are missing coordinates: \n{culprits}")


@pytest.mark.parametrize("test_case", CHAIN_TYPE_TEST_CASES)
def test_infer_chain_info_from_atom_array(test_case: dict):
    cif_path = get_pdb_path(test_case["pdb_id"])
    atom_array = parse(
        filename=cif_path,
        add_missing_atoms=False,
        remove_waters=True,
    )["asym_unit"][0]

    chain_info = initialize_chain_info_from_atom_array(atom_array)

    for chain_id, info_dict in chain_info.items():
        got = info_dict["chain_type"]
        expected = ChainType.as_enum(test_case["chain_types"][chain_id])

        if got.is_non_polymer():
            # We allow all non-polymers to be interchanged
            assert expected.is_non_polymer()
        else:
            # Enforce strict equality for polymers
            assert got == expected


if __name__ == "__main__":
    pytest.main([__file__])
