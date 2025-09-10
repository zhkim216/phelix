import pytest

from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

TEST_CASES = [
    {"pdb_id": "6wtf", "release_date": "2020-12-23", "deposition_date": "2020-05-02"},
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_ec_numbers(test_case: dict):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        add_missing_atoms=False,
        remove_waters=False,
        remove_ccds=[],
        build_assembly=None,
        fix_ligands_at_symmetry_centers=False,
        fix_arginines=False,
        convert_mse_to_met=False,
    )
    assert result["metadata"]["release_date"] == test_case["release_date"]
    assert result["metadata"]["deposition_date"] == test_case["deposition_date"]
