from typing import Any

import numpy as np
import pytest

from atomworks.io.parser import parse
from atomworks.io.transforms.atom_array import add_charge_from_ccd_codes
from tests.io.conftest import get_pdb_path

TEST_CASES = [{"pdb_id": "1jj8", "charge_sum": 7}, {"pdb_id": "2r5z", "charge_sum": 32}]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_add_charge_from_ccd_codes(test_case: dict[str, Any]):
    path = get_pdb_path(test_case["pdb_id"])

    result = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="remove",
    )

    atom_array = result["assemblies"]["1"][0]  # First bioassembly, first model
    has_resolved_coordinates = ~np.isnan(atom_array.coord).any(axis=-1)
    non_nan_array = atom_array[has_resolved_coordinates]
    non_nan_array.del_annotation("charge")

    non_nan_array = add_charge_from_ccd_codes(non_nan_array)
    charge_sum = np.sum(non_nan_array.charge)
    assert charge_sum == test_case["charge_sum"]


if __name__ == "__main__":
    pytest.main([__file__])
