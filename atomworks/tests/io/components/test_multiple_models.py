import numpy as np
import pytest

from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

NMR_TEST_CASES = [
    {"pdb_id": "1l2y", "num_models": 38},
    {"pdb_id": "1g03", "num_models": 20},
]


@pytest.mark.parametrize("test_case", NMR_TEST_CASES)
def test_multiple_models(test_case: dict):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        remove_waters=True,
        build_assembly="all",
        model=None,  # Builds all models
    )

    atom_array_stack = result["asym_unit"]
    assert atom_array_stack.stack_depth() == test_case["num_models"]

    # Assert all models have different coordiantes
    for i in range(test_case["num_models"] - 1):
        assert not np.array_equal(atom_array_stack[i].coord, atom_array_stack[i + 1].coord)
