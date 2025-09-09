"""Pytest function to test the detection and assignment of contacting PN units."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from atomworks.ml.utils.testing import get_pdb_mirror_path
from tests.ml.preprocessing.conftest import DATA_PREPROCESSOR

FIND_CONTACTS_TEST_CASES = [
    # Defined with:
    # - contact_distance = 5
    # - close_distance = 30
    {
        # Simple protein complex
        "pdb_id": "1fu2",
        "contact_information": [
            {
                "assembly_id": "1",
                "pn_unit_iid": "A_1",
                "num_contacting_pn_units": 2,
                "num_contacts": 461,  # 458 for (A,B) and 3 for (A, D)
                "num_close_pn_units": 11,  # NOTE: 8 if excluding AF-3 exclusion ligands (Na, Cl)
            }
        ],
    },
    {
        # RNA complex
        "pdb_id": "4gxy",
        "contact_information": [
            {
                "assembly_id": "1",
                "pn_unit_iid": "C_1",
                "num_contacting_pn_units": 1,
                "num_contacts": 94,  # 94 for (C, A)
                "num_close_pn_units": 10,
            }
        ],
    },
]


@pytest.mark.parametrize("test_case", FIND_CONTACTS_TEST_CASES)
def test_find_contacts(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_mirror_path(pdb_id)

    rows = DATA_PREPROCESSOR.get_rows(path)
    df = pd.DataFrame(rows)

    for example in test_case["contact_information"]:
        assembly_id = example["assembly_id"]
        pn_unit_iid = example["pn_unit_iid"]

        # Filter the DataFrame to only the PN unit of interest
        pn_unit_row = df[(df["assembly_id"] == assembly_id) & (df["q_pn_unit_iid"] == pn_unit_iid)]

        # Assert that there is only one row
        assert len(pn_unit_row) == 1

        contacting_pn_unit_iids = json.loads(pn_unit_row["q_pn_unit_contacting_pn_unit_iids"].iloc[0])
        assert len(contacting_pn_unit_iids) == example["num_contacting_pn_units"]

        # Count contacting atoms
        contacting_atoms = 0
        for partner in contacting_pn_unit_iids:
            contacting_atoms += partner["num_contacts"]
        assert example["num_contacts"] == contacting_atoms

        # Count close PN units
        num_close_pn_units = len(json.loads(pn_unit_row["q_pn_unit_close_pn_unit_iids"].iloc[0]))
        assert example["num_close_pn_units"] == num_close_pn_units


if __name__ == "__main__":
    pytest.main(["-v", __file__])
