import copy

import numpy as np
import pytest

from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.msa._msa_pairing_utils import (
    _get_matched_indices,
    _remove_extraneous_taxid_copies,
    join_multiple_msas_by_tax_id,
)
from atomworks.ml.transforms.msa.msa import (
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
)
from atomworks.ml.utils.testing import cached_parse
from tests.ml.conftest import PROTEIN_MSA_DIRS, RNA_MSA_DIRS

PAIR_MSA_TEST_CASES = [
    {
        # Simple heteromer pairing between chains A and B, where query chain tax ID is 2
        "dense": False,
        "input_msas": {
            "A": {
                "msa": np.array([["A", "C", "G"], ["A", "H", "H"], ["A", "-", "G"], ["L", "C", "G"]], dtype=np.bytes_),
                "ins": np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                "tax_ids": np.array(["2", "1", "1", "3"]),  # 2 is query
                "msa_is_padded_mask": np.zeros((4, 3), dtype=bool),
            },
            "B": {
                "msa": np.array([["G", "T"], ["G", "-"], ["G", "H"]], dtype=np.bytes_),
                "ins": np.array([[0, 0], [1, 0], [1, 0]]),
                "tax_ids": np.array(["2", "3", "1"]),  # 2 is query
                "msa_is_padded_mask": np.zeros((3, 2), dtype=bool),
            },
        },
        "output_msa": {
            "msa": np.array(
                [
                    ["A", "C", "G", "G", "T"],  # Query
                    ["L", "C", "G", "G", "-"],  # Paired
                    ["A", "-", "G", "G", "H"],  # Paired
                    ["A", "H", "H", "-", "-"],  # Unpaired
                ],
                dtype=np.bytes_,
            ),
            "ins": np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                ]
            ),
            "tax_ids": np.array(["2", "3", "1", "1"]),
            "any_paired": np.array([1, 1, 1, 0], dtype=bool),
            "all_paired": np.array([1, 1, 1, 0], dtype=bool),
            "msa_is_padded_mask": np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                ],
                dtype=bool,
            ),
        },
    },
    {
        # Dense heteromer pairing between chains A and B, where query chain tax ID is 2
        "dense": True,
        "input_msas": {
            "A": {
                "msa": np.array([["A", "C", "G"], ["A", "H", "H"], ["A", "-", "G"], ["L", "C", "G"]], dtype=np.bytes_),
                "ins": np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                "tax_ids": np.array(["query", "3", "1", "4"]),  # Test "query"
                "msa_is_padded_mask": np.zeros((4, 3), dtype=bool),
            },
            "B": {
                "msa": np.array([["G", "T"], ["G", "-"], ["G", "H"], ["G", "L"]], dtype=np.bytes_),
                "ins": np.array([[0, 0], [1, 0], [1, 0], [0, 2]]),
                "tax_ids": np.array(["query", "5", "1", "9"]),
                "msa_is_padded_mask": np.zeros((4, 2), dtype=bool),
            },
            "C": {
                "msa": np.array([["A"], ["B"], ["C"], ["D"], ["E"]], dtype=np.bytes_),
                "ins": np.array([[0], [2], [3], [4], [5]]),
                "tax_ids": np.array(["query", "1", "3", "6", "8"]),
                "msa_is_padded_mask": np.zeros((5, 1), dtype=bool),
            },
        },
        "output_msa": {
            "msa": np.array(
                [
                    # Paired
                    ["A", "C", "G", "G", "T", "A"],  # Query
                    ["A", "H", "H", "-", "-", "C"],  # Partially paired
                    ["A", "-", "G", "G", "H", "B"],  # Fully paired
                    # Unpaired (dense)
                    ["L", "C", "G", "G", "-", "D"],  # Dense
                    ["-", "-", "-", "G", "L", "E"],  # Dense
                ],
                dtype=np.bytes_,
            ),
            "ins": np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 3],
                    [1, 0, 0, 1, 0, 2],
                    [0, 1, 0, 1, 0, 4],
                    [0, 0, 0, 0, 2, 5],
                ]
            ),
            "tax_ids": np.array(["query", "3", "1", "", ""]),
            "any_paired": np.array([1, 1, 1, 0, 0], dtype=bool),
            "all_paired": np.array([1, 0, 1, 0, 0], dtype=bool),
            "msa_is_padded_mask": np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                ],
                dtype=bool,
            ),
        },
    },
    {
        # Pair A to B, then AB to C
        "dense": False,
        "input_msas": {
            "A": {
                "msa": np.array(
                    [["A", "C", "G"], ["A", "H", "H"], ["A", "-", "G"], ["L", "C", "G"], ["C", "G", "T"]],
                    dtype=np.bytes_,
                ),
                "ins": np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]]),
                "tax_ids": np.array(["2", "1", "2", "1", "3"]),  # 2 is query
                "msa_is_padded_mask": np.zeros((5, 3), dtype=bool),
            },
            "B": {
                "msa": np.array(
                    [
                        ["G", "T"],
                        ["G", "-"],
                        ["G", "H"],
                        ["G", "G"],
                        ["C", "T"],
                    ],
                    dtype=np.bytes_,
                ),
                "ins": np.array(
                    [
                        [0, 0],
                        [1, 0],
                        [1, 0],
                        [0, 0],
                        [0, 1],
                    ]
                ),
                "tax_ids": np.array(["2", "3", "2", "1", "4"]),  # 2 is query
                "msa_is_padded_mask": np.zeros((5, 2), dtype=bool),
            },
            "C": {
                "msa": np.array(
                    [
                        ["Y", "Q", "Q"],
                        ["Y", "Q", "R"],
                        ["Q", "Q", "Q"],
                    ],
                    dtype=np.bytes_,
                ),
                "ins": np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ]
                ),
                "tax_ids": np.array(["2", "3", "4"]),  # 2 is query
                "msa_is_padded_mask": np.zeros((3, 3), dtype=bool),
            },
        },
        "output_msa": {
            "msa": np.array(
                [
                    ["A", "C", "G", "G", "T", "Y", "Q", "Q"],  # Query
                    ["C", "G", "T", "G", "-", "Y", "Q", "R"],  # Fully paired (tax_id = 3)
                    ["A", "-", "G", "G", "H", "-", "-", "-"],  # Partially paired (tax_id = 2, query)
                    ["L", "C", "G", "G", "G", "-", "-", "-"],  # Partially paired (tax_id = 1)
                    ["-", "-", "-", "C", "T", "Q", "Q", "Q"],  # Partially paired (tax_id = 4)
                    ["A", "H", "H", "-", "-", "-", "-", "-"],  # Unpaired (tax_id = 1)
                ],
                dtype=np.bytes_,
            ),
            "ins": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],  # Query
                    [0, 0, 0, 1, 0, 0, 0, 1],  # Fully paired (tax_id = 3)
                    [1, 0, 0, 1, 0, 0, 0, 0],  # Partially paired (tax_id = 2, query)
                    [0, 1, 0, 0, 0, 0, 0, 0],  # Partially paired (tax_id = 1)
                    [0, 0, 0, 0, 1, 0, 0, 0],  # Partially paired (tax_id = 4)
                    [0, 0, 1, 0, 0, 0, 0, 0],  # Unpaired (tax_id = 1)
                ]
            ),
            "tax_ids": np.array(["2", "3", "2", "1", "4", "1"]),
            "any_paired": np.array([1, 1, 1, 1, 1, 0], dtype=bool),
            "all_paired": np.array([1, 1, 0, 0, 0, 0], dtype=bool),
            "msa_is_padded_mask": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1],
                ],
                dtype=bool,
            ),
        },
    },
]


def get_tuples(msa, ins, tax_id, msa_is_padded, mask):
    msa_filtered = msa[mask]
    ins_filtered = ins[mask]
    tax_id_filtered = tax_id[mask]
    msa_is_padded_filtered = msa_is_padded[mask]
    return {
        (tuple(msa_filtered[i]), tuple(ins_filtered[i]), tax_id_filtered[i], tuple(msa_is_padded_filtered[i]))
        for i in range(msa_filtered.shape[0])
    }


def assert_msa_results(result, expected_output):
    # Check fully paired MSAs
    fully_paired_result_tuples = get_tuples(
        result["msa"], result["ins"], result["tax_ids"], result["msa_is_padded_mask"], mask=result["all_paired"]
    )
    fully_paired_expected_tuples = get_tuples(
        expected_output["msa"],
        expected_output["ins"],
        expected_output["tax_ids"],
        expected_output["msa_is_padded_mask"],
        mask=expected_output["all_paired"],
    )
    assert fully_paired_result_tuples == fully_paired_expected_tuples

    # Check partially paired MSAs
    any_paired_result_tuples = get_tuples(
        result["msa"], result["ins"], result["tax_ids"], result["msa_is_padded_mask"], mask=result["any_paired"]
    )
    any_paired_expected_tuples = get_tuples(
        expected_output["msa"],
        expected_output["ins"],
        expected_output["tax_ids"],
        expected_output["msa_is_padded_mask"],
        mask=expected_output["any_paired"],
    )
    assert any_paired_result_tuples == any_paired_expected_tuples

    # Check unpaired MSAs
    unpaired_result_tuples = get_tuples(
        result["msa"], result["ins"], result["tax_ids"], result["msa_is_padded_mask"], mask=~result["any_paired"]
    )
    unpaired_expected_tuples = get_tuples(
        expected_output["msa"],
        expected_output["ins"],
        expected_output["tax_ids"],
        expected_output["msa_is_padded_mask"],
        mask=~expected_output["any_paired"],
    )
    assert unpaired_result_tuples == unpaired_expected_tuples

    # Check that the `msa_is_padded_mask` is only ever (1) true where `all_paired` is false and (2) true where a gap token (`-`) is present
    assert np.all(result["msa_is_padded_mask"][result["all_paired"]] == 0)
    assert np.all(result["msa"][result["msa_is_padded_mask"] == 1] == b"-")

    # Check that wherever `msa_is_padded_mask` is true, there are no insertion
    assert np.all(result["ins"][result["msa_is_padded_mask"] == 1] == 0)


@pytest.mark.parametrize("test_case", PAIR_MSA_TEST_CASES)
def test_pair_and_merge_polymer_msas(test_case: dict):
    input_msas = test_case["input_msas"]
    output_msa = test_case["output_msa"]
    result = join_multiple_msas_by_tax_id(
        list(input_msas.values()), unpaired_padding=np.array(["-"], dtype="S"), dense=test_case["dense"]
    )

    # ...assert original result
    assert_msa_results(result, output_msa)

    # Now, we ensure that adjusting the input msa row order only impacts the output msa row order, not the content)
    input_msas["A"]["msa"] = np.concatenate([input_msas["A"]["msa"][:1], input_msas["A"]["msa"][1:][::-1]])
    input_msas["A"]["ins"] = np.concatenate([input_msas["A"]["ins"][:1], input_msas["A"]["ins"][1:][::-1]])
    input_msas["A"]["tax_ids"] = np.concatenate([input_msas["A"]["tax_ids"][:1], input_msas["A"]["tax_ids"][1:][::-1]])
    result_inverted = join_multiple_msas_by_tax_id(
        list(input_msas.values()), unpaired_padding=np.array(["-"], dtype="S"), dense=test_case["dense"]
    )

    # ...assert inverted result
    assert_msa_results(result_inverted, output_msa)


def test_get_matched_indices():
    msa = {
        "tax_ids": np.array([101, 102, 103, 104, 102]),
        "msa": np.array(
            [
                ["A", "T", "G", "C"],  # Query sequence
                ["A", "T", "D", "A"],  # Matches 2 out of 4 with the query
                ["G", "T", "G", "C"],  # Not in shared_tax_ids
                ["A", "A", "A", "C"],  # Matches 2 out of 4 with the query
                ["A", "T", "A", "C"],  # Matches 3 out of 4 with the query
            ]
        ),
    }
    shared_tax_ids = [101, 102, 104]
    expected_output = np.array([0, 3, 4, 1])
    result = _get_matched_indices(msa, shared_tax_ids)
    assert np.array_equal(result, expected_output)


# Constants for tests
REMOVE_EXTRANEOUS_TAX_ID_COPIES_TEST_CASES = [
    # Original test case
    (
        {
            "tax_ids": np.array(["a", "a", "a", "b", "b"]),
        },
        {
            "tax_ids": np.array(["a", "a", "b", "b", "b"]),
        },
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 3, 4]),
        np.array([0, 1, 2, 3]),
    ),
    # Edge case 1: No shared tax_ids
    (
        {
            "tax_ids": np.array(["a", "a", "a", "b", "b"]),
        },
        {
            "tax_ids": np.array(["c", "c", "d", "d", "d"]),
        },
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 3, 4]),
        np.array([]),
        np.array([]),
    ),
    # Edge case 2: All tax_ids match exactly
    (
        {
            "tax_ids": np.array(["a", "a", "b", "b"]),
        },
        {
            "tax_ids": np.array(["a", "a", "b", "b"]),
        },
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
    ),
]


@pytest.mark.parametrize(
    "msa_a, msa_b, i_paired_a, i_paired_b, expected_i_paired_a, expected_i_paired_b",
    REMOVE_EXTRANEOUS_TAX_ID_COPIES_TEST_CASES,
)
def test_remove_extraneous_taxid_copies(msa_a, msa_b, i_paired_a, i_paired_b, expected_i_paired_a, expected_i_paired_b):
    result_i_paired_a, result_i_paired_b = _remove_extraneous_taxid_copies(msa_a, msa_b, i_paired_a, i_paired_b)

    assert np.array_equal(result_i_paired_a, expected_i_paired_a)
    assert np.array_equal(result_i_paired_b, expected_i_paired_b)


MSA_PAIRING_PIPELINE_TEST_CASES = ["3ejj", "1mna", "1hge"]


@pytest.mark.parametrize("pdb_id", MSA_PAIRING_PIPELINE_TEST_CASES)
def test_msa_pairing_pipline(pdb_id: str):
    # Apply initial transforms
    # fmt: off
    pipeline = Compose([
        LoadPolymerMSAs(protein_msa_dirs=PROTEIN_MSA_DIRS, rna_msa_dirs=RNA_MSA_DIRS, max_msa_sequences=100),
    ], track_rng_state=False)
    # fmt: on

    output = pipeline(cached_parse(pdb_id))
    output_before_pairing = copy.deepcopy(output)

    # Pair and merge
    output = PairAndMergePolymerMSAs()(output)

    assert "polymer_msas_by_chain_id" in output

    # Ensure that the shapes are consistent, based on the number of pairs
    chain_id_list = list(output["polymer_msas_by_chain_id"].keys())
    original_num_msa_rows = sum(
        output_before_pairing["polymer_msas_by_chain_id"][chain_id]["msa"].shape[0] for chain_id in chain_id_list
    )
    any_paired_rows = np.sum(output["polymer_msas_by_chain_id"][chain_id_list[0]]["any_paired"])
    final_num_msa_rows = output["polymer_msas_by_chain_id"][chain_id_list[0]]["msa"].shape[0]

    assert original_num_msa_rows >= final_num_msa_rows + any_paired_rows

    # The difference should be a multiple of one of the original MSA sizes (due to monomer pairing)
    msa_sizes = [
        output_before_pairing["polymer_msas_by_chain_id"][chain_id]["msa"].shape[0] for chain_id in chain_id_list
    ]
    difference = original_num_msa_rows - (final_num_msa_rows + any_paired_rows)
    if difference > 0:
        assert any(difference % msa_size == 0 for msa_size in msa_sizes)
    elif difference == 0:
        # There should be no paired rows, if the difference is zero
        assert any_paired_rows == 0
    else:
        # Impossible - there should be no negative difference
        raise AssertionError()

    for chain_id in chain_id_list:
        # Check that the first rows of the MSAs have not been changed
        original_msa_first_row = output_before_pairing["polymer_msas_by_chain_id"][chain_id]["msa"][0]
        paired_msa_first_row = output["polymer_msas_by_chain_id"][chain_id]["msa"][0]
        assert np.array_equal(original_msa_first_row, paired_msa_first_row)

        # Check that the number of insertions has not changed
        original_msa_num_insertions = np.sum(output_before_pairing["polymer_msas_by_chain_id"][chain_id]["ins"])
        paired_msa_num_insertion = np.sum(output["polymer_msas_by_chain_id"][chain_id]["ins"])
        assert original_msa_num_insertions == paired_msa_num_insertion

        # Check that the `msa_is_padded_mask` is only ever (1) true where `all_paired` is false and (2) true where a gap token (`-`) is present
        msa_is_padded_mask = output["polymer_msas_by_chain_id"][chain_id]["msa_is_padded_mask"]
        assert np.all(msa_is_padded_mask[output["polymer_msas_by_chain_id"][chain_id]["all_paired"]] == 0)

        # Check that wherever `msa_is_padded_mask` is True, there are no insertion
        assert np.all(output["polymer_msas_by_chain_id"][chain_id]["ins"][msa_is_padded_mask == 1] == 0)
