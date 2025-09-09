from string import ascii_uppercase

import pytest

from atomworks.io.utils.chain import create_chain_id_generator

NEXT_CHAIN_ID_TEST_CASES = [
    {"input": ["A", "B", "C"], "expected": "D"},
    {"input": ["A", "C", "D"], "expected": "B"},
    {"input": list(ascii_uppercase) + ["AA", "AB"], "expected": "AC"},
    {"input": list(ascii_uppercase) + ["ZY", "ZZ"], "expected": "AA"},
    {"input": ["A"], "expected": "B"},  # Single element
    {"input": list(ascii_uppercase)[:-1], "expected": "Z"},  # Single Z element
    {"input": list(ascii_uppercase), "expected": "AA"},  # Increment last element
]


@pytest.mark.parametrize("test_case", NEXT_CHAIN_ID_TEST_CASES)
def test_create_chain_id_generator(test_case):
    input_chain_ids = test_case["input"]
    expected_output = test_case["expected"]
    chain_id_generator = create_chain_id_generator(unavailable_chain_ids=input_chain_ids)
    assert next(chain_id_generator) == expected_output


if __name__ == "__main__":
    pytest.main(["-v", __file__])
