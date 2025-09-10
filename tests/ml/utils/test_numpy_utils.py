import numpy as np
import pytest

from atomworks.ml.utils.numpy import get_nearest_true_index_for_each_false, insert_data_by_id_, select_data_by_id


def test_select_data_by_id():
    to_ids = np.array([1, 5, 2, 20, 20, 2])

    from_array = np.arange(10).repeat(6).reshape(10, 6)
    from_ids = np.array([1, 2, 5, 6, 7, 21, 22, 23, 20, 25])

    solution = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    selected = select_data_by_id(to_ids, from_ids, from_array)

    assert np.all(selected == solution)


def test_insert_data_by_id():
    to_array = np.zeros((7, 6))
    to_ids = np.array([1, 5, 2, 17, 20, 20, 2])

    from_array = np.arange(10).repeat(6).reshape(10, 6)
    from_ids = np.array([1, 2, 5, 6, 7, 21, 22, 23, 20, 25])

    solution = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    insert_data_by_id_(to_array, to_ids, from_array, from_ids)

    assert np.all(to_array == solution)


@pytest.mark.parametrize(
    "input_array, expected_output",
    [
        (np.array([False, True, True, False, False, True, False]), np.array([1, 2, 5, 5])),
        (np.array([True, False, False, True, False]), np.array([0, 3, 3])),
        (np.array([False, False, False]), np.array([])),  # No Trues, so the result should be an empty array
        (np.array([True, False, True, False]), np.array([0, 2])),
        (np.array([False, True, False, False, True]), np.array([1, 1, 4])),
        (np.array([True, True, True]), np.array([])),  # All Trues, no False to replace
        (np.array([False, True, False, True, False, False]), np.array([1, 1, 3, 3])),
    ],
)
def test_replace_false_with_nearest_true_index(input_array, expected_output):
    result = get_nearest_true_index_for_each_false(input_array)
    assert np.array_equal(result, expected_output)


if __name__ == "__main__":
    pytest.main([__file__])
