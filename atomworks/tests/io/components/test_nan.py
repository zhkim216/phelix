import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack

from atomworks.io.transforms.atom_array import is_any_coord_nan, remove_nan_coords


def test_is_any_coord_nan_atom_array():
    """Test is_any_coord_nan function with AtomArray."""
    # Create an AtomArray with some NaN coordinates
    atom_array = AtomArray(5)
    atom_array.coord = np.array(
        [[1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], [1.0, np.nan, 3.0], [1.0, 2.0, np.nan], [4.0, 5.0, 6.0]]
    )

    result = is_any_coord_nan(atom_array)

    # Expected result: atoms at indices 1, 2, 3 have NaN coordinates
    expected = np.array([False, True, True, True, False])

    # Assert that the result matches the expected output
    np.testing.assert_array_equal(result, expected)


def test_is_any_coord_nan_atom_array_stack():
    """Test is_any_coord_nan function with AtomArrayStack."""
    # Create an AtomArrayStack with some NaN coordinates
    atom_array_stack = AtomArrayStack(2, 4)  # 2 models, 4 atoms

    # First model
    atom_array_stack.coord[0] = np.array([[1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Second model
    atom_array_stack.coord[1] = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])

    result = is_any_coord_nan(atom_array_stack)

    # Expected result: atoms at indices 1 and 2 have NaN coordinates in at least one model
    expected = np.array([False, True, True, False])

    # Assert that the result matches the expected output
    np.testing.assert_array_equal(result, expected)


def test_remove_nan_coords_atom_array():
    """Test remove_nan_coords function with AtomArray."""
    # Create an AtomArray with some NaN coordinates
    atom_array = AtomArray(5)
    atom_array.coord = np.array(
        [[1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], [1.0, np.nan, 3.0], [1.0, 2.0, np.nan], [4.0, 5.0, 6.0]]
    )

    # Add some annotations to ensure they're preserved
    atom_array.set_annotation("chain_id", np.array(["A", "B", "C", "D", "E"]))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))

    result = remove_nan_coords(atom_array)

    # Expected result: only atoms at indices 0 and 4 should remain
    assert result.array_length() == 2
    np.testing.assert_array_equal(result.coord, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_array_equal(result.chain_id, np.array(["A", "E"]))
    np.testing.assert_array_equal(result.res_id, np.array([1, 5]))


def test_remove_nan_coords_atom_array_stack():
    """Test remove_nan_coords function with AtomArrayStack."""
    # Create an AtomArrayStack with some NaN coordinates
    atom_array_stack = AtomArrayStack(2, 4)  # 2 models, 4 atoms

    # First model
    atom_array_stack.coord[0] = np.array([[1.0, 2.0, 3.0], [np.nan, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Second model
    atom_array_stack.coord[1] = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])

    # Add some annotations to ensure they're preserved
    atom_array_stack.set_annotation("chain_id", np.array(["A", "B", "C", "D"]))
    atom_array_stack.set_annotation("res_id", np.array([1, 2, 3, 4]))

    result = remove_nan_coords(atom_array_stack)

    # Expected result: only atoms at indices 0 and 3 should remain
    assert result.stack_depth() == 2
    assert result.array_length() == 2

    # Check coordinates for both models
    np.testing.assert_array_equal(result.coord[0], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_array_equal(result.coord[1], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    # Check annotations
    np.testing.assert_array_equal(result.chain_id, np.array(["A", "D"]))
    np.testing.assert_array_equal(result.res_id, np.array([1, 4]))


def test_all_valid_coords():
    """Test with AtomArray that has no NaN coordinates."""
    # Create an AtomArray with no NaN coordinates
    atom_array = AtomArray(3)
    atom_array.coord = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Test is_any_coord_nan
    result = is_any_coord_nan(atom_array)
    expected = np.array([False, False, False])
    np.testing.assert_array_equal(result, expected)

    # Test remove_nan_coords
    result = remove_nan_coords(atom_array)
    assert result.array_length() == 3
    np.testing.assert_array_equal(result.coord, atom_array.coord)


def test_all_nan_coords():
    """Test with AtomArray that has all NaN coordinates."""
    # Create an AtomArray with all NaN coordinates
    atom_array = AtomArray(3)
    atom_array.coord = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])

    # Test is_any_coord_nan
    result = is_any_coord_nan(atom_array)
    expected = np.array([True, True, True])
    np.testing.assert_array_equal(result, expected)

    # Test remove_nan_coords
    result = remove_nan_coords(atom_array)
    assert result.array_length() == 0


if __name__ == "__main__":
    pytest.main([__file__])
