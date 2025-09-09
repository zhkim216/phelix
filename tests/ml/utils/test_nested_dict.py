"""Tests for nested dictionary utilities."""

import pytest

from atomworks.ml.utils.nested_dict import flatten, get, getitem, unflatten


@pytest.fixture
def nested_dict():
    """Sample nested dictionary for testing."""
    return {"a": {"b": [1, 2]}, "c": {"d": {"e": 3}}, "f": [4, 5]}


@pytest.fixture
def flattened_dict():
    """Sample flattened dictionary with tuple keys."""
    return {("a", "b"): [1, 2], ("c", "d", "e"): 3, ("f",): [4, 5]}


@pytest.fixture
def flattened_dict_str():
    """Sample flattened dictionary with string keys."""
    return {"a.b": [1, 2], "c.d.e": 3, "f": [4, 5]}


class TestFlatten:
    def test_flatten_basic(self, nested_dict, flattened_dict):
        """Test basic flattening with tuple keys."""
        assert flatten(nested_dict) == flattened_dict

    def test_flatten_with_fuse_keys(self, nested_dict, flattened_dict_str):
        """Test flattening with string key fusion."""
        assert flatten(nested_dict, fuse_keys=".") == flattened_dict_str

    def test_flatten_empty_dict(self):
        """Test flattening an empty dictionary."""
        assert flatten({}) == {}

    def test_flatten_single_level(self):
        """Test flattening a single-level dictionary."""
        d = {"a": 1, "b": 2}
        expected = {("a",): 1, ("b",): 2}
        assert flatten(d) == expected


class TestUnflatten:
    def test_unflatten_basic(self, flattened_dict, nested_dict):
        """Test basic unflattening with tuple keys."""
        assert unflatten(flattened_dict) == nested_dict

    def test_unflatten_with_split_keys(self, flattened_dict_str, nested_dict):
        """Test unflattening with string key splitting."""
        assert unflatten(flattened_dict_str, split_keys=".") == nested_dict

    def test_unflatten_empty_dict(self):
        """Test unflattening an empty dictionary."""
        assert unflatten({}) == {}

    def test_unflatten_single_level(self):
        """Test unflattening a single-level dictionary."""
        d = {("a",): 1, ("b",): 2}
        expected = {"a": 1, "b": 2}
        assert unflatten(d) == expected


class TestGet:
    def test_get_existing_value(self, nested_dict):
        """Test getting an existing value."""
        assert get(nested_dict, ("a", "b")) == [1, 2]
        assert get(nested_dict, ("c", "d", "e")) == 3
        assert get(nested_dict, ("f",)) == [4, 5]

    def test_get_missing_value(self, nested_dict):
        """Test getting a missing value returns default."""
        assert get(nested_dict, ("x",), default="missing") == "missing"
        assert get(nested_dict, ("a", "x"), default="missing") == "missing"

    def test_get_empty_key(self, nested_dict):
        """Test getting with empty key tuple."""
        with pytest.raises(KeyError):
            get(nested_dict, ())


class TestGetitem:
    def test_getitem_existing_value(self, nested_dict):
        """Test getting an existing value."""
        assert getitem(nested_dict, ("a", "b")) == [1, 2]
        assert getitem(nested_dict, ("c", "d", "e")) == 3
        assert getitem(nested_dict, ("f",)) == [4, 5]

    def test_getitem_missing_value(self, nested_dict):
        """Test getting a missing value raises KeyError."""
        with pytest.raises(KeyError):
            getitem(nested_dict, ("x",))
        with pytest.raises(KeyError):
            getitem(nested_dict, ("a", "x"))

    def test_getitem_empty_key(self, nested_dict):
        """Test getting with empty key tuple."""
        with pytest.raises(KeyError):
            getitem(nested_dict, ())
