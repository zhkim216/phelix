import numpy as np
import pytest

from atomworks.ml.transforms.atom_array import AddGlobalResIdAnnotation
from atomworks.ml.transforms.cached_residue_data import (
    LoadCachedResidueLevelData,
    RandomSubsampleCachedConformers,
    load_cached_residue_level_data,
)
from atomworks.ml.utils.testing import cached_parse


@pytest.fixture
def sample_data_with_global_res_id():
    """Fixture providing sample data with global residue IDs."""
    data = cached_parse("1crn", hydrogen_policy="remove")
    global_res_id_transform = AddGlobalResIdAnnotation()
    return global_res_id_transform(data)


def test_load_with_key_filtering(cache_dir, sample_data_with_global_res_id):
    """Test loading cached data with specific keys only."""
    keys_to_load = ["descriptors", "atom_names"]
    result = load_cached_residue_level_data(
        sample_data_with_global_res_id["atom_array"],
        dir=cache_dir,
        file_extension=".pt",
        keys_to_load=keys_to_load,
        sharding_depth=1,
    )

    cached_data = result["residues"]
    for res_name, res_data in cached_data.items():
        for key in res_data:
            assert key in keys_to_load, f"Unexpected key '{key}' found in {res_name} data"


def test_random_subsample_conformers(cache_dir, sample_data_with_global_res_id):
    """Test random subsampling of conformers"""
    # Load cached residue data (conformers, descriptors, atom_names)
    load_transform = LoadCachedResidueLevelData(dir=cache_dir, sharding_depth=1)
    cached_residue_data = load_transform(sample_data_with_global_res_id)

    assert "residues" in cached_residue_data["cached_residue_level_data"]

    n_conformers_to_sample = 3
    subsample_transform = RandomSubsampleCachedConformers(n_conformers=n_conformers_to_sample)
    data = subsample_transform(cached_residue_data)

    assert "residue_conformer_indices" in data
    indices_dict = data["residue_conformer_indices"]

    atom_array = data["atom_array"]
    for global_res_id, conformer_indices in indices_dict.items():
        assert isinstance(conformer_indices, np.ndarray)
        assert len(conformer_indices) == n_conformers_to_sample

        # Verify that the conformer indices are within bounds
        res_mask = atom_array.res_id_global == global_res_id
        res_name = atom_array.res_name[res_mask][0]
        res_data = data["cached_residue_level_data"]["residues"][res_name]
        n_available = res_data["mol"].GetNumConformers()
        assert all(0 <= idx < n_available for idx in conformer_indices)


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", __file__])
