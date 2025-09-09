"""
Includes tests to assert that the data loading pipeline outputs examples that satisfy the assumptions of the AF3 model.
"""

import random

import numpy as np
import pytest
import torch

from tests.conftest import skip_if_on_github_runner
from tests.ml.conftest import TEST_DIFFUSION_BATCH_SIZE


@pytest.fixture
def dataset_config(request):
    """Return the dataset configuration based on the requested fixture."""
    fixture_name = request.param
    dataset = request.getfixturevalue(fixture_name)

    return {"dataset": dataset, "name": fixture_name}


@pytest.mark.slow
@skip_if_on_github_runner
@pytest.mark.parametrize("dataset_config", ["af3_pdb_dataset", "af3_af2fb_distillation_concat_dataset"], indirect=True)
def test_satisfies_af3_dataloading_assumptions(dataset_config):
    """
    Tests that the data loading pipeline outputs examples that satisfy the assumptions of the AF3 model.
    """
    num_random_examples = 3

    dataset = dataset_config["dataset"]
    dataset_name = dataset_config["name"]

    # Set the seed for reproducibility
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Select deterministic examples to profile
    deterministic_indices = np.random.choice(len(dataset), num_random_examples, replace=False)

    for index in deterministic_indices:
        sample = dataset[index]
        example_id = sample["example_id"]

        try:
            assert_satisfies_af3_assumptions(sample)
        except AssertionError as e:
            # Update message with sample index
            rng_state_dict = dataset.get_dataset_by_idx(dataset.id_to_idx(example_id)).transform.latest_rng_state_dict
            raise AssertionError(
                f"Assertion failed for sample {example_id} in dataset {dataset_name}." + "\n" + f"{rng_state_dict}"
            ) from e


def assert_satisfies_af3_assumptions(sample):
    """
    Asserts that the features satisfy the assumptions of the AF3 model.
    """
    n_tokens, n_atoms, n_sequences, n_templates, n_recycles = assert_input_feature_dimensions(sample["feats"])

    assert_ground_truth_dimensions(sample["ground_truth"], n_tokens, n_atoms)
    assert_coordinates_for_noising_dimensions(sample["coord_atom_lvl_to_be_noised"], n_atoms)

    assert sample["t"].shape == (TEST_DIFFUSION_BATCH_SIZE,)
    assert sample["noise"].shape == (TEST_DIFFUSION_BATCH_SIZE, n_atoms, 3)

    return True


def assert_input_feature_dimensions(feats):
    """
    Asserts that the input features have the correct dimensions for the AF3 model.
    """
    # Check the dimensions of the input features
    # assert "f" in feats
    # find I, L and N

    f = feats
    n_token = f["restype"].shape[0]
    n_types_of_tokens = f["restype"].shape[1]
    n_atoms = f["atom_to_token_map"].shape[0]
    n_templates = f["template_restype"].shape[0]
    n_sequences = f["msa_stack"].shape[1]
    n_recycles = f["msa_stack"].shape[0]

    assert f["residue_index"].shape == (n_token,)
    assert f["token_index"].shape == (n_token,)
    assert f["asym_id"].shape == (n_token,)
    assert f["entity_id"].shape == (n_token,)
    assert f["sym_id"].shape == (n_token,)
    assert f["restype"].shape == (n_token, n_types_of_tokens)
    assert f["is_protein"].shape == (n_token,)
    assert f["is_ligand"].shape == (n_token,)
    assert f["is_dna"].shape == (n_token,)
    assert f["is_rna"].shape == (n_token,)

    assert f["ref_pos"].shape == (n_atoms, 3)
    assert f["ref_mask"].shape == (n_atoms,)
    assert f["ref_element"].shape == (
        n_atoms,
        128,
    )
    assert f["ref_charge"].shape == (n_atoms,)
    assert f["ref_atom_name_chars"].shape == (n_atoms, 4, 64)
    assert f["ref_space_uid"].shape == (n_atoms,)

    # templates
    assert f["template_restype"].shape == (
        n_templates,
        n_token,
        n_types_of_tokens,
    )
    assert f["template_pseudo_beta_mask"].shape == (
        n_templates,
        n_token,
    )
    assert f["template_backbone_frame_mask"].shape == (
        n_templates,
        n_token,
    )
    assert f["template_distogram"].shape == (n_templates, n_token, n_token, 39)
    assert f["template_unit_vector"].shape == (n_templates, n_token, n_token, 3)

    # bond feats
    assert f["token_bonds"].shape == (n_token, n_token)

    # msa stack
    assert f["msa_stack"].shape == (n_recycles, n_sequences, n_token, 32 + 2)
    assert f["profile"].shape == (n_token, 32)
    assert f["deletion_mean"].shape == (n_token,)
    return n_token, n_atoms, n_templates, n_sequences, n_recycles


def assert_ground_truth_dimensions(ground_truth, n_tokens, n_atoms):
    """
    Asserts that the ground truth features have the correct dimensions for the AF3 model.
    """
    assert ground_truth["coord_atom_lvl"].shape == (n_atoms, 3)
    assert ground_truth["mask_atom_lvl"].shape == (n_atoms,)
    assert ground_truth["coord_token_lvl"].shape == (n_tokens, 3)
    assert ground_truth["mask_token_lvl"].shape == (n_tokens,)
    assert ground_truth["chain_iid_token_lvl"].shape == (n_tokens,)


def assert_coordinates_for_noising_dimensions(coord_atom_lvl_to_be_noised, n_atoms):
    """
    Asserts that the coordinates that will be noised have the correct dimensions for the AF3 model.
    """
    assert coord_atom_lvl_to_be_noised.shape == (TEST_DIFFUSION_BATCH_SIZE, n_atoms, 3)


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", "-m slow", __file__])
