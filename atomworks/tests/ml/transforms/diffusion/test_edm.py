import pickle

import torch

from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.diffusion.batch_structures import BatchStructuresForDiffusionNoising
from atomworks.ml.transforms.diffusion.edm import sample_noise_edm, sample_t_edm
from tests.ml.conftest import TEST_DATA_ML


def test_edm_t_sampling():
    torch.manual_seed(0)
    diffusion_batch_size = 1000
    sigma_data = 1.0

    t = sample_t_edm(sigma_data, diffusion_batch_size)
    assert t.shape == (diffusion_batch_size,)

    # regression test; does the distribution match?

    saved_result_path = TEST_DATA_ML / "edm_t_sampled.pkl"
    # Uncomment to save t for regression tests, as a pickle (JSON is too slow)
    # with open(saved_result_path, "wb") as f:
    # pickle.dump(t, f)

    with open(saved_result_path, "rb") as f:
        expected_t = pickle.load(f)
    assert torch.allclose(
        t,
        expected_t,
        atol=1e-4,
        rtol=1e-4,
    )


def test_edm_noise_sampling():
    torch.manual_seed(0)
    num_atoms = 10
    diffusion_batch_size = 1000
    sigma_data = 1.0

    t = torch.randn(diffusion_batch_size) * sigma_data  # spoofing t sampling to decouple tests
    noise = sample_noise_edm(t, num_atoms)
    assert noise.shape == (diffusion_batch_size, num_atoms, 3)

    # regression test; does the distribution match?

    saved_result_path = TEST_DATA_ML / "edm_noise_sampled.pkl"
    # Uncomment to save noise for regression tests, as a pickle (JSON is too slow)
    # with open(saved_result_path, "wb") as f:
    # pickle.dump(noise, f)

    with open(saved_result_path, "rb") as f:
        expected_noise = pickle.load(f)
    assert torch.allclose(
        noise,
        expected_noise,
        atol=1e-4,
        rtol=1e-4,
    )


def test_sample_edm_noise_transform():
    torch.manual_seed(0)
    num_atoms = 10
    diffusion_batch_size = 1000
    sigma_data = 1.0

    from atomworks.ml.transforms.diffusion.edm import SampleEDMNoise

    transform = Compose(
        [BatchStructuresForDiffusionNoising(diffusion_batch_size), SampleEDMNoise(sigma_data, diffusion_batch_size)]
    )
    data = {
        "ground_truth": {
            "coord_atom_lvl": torch.randn(num_atoms, 3),
            "mask_atom_lvl": torch.ones(num_atoms).bool(),
        },
        "atom_array": list(range(num_atoms)),
    }

    data = transform(data)
    assert "t" in data
    assert "noise" in data
    assert data["t"].shape == (diffusion_batch_size,)
    assert data["noise"].shape == (diffusion_batch_size, num_atoms, 3)
