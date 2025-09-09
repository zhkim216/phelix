import numpy as np
import torch

from atomworks.ml.transforms.diffusion.batch_structures import BatchStructuresForDiffusionNoising


def test_batch_structures():
    batch_size = 2
    coord_atom_lvl = torch.randn(10, 3)
    mask_atom_lvl = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]).bool()

    atom_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # ...test when we don't already have the key `coord_atom_lvl_to_be_noised`
    transform = BatchStructuresForDiffusionNoising(batch_size)
    data = {
        "ground_truth": {"coord_atom_lvl": coord_atom_lvl, "mask_atom_lvl": mask_atom_lvl},
        "atom_array": atom_array,
    }

    data = transform(data)

    assert data["ground_truth"]["coord_atom_lvl"].shape == (10, 3)  # No change to the original structure
    assert data["ground_truth"]["mask_atom_lvl"].shape == (10,)  # No change to the original structure

    assert data["coord_atom_lvl_to_be_noised"].shape == (batch_size, 10, 3)
    assert torch.allclose(data["coord_atom_lvl_to_be_noised"][0], data["ground_truth"]["coord_atom_lvl"])

    # ...test when we already have the key `coord_atom_lvl_to_be_noised`
    coord_atom_lvl_to_be_noised = coord_atom_lvl.clone()
    coord_atom_lvl_to_be_noised[0, :] = float("nan")  # Set the first atom's coordinates in the first batch to NaN
    data = {
        "ground_truth": {"coord_atom_lvl": coord_atom_lvl, "mask_atom_lvl": mask_atom_lvl},
        "atom_array": atom_array,
        "coord_atom_lvl_to_be_noised": coord_atom_lvl_to_be_noised,
    }

    data = transform(data)

    assert data["ground_truth"]["coord_atom_lvl"].shape == (10, 3)  # No change to the original structure
    assert data["ground_truth"]["mask_atom_lvl"].shape == (10,)  # No change to the original structure

    assert data["coord_atom_lvl_to_be_noised"].shape == (batch_size, 10, 3)
    assert not torch.allclose(data["coord_atom_lvl_to_be_noised"][0], data["ground_truth"]["coord_atom_lvl"])
    assert torch.allclose(data["coord_atom_lvl_to_be_noised"][1], coord_atom_lvl_to_be_noised, equal_nan=True)
