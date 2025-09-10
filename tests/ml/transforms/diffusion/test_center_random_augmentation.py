import pytest
import torch

from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.center_random_augmentation import CenterRandomAugmentation
from atomworks.ml.transforms.diffusion.batch_structures import BatchStructuresForDiffusionNoising
from atomworks.ml.utils.geometry import masked_center, random_rigid_augmentation


def test_center():
    torch.manual_seed(0)

    coord_atom_lvl = torch.randn(1, 10, 3)
    mask_atom_lvl = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])[None].bool()

    coord_atom_lvl_center = masked_center(coord_atom_lvl, mask_atom_lvl)
    assert torch.allclose(coord_atom_lvl_center[mask_atom_lvl].mean(0), torch.zeros(3), atol=1e-6, rtol=1e-6)


def test_random_augmentation():
    torch.manual_seed(0)

    batch_size = 1
    coord_atom_lvl = torch.randn(batch_size, 10, 3)

    coord_atom_lvl_augmented = random_rigid_augmentation(coord_atom_lvl, batch_size=batch_size)

    assert coord_atom_lvl_augmented.shape == (batch_size, 10, 3)
    assert not torch.allclose(coord_atom_lvl, coord_atom_lvl_augmented)
    # FUTURE: test with kabsch algorithm to align the augmented structure to the original one


@pytest.mark.parametrize("batch_size", [1, 2])
def test_center_random_augmentation(batch_size):
    torch.manual_seed(0)

    coord_atom_lvl = torch.randn(10, 3)
    mask_atom_lvl = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]).bool()
    atom_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    pipe = Compose([BatchStructuresForDiffusionNoising(batch_size=batch_size), CenterRandomAugmentation(batch_size)])
    data = {
        "ground_truth": {"coord_atom_lvl": coord_atom_lvl, "mask_atom_lvl": mask_atom_lvl},
        "atom_array": atom_array,
    }

    data = pipe(data)
    mask_atom_lvl = data["ground_truth"]["mask_atom_lvl"].expand(data["coord_atom_lvl_to_be_noised"].shape[0], -1)
    assert data["coord_atom_lvl_to_be_noised"].shape == (batch_size, 10, 3)
    # make sure the structure was translated
    assert not torch.allclose(
        data["coord_atom_lvl_to_be_noised"][mask_atom_lvl].mean(0), torch.zeros(3), atol=1e-6, rtol=1e-6
    )
    # make sure the structure was rotated
    assert not torch.allclose(coord_atom_lvl, data["coord_atom_lvl_to_be_noised"], atol=1e-6, rtol=1e-6)
