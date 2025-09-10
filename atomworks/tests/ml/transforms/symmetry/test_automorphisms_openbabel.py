import biotite.structure as struc
import pytest
import torch
from openbabel import openbabel

from atomworks.ml.transforms.openbabel_utils import (
    atom_array_to_openbabel,
    find_automorphisms,
    smiles_to_openbabel,
)
from atomworks.ml.transforms.symmetry import apply_automorphs
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state

TEST_CASES = [
    {
        "smiles": "c1ccccc1",
        "expected_automorphisms": 12,
    },
    {
        "smiles": "c1c(O)cccc1(O)",
        "expected_automorphisms": 2,
    },
    {
        "smiles": "COCO",
        "expected_automorphisms": 1,
    },
    {
        # fullerene C60
        "smiles": "c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41",
        "expected_automorphisms": 120,
    },
]


@pytest.mark.parametrize("case", TEST_CASES)
def test_find_automorphisms(case):
    smiles = case["smiles"]
    expected = case["expected_automorphisms"]
    mol = smiles_to_openbabel(smiles)
    automorphisms = find_automorphisms(mol)
    assert len(automorphisms) == expected, f"Failed for SMILES: {smiles}"


@pytest.mark.parametrize("case", TEST_CASES)
def test_create_automorph_permutations(case):
    smiles = case["smiles"]
    mol = smiles_to_openbabel(smiles)
    automorphisms = torch.as_tensor(find_automorphisms(mol))
    assert automorphisms.shape == (len(automorphisms), mol.NumAtoms(), 2)

    # Coord-like data (1 extra dim)
    data = torch.arange(mol.NumAtoms()).view(-1, 1).repeat(1, 3)
    data_automorphs = apply_automorphs(data, automorphisms)
    assert data_automorphs.shape == (len(automorphisms), mol.NumAtoms(), 3)
    for automorph, data_automorph in zip(automorphisms, data_automorphs, strict=False):
        assert automorph.shape == (mol.NumAtoms(), 2)
        assert data_automorph.shape == (mol.NumAtoms(), 3)
        assert torch.equal(data_automorph, automorph[:, 1].unsqueeze(-1).expand(mol.NumAtoms(), 3))

    # Mask-like data (no extra dim)
    data = torch.arange(mol.NumAtoms())
    data_automorphs = apply_automorphs(data, automorphisms)
    assert data_automorphs.shape == (len(automorphisms), mol.NumAtoms())
    for automorph, data_automorph in zip(automorphisms, data_automorphs, strict=False):
        assert automorph.shape == (mol.NumAtoms(), 2)
        assert data_automorph.shape == (mol.NumAtoms(),)
        assert torch.allclose(data_automorph, automorph[:, 1])


@pytest.mark.parametrize("case", TEST_CASES)
def test_manual_create_automorph(case):
    mol = smiles_to_openbabel("c1c(O)cccc1(O)")
    automorphisms = torch.as_tensor(find_automorphisms(mol))

    assert torch.equal(
        automorphisms,
        torch.tensor(
            [
                [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]],
                [[0, 0], [1, 6], [2, 7], [3, 5], [4, 4], [5, 3], [6, 1], [7, 2]],
            ]
        ),
    )

    data = torch.arange(8).view(-1, 1).repeat(1, 3)
    data_automorphs = apply_automorphs(data, automorphisms)

    assert torch.equal(
        data_automorphs,
        torch.tensor(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]],
                [[0, 0, 0], [6, 6, 6], [7, 7, 7], [5, 5, 5], [4, 4, 4], [3, 3, 3], [1, 1, 1], [2, 2, 2]],
            ]
        ),
    )


# fmt: off
def _legacy_get_automorphs(mol, xyz_sm, mask_sm, max_symm=1000):
    """Enumerate atom symmetry permutations.
    Copy pasted from: https://git.ipd.uw.edu/jue/RF2-allatom/-/blob/main/rf2aa/util.py#L1175-1199
    """
    try:
        automorphs = openbabel.vvpairUIntUInt()
        openbabel.FindAutomorphisms(mol, automorphs)

        automorphs = torch.tensor(automorphs)
        n_symmetry = automorphs.shape[0]
        if n_symmetry == 0:
            raise(ValueError("finding automorphs failed"))
        xyz_sm = xyz_sm[None].repeat(n_symmetry,1,1)
        mask_sm = mask_sm[None].repeat(n_symmetry,1)

        xyz_sm = torch.scatter(xyz_sm, 1, automorphs[:,:,0:1].repeat(1,1,3),
                                    torch.gather(xyz_sm,1,automorphs[:,:,1:2].repeat(1,1,3)))
        mask_sm = torch.scatter(mask_sm, 1, automorphs[:,:,0],
                            torch.gather(mask_sm, 1, automorphs[:,:,1]))
    except Exception:
        xyz_sm = xyz_sm[None]
        mask_sm = mask_sm[None]
    if xyz_sm.shape[0] > max_symm:
        xyz_sm = xyz_sm[:max_symm]
        mask_sm = mask_sm[:max_symm]

    return xyz_sm, mask_sm
# fmt: on


@pytest.mark.parametrize("case", TEST_CASES)
def test_vs_legacy_code(case):
    smiles = case["smiles"]
    mol = smiles_to_openbabel(smiles)
    with rng_state(create_rng_state_from_seeds(np_seed=0, torch_seed=0, py_seed=0)):
        xyz_sm = torch.randn(mol.NumAtoms(), 3)
        mask_sm = torch.randn(mol.NumAtoms()) > 0.5
        xyz_sm_legacy, mask_sm_legacy = _legacy_get_automorphs(mol, xyz_sm, mask_sm)

        automorphs = find_automorphisms(mol)
        xyz_sm_new = apply_automorphs(xyz_sm, automorphs)
        mask_sm_new = apply_automorphs(mask_sm, automorphs)

        assert torch.allclose(xyz_sm_legacy, xyz_sm_new)
        assert torch.allclose(mask_sm_legacy, mask_sm_new)
        assert torch.allclose(xyz_sm_new[0], xyz_sm), "First automorph is not the identity"
        assert torch.allclose(mask_sm_new[0], mask_sm), "First automorph is not the identity"


@pytest.mark.parametrize(
    "res_name", ["ALA", "GLY", "PRO", "VAL", "PRO", "TYR", "PHE", "R2R", "BUF", "NAG", "17F", "1I6"]
)
def test_vs_legacy_code_from_ccd(res_name):
    template = struc.info.residue(res_name)

    for atom_array in (template[template.element != "H"], template):
        mol = atom_array_to_openbabel(atom_array, infer_hydrogens=True)  # infer hydrogens removes explicit hydrogens

        # with hydrogens
        with rng_state(create_rng_state_from_seeds(np_seed=0, torch_seed=0, py_seed=0)):
            xyz_sm = torch.randn(mol.NumAtoms(), 3)
            mask_sm = torch.randn(mol.NumAtoms()) > 0.5
            xyz_sm_legacy, mask_sm_legacy = _legacy_get_automorphs(mol, xyz_sm, mask_sm)

            automorphs = find_automorphisms(mol)
            xyz_sm_new = apply_automorphs(xyz_sm, automorphs)
            mask_sm_new = apply_automorphs(mask_sm, automorphs)

            assert torch.allclose(xyz_sm_legacy, xyz_sm_new)
            assert torch.allclose(mask_sm_legacy, mask_sm_new)
            assert torch.allclose(xyz_sm_new[0], xyz_sm), "First automorph is not the identity"
            assert torch.allclose(mask_sm_new[0], mask_sm), "First automorph is not the identity"
