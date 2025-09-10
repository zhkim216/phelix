import numpy as np
import pytest
from biotite.structure import AtomArray

from atomworks.ml.encoding_definitions import RF2AA_ATOM36_ENCODING
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    SortLikeRF2AA,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.crop import CropSpatialLikeAF3
from atomworks.ml.transforms.encoding import EncodeAtomArray
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveTerminalOxygen
from atomworks.ml.transforms.openbabel_utils import AddOpenBabelMoleculesForAtomizedMolecules
from atomworks.ml.transforms.symmetry import (
    CreateSymmetryCopyAxisLikeRF2AA,
    _create_instance_to_entity_map,
    _n_possible_isomorphisms,
    get_isomorphisms_from_symmetry_groups,
    identify_isomorphic_chains_based_on_chain_entity,
    identify_isomorphic_chains_based_on_molecule_entity,
)
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse


def test_create_instance_to_entity_map():
    iids = np.array([0, 1, 2, 3, 4, 5])
    entities = np.array([1, 1, 1, 2, 3, 3])
    expected_output = {1: np.array([0, 1, 2]), 2: np.array([3]), 3: np.array([4, 5])}
    output = _create_instance_to_entity_map(iids, entities)
    for key in expected_output:
        assert np.array_equal(output[key], expected_output[key])


def test_n_possible_isomorphisms():
    group_to_instance_map = {1: [0, 1, 2], 2: [3], 3: [4, 5]}
    expected_output = 12
    output = _n_possible_isomorphisms(group_to_instance_map)
    assert output == expected_output


def test_get_isomorphisms_from_symmetry_groups():
    group_to_instance_map = {1: [0, 1, 2], 2: [3], 3: [4, 5]}
    expected_output = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 2, 1, 3, 4, 5],
            [1, 0, 2, 3, 4, 5],
            [1, 2, 0, 3, 4, 5],
            [2, 0, 1, 3, 4, 5],
            [2, 1, 0, 3, 4, 5],
            [0, 1, 2, 3, 5, 4],
            [0, 2, 1, 3, 5, 4],
            [1, 0, 2, 3, 5, 4],
            [1, 2, 0, 3, 5, 4],
            [2, 0, 1, 3, 5, 4],
            [2, 1, 0, 3, 5, 4],
        ],
        dtype=np.uint32,
    )
    output = get_isomorphisms_from_symmetry_groups(group_to_instance_map)
    assert np.array_equal(output, expected_output)


def test_get_isomorphisms_from_symmetry_groups_with_max_set():
    group_to_instance_map = {1: [0, 1, 2], 2: [3], 3: [4, 5]}
    expected_output = np.array(
        [[0, 1, 2, 3, 4, 5], [0, 2, 1, 3, 4, 5], [1, 0, 2, 3, 4, 5], [0, 1, 2, 3, 5, 4], [0, 2, 1, 3, 5, 4]],
        dtype=np.uint32,
    )
    output = get_isomorphisms_from_symmetry_groups(group_to_instance_map, max_isomorphisms=5)
    assert np.array_equal(output, expected_output)


def test_identify_isomorphic_chains_based_on_molecule_entity():
    atom_array = AtomArray(6)
    atom_array.set_annotation("molecule_iid", np.array([0, 1, 2, 3, 4, 5]))
    atom_array.set_annotation("molecule_entity", np.array([1, 1, 1, 2, 3, 3]))
    expected_output = {1: np.array([0, 1, 2]), 2: np.array([3]), 3: np.array([4, 5])}
    output = identify_isomorphic_chains_based_on_molecule_entity(atom_array)
    for key in expected_output:
        assert np.array_equal(output[key], expected_output[key])


def test_identify_isomorphic_chains_based_on_chain_entity():
    atom_array = AtomArray(6)
    atom_array.set_annotation("chain_iid", np.array([0, 1, 2, 3, 4, 5]))
    atom_array.set_annotation("chain_entity", np.array([1, 1, 1, 2, 3, 3]))
    expected_output = {1: np.array([0, 1, 2]), 2: np.array([3]), 3: np.array([4, 5])}
    output = identify_isomorphic_chains_based_on_chain_entity(atom_array)
    for key in expected_output:
        assert np.array_equal(output[key], expected_output[key])


TEST_CASES = [
    {
        "pdb_id": "5ocm",  # homo-dimer with small molecule that has automorphs
        "crop_size": 160,
        "seed": 1,
        "expected_xyz_shape": (24, 160, 36, 3),
        "expected_mask_shape": (24, 160, 36),
    },
    {
        "pdb_id": "6lyz",  # simple monomer, no symmetry, no small molecules
        "crop_size": 160,
        "seed": 1,
        "expected_xyz_shape": (1, 129, 36, 3),
        "expected_mask_shape": (1, 129, 36),
    },
    {
        "pdb_id": "3sjm",  # crop contains parts of 2 identical protein chains
        "crop_size": 120,
        "seed": 3,
        "expected_xyz_shape": (2, 120, 36, 3),
        "expected_mask_shape": (2, 120, 36),
    },
    {
        "pdb_id": "3sjm",  # only 1 of the two identical protein chains in crop
        "crop_size": 60,
        "seed": 3,
        "expected_xyz_shape": (1, 60, 36, 3),
        "expected_mask_shape": (1, 60, 36),
    },
    {
        "pdb_id": "4i7z",
        "crop_size": 200,
        "seed": 4,
        "expected_xyz_shape": (2, 200, 36, 3),
        "expected_mask_shape": (2, 200, 36),
    },
    {
        "pdb_id": "4res",
        "crop_size": 160,
        "seed": 3,
        "expected_xyz_shape": (2, 160, 36, 3),
        "expected_mask_shape": (2, 160, 36),
    },
    {
        "pdb_id": "7tmj",
        "crop_size": 160,
        "seed": 3,
        "expected_xyz_shape": (2, 72, 36, 3),
        "expected_mask_shape": (2, 72, 36),
    },
]


@pytest.mark.parametrize("case", TEST_CASES)
def test_rf2aa_like_symmetry_encoding(case: dict):
    pdb_id = case["pdb_id"]
    data = cached_parse(pdb_id)
    encoding = RF2AA_ATOM36_ENCODING

    pipe = Compose(
        [
            RemoveHydrogens(),
            AddGlobalAtomIdAnnotation(),
            RemoveTerminalOxygen(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=encoding.tokens),
            SortLikeRF2AA(),
            AddOpenBabelMoleculesForAtomizedMolecules(),
            CropSpatialLikeAF3(crop_size=case["crop_size"], keep_uncropped_atom_array=True),
            EncodeAtomArray(encoding=encoding),
            CreateSymmetryCopyAxisLikeRF2AA(encoding=encoding),
        ],
        track_rng_state=False,
    )

    seed = case["seed"]
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        data = pipe(data)
        print(pdb_id, data["encoded"]["xyz"].shape, data["encoded"]["mask"].shape)
        assert data["encoded"]["xyz"].shape == case["expected_xyz_shape"]
        assert data["encoded"]["mask"].shape == case["expected_mask_shape"]
        # TODO: Spoof the cropped indices and compare to RF2AA


if __name__ == "__main__":
    test_get_isomorphisms_from_symmetry_groups_with_max_set()
