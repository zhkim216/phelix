import os

import biotite.structure as struc
import numpy as np
import pytest

from atomworks.ml.executables import _EXECUTABLES
from atomworks.ml.executables.x3dna import X3DNAFiber
from atomworks.ml.transforms.dna.pad_dna import PadDNA, generate_bform_dna, to_reverse_complement
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse

X3DNA_PATH = os.environ.get("X3DNA", "/projects/ml/prot_dna/x3dna-v2.4")


def _has_x3dna() -> bool:
    """Check if X3DNA is installed."""
    if os.environ.get("X3DNA"):
        try:
            X3DNAFiber.get_or_initialize(os.path.join(X3DNA_PATH, "bin", "fiber"))
            return True
        except FileNotFoundError:
            return False
    return False


has_x3dna = _has_x3dna()


skip_if_no_x3dna = pytest.mark.skipif(
    not has_x3dna,
    reason="X3DNA is not installed",
)


@pytest.fixture
def cleanup_x3dna():
    """
    Reinitialize X3DNAFiber for test isolation.

    This fixture should be used only in tests that require X3DNA, to ensure the X3DNA fiber executable
    singleton is reset and does not persist state between tests.
    """
    X3DNAFiber.reinitialize(os.path.join(X3DNA_PATH, "bin", "fiber"))
    X3DNAFiber._setup(os.path.join(X3DNA_PATH, "bin", "fiber"))  # Needed because pytest deletes per-test env variables
    yield  # run test
    _EXECUTABLES.pop(X3DNAFiber.name, None)
    X3DNAFiber._is_initialized = False


def test_x3dna_manager_fail1():
    """Test the X3DNA manager."""
    with pytest.raises(FileNotFoundError):
        if X3DNAFiber.is_initialized():
            X3DNAFiber.reinitialize("/this/path/is/invalid")
        else:
            X3DNAFiber.initialize("/this/path/is/invalid")


@pytest.mark.requires_x3dna
@skip_if_no_x3dna
def test_x3dna_manager(cleanup_x3dna):
    """Test the X3DNA manager."""
    X3DNAFiber.get_or_initialize(os.path.join(X3DNA_PATH, "bin", "fiber"))
    assert X3DNAFiber.get_bin_path()


def test_to_reverse_complement():
    """Test that the reverse complement of a DNA sequence is generated correctly."""
    assert to_reverse_complement("ATCG") == "CGAT"
    assert to_reverse_complement("atcg") == "CGAT"
    assert to_reverse_complement("ATCGX") == "XCGAT"
    assert to_reverse_complement("AAAAAACCCCCC") == "GGGGGGTTTTTT"


@pytest.mark.requires_x3dna
@skip_if_no_x3dna
def test_generate_bform_dna(cleanup_x3dna):
    """Test that the bform DNA is generated correctly."""
    X3DNAFiber.get_or_initialize(os.path.join(X3DNA_PATH, "bin", "fiber"))
    target_seq = "ATCG"
    atom_array = generate_bform_dna(target_seq)
    ccd_seq = struc.get_residues(atom_array)[1]

    target_ccd_seq = [f"D{nuc}" for nuc in target_seq + to_reverse_complement(target_seq)]

    n_atoms_per_res = struc.apply_residue_wise(atom_array, np.ones(len(atom_array), dtype=int), np.sum, axis=0)
    heavy_atoms_per_res = {
        # ... excluding the OP3 phosphate oxygen, which is part of the leaving group upon polymer formation
        "DA": 21,
        "DT": 20,
        "DC": 19,
        "DG": 22,
    }
    for res_name, n_atoms, target_res_name in zip(ccd_seq, n_atoms_per_res, target_ccd_seq, strict=False):
        assert res_name == target_res_name, f"{res_name} != {target_res_name}"
        assert n_atoms == heavy_atoms_per_res[target_res_name], f"{n_atoms} != {heavy_atoms_per_res[target_res_name]}"


@pytest.mark.requires_x3dna
@skip_if_no_x3dna
@pytest.mark.parametrize("example_id", ["6w13"])
def test_augment_pad_dna(example_id: str, cleanup_x3dna, np_seed: int = 1):
    data = cached_parse(example_id)
    pipe = PadDNA(
        x3dna_path=None,  # Use environment variable instead of directory path
        p_skip=0.0,
    )
    with rng_state(create_rng_state_from_seeds(np_seed=np_seed)):
        data = pipe(data)
        atom_array = data["atom_array"]
        assert atom_array.coord.shape == (5436, 3)


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", __file__])
