import numpy as np

from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path


def test_charges():
    pdb_path = get_pdb_path("6lyz")

    result = parse(filename=pdb_path, build_assembly=None, add_missing_atoms=True, fix_formal_charges=True)
    atom_array = result["asym_unit"][0]

    # ... 6lyz is a pure protein backbone, so no charges should be present
    assert np.all(atom_array[atom_array.is_backbone_atom].charge == 0)

    # ... but the sidechains should have charges
    assert np.any(atom_array[~atom_array.is_backbone_atom].charge != 0)

    # ... assert that the charges are within the expected range
    assert np.all(atom_array.charge >= -1)
    assert np.all(atom_array.charge <= 1)
