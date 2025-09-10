import pytest

from atomworks.io.parser import parse
from tests.io.conftest import get_pdb_path

TEST_CASES = [
    "5e5j",  # Comes from more than 1 experimental method (X-ray & neutron scattering)
    "1j8z",  # Contains misordered atoms in a residue
    "2fs3",  # Contains an unusual operation expression for assembly building
    "1fp7",  # Contains bonds between crystallization aids in struct_conn
    "6lzb",  # Duplicate index problem with struct_conn (? - presumably crystallization aids/water)
    "5t39",  # Contains misordered atoms in a residue (`SAH`)
    "1nci",  # Issues with arginine resolving, seems to have differing number of NH1/NH2
    "1aym",  # Issues with patching symmetric ligands (invalid literal)
    "8bc3",  # Covalent bond involving chains that were removed during cleaning
    "1twr",  # Residue name not in biotite's CCD
    "6q9t",  # Contains residue `QUK` which uses a mix of `std` and `alt` atom ids
    "8cuy",  # Raises error when parsing CCD CIF's ('NoneType' object has no attribute 'row_count') due to UNL
    "1xvk",  # Raises error when parsing struct_conn record (index 0 is out of bounds for axis 0 with size 0)
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_prior_bugs(pdb_id: str):
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
    )
    assert result is not None  # Check if processing runs through


if __name__ == "__main__":
    pytest.main([__file__])
