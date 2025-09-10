from atomworks.io import parse
from tests.io.conftest import TEST_DATA_IO


def test_structure_with_non_ccd_ligand():
    """Test parsing a structure containing a non-CCD ligand."""
    # Fetch the test structure
    cif_path = TEST_DATA_IO / "9cox_with_unknown_ccd.cif"

    # Parse the structure without CCD mirror path
    structure = parse(cif_path, ccd_mirror_path=None)

    # Basic validation that we got a structure
    assert structure is not None
    assert "asym_unit" in structure
    assert len(structure["asym_unit"]) > 0
    assert "UNKNOWN_CCD" in structure["asym_unit"][0].res_name

    # Optional: Uncomment for visual inspection during development
    # view(structure["asym_unit"][0])
