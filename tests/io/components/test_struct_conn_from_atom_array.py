import numpy as np
import pytest

from atomworks.io.parser import parse
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.bonds import get_struct_conn_dict_from_atom_array
from atomworks.io.utils.io_utils import read_any
from tests.io.conftest import get_pdb_path

# (pdb_id, assembly_id)
TEST_CASES = [
    ("2k33", "1"),  # Contains only covalent bonds in struct_conn
    ("3u8v", "3"),  # Contains only metal bonds in struct_conn
    ("7cu5", "1"),  # Contains a mix of covalent and disulfide bonds in struct_conn
    ("1iv9", "1"),  # Does not contain struct_conn field
]


def sort_bond_array(bond_array):
    """Sort a 2D array of dimension (2, n_bonds), preserving the values within each column (bond)"""

    # Sort the elements within each bond
    bond_array = np.sort(bond_array, axis=0)

    # Sort the bonds themselves
    lex_key = np.flipud(bond_array)
    sorted_indices = np.lexsort(lex_key)
    bond_array = bond_array[:, sorted_indices]

    return bond_array


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_get_struct_conn_dict_from_atom_array(test_case: str):
    pdb_id, assembly_id = test_case
    path = get_pdb_path(pdb_id)

    # Parse AtomArray from CIF
    result = parse(
        filename=path,
        build_assembly=[assembly_id],
        add_bond_types_from_struct_conn=[
            "covale",
            "metalc",
            "disulf",
        ],
    )
    atom_array = result["assemblies"][assembly_id][0]

    # Get the struct_conn_dict from the CIF file
    # NOTE: This will not subset by assembly, so the test assembly must contain all struct_conn bonds in the CIF file
    cif_file = read_any(path)
    struct_conn_dict_from_cif = category_to_dict(cif_file.block, "struct_conn")

    # Get the struct_conn_dict from the AtomArray
    struct_conn_dict_from_atom_array = get_struct_conn_dict_from_atom_array(atom_array)

    # Compare the two struct_conn_dicts
    for key in struct_conn_dict_from_atom_array:
        assert key in struct_conn_dict_from_cif

        # Get the corresponding partner annotation
        if "ptnr1" in key:
            # Sort the bond arrays, since they may be reordered during parsing
            partner_key = key.replace("ptnr1", "ptnr2")
            arr_from_cif = np.stack(
                (
                    struct_conn_dict_from_cif[key],
                    struct_conn_dict_from_cif[partner_key],
                ),
                axis=0,
            )

            # Convert missing or inapplicable ins_codes to empty strings, as is done
            # downstream in get_struct_conn_bonds. This is done here to facilitate testing
            if key.endswith("PDB_ins_code"):
                mask = np.isin(arr_from_cif, np.array([".", "?"]))
                arr_from_cif[mask] = ""

            arr_from_cif = sort_bond_array(arr_from_cif)

            arr_from_atom_array = np.stack(
                (
                    struct_conn_dict_from_atom_array[key],
                    struct_conn_dict_from_atom_array[partner_key],
                ),
                axis=0,
            )
            arr_from_atom_array = sort_bond_array(arr_from_atom_array)
        elif "ptnr2" in key:
            continue
        else:
            # We convert disulfides to covalent if parsed
            if key == "conn_type_id":
                arr_from_cif = np.sort(np.char.replace(struct_conn_dict_from_cif[key], "disulf", "covale"))
            else:
                arr_from_cif = np.sort(struct_conn_dict_from_cif[key])

            arr_from_atom_array = np.sort(struct_conn_dict_from_atom_array[key])

        # Handle the known issue where the label_seq_id is uninformative for non-polymers
        # This leads to mismatches since we sometimes re-label these cases during CIF parsing
        # This makes sorting unreliable, so we fall back to an isin assertion
        if key.endswith("label_seq_id"):
            flattened_arr_from_cif = arr_from_cif.flatten()
            flattened_arr_from_atom_array = arr_from_atom_array.flatten()
            mask = flattened_arr_from_cif != "."
            assert np.isin(
                flattened_arr_from_cif[mask],
                flattened_arr_from_atom_array,
            ).all(), f"Mismatch in {key} for {test_case}"
        else:
            assert np.array_equal(
                arr_from_atom_array,
                arr_from_cif,
            ), f"Mismatch in {key} for {test_case}"


if __name__ == "__main__":
    pytest.main([__file__])
