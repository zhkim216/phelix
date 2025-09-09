import biotite.structure as struc
import numpy as np
import pytest

from atomworks.io.parser import parse
from atomworks.io.template import get_empty_ccd_template
from atomworks.io.utils.bonds import (
    _get_bond_degree_per_atom,
    correct_formal_charges_for_specified_atoms,
    get_inferred_polymer_bonds,
    hash_atom_array,
)
from atomworks.io.utils.ccd import get_chem_comp_leaving_atom_names
from tests.io.conftest import get_pdb_path

LEAVING_GROUP_TEST_CASES = {
    "ALA": {"N": ("H2",), "C": ("OXT", "HXT"), "OXT": ("HXT",)},
    "TYR": {"N": ("H2",), "C": ("OXT", "HXT"), "OXT": ("HXT",)},
}


@pytest.mark.parametrize("ccd_code, expected_leaving_groups", LEAVING_GROUP_TEST_CASES.items())
def test_leaving_group_computation(ccd_code, expected_leaving_groups):
    assert get_chem_comp_leaving_atom_names(ccd_code) == expected_leaving_groups


def test_fix_formal_charge_of_deprotonated_alanine():
    ala = get_empty_ccd_template("ALA", res_id=1, remove_hydrogens=False)
    assert np.array_equal(ala.charge, np.zeros(len(ala)))

    ala_oxt_deprotonated = ala[ala.atom_name != "HXT"]
    assert (
        correct_formal_charges_for_specified_atoms(ala_oxt_deprotonated, np.ones(len(ala) - 1, dtype=bool))[
            ala_oxt_deprotonated.atom_name == "OXT"
        ].charge
        == -1
    )


def test_infer_polymer_bonds():
    residues = []
    n_intra_bonds = []
    for i, ccd_code in enumerate(["ALA", "TYR", "GLY", "SER"]):
        residue = get_empty_ccd_template(ccd_code, res_id=i + 1, chain_id="A", remove_hydrogens=False)
        residues.append(residue)
        n_intra_bonds.append(len(residue.bonds.as_array()))
    atom_array = struc.concatenate(residues)

    assert np.array_equal(atom_array.charge, np.zeros(len(atom_array)))
    assert len(atom_array[atom_array.atom_name == "OXT"]) == 4
    assert sum(n_intra_bonds) == len(atom_array.bonds.as_array())

    # ... make polymer bonds
    polymer_bonds, leaving_atom_idxs = get_inferred_polymer_bonds(atom_array)
    assert len(polymer_bonds) == 3
    assert all(atom_array[leaving_atom_idxs].is_leaving_atom)

    # ... add those bonds to the atom array and remove the leaving atoms
    atom_array.bonds = atom_array.bonds.merge(struc.BondList(len(atom_array), polymer_bonds))
    is_leaving = np.zeros(len(atom_array), dtype=bool)
    is_leaving[leaving_atom_idxs] = True
    atom_array = atom_array[~is_leaving]

    assert len(atom_array[atom_array.atom_name == "OXT"]) == 1
    assert len(atom_array[atom_array.atom_name == "HXT"]) == 1
    assert len(atom_array[atom_array.atom_name == "H2"]) == 1

    # ... fix formal charges
    atom_array = correct_formal_charges_for_specified_atoms(atom_array, np.ones(len(atom_array), dtype=bool))
    assert np.array_equal(atom_array.charge, np.zeros(len(atom_array)))


def test_hash_atom_array():
    arr1 = get_empty_ccd_template("ALA", res_id=1, chain_id="A", remove_hydrogens=False)
    arr2 = arr1.copy()
    assert hash_atom_array(arr1, annotations=None) == hash_atom_array(arr2, annotations=None)
    assert hash_atom_array(arr1, annotations=["atom_name"]) == hash_atom_array(arr2, annotations=["atom_name"])
    assert hash_atom_array(arr1, annotations=["atom_name"], bond_order=True) == hash_atom_array(
        arr2, annotations=["atom_name"], bond_order=True
    )
    # ... invert the order
    invert_order = np.arange(len(arr1))[::-1]
    arr2 = arr1[invert_order]

    # DEBUG: Uncomment for manual inspection
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # from atomworks.io.utils.bonds import _atom_array_to_networkx_graph
    # gs = []
    # annotations = ["element"]
    # for arr in [arr1, arr2]:
    #     g = _atom_array_to_networkx_graph(arr, annotations=annotations, bond_order=True)
    #     gs.append(g)
    # def show_graph(G, figsize=(10, 10), node_attr="node_data", edge_attr="bond_type"):
    #     fig, ax = plt.subplots(figsize=figsize)
    #     pos = nx.kamada_kawai_layout(G)
    #     node_values = [G.nodes[node].get(node_attr, "") for node in G.nodes()]
    #     edge_values = [G[u][v].get(edge_attr, "") for u, v in G.edges()]
    #     unique_node_values = list(set(node_values))
    #     unique_edge_values = list(set(edge_values))
    #     node_colors = [unique_node_values.index(val) for val in node_values]
    #     edge_colors = [unique_edge_values.index(val) for val in edge_values]
    #     nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.tab20)
    #     nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.tab20)
    #     node_labels = {node: G.nodes[node].get(node_attr, "") for node in G.nodes()}
    #     nx.draw_networkx_labels(G, pos, labels=node_labels)
    #     edge_labels = {(u, v): G[u][v].get(edge_attr, "") for u, v in G.edges()}
    #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #     plt.axis("off")
    #     return fig, ax
    # show_graph(gs[0])
    # show_graph(gs[1])

    assert hash_atom_array(arr1, annotations=["atom_name"], bond_order=True) == hash_atom_array(
        arr2, annotations=["atom_name"], bond_order=True
    )
    # ... swap first two atoms
    swap_first_two = np.arange(len(arr1))
    swap_first_two[0], swap_first_two[1] = swap_first_two[1], swap_first_two[0]
    arr2 = arr1[swap_first_two]
    assert hash_atom_array(arr1, annotations=["atom_name"], bond_order=True) == hash_atom_array(
        arr2, annotations=["atom_name"], bond_order=True
    )


@pytest.mark.parametrize("pdb_id", ["1TQH"])
def test_correct_bond_types_for_nucleophilic_additions(pdb_id: str):
    # Example with a nucleophilic addition to a carbonyl carbon that the PDB incorrectly shows as a double bond
    path = get_pdb_path(pdb_id)

    result = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="remove",
        fix_bond_types=False,
    )

    atom_array = result["assemblies"]["1"][0]  # First bioassembly, first model
    carbon_mask = atom_array.element == "C"
    degrees = _get_bond_degree_per_atom(atom_array)
    assert not np.all(degrees[carbon_mask] <= 4), "Example does not show a nucleophilic addition!"

    # Try again, with bond type correction
    result = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="remove",
        fix_bond_types=True,
    )
    atom_array = result["assemblies"]["1"][0]  # First bioassembly, first model
    carbon_mask = atom_array.element == "C"
    degrees = _get_bond_degree_per_atom(atom_array)
    assert np.all(degrees[carbon_mask] <= 4), "Example does not show a nucleophilic addition!"


if __name__ == "__main__":
    test_correct_bond_types_for_nucleophilic_additions("1j8z")
