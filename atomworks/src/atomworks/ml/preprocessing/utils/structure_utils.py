"""Utilities to create the dataset to be loaded by RF2AA. See the main script for term Glossary."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Collection, Final

import networkx as nx
import numpy as np
import requests
from biotite.structure import AtomArray, CellList
from scipy.spatial.distance import cdist

from atomworks.common import default, not_isin
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER, METAL_ELEMENTS
from atomworks.ml.preprocessing.constants import ClashSeverity

logger = logging.getLogger("preprocess")


def get_pn_units_with_non_biological_bonds(atom_array: AtomArray, bond_mask: np.ndarray) -> np.ndarray:
    """
    Checks for non-biological bonds between PN units within an assembly.
    Note that "inter-PN-unit bonds" in this instance do not include bonds between non-polymers,
    which are treated as the same PN unit (by definition).

    Specifically, this function looks for inter-molecular:
    - Oxygen-oxygen bonds
    - Fluorine-fluorine bonds
    - Bonds involving free oxygen or hydroxyl groups (e.g. HOH, OH, O)

    Args:
        atom_array: AtomArray containing the relevant structure
        bond_mask: Mask of inter-molecular bonds

    Returns:
        numpy.ndarray: Array of polymer/non-polymer unit instance IDs that contain non-biological bonds
    """
    # Get atoms and residues involved in the inter-molecular bonds
    bonds_to_check = atom_array.bonds.as_array()[bond_mask]
    atom_a_elements = atom_array.atomic_number[bonds_to_check[:, 0]]
    atom_b_elements = atom_array.atomic_number[bonds_to_check[:, 1]]
    atom_a_res_names = atom_array.res_name[bonds_to_check[:, 0]]
    atom_b_res_names = atom_array.res_name[bonds_to_check[:, 1]]

    # Check for non-biological bonds
    non_biological_bonds = (
        (atom_a_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["O"])
        & (atom_b_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["O"])  # Oxygen-oxygen bonds
        | (atom_a_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["F"])
        & (atom_b_elements == ELEMENT_NAME_TO_ATOMIC_NUMBER["F"])  # Fluorine-fluorine bonds
        | np.isin(atom_a_res_names, ["HOH", "OH", "O"])  # Bonds involving free oxygen or hydroxyl groups
        | np.isin(atom_b_res_names, ["HOH", "OH", "O"])
    )

    # Get unique polymer/non-polymer unit IDs with non-biological bonds
    return np.unique(
        np.concatenate(
            [
                atom_array.pn_unit_iid[bonds_to_check[:, 0]][non_biological_bonds],
                atom_array.pn_unit_iid[bonds_to_check[:, 1]][non_biological_bonds],
            ]
        )
    )


def get_clashing_pn_units(
    pn_unit_iids_to_consider: np.Array, atom_array: AtomArray, cell_list: CellList, clash_distance: float
) -> tuple[set[str], dict[str, set[str]]]:
    """Wrapper function to find clashing PN units within an atom array."""
    # Process PN unit-by-PN unit to avoid memory issues
    clashing_pn_units_dict = defaultdict(set)
    clashing_pn_units_set = set()
    for query_pn_unit_iid in pn_unit_iids_to_consider:
        query_pn_unit_atom_array = atom_array[atom_array.pn_unit_iid == query_pn_unit_iid]
        clashing_pn_units = get_pn_units_clashing_with_pn_unit(
            query_pn_unit_atom_array, atom_array, cell_list, clash_distance
        )
        if clashing_pn_units:
            clashing_pn_units_set.update(clashing_pn_units | {query_pn_unit_iid})
            clashing_pn_units_dict[query_pn_unit_iid] = set(clashing_pn_units)
    return clashing_pn_units_set, clashing_pn_units_dict


def get_atom_mask_from_cell_list(
    coord: np.array, cell_list: CellList, cell_list_size: int, cutoff: float, chunk_size: int = int(2e9)
) -> np.ndarray:
    """
    Builds a mask indicating which atoms clash with the query PN unit. If the number of comparisons is too large,
    the computation is split into manageable chunks along the rows of `coord`.

    TODO: Update documentation since this is not specific to PN units or clashes.

    Args:
        coord (ndarray): The coordinates of the query PN unit. Shape is (n, 3).
        cell_list (CellList): A CellList object that allows efficient vicinity searches.
        cell_list_size (int): The number of atoms in the cell list.
        clash_distance (float): The distance threshold below which atoms are considered to be clashing.
        chunk_size (int): The maximum number of comparisons allowed in a single chunk.

    Returns:
        ndarray: Mask indicating which atoms in `cell_list` clash with the  atoms in `coord`. Shape is (n, cell_list_size), dtype is bool.
    """
    num_coords = coord.shape[0]
    clashing_atom_mask = np.zeros((num_coords, cell_list_size), dtype=bool)

    max_rows_per_chunk = max(1, chunk_size // cell_list_size)  # Ensure at least 1 row per chunk
    if num_coords * cell_list_size > chunk_size:
        logger.info(
            f"{num_coords * cell_list_size:,} comparisons needed; distance computation split into {math.ceil(num_coords / max_rows_per_chunk)} chunks."
        )
        for i in range(0, num_coords, max_rows_per_chunk):
            end = min(i + max_rows_per_chunk, num_coords)
            clashing_atom_mask[i:end, :] = cell_list.get_atoms(coord[i:end], cutoff, as_mask=True)
    else:
        clashing_atom_mask = cell_list.get_atoms(coord, cutoff, as_mask=True)

    return clashing_atom_mask


def get_pn_units_clashing_with_pn_unit(
    query_pn_unit: AtomArray, filtered_atom_array: AtomArray, cell_list: CellList, clash_distance: float
) -> set[str]:
    """
    Finds clashes between a query PN unit and the rest of the structure.
    A clash is defined as any pair of atoms from the query PN unit and the rest of the structure that are closer than `clash_distance`.

    Args:
        query_pn_unit: AtomArray containing the query PN unit that we want to check for clashes
        filtered_atom_array: AtomArray containing atoms with non-zero occupancy
        cell_list: CellList of structure for rapid distance computations
        clash_distance: Distance threshold for clashing atoms

    Returns:
        Set of polymer/non-polymer unit instance IDs that are clashing with the query PN unit
    """
    clashing_atom_mask = get_atom_mask_from_cell_list(
        query_pn_unit.coord, cell_list, len(filtered_atom_array), clash_distance
    )
    collapsed_mask = np.any(clashing_atom_mask, axis=0)
    clashing_atoms = filtered_atom_array[collapsed_mask]

    # Filter out query PN unit atoms
    query_pn_unit_iids = np.unique(query_pn_unit.pn_unit_iid)
    clashing_atoms = clashing_atoms[not_isin(clashing_atoms.pn_unit_iid, query_pn_unit_iids)]
    return set(np.unique(clashing_atoms.pn_unit_iid))


def handle_clashing_pn_units(
    clashing_pn_units_set: set, clashing_pn_units_dict: dict, atom_array: AtomArray, pn_unit_iid_map: dict
) -> AtomArray:
    """
    Resolves clashing PN units according to the following process:
    1. Sort clashing PN units by the number of atoms within the PN unit
    2. Iterate through the sorted list keeping (1) the larger PN unit (2) the lower transformation number until all clashes are resolved

    Args:
        clashing_pn_units_set: Set of polymer/non-polymer unit instance IDs that contain clashing atoms
        clashing_pn_units_dict: Dictionary mapping polymer/non-polymer unit instance IDs to a list of clashing polymer/non-polymer unit instance IDs
        atom_array: AtomArray containing atoms with non-zero occupancy
        pn_unit_iid_map: Dictionary mapping integer representations of polymer/non-polymer unit instance IDs to the polymer/non-polymer unit instance IDs themselves

    Returns:
        AtomArray: AtomArray with clashing atoms removed
        ClashSeverity: Enum representing the severity of the clash
    """
    pn_units_to_remove = set()
    pn_units_to_keep = set()

    # Build a dictionary of clashing PN unit details
    clashing_pn_unit_details = {}
    for pn_unit in clashing_pn_units_set:
        pn_unit_atom_array = atom_array[atom_array.pn_unit_iid == pn_unit]
        clashing_pn_unit_details[pn_unit] = {
            "num_atoms": len(pn_unit_atom_array),
            "is_metal": pn_unit_atom_array[0].element.upper() in METAL_ELEMENTS,  # JH changed: .upper -> .upper() (was always False)
            "is_polymer": pn_unit_atom_array[0].is_polymer,
        }

    # Sort clashing PN units by the number of atoms in the reference PN unit (high to low) and then the transformation ID (low to high)
    # The last character in the polymer/non-polymer unit instance ID is the transformation ID, even in cases where we have covalently bound chains (e.g., A_1,B_1)
    def sort_key(x: tuple[str, dict[str, Any]]) -> tuple[int, int]:
        try:
            return (x[1]["num_atoms"], -int(pn_unit_iid_map[x[0]][-1]))
        except ValueError:
            return (x[1]["num_atoms"], -ord(pn_unit_iid_map[x[0]][-1]))

    sorted_clashing_pn_units = [x[0] for x in sorted(clashing_pn_unit_details.items(), key=sort_key, reverse=True)]

    # Define the ClashSeverity
    num_polymers = len(
        np.unique(atom_array.pn_unit_iid[atom_array.is_polymer])
    )  # Number of polymer PN units in the structure
    num_clashing_polymers = len(
        np.unique([pn_unit for pn_unit in clashing_pn_units_set if clashing_pn_unit_details[pn_unit]["is_polymer"]])
    )  # Number of clashing polymer PN units

    if num_clashing_polymers / num_polymers > 0.5:
        clash_severity = ClashSeverity.SEVERE
    elif num_clashing_polymers > 0:
        clash_severity = ClashSeverity.MODERATE
    else:
        clash_severity = ClashSeverity.MILD

    # Keep the larger PN unit
    for pn_unit in sorted_clashing_pn_units:
        if pn_unit not in pn_units_to_remove:
            pn_units_to_keep.add(pn_unit)
            pn_units_to_remove.update(clashing_pn_units_dict[pn_unit] - pn_units_to_keep)
    logger.warning(
        f"Removing clashing PN units: {[pn_unit_iid_map[pn_unit] for pn_unit in pn_units_to_remove]} from the structure."
    )

    # Remove clashing PN units
    atom_array = atom_array[not_isin(atom_array.pn_unit_iid, list(pn_units_to_remove))]
    return atom_array, clash_severity


def get_contacting_pn_units(
    query_pn_unit: AtomArray,
    filtered_atom_array: AtomArray,
    cell_list: CellList,
    contact_distance: float = 4.5,
    min_contacts_required: float = 1,
    min_contacts_required_for_metals: float = 3,
    mask: np.ndarray = None,
    calculate_min_distance: bool = False,
    second_shell: bool = False,
) -> list[dict[str, str]]:
    """
    Finds PN units (proteins, nucleic acids, or small molecules) with a minimum number of atoms within a given distance of the query PN unit.

    Args:
        query_pn_unit: AtomArray containing the query PN unit that we want to check for contacts (could be a single chain or multiple covalently bonded chains)
        filtered_atom_array: AtomArray containing the set of atoms to consider when looking for contacts (must correspond to the `cell_list`)
        cell_list: CellList of `atom_array` for rapid distance computations; must correspond to the `filtered_atom_array`
        contact_distance: Distance threshold for contacting atoms
        min_contacts_required: Minimum number of atoms within the cutoff distance to consider a PN unit as a potential partner
        mask: Mask of PN units to consider as potential partners within the atom_array. If None, all PN units (except the query) are considered.
        calculate_min_distance: Whether to calculate the minimum distance between the query PN unit and each PN unit within the cutoff distance

    Returns:
        list: A list of dictionaries, each containing:
            - 'pn_unit_iid' (str): A string representing a unique partner PN unit in contact with the query PN unit.
            - 'num_atoms' (int): Number of non-zero occupancy atoms in the partner PN unit.
            - 'num_contacts' (int): Number of atoms in the partner PN unit within the contact_distance of the query PN unit.
            - 'min_distance' (float): Minimum distance between the query PN unit and the partner PN unit.
    """
    contacting_pn_unit_summary = []

    # ---------- Step 1: Find and count all atoms within contact_distance ---------- #
    full_contacting_atom_mask = get_atom_mask_from_cell_list(
        query_pn_unit.coord, cell_list, len(filtered_atom_array), contact_distance
    )  # (n_query, n_cell_list)
    collapsed_contacting_atom_mask = np.any(full_contacting_atom_mask, axis=0)  # (n_cell_list,)

    # Filter out unwanted atoms (either those in the query PN unit, or those not in the potential partner mask)
    contacting_atoms_mask = (
        mask & collapsed_contacting_atom_mask if mask is not None else collapsed_contacting_atom_mask
    )
    # Create a mask for atoms that are not part of the query PN unit
    non_query_atoms_mask = not_isin(filtered_atom_array.pn_unit_iid, np.unique(query_pn_unit.pn_unit_iid))
    contacting_atoms_mask = contacting_atoms_mask & non_query_atoms_mask
    # Using the final mask to get contacting atoms
    contacting_atoms = filtered_atom_array[contacting_atoms_mask]

    # ---------- Step 2: Calculate the minimum distance to each contacting PN unit ---------- #
    min_distances = {}
    if calculate_min_distance and len(contacting_atoms) > 0:
        # Get a list of PN units within `contact_distance`
        close_atom_pn_unit_ids = np.unique(contacting_atoms.pn_unit_iid)
        atoms_to_check = contacting_atoms

        # Perform distance calculation and store the minimum distance for each PN unit
        distances = cdist(query_pn_unit.coord, atoms_to_check.coord)
        for pn_unit_iid in close_atom_pn_unit_ids:
            mask = atoms_to_check.pn_unit_iid == pn_unit_iid
            min_distances[pn_unit_iid] = np.min(distances[:, mask]) if np.sum(mask) > 0 else None

    # ---------- Step 3: Apply criteria and add to list ---------- #
    for pn_unit_iid in np.unique(contacting_atoms.pn_unit_iid):
        contacting_pn_unit_atoms = contacting_atoms_mask & (filtered_atom_array.pn_unit_iid == pn_unit_iid)
        # Count number of pairwise contacts (which is lower bounded by the number of contacting atoms)
        pairwise_contacts = int(np.sum(full_contacting_atom_mask[:, contacting_pn_unit_atoms]))

        if pairwise_contacts >= min_contacts_required:
            contacting_pn_unit_summary.append(
                {
                    "pn_unit_iid": pn_unit_iid,
                    "num_atoms": len(filtered_atom_array[filtered_atom_array.pn_unit_iid == pn_unit_iid]),
                    "num_contacts": pairwise_contacts,
                    "min_distance": min_distances[pn_unit_iid] if min_distances else None,
                }
            )

    return contacting_pn_unit_summary


# JH changed: added count_metal_coordination_partners
def count_metal_coordination_partners(
    filtered_atom_array: AtomArray,
    cell_list: CellList,
    coordination_distance: float = 3.2,
    donor_elements: Collection[str] = ("N", "O", "F", "P", "S", "Cl", "As", "Se", "Br", "I"),
) -> dict[int, int]:
    """
    Count coordination partners for each single-atom metal PN unit.

    A coordination partner is an atom within ``coordination_distance`` that is either:
    - A donor element (from ``donor_elements``), OR
    - A metal atom from another PN unit.

    The caller is expected to have already applied ``occupancy > 0`` filtering upstream
    (see ``DataPreprocessor._apply_filters``), so no per-partner occupancy threshold is
    applied here.

    Args:
        filtered_atom_array: AtomArray with non-zero occupancy atoms.
        cell_list: CellList of ``filtered_atom_array`` for distance computations.
        coordination_distance: Distance threshold for coordination partners (Angstrom).
        donor_elements: Element symbols considered as donor atoms.

    Returns:
        Dictionary mapping metal ``pn_unit_iid`` → number of coordination partners.
    """
    donor_elements_upper = frozenset(e.upper() for e in donor_elements)

    # 1. Identify all single-atom metal PN units
    pn_unit_iids = np.unique(filtered_atom_array.pn_unit_iid)
    metal_iids: list[int] = []
    for iid in pn_unit_iids:
        atoms = filtered_atom_array[filtered_atom_array.pn_unit_iid == iid]
        if len(atoms) == 1 and atoms[0].element.upper() in METAL_ELEMENTS:
            metal_iids.append(iid)

    if not metal_iids:
        return {}

    # 2. Build valid-partner mask: donor element OR any metal atom (from any metal PN unit).
    # Self is excluded per metal below via `non_self`.
    elements_upper = np.array([e.upper() for e in filtered_atom_array.element])
    is_donor = np.isin(elements_upper, list(donor_elements_upper))

    metal_iid_set = set(metal_iids)
    is_metal_atom = np.array([iid in metal_iid_set for iid in filtered_atom_array.pn_unit_iid])

    valid_partner_mask = is_donor | is_metal_atom

    # 3. Per metal: count coordination partners within distance, excluding self
    result: dict[int, int] = {}
    for metal_iid in metal_iids:
        metal_atoms = filtered_atom_array[filtered_atom_array.pn_unit_iid == metal_iid]
        neighbor_mask = get_atom_mask_from_cell_list(
            metal_atoms.coord, cell_list, len(filtered_atom_array), coordination_distance
        )
        collapsed = np.any(neighbor_mask, axis=0)
        non_self = filtered_atom_array.pn_unit_iid != metal_iid
        result[metal_iid] = int(np.sum(collapsed & non_self & valid_partner_mask))

    return result


# JH changed: added count_halide_coordination_partners
def count_halide_coordination_partners(
    filtered_atom_array: AtomArray,
    cell_list: CellList,
    coordination_distance: float = 5.0,
    halide_res_names: Collection[str] = ("F", "CL", "BR", "IOD"),
) -> dict[int, int]:
    """
    Count coordination partners for each single-atom halide-ion PN unit.

    A halide-ion PN unit is a non-polymer single-atom PN unit whose residue name is one of
    ``halide_res_names`` (PDB ``comp_id``: F, CL, BR, IOD for fluoride/chloride/bromide/iodide).

    A coordination partner is any non-carbon atom within ``coordination_distance`` (regardless
    of modality), excluding atoms belonging to the halide's own PN unit. Hydrogens are assumed
    to have been removed upstream (``hydrogen_policy="remove"``), so "non-C" effectively means
    "heavy non-C".

    Args:
        filtered_atom_array: AtomArray with non-zero occupancy atoms.
        cell_list: CellList of ``filtered_atom_array`` for distance computations.
        coordination_distance: Distance threshold for coordination partners (Angstrom).
        halide_res_names: Residue names (``comp_id``) treated as halide ions.

    Returns:
        Dictionary mapping halide ``pn_unit_iid`` → number of non-C atoms within
        ``coordination_distance``.
    """
    halide_res_names_upper = frozenset(n.upper() for n in halide_res_names)

    # 1. Identify single-atom halide PN units by residue name
    pn_unit_iids = np.unique(filtered_atom_array.pn_unit_iid)
    halide_iids: list[int] = []
    for iid in pn_unit_iids:
        atoms = filtered_atom_array[filtered_atom_array.pn_unit_iid == iid]
        if len(atoms) == 1 and atoms[0].res_name.upper() in halide_res_names_upper:
            halide_iids.append(iid)

    if not halide_iids:
        return {}

    # 2. Build non-C partner mask once
    elements_upper = np.array([e.upper() for e in filtered_atom_array.element])
    is_non_carbon = elements_upper != "C"

    # 3. Per halide: count non-C atoms within distance, excluding self
    result: dict[int, int] = {}
    for halide_iid in halide_iids:
        halide_atoms = filtered_atom_array[filtered_atom_array.pn_unit_iid == halide_iid]
        neighbor_mask = get_atom_mask_from_cell_list(
            halide_atoms.coord, cell_list, len(filtered_atom_array), coordination_distance
        )
        collapsed = np.any(neighbor_mask, axis=0)
        non_self = filtered_atom_array.pn_unit_iid != halide_iid
        result[halide_iid] = int(np.sum(collapsed & non_self & is_non_carbon))

    return result


# JH changed: added count_per_partner_contacts
def count_per_partner_contacts(
    query_coord: np.ndarray,
    query_pn_unit_iids: Collection[int],
    filtered_atom_array: AtomArray,
    cell_list: CellList,
    distance: float,
    partner_mask: np.ndarray | None = None,
) -> list[dict]:
    """
    For a single query (small molecule / metal / halide), list how many atoms each partner
    PN unit contributes within ``distance``, broken down by (pn_unit_iid, chain_iid).

    Args:
        query_coord: Coordinates of the query atoms; shape ``(n_q, 3)``.
        query_pn_unit_iids: pn_unit_iid(s) considered "self" and excluded from the breakdown.
            For a single-pn_unit query, pass ``[query_pn_unit_iid]``.
        filtered_atom_array: AtomArray with ``pn_unit_iid`` and ``chain_iid`` annotations;
            post-filter (occupancy > 0, clashes resolved).
        cell_list: CellList of ``filtered_atom_array``.
        distance: Distance threshold (Angstrom).
        partner_mask: Optional bool mask over ``filtered_atom_array`` restricting eligible
            partner atoms (e.g. donor elements only, or non-C only). Default: all atoms.

    Returns:
        List of ``{"pn_unit_iid": int, "chain_iid": str, "count": int}``, sorted by count
        descending. Empty list if no contacts. ``pn_unit_iid`` is returned as the raw
        remapped integer; callers are expected to decode via ``id_map_dict["pn_unit_iid"]``
        to the verbose string before persisting.
    """
    if len(query_coord) == 0:
        return []

    neighbor_mask = get_atom_mask_from_cell_list(
        query_coord, cell_list, len(filtered_atom_array), distance
    )
    collapsed = np.any(neighbor_mask, axis=0)  # (n_cell_list,)

    # Exclude query's own pn_unit(s) to avoid self-contacts in multi-chain covalent cases
    self_mask = np.isin(filtered_atom_array.pn_unit_iid, list(query_pn_unit_iids))
    eligible = collapsed & ~self_mask
    if partner_mask is not None:
        eligible = eligible & partner_mask

    if not np.any(eligible):
        return []

    partner_pn_unit_iids = filtered_atom_array.pn_unit_iid[eligible]
    partner_chain_iids = filtered_atom_array.chain_iid[eligible]

    # Group (pn_unit_iid, chain_iid) -> atom count
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for p, c in zip(partner_pn_unit_iids.tolist(), partner_chain_iids.tolist()):
        counts[(int(p), str(c))] += 1

    return [
        {"pn_unit_iid": p, "chain_iid": c, "count": n}
        for (p, c), n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]


def get_intra_pn_unit_bonds(pn_unit_iid: str, full_atom_array: AtomArray) -> np.ndarray:
    # NOTE: Not currently used; kept for potential future use
    """
    Retrieve all bonds within the PN unit.
    Includes inter-chain bonds if the PN unit is composed of multiple chains.
    Does NOT include bonds between the PN unit and any other PN units (e.g., protein-ligand bonds).

    Args:
        pn_unit_iid (str): The polymer/non-polymer unit instance ID (e.g., 'A1', 'C2,B3', etc.)
        full_atom_array (AtomArray): The full structure AtomArray. Must include all atoms, including those with zero occupancy.

    Returns:
        numpy.ndarray: Array of Biotite bond objects (atom_a_index, atom_b_index, bond_type) within the specified PN unit.
    """
    all_bonds = full_atom_array.bonds.as_array()
    atom_a_indices, atom_b_indices = all_bonds[:, 0], all_bonds[:, 1]
    mask = (full_atom_array.pn_unit_iid[atom_a_indices] == pn_unit_iid) & (
        full_atom_array.pn_unit_iid[atom_b_indices] == pn_unit_iid
    )
    return all_bonds[mask]


def calculate_molecule_diameter(full_molecule_atoms: AtomArray) -> float:
    # NOTE: Not currently used; kept for potential future use
    """
    Calculates the molecular diameter, defined as the maximum number of bonds between any two vertices
    in the molecule, using a maximum spanning tree.

    Args:
        full_molecule_atoms (AtomArray): The array of atoms representing the molecule. Must include all atoms, including those with zero occupancy.

    Returns:
        float: The molecular diameter. If the diameter cannot be computed, returns 0.0.
    """
    if len(full_molecule_atoms.bonds.as_array() > 0):
        try:
            nx_graph = full_molecule_atoms.bonds.as_graph()  # as a NetworkX graph
            tree = nx.maximum_spanning_tree(nx_graph)
            return nx.diameter(tree)
        except nx.exception.NetworkXError:
            logger.warning("Could not compute diameter.")
            return 0.0
    else:
        return 0.0


def get_soi_ligands_from_pdb_id(pdb_id: str) -> set[str]:
    """
    This function takes a PDB ID and returns a set of ligand names that are annotated as subject of
    investigation (SOI) in the PDB. Such ligands are often considered
    biologically meaningful.

    NOTE: This function is kept from the old pipeline for convenience and testing,
    but not used in the current processing pipeline, where the LOI is extracted directly
    from the cif file instead of via an FTP query.

    Args:
        pdb_id (str): The PDB ID to query.

    Returns:
        Set[str]: A set of ligand names that are annotated as SOI in the PDB.
    """
    try:
        pdb_id = pdb_id.lower()
        core_response = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
        core_json_dict = core_response.json()
        nonpolymer_ids = core_json_dict["rcsb_entry_container_identifiers"].get("non_polymer_entity_ids", [])
        soi_ligand_names = []

        for nonpolymer_id in nonpolymer_ids:
            ligand_response = requests.get(
                f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{nonpolymer_id}"
            )
            ligand_json_dict = ligand_response.json()
            ligand_name = ligand_json_dict["pdbx_entity_nonpoly"]["comp_id"]

            if "rcsb_nonpolymer_entity_annotation" in ligand_json_dict:
                for annotation_dict in ligand_json_dict["rcsb_nonpolymer_entity_annotation"]:
                    if annotation_dict["type"] == "SUBJECT_OF_INVESTIGATION" and annotation_dict["name"] == ligand_name:
                        soi_ligand_names.append(ligand_name)
                        break
    except Exception:
        return set()
    return set(soi_ligand_names)


def get_ligand_validity_scores_from_pdb_id(
    pdb_id: str,
    *,
    timeout: float | tuple[float, float] = (5.0, 30.0),
    num_retries: int = 2,
) -> list[dict[str, str | int | float | None]]:
    """
    Query the RCSB PDB for ligand validity scores for a given PDB ID.

    Args:
        pdb_id (str): The PDB ID to query.
        timeout: requests timeout in seconds. If a tuple, interpreted as (connect, read).
        num_retries: number of retries on network errors.

    Returns:
        records: (list[dict[str, str | int | float | None]]): A list of dictionaries, each containing
            the ligand validity scores for a ligand (e.g. RSCC, RSR) as well identifiers such as the
            residue name, chain ID, and entity ID. Can easily be converted to a pandas DataFrame for
            easier handling via `pd.DataFrame(records)`.

    Reference:
        `RCSB Ligand Structure Quality Guide <https://www.rcsb.org/docs/general-help/ligand-structure-quality-in-pdb-structures>`_
    """
    pdb_graphql_url: Final[str] = "https://data.rcsb.org/graphql"

    # Query string in graphql language to get ligand validity scores from a PDB entry
    ligand_validity_query: Final[str] = """
    query ($id: String!) {
        entry(entry_id:$id){
            nonpolymer_entities {
                rcsb_nonpolymer_entity_container_identifiers {
                    nonpolymer_comp_id
                    rcsb_id
                }
                rcsb_nonpolymer_entity_annotation {
                    type
                }
                nonpolymer_entity_instances {
                    rcsb_nonpolymer_entity_instance_container_identifiers {
                        auth_seq_id
                        auth_asym_id
                        asym_id
                        entity_id
                        entry_id
                    }
                    rcsb_nonpolymer_instance_validation_score {
                        RSCC
                        RSR
                        alt_id
                        completeness
                        intermolecular_clashes
                        is_best_instance
                        mogul_angle_outliers
                        mogul_angles_RMSZ
                        mogul_bond_outliers
                        mogul_bonds_RMSZ
                        ranking_model_fit
                        ranking_model_geometry
                        score_model_fit
                        score_model_geometry
                        stereo_outliers
                        average_occupancy
                        type
                        is_subject_of_investigation
                        is_subject_of_investigation_provenance
                    }
                }
            }
        }
    }
    """

    # Perform the actual query for the target PDB ID.
    # NOTE: Use an explicit timeout + small retry count to avoid indefinite hangs on clusters.
    last_exc: Exception | None = None
    response = None
    for _ in range(max(1, int(num_retries) + 1)):
        try:
            response = requests.post(
                pdb_graphql_url,
                json={"query": ligand_validity_query, "variables": {"id": pdb_id}},
                timeout=timeout,
            )
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            continue

    if response is None:
        logger.debug(f"Query failed for PDB ID {pdb_id}: {repr(last_exc)}")
        return []

    # Extract the records from the response
    records = []
    if response.status_code == 200:
        data = response.json()
        try:
            nonpolymer_entities = default(data["data"]["entry"]["nonpolymer_entities"], [])
            for entity in nonpolymer_entities:
                res_name = entity["rcsb_nonpolymer_entity_container_identifiers"]["nonpolymer_comp_id"]
                for instance in entity.get("nonpolymer_entity_instances", []):
                    record_template = {"res_name": res_name}
                    record_template.update(instance.get("rcsb_nonpolymer_entity_instance_container_identifiers", {}))
                    validation_scores = default(instance["rcsb_nonpolymer_instance_validation_score"], [])
                    for score in validation_scores:
                        record = record_template.copy()
                        record.update(score)
                        records.append(record)
        except KeyError:
            logger.debug(f"No validation scores found for PDB ID: {pdb_id}")
        except TypeError:
            logger.debug(f"No validation scores found for PDB ID: {pdb_id}")
    else:
        logger.debug(f"Query failed with status code {response.status_code} and response: {response.text}")
    return records


def get_inter_pn_unit_bond_mask(atom_array: AtomArray) -> np.ndarray:
    """
    Returns a mask indicating which bonds in `atom_array.bonds` are between two distinct PN units.
    Because we are operating at the PN unit-level, such bonds cannot be bonds between non-polymers.

    WARNING: Must be applied before reassigning PN unit IIDs (e.g., as is done for covalent modifications).

    Args:
        atom_array (AtomArray): The full atom array. Must have PN unit-level annotations.

    Returns:
        numpy.ndarray: A boolean mask indicating which bonds are between two PN units.
    """
    bond_pn_unit_a = atom_array.pn_unit_iid[atom_array.bonds.as_array()[:, 0]]
    bond_pn_unit_b = atom_array.pn_unit_iid[atom_array.bonds.as_array()[:, 1]]
    return bond_pn_unit_a != bond_pn_unit_b


def get_bonded_polymer_pn_units(query_pn_unit_iid: str, filtered_atom_array: AtomArray) -> set[str]:
    """
    Returns a set of polymer PN units that are covalently bonded to a given PN unit.
    For example, useful to detect oligosaccharides that are covalently bonded to a protein.

    Args:
        query_pn_unit_iid (str): The full ID of the non-polymer PN unit to check for bonds.
        filtered_atom_array (AtomArray): AtomArray with non-zero occupancy

    Returns:
        set[str]: A set of full IDs of polymer PN units that are covalently bonded to the query PN unit.
    """
    # Check if the non polymer is covalently bonded to a polymer
    inter_pn_unit_bonds = filtered_atom_array.bonds.as_array()[get_inter_pn_unit_bond_mask(filtered_atom_array)]
    bond_atom_a_pn_unit_iids = filtered_atom_array.pn_unit_iid[inter_pn_unit_bonds[:, 0]]
    bond_atom_b_pn_unit_iids = filtered_atom_array.pn_unit_iid[inter_pn_unit_bonds[:, 1]]

    # Find bonded PN units
    bonded_pn_units = set(bond_atom_b_pn_unit_iids[np.where(bond_atom_a_pn_unit_iids == query_pn_unit_iid)[0]]) | set(
        bond_atom_a_pn_unit_iids[np.where(bond_atom_b_pn_unit_iids == query_pn_unit_iid)[0]]
    )

    # Get the set of polymers
    polymer_pn_unit_iids = set(filtered_atom_array.pn_unit_iid[filtered_atom_array.is_polymer])

    # Set intersection between bonded PN units and polymer PN units
    return bonded_pn_units & polymer_pn_unit_iids
