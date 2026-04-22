"""Pre-process mmCIF files and return a dataframe containing a record for each PN unit in the structure.

See the `atomworks` README for a term glosssary.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
import pandas as pd
from biotite.structure import AtomArray

import atomworks.ml.preprocessing.utils.structure_utils as dp  # to avoid circular imports
from atomworks.common import exists, not_isin
from atomworks.constants import CRYSTALLIZATION_AIDS, METAL_ELEMENTS
from atomworks.enums import ChainType
from atomworks.io import parse
from atomworks.ml.preprocessing.constants import CELL_SIZE, ClashSeverity
from atomworks.ml.utils.misc import hash_sequence
from allatom_design.data.const import VDW_DICT

logger = logging.getLogger("preprocess")


@dataclass
class DataPreprocessor:
    # (Whether examples are from the PDB)
    from_rcsb: bool = True
    # (Cutoff distances)
    close_distance: float = 30.0
    contact_distance: float = 5
    second_shell_distance: float = 8
    clash_distance: float = 1.0
    # (Misc)
    ignore_residues: list[str] = field(default_factory=list)
    # (Efficiency)
    polymer_pn_unit_limit: int = 1000
    # (CIFUtils defaults)
    add_missing_atoms: bool = True
    remove_waters: bool = True
    remove_ccds: list = field(default_factory=lambda: CRYSTALLIZATION_AIDS)
    fix_ligands_at_symmetry_centers: bool = True
    build_assembly: str = "all"
    fix_arginines: bool = True
    convert_mse_to_met: bool = True
    hydrogen_policy: Literal["remove", "infer", "keep"] = "remove"
    add_bond_types_from_struct_conn: list[str] = field(default_factory=lambda: ["covale"]) #! (JH) changed 251016
    # JH changed: metal coordination and small molecule neighboring config
    metal_coordination_cfg: dict | None = None  # e.g. {"coordination_distance": 3.2, "donor_elements": [...]}
    # JH changed: halide coordination config (5A non-C atom count)
    halide_coordination_cfg: dict | None = None  # e.g. {"coordination_distance": 5.0, "halide_res_names": ["F","CL","BR","IOD"]}
    small_molecule_neighboring_distance: float = 5.0  # distance for n_neighboring_heavy_atoms computation

    def __post_init__(self):
        logger.info(f"Initialized DataPreprocessor with the following parameters: {self.__dict__}")

    def _load_structure_with_atomworks(self, path: PathLike) -> dict[str, Any]:
        """Load structure file using CIFUtils parser.

        Supported file types: .cif, .cif.gz, .pdb, .pdb.gz

        Args:
            path (PathLike): The path to the structure file to load.
        """
        self.path = path  # for logging
        return parse(
            filename=path,
            build_assembly=self.build_assembly,
            add_missing_atoms=self.add_missing_atoms,
            remove_waters=self.remove_waters,
            remove_ccds=self.remove_ccds,
            fix_ligands_at_symmetry_centers=self.fix_ligands_at_symmetry_centers,
            fix_arginines=self.fix_arginines,
            convert_mse_to_met=self.convert_mse_to_met,
            hydrogen_policy=self.hydrogen_policy,            
            add_bond_types_from_struct_conn=self.add_bond_types_from_struct_conn, #! (JH) changed 251016
        )

    def _apply_filters(self, atom_array: AtomArray) -> AtomArray:
        """Apply filters to the AtomArray to remove non-biological bonds, ignore residues, and filter out atoms with zero occupancy."""
        # Avoid modifying the original AtomArray
        filtered_atom_array = copy.deepcopy(atom_array)

        # ----- Filter A: Filter out non-polymers with non-biological bonds to polymers ------
        # ... check for non-biological bonds between the current non-polymer PN unit and any polymer (e.g., oxygen-oxygen, etc.)
        inter_pn_unit_bond_mask = dp.get_inter_pn_unit_bond_mask(atom_array)
        if np.sum(inter_pn_unit_bond_mask) > 0:
            pn_units_with_non_biological_bonds = dp.get_pn_units_with_non_biological_bonds(
                atom_array, inter_pn_unit_bond_mask
            )
            if len(pn_units_with_non_biological_bonds) > 0:
                # ... filter out non-polymer chains with non-biological bonds to polymer chains
                non_biological_mask = (
                    np.isin(atom_array.pn_unit_iid, pn_units_with_non_biological_bonds) & ~atom_array.is_polymer
                )
                filtered_atom_array = atom_array[~non_biological_mask]
                logger.warning(f"{self.path}: Non-biological bonds detected between non-polymer and polymer PN units.")

        # ----- Filter B: Apply ignore residue list ------
        processed_ignore_residues = [c.strip() for c in self.ignore_residues]
        mask = not_isin(
            filtered_atom_array.res_name, processed_ignore_residues
        )  # (Applying filter also remove impacted bonds)
        filtered_atom_array = filtered_atom_array[mask]

        # ----- Filter C: Filter out atoms with zero occupancy ------
        filtered_atom_array = filtered_atom_array[filtered_atom_array.occupancy > 0.0]

        return filtered_atom_array

    def get_rows(
        self,
        path_to_structure: PathLike,
        ligand_scores: list[str] = [
            "RSCC",
            "RSR",
            "completeness",
            "intermolecular_clashes",
            "is_best_instance",
            "ranking_model_fit",
            "ranking_model_geometry",
        ],
    ) -> list[dict[str, Any]]:
        """Processes a structure file, applies filters, and generates a list of records to be loaded at train-time.

        We create a record for each PN unit (protein, nucleic acid, or non-polymer) in the structure.
        Each record contains information about a query PN unit and its partner (contacting) PN units in the structure.

        Args:
            path_to_structure (PathLike): The path to the structure file to process. Must be readable by CIFUtils.

        Returns:
            list: A list of dictionaries. Each dictionary contains information about a query PN unit and its partner PN units.
        """
        path_to_structure = Path(path_to_structure)
        if not path_to_structure.exists():
            raise FileNotFoundError(f"File not found: {path_to_structure}")

        result_dict = self._load_structure_with_atomworks(path_to_structure)
        id = result_dict["metadata"]["id"]
        self.id = id  # For logging

        # If applicable, query the RCSB for ligand validity scores
        if self.from_rcsb and exists(ligand_scores) and len(ligand_scores) > 0:
            # Attempt fetching ligand validity scores
            ligand_validity_scores = dp.get_ligand_validity_scores_from_pdb_id(id)
            if exists(ligand_validity_scores) and len(ligand_validity_scores) > 0:
                # ... if successful, filter and format the data
                ligand_validity_scores = pd.DataFrame(ligand_validity_scores)
                ligand_validity_scores = ligand_validity_scores.set_index(["asym_id", "res_name"])[ligand_scores]
                #? (JH) there can be more than two ligands that have the same comp_id (same type of ligand)

                # Ensure the index is sorted lexicographically
                ligand_validity_scores.sort_index(inplace=True)
            else:
                # ... if unsuccessful, log and set to None
                logger.debug(f"Failed to fetch ligand validity scores for ID {id}")
                ligand_validity_scores = None
        else:
            ligand_validity_scores = None

        # Process each assembly
        records = []        
        for assembly_id in result_dict["assemblies"]:
            result = self._generate_pn_unit_metadata_for_assembly(
                result_dict,
                assembly_id,
                ligand_validity_scores,
            )
            if result is not None:
                # The path info should be saved as well
                for row in result:
                    row["path"] = path_to_structure
                records.extend(result)
                                
        return records

    def _generate_pn_unit_metadata_for_assembly(
        self,
        result_dict: dict,
        assembly_id: str,
        ligand_validity_scores: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """Processes an atom array that represents a single assembly and generate a list of metadata records for each PN unit.

        Arguments:
            result_dict (dict): The dictionary containing the output of CIFUtils parser.
            assembly_id (str): The ID of the assembly to process.
            ligand_validity_scores (pd.DataFrame | None): A DataFrame containing ligand validity scores, otherwise None.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each containing metadata about a PN unit in the assembly.
        """
        id = result_dict["metadata"]["id"]
        full_atom_array = result_dict["assemblies"][assembly_id][0]  # Choose the first model

        chain_info_dict = result_dict["chain_info"]

        # ---------- Step 1: Upfront pre-processing ---------- #

        # Re-map PN unit IDs to integers, and keep a dictionary that maps back to the verbose PN unit IDs
        # (We will use the integer IDs for memory efficiency downstream)
        ids_to_remap = ["pn_unit_id", "pn_unit_iid"]
        id_map_dict = {}
        for id_to_remap in ids_to_remap:
            ids = np.unique(full_atom_array.get_annotation(id_to_remap))

            # ... create the new map
            mapped_ids = {old_id: new_id for new_id, old_id in enumerate(ids)}

            # ... apply the new map
            new_ids = np.array(
                [mapped_ids[old_id] for old_id in full_atom_array.get_annotation(id_to_remap)], dtype=np.int16
            )
            full_atom_array.del_annotation(id_to_remap)  # Remove the old annotation (so that we can change the type)
            full_atom_array.set_annotation(id_to_remap, new_ids)

            # ... set the reverse map so we can look up the verbose IDs later
            id_map_dict[id_to_remap] = {new_id: old_id for old_id, new_id in mapped_ids.items()}

        # ---------- Step 2: Apply filters to the AtomArray ---------- #
        filtered_atom_array = self._apply_filters(full_atom_array)

        # ---------- Step 3: Load entry-level criteria ---------- #
        loi_ligand_set = set(result_dict["ligand_info"]["ligand_of_interest"])
        num_polymer_pn_units = len(np.unique(filtered_atom_array.pn_unit_iid[filtered_atom_array.is_polymer]))

        if num_polymer_pn_units > self.polymer_pn_unit_limit:
            logger.warning(
                f"(Example {self.id}): {num_polymer_pn_units} polymer PN units in entry; skipping for performance reasons."
            )
            return None

        # ---------- Step 4: Detect and resolve clashes ---------- #

        # Build cell list for rapid distance computations
        if len(filtered_atom_array) == 0:
            logger.warning(f"(Example {self.id}): No atoms remaining after filtering.")
            return []
        cell_list = struc.CellList(filtered_atom_array, cell_size=CELL_SIZE)

        # Get the unique polymer/non-polymer unit IDs from the AtomArray (we will consider process each moving forward)
        pn_unit_iids_to_consider = np.unique(filtered_atom_array.pn_unit_iid)

        # Find inter-PN unit clashes
        clash_severity = ClashSeverity.NO_CLASH
        clashing_pn_units_set, clashing_pn_units_dict = dp.get_clashing_pn_units(
            pn_unit_iids_to_consider, filtered_atom_array, cell_list, clash_distance=self.clash_distance
        )
        if clashing_pn_units_set:
            logger.warning(
                f"(Example {self.id}): Clash detected between PN units: {[id_map_dict['pn_unit_iid'][pn_unit] for pn_unit in clashing_pn_units_set]}"
            )
            filtered_atom_array, clash_severity = dp.handle_clashing_pn_units(
                clashing_pn_units_set, clashing_pn_units_dict, filtered_atom_array, id_map_dict["pn_unit_iid"]
            )

            # Remake the cell list, since we have removed atoms
            cell_list = struc.CellList(filtered_atom_array, cell_size=CELL_SIZE)

        # JH changed: pre-compute metal coordination partner counts
        if self.metal_coordination_cfg is not None:
            metal_coordination_counts = dp.count_metal_coordination_partners(
                filtered_atom_array, cell_list, **self.metal_coordination_cfg
            )
            metal_coord_distance = self.metal_coordination_cfg.get("coordination_distance", 3.2)
            metal_donor_elements = self.metal_coordination_cfg.get(
                "donor_elements", ("N", "O", "F", "P", "S", "Cl", "As", "Se", "Br", "I")
            )
        else:
            metal_coordination_counts = {}
            metal_coord_distance = 3.2
            metal_donor_elements = ("N", "O", "F", "P", "S", "Cl", "As", "Se", "Br", "I")

        # JH changed: pre-compute halide coordination partner counts
        if self.halide_coordination_cfg is not None:
            halide_coordination_counts = dp.count_halide_coordination_partners(
                filtered_atom_array, cell_list, **self.halide_coordination_cfg
            )
            halide_coord_distance = self.halide_coordination_cfg.get("coordination_distance", 5.0)
            halide_res_names_upper = frozenset(
                n.upper()
                for n in self.halide_coordination_cfg.get(
                    "halide_res_names", ("F", "CL", "BR", "IOD")
                )
            )
        else:
            halide_coordination_counts = {}
            halide_coord_distance = 5.0
            halide_res_names_upper = frozenset()

        # JH changed: pre-compute per-partner masks (donor / non-carbon) over filtered_atom_array.
        # Hydrogens are removed upstream (hydrogen_policy="remove"), so "non-C" == "heavy non-C".
        if len(filtered_atom_array) > 0:
            _elements_upper = np.array([e.upper() for e in filtered_atom_array.element])
            is_donor_partner_mask = np.isin(
                _elements_upper, list(frozenset(e.upper() for e in metal_donor_elements))
            )
            is_non_carbon_partner_mask = _elements_upper != "C"
        else:
            is_donor_partner_mask = np.zeros(0, dtype=bool)
            is_non_carbon_partner_mask = np.zeros(0, dtype=bool)

        # ---------- Step 5: Find contacting/partner PN units and build dataframes ---------- #
        # Loop through all considered query PN units, including proteins (single-chain), nucleic acids (single-chain), and ligands (single- or multi-chain)
        assembly_records = []

        # Filter out pn_units that have length 0 atom arrays
        pn_unit_iids_to_consider = [
            pn_unit_iid
            for pn_unit_iid in pn_unit_iids_to_consider
            if len(filtered_atom_array[filtered_atom_array.pn_unit_iid == pn_unit_iid]) > 0
        ]
        
        for query_pn_unit_iid in pn_unit_iids_to_consider:
            query_pn_unit_atom_array = filtered_atom_array[filtered_atom_array.pn_unit_iid == query_pn_unit_iid]
                        
            assert len(query_pn_unit_atom_array) > 0, f"Query PN unit {query_pn_unit_iid} has zero atoms"

            query_pn_unit_type = ChainType(
                query_pn_unit_atom_array.chain_type[0]
            )  # All chains in a PN unit have the same type

            # Find contacting PN units, which we will use to construct interfaces
            contacting_pn_unit_iids = dp.get_contacting_pn_units(
                query_pn_unit=query_pn_unit_atom_array,
                filtered_atom_array=filtered_atom_array,
                cell_list=cell_list,
                contact_distance=self.contact_distance,
                min_contacts_required=1,
                calculate_min_distance=True,
            )
                        
            # Find close PN units, which will be used to determine which PN units to load at train-time
            close_pn_unit_iids = dp.get_contacting_pn_units(
                query_pn_unit=query_pn_unit_atom_array,
                filtered_atom_array=filtered_atom_array,
                cell_list=cell_list,
                contact_distance=self.close_distance,
                min_contacts_required=1,
                calculate_min_distance=False,
            )
            close_pn_unit_iids = [id_map_dict["pn_unit_iid"][pn_unit["pn_unit_iid"]] for pn_unit in close_pn_unit_iids]

            # Sort contacting PN units by number of contacting atoms and then by minimum distance
            contacting_pn_unit_iids = sorted(
                contacting_pn_unit_iids,
                key=lambda x: (x["num_contacts"], -x["min_distance"]),
                reverse=True,
            )

            # Determine the primary polymer chain, which is the first partner polymer PN unit for non-polymers, or the query PN unit itself for polymers
            if not query_pn_unit_type.is_non_polymer():
                primary_polymer_partner_pn_unit_iid = query_pn_unit_iid
            else:
                polymer_pn_unit_iids = np.unique(filtered_atom_array.pn_unit_iid[filtered_atom_array.is_polymer])
                # Find the first polymer PN unit in the contacting_pn_unit_iids sorted list
                partner_polymer_pn_units = [
                    partner for partner in contacting_pn_unit_iids if partner["pn_unit_iid"] in polymer_pn_unit_iids
                ]
                primary_polymer_partner_pn_unit_iid = (
                    partner_polymer_pn_units[0]["pn_unit_iid"] if len(partner_polymer_pn_units) > 0 else None
                )

            type_specific_criteria = {}
            # For non-polymers, calculate additional information
            if query_pn_unit_type.is_non_polymer():
                bonded_polymer_pn_units = dp.get_bonded_polymer_pn_units(query_pn_unit_iid, filtered_atom_array)

                # ... check is LOI
                residue_names = np.unique(query_pn_unit_atom_array.res_name)
                is_loi = len(loi_ligand_set.intersection(residue_names)) > 0

                # ... check if metal (single atom + metal element)
                is_metal = bool(
                    len(query_pn_unit_atom_array) == 1 and query_pn_unit_atom_array[0].element.upper() in METAL_ELEMENTS
                )
                # JH changed: check if halide (single atom + halide res_name; physically disjoint from metal)
                is_halide = bool(
                    not is_metal
                    and len(query_pn_unit_atom_array) == 1
                    and query_pn_unit_atom_array[0].res_name.upper() in halide_res_names_upper
                )

                # JH changed: per-type total counts and per-partner contact breakdowns
                # (mutually exclusive: metal XOR halide XOR small molecule)
                n_coordination_partners_metal = None
                n_coordination_partners_halide = None
                n_neighboring_heavy_atoms_sm = None
                per_partner_contacts_metal_raw = None
                per_partner_contacts_halide_raw = None
                per_partner_contacts_small_molecule_raw = None

                if is_metal:
                    n_coordination_partners_metal = metal_coordination_counts.get(query_pn_unit_iid, 0)
                    per_partner_contacts_metal_raw = dp.count_per_partner_contacts(
                        query_coord=query_pn_unit_atom_array.coord,
                        query_pn_unit_iids=[query_pn_unit_iid],
                        filtered_atom_array=filtered_atom_array,
                        cell_list=cell_list,
                        distance=metal_coord_distance,
                        partner_mask=is_donor_partner_mask,
                    )
                elif is_halide:
                    n_coordination_partners_halide = halide_coordination_counts.get(query_pn_unit_iid, 0)
                    per_partner_contacts_halide_raw = dp.count_per_partner_contacts(
                        query_coord=query_pn_unit_atom_array.coord,
                        query_pn_unit_iids=[query_pn_unit_iid],
                        filtered_atom_array=filtered_atom_array,
                        cell_list=cell_list,
                        distance=halide_coord_distance,
                        partner_mask=is_non_carbon_partner_mask,
                    )
                else:
                    # Small molecule (non-metal, non-halide, non-polymer)
                    neighbor_mask = dp.get_atom_mask_from_cell_list(
                        query_pn_unit_atom_array.coord, cell_list,
                        len(filtered_atom_array), self.small_molecule_neighboring_distance,
                    )
                    collapsed = np.any(neighbor_mask, axis=0)
                    non_query = not_isin(
                        filtered_atom_array.pn_unit_iid,
                        np.unique(query_pn_unit_atom_array.pn_unit_iid),
                    )
                    n_neighboring_heavy_atoms_sm = int(np.sum(collapsed & non_query))
                    per_partner_contacts_small_molecule_raw = dp.count_per_partner_contacts(
                        query_coord=query_pn_unit_atom_array.coord,
                        query_pn_unit_iids=[query_pn_unit_iid],
                        filtered_atom_array=filtered_atom_array,
                        cell_list=cell_list,
                        distance=self.small_molecule_neighboring_distance,
                        partner_mask=None,
                    )

                # JH changed: decode raw int pn_unit_iid → verbose string via id_map_dict
                def _decode_per_partner(raw_list: list[dict] | None) -> list[dict] | None:
                    if raw_list is None:
                        return None
                    return [
                        {
                            "pn_unit_iid": id_map_dict["pn_unit_iid"][p["pn_unit_iid"]],
                            "chain_iid": p["chain_iid"],
                            "count": p["count"],
                        }
                        for p in raw_list
                    ]

                per_partner_contacts_metal = _decode_per_partner(per_partner_contacts_metal_raw)
                per_partner_contacts_halide = _decode_per_partner(per_partner_contacts_halide_raw)
                per_partner_contacts_small_molecule = _decode_per_partner(
                    per_partner_contacts_small_molecule_raw
                )

                # JH changed: avg_occupancy including missing atoms (occ=0) from full_atom_array
                full_query_atoms = full_atom_array[full_atom_array.pn_unit_iid == query_pn_unit_iid]
                avg_occupancy = float(np.mean(full_query_atoms.occupancy)) if len(full_query_atoms) > 0 else None

                if exists(ligand_validity_scores):
                    _query_pn_unit_ligand_ids = sorted(
                        set(zip(query_pn_unit_atom_array.chain_id, query_pn_unit_atom_array.res_name, strict=False))
                    )

                    # ... subset to the ids that have validity scores
                    _ligands_with_scores = [
                        _id for _id in _query_pn_unit_ligand_ids if _id in ligand_validity_scores.index
                    ]

                    ligand_validity = ligand_validity_scores.loc[_ligands_with_scores].to_dict()
                else:
                    ligand_validity = {}

                # Get the sequence of residues in the non-polymer PN unit
                non_polymer_res_names = struc.get_residues(query_pn_unit_atom_array)[
                    1
                ]  # (`get_residues` returns a tuple of (ids, names))

                # Other options to consider for criteria:
                # -- Get the diameter of the PN unit
                # -- Get whether the query PN unit is coordinated
                # -- Get polar contacts
                # -- Get fraction of atoms in hull

                type_specific_criteria = {
                    "is_metal": is_metal,
                    # JH changed: halide flag
                    "is_halide": is_halide,
                    "is_loi": is_loi,
                    "bonded_polymer_pn_units": bonded_polymer_pn_units,
                    "ligand_validity": ligand_validity,
                    "non_polymer_res_names": non_polymer_res_names,
                    # JH changed: per-type coordination counts
                    "n_coordination_partners_metal": n_coordination_partners_metal,
                    "n_coordination_partners_halide": n_coordination_partners_halide,
                    "n_neighboring_heavy_atoms_small_molecule": n_neighboring_heavy_atoms_sm,
                    "avg_occupancy": avg_occupancy,
                    # JH changed: per-partner contact breakdowns (list of {pn_unit_iid, chain_iid, count})
                    "per_partner_contacts_metal": per_partner_contacts_metal,
                    "per_partner_contacts_halide": per_partner_contacts_halide,
                    "per_partner_contacts_small_molecule": per_partner_contacts_small_molecule,
                }
                
            elif query_pn_unit_type.is_polymer():
                chain_id = query_pn_unit_atom_array.chain_id[0]  # (Polymers have only one chain)
                ec_numbers = chain_info_dict[chain_id].get("ec_numbers", None)
                type_specific_criteria = {
                    "ec_numbers": json.dumps(ec_numbers),
                    "sequence_length": len(
                        chain_info_dict[chain_id]["processed_entity_canonical_sequence"]
                    ),  # (We need to use the processed sequence)
                    "processed_entity_canonical_sequence": chain_info_dict[chain_id][
                        "processed_entity_canonical_sequence"
                    ],
                    "processed_entity_non_canonical_sequence": chain_info_dict[chain_id][
                        "processed_entity_non_canonical_sequence"
                    ],
                }

            # Sequence information
            q_pn_unit_processed_entity_canonical_sequence = type_specific_criteria.get(
                "processed_entity_canonical_sequence", ""
            )
            q_pn_unit_processed_entity_non_canonical_sequence = type_specific_criteria.get(
                "processed_entity_non_canonical_sequence", ""
            )

            # Sequence hashes
            q_pn_unit_processed_entity_canonical_sequence_hash = (
                hash_sequence(q_pn_unit_processed_entity_canonical_sequence)
                if q_pn_unit_processed_entity_canonical_sequence
                else None
            )
            q_pn_unit_processed_entity_non_canonical_sequence_hash = (
                hash_sequence(q_pn_unit_processed_entity_non_canonical_sequence)
                if q_pn_unit_processed_entity_non_canonical_sequence
                else None
            )

            # Resolved residues (we already removed atoms with zero occupancy)
            num_resolved_residues = struc.get_residue_count(query_pn_unit_atom_array)

            # fmt: off
            pn_unit_record = {
                # ...add entry-level data  to the record (e.g., resolution, deposition date, etc.)
                "pdb_id": id if self.from_rcsb else None,
                "assembly_id": assembly_id,
                "clash_severity": clash_severity.value,
                "resolution": result_dict["metadata"]["resolution"],
                "deposition_date": result_dict["metadata"]["deposition_date"],
                "release_date": result_dict["metadata"]["release_date"],
                "method": result_dict["metadata"]["method"],
                "num_polymer_pn_units": num_polymer_pn_units,
                "num_resolved_atoms_in_processed_assembly": len(filtered_atom_array),
                "total_num_atoms_in_unprocessed_assembly": len(full_atom_array),
                "all_pn_unit_iids_after_processing": json.dumps([id_map_dict["pn_unit_iid"][pn_unit_iid] for pn_unit_iid in np.unique(filtered_atom_array.pn_unit_iid)]), # e.g., what we should load at train-time, after resolving clashes

                # ...add the fundamental PN unit-level data to the record directly from the AtomArray
                "q_pn_unit_id": id_map_dict["pn_unit_id"][query_pn_unit_atom_array.pn_unit_id[0]],
                "q_pn_unit_iid": id_map_dict["pn_unit_iid"][query_pn_unit_iid],
                "q_pn_unit_molecule_id": query_pn_unit_atom_array.molecule_id[0],  # All atoms in a PN unit have the same molecule ID
                "q_pn_unit_molecule_iid": query_pn_unit_atom_array.molecule_iid[0], # All atoms in a PN unit have the same molecule IID
                "q_pn_unit_transformation_id": query_pn_unit_atom_array.transformation_id[
                    0
                ],  # All atoms in a PN unit have the same transformation ID
                "q_pn_unit_type": query_pn_unit_type.value,
                "q_pn_unit_is_polymer": query_pn_unit_type.is_polymer(),

                # ...add derived PN unit-level data to the record
                "q_pn_unit_num_resolved_atoms": len(query_pn_unit_atom_array),
                "q_pn_unit_is_multichain": len(np.unique(query_pn_unit_atom_array.chain_id)) > 1,
                "q_pn_unit_is_multiresidue": len(np.unique(query_pn_unit_atom_array.res_id)) > 1,
                "q_pn_unit_num_resolved_residues": num_resolved_residues,
                # (Type-specific) Non-polymer criteria
                "q_pn_unit_is_metal": type_specific_criteria.get("is_metal", False),
                # JH changed: halide flag (False for polymers and non-halide non-polymers)
                "q_pn_unit_is_halide": type_specific_criteria.get("is_halide", False),
                "q_pn_unit_is_loi": type_specific_criteria.get("is_loi", False),
                "q_pn_unit_ligand_validity": type_specific_criteria.get("ligand_validity", {}),
                "q_pn_unit_bonded_polymer_pn_units": {
                    id_map_dict["pn_unit_iid"][pn_unit]
                    for pn_unit in type_specific_criteria.get("bonded_polymer_pn_units", set())
                },  # Covalent modifications
                "q_pn_unit_non_polymer_res_names": ",".join(type_specific_criteria.get("non_polymer_res_names", [])),
                # JH changed: physical annotation columns (None for polymers / non-applicable types)
                "q_pn_unit_n_coordination_partners_metal": type_specific_criteria.get("n_coordination_partners_metal", None),
                "q_pn_unit_n_coordination_partners_halide": type_specific_criteria.get("n_coordination_partners_halide", None),
                "q_pn_unit_n_neighboring_heavy_atoms_small_molecule": type_specific_criteria.get("n_neighboring_heavy_atoms_small_molecule", None),
                "q_pn_unit_avg_occupancy_nonpolymer": type_specific_criteria.get("avg_occupancy", None),
                # JH changed: per-partner contact breakdowns serialized as JSON strings
                # (None for non-applicable rows, json.dumps([]) for applicable rows with no partners)
                "q_pn_unit_per_partner_contacts_metal": (
                    json.dumps(type_specific_criteria["per_partner_contacts_metal"])
                    if type_specific_criteria.get("per_partner_contacts_metal") is not None
                    else None
                ),
                "q_pn_unit_per_partner_contacts_halide": (
                    json.dumps(type_specific_criteria["per_partner_contacts_halide"])
                    if type_specific_criteria.get("per_partner_contacts_halide") is not None
                    else None
                ),
                "q_pn_unit_per_partner_contacts_small_molecule": (
                    json.dumps(type_specific_criteria["per_partner_contacts_small_molecule"])
                    if type_specific_criteria.get("per_partner_contacts_small_molecule") is not None
                    else None
                ),
                # (Type-specific) Polymer criteria
                "q_pn_unit_ec_numbers": type_specific_criteria.get("ec_numbers", []),
                "q_pn_unit_sequence_length": type_specific_criteria.get("sequence_length", None),

                # ...sequences
                "q_pn_unit_processed_entity_canonical_sequence": q_pn_unit_processed_entity_canonical_sequence,
                "q_pn_unit_processed_entity_non_canonical_sequence": q_pn_unit_processed_entity_non_canonical_sequence,

                # ...sequence hashes
                "q_pn_unit_processed_entity_canonical_sequence_hash": q_pn_unit_processed_entity_canonical_sequence_hash,
                "q_pn_unit_processed_entity_non_canonical_sequence_hash": q_pn_unit_processed_entity_non_canonical_sequence_hash,

                # ...partners
                "q_pn_unit_primary_polymer_partner": id_map_dict["pn_unit_iid"][primary_polymer_partner_pn_unit_iid]
                if primary_polymer_partner_pn_unit_iid
                else None,
                "q_pn_unit_contacting_pn_unit_iids": json.dumps(
                    [
                        {
                            "pn_unit_iid": id_map_dict["pn_unit_iid"][partner["pn_unit_iid"]],
                            **{k: v for k, v in partner.items() if k != "pn_unit_iid"},
                        }
                        for partner in contacting_pn_unit_iids
                    ]
                ),
                "q_pn_unit_close_pn_unit_iids": json.dumps(close_pn_unit_iids),
            }
            # fmt: on
            assembly_records.append(pn_unit_record)
        
        return assembly_records