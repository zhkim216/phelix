import json

import numpy as np
import pandas as pd
import atomworks.enums as aw_enums
from biotite.structure import AtomArray

from allatom_design.data.datasets.atomworks_sd_dataset import (
    _parse_pn_unit_iids_value,
    add_chain_counts_info,
    add_cluster_balanced_sampling_weights,
    build_interface_df,
)
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise


def _contacts(*pairs):
    return json.dumps([
        {"pn_unit_iid": iid, "min_distance": distance}
        for iid, distance in pairs
    ])


def _row(
    iid,
    *,
    chain_type,
    cluster_id,
    contacts="[]",
    is_protein=False,
    is_nuc=False,
    is_small_molecule=False,
    is_metal=False,
    is_bmsm=False,
    is_bmm=False,
    is_nuc_ligand=False,
    nuc_group_id=None,
    nuc_group_iids=None,
    nuc_group_residues=0,
    nuc_group_cluster_ids=None,
    non_polymer_res_names=None,
    bmsm_ligand_cluster_id=-1,
):
    return {
        "pdb_id": "1abc",
        "assembly_id": "1",
        "path": "/tmp/1abc.cif",
        "q_pn_unit_id": iid.split("_")[0],
        "q_pn_unit_iid": iid,
        "q_pn_unit_type": chain_type,
        "q_pn_unit_sequence_length": nuc_group_residues or 20,
        "q_pn_unit_contacting_pn_unit_iids": contacts,
        "q_pn_unit_is_protein": is_protein,
        "q_pn_unit_is_peptide": False,
        "q_pn_unit_is_nuc": is_nuc,
        "q_pn_unit_is_small_molecule": is_small_molecule,
        "q_pn_unit_is_metal": is_metal,
        "q_pn_unit_is_polymer": is_protein or is_nuc,
        "q_pn_unit_is_biologically_meaningful_small_molecule": is_bmsm,
        "q_pn_unit_is_biologically_meaningful_metal": is_bmm,
        "q_pn_unit_is_nuc_ligand": is_nuc_ligand,
        "q_pn_unit_is_nuc_polymer": is_nuc and not is_nuc_ligand,
        "q_pn_unit_nucleic_acid_group_id": nuc_group_id,
        "q_pn_unit_nucleic_acid_group_iids": nuc_group_iids,
        "q_pn_unit_num_resolved_residues_in_nucleic_acid_group": nuc_group_residues,
        "q_pn_unit_nucleic_acid_group_cluster_ids": nuc_group_cluster_ids,
        "q_pn_unit_cluster_id": cluster_id,
        "q_pn_unit_non_polymer_res_names": non_polymer_res_names,
        "q_pn_unit_bmsm_ligand_cluster_id": bmsm_ligand_cluster_id,
        "biologically_meaningful_pn_unit_iids": ["P_1", "S_1", "A_1", "B_1"],
    }


def test_build_interface_df_adds_one_row_per_nucleic_acid_ligand_group():
    df = pd.DataFrame(
        [
            _row("P_1", chain_type=6, cluster_id=100, is_protein=True),
            _row(
                "S_1",
                chain_type=8,
                cluster_id=200,
                contacts=_contacts(("P_1", 3.8)),
                is_small_molecule=True,
                is_bmsm=True,
                non_polymer_res_names="ATP",
                bmsm_ligand_cluster_id=7,
            ),
            _row(
                "A_1",
                chain_type=3,
                cluster_id=10,
                contacts=_contacts(("B_1", 4.5), ("P_1", 4.0)),
                is_nuc=True,
                is_nuc_ligand=True,
                nuc_group_id="(A_1, B_1)",
                nuc_group_iids="A_1, B_1",
                nuc_group_residues=10,
                nuc_group_cluster_ids="10, 11",
            ),
            _row(
                "B_1",
                chain_type=7,
                cluster_id=11,
                contacts=_contacts(("A_1", 4.5)),
                is_nuc=True,
                is_nuc_ligand=True,
                nuc_group_id="(A_1, B_1)",
                nuc_group_iids="A_1, B_1",
                nuc_group_residues=10,
                nuc_group_cluster_ids="10, 11",
            ),
        ]
    )

    interface_df = build_interface_df(
        df,
        dataset_name="toy",
        ligand_cluster_col="q_pn_unit_bmsm_ligand_cluster_id",
    )

    assert set(interface_df["interface_type"]) == {"bmsm_protein", "nuc_lig_protein"}

    nuc_row = interface_df[interface_df["interface_type"] == "nuc_lig_protein"].iloc[0]
    assert nuc_row["ligand_pn_unit_iids"] == ("A_1", "B_1")
    assert nuc_row["protein_pn_unit_iids"] == ("P_1",)
    assert nuc_row["query_pn_unit_iids"] == ["A_1", "B_1", "P_1"]
    assert nuc_row["ligand_ccd_key"] == ("nuc_seq_cluster", (10, 11))

    counted = add_chain_counts_info(interface_df.copy())
    nuc_counted = counted[counted["interface_type"] == "nuc_lig_protein"].iloc[0]
    assert nuc_counted["n_nuc"] == 1
    assert nuc_counted["n_small_molecule"] == 0


def test_nuc_lig_interface_uses_alpha_protein_nuc_lig():
    monomer_df = pd.DataFrame(
        [{"q_pn_unit_cluster_id": 100, "q_pn_unit_is_protein": True}],
        index=["monomer"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "interface_type": "nuc_lig_protein",
                "protein_cluster_multiset": (100,),
                "ligand_ccd_key": ("nuc_seq_cluster", (10, 11)),
            }
        ],
        index=["iface"],
    )

    _, weighted_interface = add_cluster_balanced_sampling_weights(
        monomer_df=monomer_df,
        interface_df=interface_df,
        alphas_interface={
            "alpha_protein_small_molecule": 0.0,
            "alpha_protein_nuc_lig": 2.0,
        },
        k_percentile=100.0,
    )

    assert weighted_interface.loc["iface", "alpha"] == 2.0
    assert weighted_interface.loc["iface", "sampling_weight"] == 2.0


def test_build_interface_df_adds_bmm_protein_rows():
    df = pd.DataFrame(
        [
            _row("P_1", chain_type=6, cluster_id=100, is_protein=True),
            _row(
                "M_1",
                chain_type=10,
                cluster_id=300,
                contacts=_contacts(("P_1", 2.2)),
                is_metal=True,
                is_bmm=True,
                non_polymer_res_names="MG",
            ),
        ]
    )

    interface_df = build_interface_df(df, dataset_name="toy")

    assert set(interface_df["interface_type"]) == {"bmm_protein"}
    row = interface_df.iloc[0]
    assert row["query_pn_unit_iids"] == ["M_1", "P_1"]
    assert row["ligand_pn_unit_iids"] == ("M_1",)
    assert row["protein_pn_unit_iids"] == ("P_1",)
    assert row["ligand_ccd_key"] == ("ccd", "MG")

    counted = add_chain_counts_info(interface_df.copy())
    assert counted.iloc[0]["n_metal"] == 1
    assert counted.iloc[0]["n_small_molecule"] == 0


def test_bmm_interface_uses_alpha_protein_metal():
    monomer_df = pd.DataFrame(
        [{"q_pn_unit_cluster_id": 100, "q_pn_unit_is_protein": True}],
        index=["monomer"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "interface_type": "bmm_protein",
                "protein_cluster_multiset": (100,),
                "ligand_ccd_key": ("ccd", "MG"),
            }
        ],
        index=["iface"],
    )

    _, weighted_interface = add_cluster_balanced_sampling_weights(
        monomer_df=monomer_df,
        interface_df=interface_df,
        alphas_interface={
            "alpha_protein_metal": 3.0,
        },
        k_percentile=100.0,
    )

    assert weighted_interface.loc["iface", "alpha"] == 3.0
    assert weighted_interface.loc["iface", "sampling_weight"] == 3.0


def test_parse_pn_unit_iids_accepts_numpy_arrays_and_strings():
    assert _parse_pn_unit_iids_value(np.array(["A_1", "B_1"], dtype=object)) == ["A_1", "B_1"]
    assert _parse_pn_unit_iids_value("['A_1', 'B_1']") == ["A_1", "B_1"]
    assert _parse_pn_unit_iids_value(("A_1", "B_1")) == ["A_1", "B_1"]


def _protein_atom_array(res_names):
    atom_names = ["N", "CA", "C", "O"]
    n_protein_atoms = len(res_names) * len(atom_names)
    arr = AtomArray(n_protein_atoms + 1)
    coords = []
    res_id = []
    res_name = []
    atom_name = []
    chain_type = []
    hetero = []
    occupancy = []
    chain_id = []
    pn_unit_iid = []
    is_polymer = []
    is_covalent_modification = []

    for i, rn in enumerate(res_names, start=1):
        base_x = float(i - 1) * 6.0
        for atom_idx, an in enumerate(atom_names):
            coords.append([base_x + atom_idx * 0.1, 0.0, 0.0])
            res_id.append(i)
            res_name.append(rn)
            atom_name.append(an)
            chain_type.append(aw_enums.ChainType.POLYPEPTIDE_L)
            hetero.append(False)
            occupancy.append(1.0)
            chain_id.append("A")
            pn_unit_iid.append("A_1")
            is_polymer.append(True)
            is_covalent_modification.append(False)

    coords.append([0.2, 0.0, 0.0])
    res_id.append(1)
    res_name.append("MG")
    atom_name.append("MG")
    chain_type.append(aw_enums.ChainType.NON_POLYMER)
    hetero.append(True)
    occupancy.append(1.0)
    chain_id.append("B")
    pn_unit_iid.append("B_1")
    is_polymer.append(False)
    is_covalent_modification.append(False)

    arr.coord = np.array(coords, dtype=float)
    arr.res_id = np.array(res_id)
    arr.res_name = np.array(res_name, dtype=object)
    arr.atom_name = np.array(atom_name, dtype=object)
    arr.hetero = np.array(hetero, dtype=bool)
    arr.occupancy = np.array(occupancy, dtype=float)
    arr.chain_id = np.array(chain_id, dtype=object)
    arr.set_annotation("chain_type", np.array(chain_type, dtype=object))
    arr.set_annotation("pn_unit_iid", np.array(pn_unit_iid, dtype=object))
    arr.set_annotation("is_polymer", np.array(is_polymer, dtype=bool))
    arr.set_annotation(
        "is_covalent_modification",
        np.array(is_covalent_modification, dtype=bool),
    )
    return arr


def test_mg_single_atom_pocket_annotation_uses_n_min_ligand_atoms():
    atom_array = _protein_atom_array(["ALA", "GLY"])

    default_annotated = annotate_ligand_pockets(
        atom_array.copy(),
        pocket_distance=5.0,
        n_min_ligand_atoms=5,
        annotation_name="is_ligand_pocket_5_default",
    )
    mg_annotated = annotate_ligand_pockets(
        atom_array.copy(),
        pocket_distance=5.0,
        n_min_ligand_atoms=1,
        annotation_name="is_ligand_pocket_5_mg",
    )

    default_residue_mask = apply_and_spread_residue_wise(
        default_annotated,
        default_annotated.get_annotation("is_ligand_pocket_5_default"),
        function=np.any,
    )
    mg_residue_mask = apply_and_spread_residue_wise(
        mg_annotated,
        mg_annotated.get_annotation("is_ligand_pocket_5_mg"),
        function=np.any,
    )
    assert int(default_residue_mask.sum()) == 0
    assert int(mg_residue_mask[mg_annotated.atom_name == "CA"].sum()) == 1
