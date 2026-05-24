import json

import pandas as pd

from allatom_design.utils.metadata_utils import compute_nucleic_acid_groups_per_pdb


def _contacts(*pairs):
    return json.dumps([
        {"pn_unit_iid": iid, "min_distance": distance}
        for iid, distance in pairs
    ])


def test_nucleic_acid_groups_use_connected_group_residue_count():
    df = pd.DataFrame(
        [
            {
                "q_pn_unit_iid": "A_1",
                "q_pn_unit_type": 3,
                "q_pn_unit_contacting_pn_unit_iids": _contacts(("B_1", 4.5)),
                "q_pn_unit_num_resolved_residues": 6,
                "q_pn_unit_cluster_id": 10,
            },
            {
                "q_pn_unit_iid": "B_1",
                "q_pn_unit_type": 3,
                "q_pn_unit_contacting_pn_unit_iids": _contacts(("A_1", 4.5)),
                "q_pn_unit_num_resolved_residues": 6,
                "q_pn_unit_cluster_id": 11,
            },
            {
                "q_pn_unit_iid": "C_1",
                "q_pn_unit_type": 7,
                "q_pn_unit_contacting_pn_unit_iids": _contacts(("D_1", 4.5)),
                "q_pn_unit_num_resolved_residues": 5,
                "q_pn_unit_cluster_id": 12,
            },
            {
                "q_pn_unit_iid": "D_1",
                "q_pn_unit_type": 4,
                "q_pn_unit_contacting_pn_unit_iids": _contacts(("C_1", 4.5)),
                "q_pn_unit_num_resolved_residues": 5,
                "q_pn_unit_cluster_id": 13,
            },
            {
                "q_pn_unit_iid": "E_1",
                "q_pn_unit_type": 3,
                "q_pn_unit_contacting_pn_unit_iids": _contacts(("F_1", 4.6)),
                "q_pn_unit_num_resolved_residues": 5,
                "q_pn_unit_cluster_id": 14,
            },
            {
                "q_pn_unit_iid": "F_1",
                "q_pn_unit_type": 3,
                "q_pn_unit_contacting_pn_unit_iids": _contacts(("E_1", 4.6)),
                "q_pn_unit_num_resolved_residues": 5,
                "q_pn_unit_cluster_id": 15,
            },
        ]
    )

    out = compute_nucleic_acid_groups_per_pdb(df, nucleic_acid_dist_threshold=4.5)

    ab = out.loc[df["q_pn_unit_iid"].isin(["A_1", "B_1"])]
    assert set(ab["q_pn_unit_nucleic_acid_group_id"]) == {"(A_1, B_1)"}
    assert set(ab["q_pn_unit_num_resolved_residues_in_nucleic_acid_group"]) == {12}
    assert not ab["q_pn_unit_is_nuc_ligand"].any()
    assert ab["q_pn_unit_is_nuc_polymer"].all()

    cd = out.loc[df["q_pn_unit_iid"].isin(["C_1", "D_1"])]
    assert set(cd["q_pn_unit_nucleic_acid_group_id"]) == {"(C_1, D_1)"}
    assert set(cd["q_pn_unit_num_resolved_residues_in_nucleic_acid_group"]) == {10}
    assert cd["q_pn_unit_is_nuc_ligand"].all()

    e = out.loc[df["q_pn_unit_iid"] == "E_1"].iloc[0]
    f = out.loc[df["q_pn_unit_iid"] == "F_1"].iloc[0]
    assert e["q_pn_unit_nucleic_acid_group_id"] == "(E_1)"
    assert f["q_pn_unit_nucleic_acid_group_id"] == "(F_1)"
    assert e["q_pn_unit_is_nuc_ligand"]
    assert f["q_pn_unit_is_nuc_ligand"]
