import json

import numpy as np
import pandas as pd
import pytest

from allatom_design.data.datasets.atomworks_sd_dataset_mg_proto import (
    MGProtoSDDataset,
    add_mg_proto_sampling_weights,
    attach_mg_external_evidence_flag,
    build_mg_proto_interface_df,
    _mg_proto_center_mask,
)
from allatom_design.train_seq_denoiser import build_sd_datamodule
from omegaconf import OmegaConf


def _contacts(*items):
    return json.dumps(
        [
            {"pn_unit_iid": pn_unit_iid, "chain_iid": pn_unit_iid, "count": count}
            for pn_unit_iid, count in items
        ]
    )


def _row(
    iid,
    *,
    is_protein=False,
    is_metal=False,
    ccd=None,
    cluster_id=1,
    contacts="[]",
    substring_evidence=False,
    gpt_evidence=False,
):
    return {
        "pdb_id": "1abc",
        "assembly_id": "1",
        "path": "/tmp/1abc.cif",
        "example_id": f"raw-{iid}",
        "q_pn_unit_id": iid.split("_")[0],
        "q_pn_unit_iid": iid,
        "q_pn_unit_type": 6 if is_protein else 8,
        "q_pn_unit_sequence_length": 50 if is_protein else np.nan,
        "q_pn_unit_num_resolved_residues": 50 if is_protein else 1,
        "q_pn_unit_is_protein": is_protein,
        "q_pn_unit_is_metal": is_metal,
        "q_pn_unit_is_polymer": is_protein,
        "q_pn_unit_is_halide": False,
        "q_pn_unit_non_polymer_res_names": ccd,
        "q_pn_unit_avg_occupancy_nonpolymer": 1.0 if is_metal else np.nan,
        "q_pn_unit_per_partner_contacts_metal": contacts,
        "q_pn_unit_cluster_id": cluster_id,
        "q_pn_unit_has_pubmed_evidence_substring": substring_evidence,
        "q_pn_unit_has_pubmed_evidence_gpt": gpt_evidence,
    }


def test_attach_mg_external_evidence_flag_maps_policies_and_exact_mg_only():
    df = pd.DataFrame(
        [
            _row("M_1", is_metal=True, ccd="MG", substring_evidence=True),
            _row("X_1", is_metal=True, ccd="MGX", substring_evidence=True),
            _row("S_1", is_metal=False, ccd="ATP", substring_evidence=True),
        ]
    )

    no_filter = attach_mg_external_evidence_flag(df, {"external_evidence_policy": "no_filter"})
    assert no_filter["q_pn_unit_has_external_evidence"].tolist() == [True, False, False]

    substring = attach_mg_external_evidence_flag(df, {"external_evidence_policy": "substring"})
    assert substring["q_pn_unit_has_external_evidence"].tolist() == [True, False, False]

    gpt = attach_mg_external_evidence_flag(df, {"external_evidence_policy": "gpt"})
    assert gpt["q_pn_unit_has_external_evidence"].tolist() == [False, False, False]


def test_attach_mg_external_evidence_flag_requires_policy_source_column():
    df = pd.DataFrame([_row("M_1", is_metal=True, ccd="MG")]).drop(
        columns=["q_pn_unit_has_pubmed_evidence_gpt"]
    )
    with pytest.raises(KeyError, match="q_pn_unit_has_pubmed_evidence_gpt"):
        attach_mg_external_evidence_flag(df, {"external_evidence_policy": "gpt"})


def test_attach_mg_external_evidence_flag_rejects_old_policy_names():
    df = pd.DataFrame([_row("M_1", is_metal=True, ccd="MG")])

    with pytest.raises(ValueError, match="Supported values: 'no_filter', 'substring', 'gpt'"):
        attach_mg_external_evidence_flag(df, {"external_evidence_policy": "biolip2_style"})

    with pytest.raises(ValueError, match="Supported values: 'no_filter', 'substring', 'gpt'"):
        attach_mg_external_evidence_flag(df, {"external_evidence_policy": "llm_assisted"})


def test_build_mg_proto_interface_df_counts_only_protein_donors_and_omits_crop_center_override():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("S_1", is_protein=False, cluster_id=200, ccd="ATP"),
            _row(
                "M_1",
                is_metal=True,
                ccd="MG",
                cluster_id=300,
                contacts=_contacts(("P_1", 3), ("S_1", 99), ("missing_1", 99)),
            ),
        ]
    )
    metadata_df["crop_center_pn_unit_iids"] = [["STALE"]] * len(metadata_df)
    metadata_df = attach_mg_external_evidence_flag(metadata_df, {"external_evidence_policy": "no_filter"})
    protein_df = metadata_df[metadata_df["q_pn_unit_is_protein"]].copy()

    interface_df = build_mg_proto_interface_df(
        metadata_df,
        protein_df=protein_df,
        dataset_name="toy",
        mg_cfg={
            "external_evidence_policy": "no_filter",
            "allowed_ccd_codes": ["MG"],
            "min_protein_donor_atoms": 3,
            "min_avg_occupancy_nonpolymer": 0.5,
        },
    )

    assert len(interface_df) == 1
    row = interface_df.iloc[0]
    assert row["query_pn_unit_iids"] == ["M_1", "P_1"]
    assert row["biologically_meaningful_pn_unit_iids"] == ["M_1", "P_1"]
    assert "crop_center_pn_unit_iids" not in interface_df.columns
    assert row["ligand_pn_unit_iids"] == ("M_1",)
    assert row["protein_pn_unit_iids"] == ("P_1",)
    assert row["protein_cluster_multiset"] == (100,)
    assert row["ligand_ccd_key"] == ("ccd", "MG")
    assert row["n_coordinating_protein_donor_atoms"] == 3
    assert "interface" in row["example_id"]
    assert "S_1" not in row["query_pn_unit_iids"]


def test_build_mg_proto_interface_df_counts_composite_contact_once_and_prefers_chain_iid():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("P_2", is_protein=True, cluster_id=101),
            _row(
                "M_1",
                is_metal=True,
                ccd="MG",
                cluster_id=300,
                contacts=json.dumps(
                    [
                        {
                            "pn_unit_iid": "P_1,P_2",
                            "chain_iid": "P_1",
                            "count": 3,
                        }
                    ]
                ),
            ),
        ]
    )
    metadata_df = attach_mg_external_evidence_flag(metadata_df, {"external_evidence_policy": "no_filter"})

    interface_df = build_mg_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        mg_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )

    assert len(interface_df) == 1
    row = interface_df.iloc[0]
    assert row["query_pn_unit_iids"] == ["M_1", "P_1"]
    assert row["protein_pn_unit_iids"] == ("P_1",)
    assert row["n_coordinating_protein_donor_atoms"] == 3


def test_mg_proto_center_mask_uses_exact_ccd_evidence_and_occupancy():
    df = pd.DataFrame(
        [
            _row("M_1", is_metal=True, ccd="MG", cluster_id=1),
            _row("X_1", is_metal=True, ccd="MGX", cluster_id=2),
            _row("C_1", is_metal=True, ccd="CA", cluster_id=3),
            _row("S_1", is_protein=False, ccd="ATP", cluster_id=4),
            _row("M_2", is_metal=True, ccd="MG", cluster_id=5),
            _row("M_3", is_metal=True, ccd="MG", cluster_id=6),
        ]
    )
    df["q_pn_unit_has_external_evidence"] = [True, True, True, True, False, True]
    df.loc[df["q_pn_unit_iid"] == "M_3", "q_pn_unit_avg_occupancy_nonpolymer"] = 0.25

    mask = _mg_proto_center_mask(
        df,
        {"allowed_ccd_codes": ["MG"], "min_avg_occupancy_nonpolymer": 0.5},
    )

    assert mask.tolist() == [True, False, False, False, False, False]

    with pytest.raises(ValueError, match="allowed_ccd_codes"):
        _mg_proto_center_mask(df, {"allowed_ccd_codes": []})


def test_filter_metadata_to_mg_proto_scope_keeps_only_protein_monomers_and_filtered_mg():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("P_short", is_protein=True, cluster_id=101),
            _row("M_1", is_metal=True, ccd="MG", cluster_id=200),
            _row("X_1", is_metal=True, ccd="MGX", cluster_id=201),
            _row("C_1", is_metal=True, ccd="CA", cluster_id=202),
            _row("S_1", is_protein=False, ccd="ATP", cluster_id=203),
        ]
    )
    metadata_df.loc[metadata_df["q_pn_unit_iid"] == "P_short", "q_pn_unit_num_resolved_residues"] = 5
    metadata_df = attach_mg_external_evidence_flag(metadata_df, {"external_evidence_policy": "no_filter"})
    metadata_df.set_index("example_id", inplace=True, drop=False)

    dataset = object.__new__(MGProtoSDDataset)
    dataset.mg_cfg = {"external_evidence_policy": "no_filter", "allowed_ccd_codes": ["MG"]}
    dataset.cfg = OmegaConf.create(
        {
            "train_filters": {
                "protein_monomer_chain_filter": [
                    "(q_pn_unit_is_protein and 20 <= q_pn_unit_num_resolved_residues < 2048)"
                ]
            }
        }
    )

    scoped = dataset._filter_metadata_to_mg_proto_scope(metadata_df)

    assert scoped["q_pn_unit_iid"].tolist() == ["P_1", "M_1"]


def test_parse_train_dfs_does_not_set_crop_center_override_for_interfaces():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("M_1", is_metal=True, ccd="MG", cluster_id=300, contacts=_contacts(("P_1", 3))),
        ]
    )
    metadata_df = attach_mg_external_evidence_flag(metadata_df, {"external_evidence_policy": "no_filter"})
    interface_df = build_mg_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        mg_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )
    monomer_df = pd.DataFrame([_row("P_9", is_protein=True, cluster_id=999)])
    monomer_df["example_id"] = "toy-protein-monomer"
    monomer_df.set_index("example_id", inplace=True, drop=False)

    dataset = object.__new__(MGProtoSDDataset)
    dataset.protein_monomer_chain_df = monomer_df
    dataset.interface_df = interface_df

    parsed_df = dataset._parse_train_dfs()
    parsed = parsed_df.loc[interface_df.index[0]]

    assert parsed["query_pn_unit_iids"] == ["M_1", "P_1"]
    assert "crop_center_pn_unit_iids" not in parsed
    assert "crop_center_pn_unit_iids" not in parsed["extra_info"]


def test_build_mg_proto_interface_df_excludes_multi_protein_mg_complexes():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("P_2", is_protein=True, cluster_id=101),
            _row(
                "M_1",
                is_metal=True,
                ccd="MG",
                cluster_id=300,
                contacts=_contacts(("P_1", 3), ("P_2", 3)),
            ),
        ]
    )
    metadata_df = attach_mg_external_evidence_flag(metadata_df, {"external_evidence_policy": "no_filter"})

    interface_df = build_mg_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        mg_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )

    assert interface_df.empty


def test_build_mg_proto_interface_df_respects_donor_threshold():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("M_1", is_metal=True, ccd="MG", cluster_id=300, contacts=_contacts(("P_1", 2))),
        ]
    )
    metadata_df = attach_mg_external_evidence_flag(metadata_df, {"external_evidence_policy": "no_filter"})

    interface_df = build_mg_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        mg_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )

    assert interface_df.empty


def test_add_mg_proto_sampling_weights_are_finite_nonnegative_and_nonzero():
    monomer_df = pd.DataFrame(
        [
            {"q_pn_unit_cluster_id": 100},
            {"q_pn_unit_cluster_id": 200},
        ],
        index=["mono1", "mono2"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "protein_cluster_multiset": (100,),
                "interface_type": "bmm_protein",
                "ligand_ccd_key": ("ccd", "MG"),
            }
        ],
        index=["iface"],
    )

    weighted_monomer, weighted_interface = add_mg_proto_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface={"alpha_protein_metal": 1.0},
        k_percentile=100.0,
    )

    weights = np.concatenate(
        [
            weighted_monomer["sampling_weight"].to_numpy(dtype=float),
            weighted_interface["sampling_weight"].to_numpy(dtype=float),
        ]
    )
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert weights.sum() > 0
    assert weighted_interface["sampling_weight"].sum() > 0
    assert weighted_monomer["sampling_weight"].sum() > 0


def test_build_sd_datamodule_rejects_unknown_selector():
    with pytest.raises(ValueError, match="Unknown data.dataset_impl"):
        build_sd_datamodule(OmegaConf.create({"dataset_impl": "unknown"}))
