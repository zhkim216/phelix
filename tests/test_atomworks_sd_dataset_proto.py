import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from allatom_design.data.datasets.atomworks_sd_dataset_proto import (
    ProtoSDDataset,
    add_proto_sampling_weights,
    build_proto_interface_df,
    collect_external_evidence,
    _proto_center_mask,
)
from allatom_design.train_seq_denoiser import build_sd_datamodule


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
    medba_evidence=False,
    pubmed_evidence=False,
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
        "metal_medba_evidence": medba_evidence,
        "metal_pubmed_evidence": pubmed_evidence,
    }


def test_collect_external_evidence_ors_medba_and_pubmed_for_metal_rows_only():
    df = pd.DataFrame(
        [
            _row("Z_1", is_metal=True, ccd="ZN", medba_evidence=True),
            _row("F_1", is_metal=True, ccd="FE", pubmed_evidence=True),
            _row("M_1", is_metal=True, ccd="MG"),
            _row("P_1", is_protein=True, medba_evidence=True, pubmed_evidence=True),
        ]
    )

    out = collect_external_evidence(df)

    assert out["has_external_evidence"].tolist() == [True, True, False, False]


def test_collect_external_evidence_requires_sources_only_when_requested():
    df = pd.DataFrame([_row("Z_1", is_metal=True, ccd="ZN")]).drop(columns=["metal_pubmed_evidence"])

    with pytest.raises(KeyError, match="metal_pubmed_evidence"):
        collect_external_evidence(df, require_columns=True)

    out = collect_external_evidence(df, require_columns=False)
    assert out["has_external_evidence"].tolist() == [False]


def test_proto_center_mask_supports_all_metal_default_ccd_filter_and_external_evidence():
    df = pd.DataFrame(
        [
            _row("M_1", is_metal=True, ccd="MG"),
            _row("Z_1", is_metal=True, ccd="ZN", pubmed_evidence=True),
            _row("F_1", is_metal=True, ccd="FE"),
            _row("S_1", is_protein=False, ccd="ATP", pubmed_evidence=True),
        ]
    )
    df.loc[df["q_pn_unit_iid"] == "F_1", "q_pn_unit_avg_occupancy_nonpolymer"] = 0.25
    df = collect_external_evidence(df)

    no_filter = _proto_center_mask(
        df,
        {"external_evidence_policy": "no_filter", "allowed_ccd_codes": None, "min_avg_occupancy_nonpolymer": 0.5},
    )
    assert no_filter.tolist() == [True, True, False, False]

    external = _proto_center_mask(
        df,
        {"external_evidence_policy": "external_evidence", "allowed_ccd_codes": None, "min_avg_occupancy_nonpolymer": 0.5},
    )
    assert external.tolist() == [False, True, False, False]

    mg_only = _proto_center_mask(
        df,
        {"external_evidence_policy": "no_filter", "allowed_ccd_codes": ["MG"], "min_avg_occupancy_nonpolymer": 0.5},
    )
    assert mg_only.tolist() == [True, False, False, False]

    with pytest.raises(ValueError, match="allowed_ccd_codes"):
        _proto_center_mask(df, {"allowed_ccd_codes": []})


def test_filter_metadata_to_proto_scope_keeps_protein_monomers_and_selected_metals():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("P_short", is_protein=True, cluster_id=101),
            _row("M_1", is_metal=True, ccd="MG", cluster_id=200),
            _row("Z_1", is_metal=True, ccd="ZN", cluster_id=201),
            _row("S_1", is_protein=False, ccd="ATP", cluster_id=203),
        ]
    )
    metadata_df.loc[metadata_df["q_pn_unit_iid"] == "P_short", "q_pn_unit_num_resolved_residues"] = 5
    metadata_df = collect_external_evidence(metadata_df)
    metadata_df.set_index("example_id", inplace=True, drop=False)

    dataset = object.__new__(ProtoSDDataset)
    dataset.proto_cfg = {"external_evidence_policy": "no_filter", "allowed_ccd_codes": ["ZN"]}
    dataset.cfg = OmegaConf.create(
        {
            "train_filters": {
                "protein_monomer_chain_filter": [
                    "(q_pn_unit_is_protein and 20 <= q_pn_unit_num_resolved_residues < 2048)"
                ]
            }
        }
    )

    scoped = dataset._filter_metadata_to_proto_scope(metadata_df)

    assert scoped["q_pn_unit_iid"].tolist() == ["P_1", "Z_1"]


def test_build_proto_interface_df_includes_multi_protein_metal_interfaces_and_actual_ccd_key():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("P_2", is_protein=True, cluster_id=101),
            _row(
                "Z_1",
                is_metal=True,
                ccd="ZN",
                cluster_id=300,
                contacts=_contacts(("P_1", 2), ("P_2", 2)),
            ),
        ]
    )
    metadata_df = collect_external_evidence(metadata_df)

    interface_df = build_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        proto_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )

    assert len(interface_df) == 1
    row = interface_df.iloc[0]
    assert row["query_pn_unit_iids"] == ["Z_1", "P_1", "P_2"]
    assert row["ligand_pn_unit_iids"] == ("Z_1",)
    assert row["protein_pn_unit_iids"] == ("P_1", "P_2")
    assert row["protein_cluster_multiset"] == (100, 101)
    assert row["ligand_ccd_key"] == ("ccd", "ZN")
    assert row["n_coordinating_protein_donor_atoms"] == 4


def test_add_proto_sampling_weights_uses_actual_metal_ccd_key():
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
                "ligand_ccd_key": ("ccd", "ZN"),
            },
            {
                "protein_cluster_multiset": (200,),
                "interface_type": "bmm_protein",
                "ligand_ccd_key": ("ccd", "FE"),
            },
        ],
        index=["zn_iface", "fe_iface"],
    )

    _, weighted_interface = add_proto_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface={"alpha_protein_metal": 1.0},
        k_percentile=100.0,
    )

    assert weighted_interface.loc["zn_iface", "pair_cluster"][0] == ("ccd", "ZN")
    assert weighted_interface.loc["fe_iface", "pair_cluster"][0] == ("ccd", "FE")


def test_getitem_leaves_cached_atom_array_filtering_to_featurizer():
    dataset = object.__new__(ProtoSDDataset)
    dataset.phase = "val"
    dataset.parsed_df = pd.DataFrame(
        [
            {
                "example_id": "toy-example",
                "query_pn_unit_iids": ["Z_1", "P_1"],
                "extra_info": {"pdb_id": "1abc"},
            }
        ],
        index=["toy-example"],
    )
    dataset._load_cached_example = lambda pdb_id: {"atom_array": "full cached assembly"}
    dataset.featurizer = lambda example: example

    result = ProtoSDDataset.__getitem__(dataset, 0)

    assert result["atom_array"] == "full cached assembly"
    assert result["query_pn_unit_iids"] == ["Z_1", "P_1"]
    assert result["phase"] == "val"


def test_build_sd_datamodule_accepts_proto_selector(monkeypatch):
    import allatom_design.data.datasets.atomworks_sd_dataset_proto as proto_module

    def fake_init(self, cfg):
        self.cfg = cfg

    monkeypatch.setattr(proto_module.AtomworksSDProtoDataModule, "__init__", fake_init)

    datamodule = build_sd_datamodule(OmegaConf.create({"dataset_impl": "proto"}))

    assert isinstance(datamodule, proto_module.AtomworksSDProtoDataModule)
