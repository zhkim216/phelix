import json
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray

import atomworks.enums as aw_enums
from allatom_design.eval.eval_utils.eval_metrics import (
    _selected_metal_pn_unit_iids,
    compute_docking_metrics_atomarray,
)
from allatom_design.eval.eval_utils.folding_utils import (
    _aggregate_best_docking_metrics_per_designed_sample,
    _aggregate_best_docking_metrics_per_input_sample,
)


def _make_atom_array(records: list[dict]) -> AtomArray:
    atom_array = AtomArray(len(records))
    atom_array.coord = np.array([record["coord"] for record in records], dtype=np.float32)
    atom_array.chain_id = np.array([record["chain_id"] for record in records])
    atom_array.res_id = np.array([record["res_id"] for record in records])
    atom_array.ins_code = np.array([""] * len(records))
    atom_array.res_name = np.array([record["res_name"] for record in records])
    atom_array.hetero = np.array([record["hetero"] for record in records], dtype=bool)
    atom_array.atom_name = np.array([record["atom_name"] for record in records])
    atom_array.element = np.array([record["element"] for record in records])
    atom_array.occupancy = np.ones(len(records), dtype=np.float32)
    atom_array.set_annotation("pn_unit_iid", np.array([record["pn_unit_iid"] for record in records]))
    atom_array.set_annotation("chain_type", np.array([record["chain_type"] for record in records], dtype=np.int8))
    atom_array.set_annotation("is_polymer", np.array([not record["hetero"] for record in records], dtype=bool))
    return atom_array


def _protein_residue_records(res_id: int, ca_coord: np.ndarray, shift: np.ndarray | None = None) -> list[dict]:
    if shift is None:
        shift = np.zeros(3, dtype=np.float32)
    atom_offsets = {
        "N": np.array([-0.5, 0.0, 0.0], dtype=np.float32),
        "CA": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "C": np.array([0.6, 0.0, 0.0], dtype=np.float32),
        "O": np.array([0.9, 0.0, 0.4], dtype=np.float32),
        "CB": np.array([0.0, 0.8, 0.0], dtype=np.float32),
    }
    elements = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C"}
    return [
        {
            "coord": ca_coord + atom_offset + shift,
            "chain_id": "A",
            "res_id": res_id,
            "res_name": "ALA",
            "hetero": False,
            "atom_name": atom_name,
            "element": elements[atom_name],
            "pn_unit_iid": "A_1",
            "chain_type": int(aw_enums.ChainType.POLYPEPTIDE_L),
        }
        for atom_name, atom_offset in atom_offsets.items()
    ]


def _metal_record(coord: np.ndarray, shift: np.ndarray | None = None) -> dict:
    if shift is None:
        shift = np.zeros(3, dtype=np.float32)
    return {
        "coord": coord + shift,
        "chain_id": "L",
        "res_id": 1,
        "res_name": "MG",
        "hetero": True,
        "atom_name": "MG",
        "element": "MG",
        "pn_unit_iid": "L_1",
        "chain_type": int(aw_enums.ChainType.NON_POLYMER),
    }


def _make_reference_and_pred_arrays() -> tuple[AtomArray, AtomArray, list[float]]:
    ref_ca_coords = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 3.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 3.0], dtype=np.float32),
        np.array([20.0, 0.0, 0.0], dtype=np.float32),
    ]
    ref_metal_coord = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pred_shift = np.array([10.0, 5.0, -2.0], dtype=np.float32)
    pred_metal_offset = np.array([19.0, 0.0, 0.0], dtype=np.float32)

    ref_records = []
    pred_records = []
    pred_plddts = []
    for res_idx, ca_coord in enumerate(ref_ca_coords, start=1):
        ref_records.extend(_protein_residue_records(res_idx, ca_coord))
        residue_shift = pred_shift
        if res_idx == 4:
            residue_shift = pred_shift + np.array([4.0, 0.0, 0.0], dtype=np.float32)
        pred_records.extend(_protein_residue_records(res_idx, ca_coord, shift=residue_shift))
        pred_plddts.extend([10.0 if res_idx <= 3 else 90.0] * 5)

    ref_records.append(_metal_record(ref_metal_coord))
    pred_records.append(_metal_record(ref_metal_coord + pred_metal_offset, shift=pred_shift))
    pred_plddts.append(70.0)

    return _make_atom_array(ref_records), _make_atom_array(pred_records), pred_plddts


def _write_confidence_files(tmp_path: Path, pred_plddts: list[float]) -> Path:
    pred_sample_path = tmp_path / "synthetic_model.cif"
    (tmp_path / "synthetic_confidences.json").write_text(json.dumps({"atom_plddts": pred_plddts}))
    (tmp_path / "synthetic_summary_confidences.json").write_text(
        json.dumps({"iptm": 0.5, "chain_pair_pae_min": [[0.0, 4.0], [3.0, 0.0]]})
    )
    return pred_sample_path


def test_metal_docking_uses_reference_5a_site_for_alignment_and_plddt(tmp_path):
    sample_atom_array, pred_atom_array, pred_plddts = _make_reference_and_pred_arrays()
    pred_sample_path = _write_confidence_files(tmp_path, pred_plddts)

    metrics = compute_docking_metrics_atomarray(
        pred_atom_array=pred_atom_array,
        sample_atom_array=sample_atom_array,
        pred_sample_path=pred_sample_path,
        pocket_distance_for_docking_metrics=5.0,
        receptor_pn_unit_iids=["A_1"],
        ligand_pn_unit_iids=["L_1"],
        ligand_ccd_codes=["MG"],
        save_aligned=False,
    )

    assert "error" not in metrics
    assert metrics["ligand_ccd_code"] == "MG"
    assert metrics["num_bs_residues"] == 3
    assert metrics["binding_site_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["ligand_rmsd"] == pytest.approx(19.0, abs=1e-4)
    assert metrics["ligand_plddt"] == pytest.approx(70.0)
    assert metrics["binding_site_plddt"] == pytest.approx(10.0)
    assert metrics["iptm"] == pytest.approx(0.5)
    assert metrics["interface_min_pae"] == pytest.approx(3.0)


def test_selected_metal_pn_unit_iids_ignores_non_metal_ligands():
    sample_atom_array, _, _ = _make_reference_and_pred_arrays()
    ligand_records = [
        {
            "coord": np.array([5.0, 5.0, 5.0], dtype=np.float32),
            "chain_id": "S",
            "res_id": 1,
            "res_name": "LIG",
            "hetero": True,
            "atom_name": "C1",
            "element": "C",
            "pn_unit_iid": "S_1",
            "chain_type": int(aw_enums.ChainType.NON_POLYMER),
        }
    ]
    atom_array = _make_atom_array(
        [
            {
                "coord": sample_atom_array.coord[i],
                "chain_id": sample_atom_array.chain_id[i],
                "res_id": sample_atom_array.res_id[i],
                "res_name": sample_atom_array.res_name[i],
                "hetero": bool(sample_atom_array.hetero[i]),
                "atom_name": sample_atom_array.atom_name[i],
                "element": sample_atom_array.element[i],
                "pn_unit_iid": sample_atom_array.pn_unit_iid[i],
                "chain_type": int(sample_atom_array.chain_type[i]),
            }
            for i in range(len(sample_atom_array))
        ]
        + ligand_records
    )

    assert _selected_metal_pn_unit_iids(atom_array, ["S_1"]) == []
    assert _selected_metal_pn_unit_iids(atom_array, ["S_1", "L_1"]) == ["L_1"]


def test_docking_aggregators_skip_error_diffusion_results_and_propagate_ccd():
    per_pred = {
        "design_0": {
            "input_sample_id": "input_0",
            "diffusion_0": {
                "error": "bad metal match",
                "ligand_plddt": 99.0,
                "ligand_rmsd": None,
            },
            "diffusion_1": {
                "ligand_rmsd": 1.2,
                "binding_site_rmsd": 0.4,
                "ligand_plddt": 40.0,
                "binding_site_plddt": 35.0,
                "iptm": 0.3,
                "interface_min_pae": 5.0,
                "ligand_ccd_code": "MG",
            },
        }
    }

    per_design = _aggregate_best_docking_metrics_per_designed_sample(per_pred)
    assert per_design["design_0"]["ligand_rmsd"] == pytest.approx(1.2)
    assert per_design["design_0"]["ligand_ccd_code"] == "MG"

    per_input = _aggregate_best_docking_metrics_per_input_sample(per_design)
    assert per_input["input_0"]["best_designed_sample_id"] == "design_0"
    assert per_input["input_0"]["ligand_ccd_code"] == "MG"
