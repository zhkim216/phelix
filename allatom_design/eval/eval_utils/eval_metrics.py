import ast
import os
import pickle
import re
import shutil
import subprocess
import uuid
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

# import gemmi
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from einops import rearrange
from omegaconf import DictConfig
from scipy import linalg
from torchtyping import TensorType
from tqdm import tqdm

from allatom_design.eval.eval_utils.sd_data_utils import get_sd_example, prepare_af3_prediction
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets, annotate_ligand_pockets_pseudocb
from allatom_design.utils.sample_io_utils import save_cif_file
from allatom_design.utils.atom_array_utils import get_valid_standard_aa_residue_mask

# Atomworks imports
from atomworks.constants import METAL_ELEMENTS, STANDARD_AA
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from atomworks.io.utils.io_utils import to_cif_string
from atomworks.io.parser import parse as aw_parse
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.ml.utils.geometry import align_atom_arrays
from atomworks.io.utils.io_utils import to_cif_file
import atomworks.enums as aw_enums

from biotite.structure import AtomArray, get_residue_count, spread_residue_wise
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign


def _docking_metric_error(message: str, ligand_ccd_code: str | None = None) -> dict[str, float | int | str | None]:
    return {
        "error": message,
        "ligand_rmsd": None,
        "binding_site_rmsd": None,
        "num_bs_residues": 0,
        "ligand_plddt": None,
        "binding_site_plddt": None,
        "iptm": None,
        "interface_min_pae": None,
        "ligand_ccd_code": ligand_ccd_code,
    }


def _normalise_element(element: object) -> str:
    return str(element).strip().upper()


def _join_unique(values: list[str]) -> str | None:
    unique_values = []
    for value in values:
        if value and value not in unique_values:
            unique_values.append(value)
    return ";".join(unique_values) if unique_values else None


def _ligand_ccd_code_for_iids(
    atom_array: AtomArray,
    ligand_pn_unit_iids: list[str],
    ligand_ccd_codes: list[str] | None = None,
) -> str | None:
    if ligand_ccd_codes is not None and len(ligand_ccd_codes) == len(ligand_pn_unit_iids):
        return _join_unique([str(code) for code in ligand_ccd_codes])

    codes = []
    for ligand_pn_unit_iid in ligand_pn_unit_iids:
        mask = atom_array.pn_unit_iid == ligand_pn_unit_iid
        if np.any(mask):
            codes.append(str(atom_array.res_name[mask][0]))
    return _join_unique(codes)


def _selected_metal_pn_unit_iids(atom_array: AtomArray, ligand_pn_unit_iids: list[str]) -> list[str]:
    metal_pn_unit_iids = []
    for ligand_pn_unit_iid in ligand_pn_unit_iids:
        mask = (atom_array.pn_unit_iid == ligand_pn_unit_iid) & (atom_array.element != "H")
        if not np.any(mask):
            continue
        ligand_atoms = atom_array[mask]
        ligand_elements = [_normalise_element(element) for element in ligand_atoms.element]
        if len(ligand_elements) == 1 and ligand_elements[0] in METAL_ELEMENTS:
            metal_pn_unit_iids.append(str(ligand_pn_unit_iid))
    return metal_pn_unit_iids


def _spread_atom_mask_by_residue(atom_array: AtomArray, atom_mask: np.ndarray) -> np.ndarray:
    residue_mask = np.zeros(len(atom_array), dtype=bool)
    for idx in np.where(atom_mask)[0]:
        same_residue = (
            (atom_array.pn_unit_iid == atom_array.pn_unit_iid[idx])
            & (atom_array.res_id == atom_array.res_id[idx])
        )
        if hasattr(atom_array, "ins_code"):
            same_residue &= atom_array.ins_code == atom_array.ins_code[idx]
        residue_mask |= same_residue
    return residue_mask


def _joint_resolved_receptor_ca_masks(
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    receptor_pn_unit_iids: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    sample_receptor_mask = np.isin(sample_atom_array.pn_unit_iid, receptor_pn_unit_iids)
    pred_receptor_mask = np.isin(pred_atom_array.pn_unit_iid, receptor_pn_unit_iids)

    sample_ca_mask_initial = sample_receptor_mask & (sample_atom_array.atom_name == "CA") & (sample_atom_array.res_name != "UNK")
    pred_ca_mask_initial = pred_receptor_mask & (pred_atom_array.atom_name == "CA") & (pred_atom_array.res_name != "UNK")

    if sample_ca_mask_initial.sum() != pred_ca_mask_initial.sum():
        return None, None, "Number of CA atoms in sample and pred must match"

    if sample_ca_mask_initial.sum() == 0:
        return None, None, "No CA atoms found"

    sample_ca_resolved_mask = ~np.isnan(sample_atom_array[sample_ca_mask_initial].coord[:, 0])
    pred_ca_resolved_mask = ~np.isnan(pred_atom_array[pred_ca_mask_initial].coord[:, 0])
    ca_resolved_mask = sample_ca_resolved_mask & pred_ca_resolved_mask

    sample_ca_indices = np.where(sample_ca_mask_initial)[0]
    sample_ca_mask = np.zeros(len(sample_atom_array), dtype=bool)
    sample_ca_mask[sample_ca_indices[ca_resolved_mask]] = True

    pred_ca_indices = np.where(pred_ca_mask_initial)[0]
    pred_ca_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_ca_mask[pred_ca_indices[ca_resolved_mask]] = True

    if sample_ca_mask.sum() == 0:
        return None, None, "No resolved CA atoms found"

    return sample_ca_mask, pred_ca_mask, None


def _metal_atom_key(atom_array: AtomArray, idx: int) -> tuple[str, str, str, str]:
    return (
        str(atom_array.pn_unit_iid[idx]),
        str(atom_array.res_name[idx]),
        str(atom_array.atom_name[idx]),
        _normalise_element(atom_array.element[idx]),
    )


def _matched_metal_atom_masks(
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    metal_pn_unit_iids: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[str, str, str, str]] | None, str | None]:
    sample_indices = np.where(
        np.isin(sample_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin([_normalise_element(element) for element in sample_atom_array.element], list(METAL_ELEMENTS))
    )[0]
    pred_indices = np.where(
        np.isin(pred_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin([_normalise_element(element) for element in pred_atom_array.element], list(METAL_ELEMENTS))
    )[0]

    sample_by_key = {}
    for idx in sample_indices:
        key = _metal_atom_key(sample_atom_array, idx)
        if key in sample_by_key:
            return None, None, None, f"Ambiguous reference metal atom key: {key}"
        sample_by_key[key] = idx

    pred_by_key = {}
    for idx in pred_indices:
        key = _metal_atom_key(pred_atom_array, idx)
        if key in pred_by_key:
            return None, None, None, f"Ambiguous predicted metal atom key: {key}"
        pred_by_key[key] = idx

    if not sample_by_key:
        return None, None, None, "No reference metal atoms found"
    if set(sample_by_key) != set(pred_by_key):
        return None, None, None, "Reference and predicted metal atom keys do not match"

    matched_keys = list(sample_by_key.keys())
    sample_mask = np.zeros(len(sample_atom_array), dtype=bool)
    pred_mask = np.zeros(len(pred_atom_array), dtype=bool)
    sample_mask[[sample_by_key[key] for key in matched_keys]] = True
    pred_mask[[pred_by_key[key] for key in matched_keys]] = True

    return sample_mask, pred_mask, matched_keys, None


def _compute_metal_docking_metrics_atomarray(
    *,
    pred_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    pred_sample_path: str | Path,
    pocket_distance_for_docking_metrics: float,
    receptor_pn_unit_iids: list[str],
    metal_pn_unit_iids: list[str],
    ligand_ccd_code: str | None,
    save_aligned: bool = True,
) -> dict[str, float | int | str | None]:
    sample_metal_mask, _, _, error = _matched_metal_atom_masks(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        metal_pn_unit_iids=metal_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(error, ligand_ccd_code=ligand_ccd_code)

    sample_receptor_heavy_mask = (
        np.isin(sample_atom_array.pn_unit_iid, receptor_pn_unit_iids)
        & (sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
        & (sample_atom_array.element != "H")
        & ~np.isnan(sample_atom_array.coord).any(axis=1)
    )
    sample_metal_coords = sample_atom_array[sample_metal_mask].coord
    valid_sample_metal_coords = sample_metal_coords[~np.isnan(sample_metal_coords).any(axis=1)]
    if len(valid_sample_metal_coords) == 0:
        return _docking_metric_error("No resolved reference metal coordinates found", ligand_ccd_code=ligand_ccd_code)

    receptor_coords = sample_atom_array.coord[sample_receptor_heavy_mask]
    if len(receptor_coords) == 0:
        return _docking_metric_error("No receptor heavy atoms found", ligand_ccd_code=ligand_ccd_code)

    distances = np.linalg.norm(
        receptor_coords[:, None, :] - valid_sample_metal_coords[None, :, :],
        axis=2,
    )
    near_metal_receptor_atoms = np.any(distances <= pocket_distance_for_docking_metrics, axis=1)

    sample_binding_site_atom_mask = np.zeros(len(sample_atom_array), dtype=bool)
    sample_receptor_indices = np.where(sample_receptor_heavy_mask)[0]
    sample_binding_site_atom_mask[sample_receptor_indices[near_metal_receptor_atoms]] = True
    sample_binding_site_residue_mask = _spread_atom_mask_by_residue(
        sample_atom_array,
        sample_binding_site_atom_mask,
    ) & np.isin(sample_atom_array.pn_unit_iid, receptor_pn_unit_iids)

    sample_ca_mask, pred_ca_mask, error = _joint_resolved_receptor_ca_masks(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(error, ligand_ccd_code=ligand_ccd_code)

    sample_ca = sample_atom_array[sample_ca_mask]
    pred_ca = pred_atom_array[pred_ca_mask]
    sample_binding_site_ca_mask = sample_binding_site_residue_mask[sample_ca_mask]

    sample_binding_site_ca = sample_ca[sample_binding_site_ca_mask]
    pred_binding_site_ca = pred_ca[sample_binding_site_ca_mask]

    if len(sample_binding_site_ca) == 0:
        return _docking_metric_error("No binding site CA atoms found", ligand_ccd_code=ligand_ccd_code)
    if not (sample_binding_site_ca.res_name == pred_binding_site_ca.res_name).all():
        return _docking_metric_error(
            "Amino acid residues in sample and pred binding site must match",
            ligand_ccd_code=ligand_ccd_code,
        )

    pred_aligned_atom_array, bs_rmsd = align_atom_arrays(
        mbl_sele=pred_binding_site_ca,
        tgt_sele=sample_binding_site_ca,
        mbl_full=pred_atom_array,
    )

    pred_binding_site_ca_full_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_ca_indices = np.where(pred_ca_mask)[0]
    pred_binding_site_ca_full_mask[pred_ca_indices[sample_binding_site_ca_mask]] = True
    pred_binding_site_mask = _spread_atom_mask_by_residue(
        pred_aligned_atom_array,
        pred_binding_site_ca_full_mask,
    ) & np.isin(pred_aligned_atom_array.pn_unit_iid, receptor_pn_unit_iids)

    sample_metal_mask, pred_metal_mask, _, error = _matched_metal_atom_masks(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_aligned_atom_array,
        metal_pn_unit_iids=metal_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(error, ligand_ccd_code=ligand_ccd_code)

    sample_coords = sample_atom_array[sample_metal_mask].coord
    pred_coords = pred_aligned_atom_array[pred_metal_mask].coord
    valid_pair_mask = ~np.isnan(sample_coords).any(axis=1) & ~np.isnan(pred_coords).any(axis=1)
    if not np.any(valid_pair_mask):
        return _docking_metric_error("No resolved matched metal atom pairs found", ligand_ccd_code=ligand_ccd_code)

    coord_delta = pred_coords[valid_pair_mask] - sample_coords[valid_pair_mask]
    ligand_rmsd = float(np.sqrt(np.mean(np.sum(coord_delta * coord_delta, axis=1))))

    confidence_dir = str(Path(pred_sample_path).parent)
    full_confidence_file_path = f"{confidence_dir}/{re.sub(r'_model$', '_confidences', str(Path(pred_sample_path).stem))}.json"
    summary_confidence_file_path = f"{confidence_dir}/{re.sub(r'_model$', '_summary_confidences', str(Path(pred_sample_path).stem))}.json"

    ligand_plddt = extract_af3_confidence_metrics(
        confidence_file_path=full_confidence_file_path,
        atom_array=pred_aligned_atom_array,
        mask=pred_metal_mask,
        metrics_to_extract="atom_plddts",
        return_mean=True,
    )
    binding_site_plddt = extract_af3_confidence_metrics(
        confidence_file_path=full_confidence_file_path,
        atom_array=pred_aligned_atom_array,
        mask=pred_binding_site_mask,
        metrics_to_extract="atom_plddts",
        return_mean=True,
    )
    iptm = extract_af3_confidence_metrics(
        confidence_file_path=summary_confidence_file_path,
        atom_array=pred_aligned_atom_array,
        metrics_to_extract="iptm",
        return_mean=True,
    )
    interface_min_pae = extract_af3_confidence_metrics(
        confidence_file_path=summary_confidence_file_path,
        atom_array=pred_aligned_atom_array,
        metrics_to_extract="interface_min_pae",
        return_mean=True,
    )

    if save_aligned:
        out_file = Path(pred_sample_path).parent / f"{Path(pred_sample_path).stem}_pocket_aligned.cif"
        try:
            save_cif_file(pred_aligned_atom_array, out_file)
        except Exception as exc:
            print(f"Warning: Failed to save aligned structure: {exc}")

    return {
        "ligand_rmsd": ligand_rmsd,
        "binding_site_rmsd": float(bs_rmsd),
        "num_bs_residues": int(sample_binding_site_ca_mask.sum()),
        "ligand_plddt": ligand_plddt,
        "binding_site_plddt": binding_site_plddt,
        "iptm": iptm,
        "interface_min_pae": interface_min_pae,
        "ligand_ccd_code": ligand_ccd_code,
    }

# ============================================================================
# Sequence recovery
# ============================================================================

def calculate_sequence_recovery(input_atom_array: AtomArray, designed_atom_array: AtomArray,
                                pocket_distances_for_seq_recovery: list[float] = [4.0, 5.0, 6.0],
                                pocket_distance_bins: list[tuple[float, float]] | None = None,
                                n_min_ligand_atoms: int = 5) -> dict[str, float]:

    """
    Calculate sequence recovery and pocket sequence recovery between input and designed atom arrays.

    Args:
        pocket_distances_for_seq_recovery: cumulative pocket cutoffs (residues within d Å).
        pocket_distance_bins: optional list of (lo, hi) disjoint distance shells (lo, hi]
            applied as residue masks via `mask_hi & ~mask_lo` (special-cased lo == 0).
            Adds `pocket_recovery_bin_{lo}_to_{hi}` and `pocket_n_residues_bin_{lo}_to_{hi}`.
    """
    seq_recovery_metrics = {}

    input_valid_residue_mask = get_valid_standard_aa_residue_mask(input_atom_array)

    # Get sequence of the input sample
    input_seq_mask = input_valid_residue_mask & (input_atom_array.atom_name == "CA")
    input_res_ids = input_atom_array[input_seq_mask].res_id
    input_res_names = input_atom_array[input_seq_mask].res_name

    # Get sequence of the designed sample
    designed_valid_residue_mask = get_valid_standard_aa_residue_mask(designed_atom_array)
    designed_seq_mask = designed_valid_residue_mask & np.isin(designed_atom_array.res_id, input_res_ids) & (designed_atom_array.atom_name == "CA")
    designed_res_names = designed_atom_array[designed_seq_mask].res_name

    # Calculate sequence recovery ratio and save to the metrics dictionary
    seq_recovery_ratio = (input_res_names == designed_res_names).mean()
    seq_recovery_metrics["seq_recovery_ratio"] = seq_recovery_ratio

    # Cache residue-level pocket masks per distance for reuse by binning block.
    edge_to_residue_mask: dict[float, np.ndarray] = {}

    # Annotate ligand pockets at different distances
    for pocket_distance in pocket_distances_for_seq_recovery:
        # Input sample
        input_atom_array = annotate_ligand_pockets(input_atom_array, pocket_distance=pocket_distance, n_min_ligand_atoms=n_min_ligand_atoms, annotation_name=f"is_ligand_pocket_{pocket_distance}")
        input_pocket_residue_mask = apply_and_spread_residue_wise(input_atom_array, input_atom_array.get_annotation(f"is_ligand_pocket_{pocket_distance}"), function=np.any)
        edge_to_residue_mask[float(pocket_distance)] = input_pocket_residue_mask
        input_pocket_seq_mask = input_seq_mask & input_pocket_residue_mask

        input_pocket_res_ids = input_atom_array[input_pocket_seq_mask].res_id
        input_pocket_res_names = input_atom_array[input_pocket_seq_mask].res_name

        # Designed sample
        designed_pocket_seq_mask = np.isin(designed_atom_array.res_id, input_pocket_res_ids) & (designed_atom_array.atom_name == "CA")
        designed_pocket_res_names = designed_atom_array[designed_pocket_seq_mask].res_name

        pocket_recovery_ratio = (input_pocket_res_names == designed_pocket_res_names).mean()
        seq_recovery_metrics[f"pocket_recovery_ratio_{pocket_distance}"] = pocket_recovery_ratio
        seq_recovery_metrics[f"pocket_n_residues_{pocket_distance}"] = int(len(input_pocket_res_names))

    # Disjoint distance-bin pocket recovery: residues with min-distance-to-ligand in (lo, hi]
    if pocket_distance_bins:
        for lo, hi in pocket_distance_bins:
            lo_f, hi_f = float(lo), float(hi)
            for d in (lo_f, hi_f):
                if d == 0.0:
                    continue
                if d not in edge_to_residue_mask:
                    ann = f"is_ligand_pocket_{d}"
                    input_atom_array = annotate_ligand_pockets(
                        input_atom_array, pocket_distance=d,
                        n_min_ligand_atoms=n_min_ligand_atoms, annotation_name=ann,
                    )
                    edge_to_residue_mask[d] = apply_and_spread_residue_wise(
                        input_atom_array, input_atom_array.get_annotation(ann), function=np.any,
                    )

            if lo_f == 0.0:
                bin_residue_mask = edge_to_residue_mask[hi_f]
            else:
                bin_residue_mask = edge_to_residue_mask[hi_f] & ~edge_to_residue_mask[lo_f]

            input_bin_seq_mask = input_seq_mask & bin_residue_mask
            input_bin_res_ids = input_atom_array[input_bin_seq_mask].res_id
            input_bin_res_names = input_atom_array[input_bin_seq_mask].res_name

            designed_bin_seq_mask = np.isin(designed_atom_array.res_id, input_bin_res_ids) & (designed_atom_array.atom_name == "CA")
            designed_bin_res_names = designed_atom_array[designed_bin_seq_mask].res_name

            key = f"pocket_recovery_bin_{lo_f}_to_{hi_f}"
            n_key = f"pocket_n_residues_bin_{lo_f}_to_{hi_f}"
            if len(input_bin_res_names) == 0:
                seq_recovery_metrics[key] = float("nan")
            else:
                seq_recovery_metrics[key] = float((input_bin_res_names == designed_bin_res_names).mean())
            seq_recovery_metrics[n_key] = int(len(input_bin_res_names))

    return seq_recovery_metrics

# ============================================================================
# Self-consistency metrics
# ============================================================================

def compute_self_consistency_metrics_atomarray(*, pred_atom_array: AtomArray,
                                                sample_atom_array: AtomArray,
                                                pred_sample_path: str = None,
                                                save_aligned: bool = True,
                                                ) -> dict[str, float]:
    """
    Compute self-consistency metrics between a designed structure and its predicted structure, using atom array.

    Uses atomworks align_atom_arrays to handle structures with different atom sets
    (e.g., sample with backbone only vs pred with full sidechain atoms).
    """
    metrics = {}

    # Build initial CA masks (without NaN filtering) to identify matching residue positions
    sample_ca_mask_initial = (sample_atom_array.atom_name == "CA") & (sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
    pred_ca_mask_initial = (pred_atom_array.atom_name == "CA") & (pred_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L) & (pred_atom_array.res_name != "UNK")

    # Compute joint resolved mask: exclude positions where EITHER array has NaN coordinates
    # (NaN occurs in native structures for unresolved residues; AF3 predictions never have NaN)
    sample_ca_resolved_mask = ~np.isnan(sample_atom_array[sample_ca_mask_initial].coord[:, 0])
    pred_ca_resolved_mask = ~np.isnan(pred_atom_array[pred_ca_mask_initial].coord[:, 0])
    ca_resolved_mask = sample_ca_resolved_mask & pred_ca_resolved_mask

    # Apply joint resolved mask back to full atom_array-level masks
    sample_ca_indices = np.where(sample_ca_mask_initial)[0]
    sample_ca_mask = np.zeros(len(sample_atom_array), dtype=bool)
    sample_ca_mask[sample_ca_indices[ca_resolved_mask]] = True

    pred_ca_indices = np.where(pred_ca_mask_initial)[0]
    pred_ca_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_ca_mask[pred_ca_indices[ca_resolved_mask]] = True

    sample_ca = sample_atom_array[sample_ca_mask]
    pred_ca = pred_atom_array[pred_ca_mask]

    assert (sample_ca.res_name == pred_ca.res_name).all(), "Sample and pred CA residues must match"

    # Align pred CA to sample CA using atomworks align_atom_arrays
    # This aligns pred_ca to sample_ca and applies the transformation to the full pred_atom_array
    aligned_pred_atom_array, ca_rmsd = align_atom_arrays(
        mbl_sele=pred_ca,           # CA atoms from pred to align
        tgt_sele=sample_ca,         # CA atoms from sample as target
        mbl_full=pred_atom_array    # Full pred structure to transform
    )

    # Write aligned coords to mmcif
    if save_aligned:
        out_file = f"{Path(pred_sample_path).parent}/{Path(pred_sample_path).stem}_ca_aligned.cif"
        save_cif_file(aligned_pred_atom_array, out_file)

    # Create CA atom mask for pLDDT extraction (matching aligned structure)
    ca_atom_mask = torch.tensor(pred_ca_mask, dtype=torch.bool)

    # Compute metrics.
    # for metric in ["sc_ca_rmsd", "avg_ca_plddt", "tmalign_score"]:
    for metric in ["sc_ca_rmsd", "avg_ca_plddt"]:
        if metric == "sc_ca_rmsd":
            # CA RMSD computed via align_atom_arrays (already a float)
            if type(ca_rmsd) != float:
                try:
                    ca_rmsd = ca_rmsd.item()
                except:
                    ca_rmsd = float(ca_rmsd)

            metrics[metric] = ca_rmsd

        elif metric == "avg_ca_plddt":
            # Compute average pLDDT across all CA atoms.
            confidence_dir = str(pred_sample_path.parent)
            confidence_file_name = re.sub(r'_model$', '_confidences', str(pred_sample_path.stem)) + '.json'

            avg_ca_plddt = extract_af3_confidence_metrics(confidence_file_path=f"{confidence_dir}/{confidence_file_name}",
                                                        atom_array=pred_atom_array,
                                                        mask=ca_atom_mask,
                                                        metrics_to_extract="atom_plddts",
                                                        return_mean=True)
            metrics[metric] = avg_ca_plddt

        # elif metric == "tmalign_score":
        #     # Compute TM-score using TM-align.
        #     tmalign_score, _ = _compute_tmalign_score(pred_pdb, design_pdb)
        #     metrics[metric] = tmalign_score

    return metrics


def compute_docking_metrics_atomarray(*, pred_atom_array: AtomArray,
                                       sample_atom_array: AtomArray,
                                       pred_sample_path: str = None,
                                       pocket_distance_for_docking_metrics: float = 6.0,
                                       receptor_pn_unit_iids: list = ["A_1"],
                                       ligand_pn_unit_iids: list = ["C_1"],
                                       ligand_ccd_codes: list[str] | None = None,
                                       save_aligned: bool = True,
                                       ref_sample_is_designed: bool = True,
                                       ) -> dict[str, float]:
    """
    Compute docking metrics between a designed structure and its predicted structure, using atom array.
    """
    ligand_ccd_code = _ligand_ccd_code_for_iids(sample_atom_array, ligand_pn_unit_iids, ligand_ccd_codes)
    metal_pn_unit_iids = _selected_metal_pn_unit_iids(sample_atom_array, ligand_pn_unit_iids)
    if metal_pn_unit_iids:
        metal_ligand_ccd_code = _ligand_ccd_code_for_iids(
            sample_atom_array,
            metal_pn_unit_iids,
            [
                str(code)
                for pn_unit_iid, code in zip(ligand_pn_unit_iids, ligand_ccd_codes or [])
                if str(pn_unit_iid) in set(metal_pn_unit_iids)
            ] or None,
        )
        return _compute_metal_docking_metrics_atomarray(
            pred_atom_array=pred_atom_array,
            sample_atom_array=sample_atom_array,
            pred_sample_path=pred_sample_path,
            pocket_distance_for_docking_metrics=pocket_distance_for_docking_metrics,
            receptor_pn_unit_iids=receptor_pn_unit_iids,
            metal_pn_unit_iids=metal_pn_unit_iids,
            ligand_ccd_code=metal_ligand_ccd_code,
            save_aligned=save_aligned,
        )

    # Annotate ligand pockets (binding site residues)
    if ref_sample_is_designed:
        sample_atom_array = annotate_ligand_pockets_pseudocb(atom_array=sample_atom_array,
                                                           pocket_distance=pocket_distance_for_docking_metrics,
                                                           annotation_name="is_ligand_pocket_for_metrics",
                                                           receptor_pn_unit_iids=receptor_pn_unit_iids,
                                                           ligand_pn_unit_iids=ligand_pn_unit_iids)
    else:
        sample_atom_array = annotate_ligand_pockets(atom_array=sample_atom_array,
                                            pocket_distance=pocket_distance_for_docking_metrics,
                                            annotation_name="is_ligand_pocket_for_metrics",
                                            receptor_pn_unit_iids=receptor_pn_unit_iids,
                                            ligand_pn_unit_iids=ligand_pn_unit_iids)

    # Apply and spread residue-wise to get pocket mask
    sample_atom_array_pocket_mask = apply_and_spread_residue_wise(sample_atom_array, sample_atom_array.get_annotation("is_ligand_pocket_for_metrics"), function=np.any)
    sample_atom_array.set_annotation("is_ligand_pocket_for_metrics", sample_atom_array_pocket_mask)

    if ref_sample_is_designed:
        pred_atom_array = annotate_ligand_pockets_pseudocb(atom_array=pred_atom_array,
                                                           pocket_distance=pocket_distance_for_docking_metrics,
                                                           annotation_name="is_ligand_pocket_for_metrics",
                                                           receptor_pn_unit_iids=receptor_pn_unit_iids,
                                                           ligand_pn_unit_iids=ligand_pn_unit_iids)
    else:
        pred_atom_array = annotate_ligand_pockets(atom_array=pred_atom_array,
                                            pocket_distance=pocket_distance_for_docking_metrics,
                                            annotation_name="is_ligand_pocket_for_metrics",
                                            receptor_pn_unit_iids=receptor_pn_unit_iids,
                                            ligand_pn_unit_iids=ligand_pn_unit_iids)

    # Apply and spread residue-wise to get pocket mask
    pred_atom_array_pocket_mask = apply_and_spread_residue_wise(pred_atom_array, pred_atom_array.get_annotation("is_ligand_pocket_for_metrics"), function=np.any)
    pred_atom_array.set_annotation("is_ligand_pocket_for_metrics", pred_atom_array_pocket_mask)

    # Get binding site CA atoms for superposition
    # Use sequential residue index (order in chain) instead of res_id for matching
    # because res_id may differ between structures (ref vs AF3 prediction)
    sample_receptor_mask = np.isin(sample_atom_array.pn_unit_iid, receptor_pn_unit_iids)
    pred_receptor_mask = np.isin(pred_atom_array.pn_unit_iid, receptor_pn_unit_iids)

    # Build initial CA masks (without NaN filtering) to identify matching residue positions
    sample_ca_mask_initial = sample_receptor_mask & (sample_atom_array.atom_name == "CA") & (sample_atom_array.res_name != "UNK")
    pred_ca_mask_initial = pred_receptor_mask & (pred_atom_array.atom_name == "CA") & (pred_atom_array.res_name != "UNK")

    # Compute joint resolved mask: exclude positions where EITHER array has NaN coordinates
    sample_ca_resolved_mask = ~np.isnan(sample_atom_array[sample_ca_mask_initial].coord[:, 0])
    pred_ca_resolved_mask = ~np.isnan(pred_atom_array[pred_ca_mask_initial].coord[:, 0])
    ca_resolved_mask = sample_ca_resolved_mask & pred_ca_resolved_mask

    # Apply joint resolved mask back to full atom_array-level masks
    sample_ca_indices = np.where(sample_ca_mask_initial)[0]
    sample_ca_mask = np.zeros(len(sample_atom_array), dtype=bool)
    sample_ca_mask[sample_ca_indices[ca_resolved_mask]] = True

    pred_ca_indices = np.where(pred_ca_mask_initial)[0]
    pred_ca_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_ca_mask[pred_ca_indices[ca_resolved_mask]] = True

    sample_ca = sample_atom_array[sample_ca_mask]
    pred_ca = pred_atom_array[pred_ca_mask]

    # Check if the number of CA atoms in sample and pred match
    assert len(sample_ca) == len(pred_ca), "Number of CA atoms in sample and pred must match"

    if len(sample_ca) == 0 or len(pred_ca) == 0:
        return _docking_metric_error("No CA atoms found", ligand_ccd_code=ligand_ccd_code)

    # Get binding site mask for CA atoms
    sample_bs_ca_mask = sample_atom_array.is_ligand_pocket_for_metrics[sample_ca_mask]

    # Get binding site CA atoms by sequential index
    sample_bs_sorted = sample_ca[sample_bs_ca_mask]
    pred_bs_sorted = pred_ca[sample_bs_ca_mask]  # Use sample's binding site mask for pred

    # check if the binding site residues in sample and pred match
    assert (sample_bs_sorted.res_name == pred_bs_sorted.res_name).all(), "amino acid residues in sample and pred binding site must match"

    num_bs_residues = np.sum(sample_bs_ca_mask)

    if len(sample_bs_sorted) == 0:
        return _docking_metric_error("No binding site CA atoms found", ligand_ccd_code=ligand_ccd_code)

    # Align pred onto ref using binding site CA atoms
    # align_atom_arrays: aligns mbl_sele to tgt_sele, applies transform to mbl_full
    pred_aligned_atom_array, bs_rmsd = align_atom_arrays(
        mbl_sele=pred_bs_sorted,  # pred binding site (to be aligned)
        tgt_sele=sample_bs_sorted,   # ref binding site (target)
        mbl_full=pred_atom_array       # full pred structure (to be transformed)
    )

    # Prepare masks for ligand and binding site
    sample_ligand_mask = np.isin(sample_atom_array.pn_unit_iid, ligand_pn_unit_iids) & (sample_atom_array.element != "H")
    pred_ligand_mask = np.isin(pred_aligned_atom_array.pn_unit_iid, ligand_pn_unit_iids) & (pred_aligned_atom_array.element != "H")
    pred_binding_site_mask = (pred_aligned_atom_array.is_ligand_pocket_for_metrics == True) & (pred_aligned_atom_array.res_name != "UNK")

    # Get ligand atom arrays from sample and pred
    sample_ligand_atom_array = sample_atom_array[sample_ligand_mask]
    pred_ligand_atom_array = pred_aligned_atom_array[pred_ligand_mask]

    if len(sample_ligand_atom_array) == 0 or len(pred_ligand_atom_array) == 0:
        return _docking_metric_error("No ligand atoms found", ligand_ccd_code=ligand_ccd_code)

    # Match ligand atoms by name
    sample_ligand_atom_names = sample_ligand_atom_array.atom_name
    pred_ligand_atom_names = pred_ligand_atom_array.atom_name
    common_atom_names = np.intersect1d(sample_ligand_atom_names, pred_ligand_atom_names)

    if len(common_atom_names) == 0:
        return _docking_metric_error("No common ligand atoms", ligand_ccd_code=ligand_ccd_code)

    # Calculate symmetry-corrected RMSD using RDKit
    ligand_rmsd = None
    try:
        try:
            sample_mol = atom_array_to_rdkit(sample_ligand_atom_array, sanitize=True)
        except Exception:
            sample_mol = atom_array_to_rdkit(sample_ligand_atom_array, sanitize=False)
        try:
            pred_mol = atom_array_to_rdkit(pred_ligand_atom_array, sanitize=True)
        except Exception:
            pred_mol = atom_array_to_rdkit(pred_ligand_atom_array, sanitize=False)

        if sample_mol and pred_mol:
            # Remove hydrogens for RMSD calculation
            sample_mol = Chem.RemoveHs(sample_mol)
            pred_mol = Chem.RemoveHs(pred_mol)

            try:
                # Use CalcRMS instead of GetBestRMS to compute symmetry-aware RMSD
                # WITHOUT additional alignment (in-place calculation)
                # This is what we want for docking poses after binding site superposition
                ligand_rmsd = rdMolAlign.CalcRMS(sample_mol, pred_mol)
                print(f"using CalcRMS (no alignment, symmetry-aware): {ligand_rmsd:.4f} Å")
            except:
                print(f"CalcRMS failed using (sample_mol, pred_mol), sample_mol: {sample_mol.GetNumHeavyAtoms()}, pred_mol: {pred_mol.GetNumHeavyAtoms()}")
                print(f"This is i) because the number of heavy atoms of sample_mol can be modified because of atomworks preprocessing")
                print(f"This is ii) or the ligand structure of AF3 prediction is wrong, e.g.) RI2 in 5yft")
                print(f"In this case, sample_mol can be not a substructure of pred_mol, thus giving CalcRMS error")
                print(f"So trying (pred_mol, sample_mol) instead")
                try:
                    ligand_rmsd = rdMolAlign.CalcRMS(pred_mol, sample_mol)
                    print(f"using CalcRMS (no alignment, symmetry-aware): {ligand_rmsd:.4f} Å")
                except Exception as e:
                    print(f"Both directions failed, cannot compute RMSD")
                    print(f"Error: {e}")

    except Exception as e:
        print(f"Failed to calculate ligand RMSD: {e}")
        return _docking_metric_error("Failed to calculate ligand RMSD using RDKit", ligand_ccd_code=ligand_ccd_code)


    # Calculate AF3 confidence metrics using the aligned pred structure
    confidence_dir = str(pred_sample_path.parent)
    full_confidence_file_path = f"{confidence_dir}/{re.sub(r'_model$', '_confidences', str(pred_sample_path.stem))}.json"
    summary_confidence_file_path = f"{confidence_dir}/{re.sub(r'_model$', '_summary_confidences', str(pred_sample_path.stem))}.json"
    ligand_plddt = extract_af3_confidence_metrics(confidence_file_path=full_confidence_file_path,
                                                   atom_array=pred_aligned_atom_array,
                                                   mask=pred_ligand_mask,
                                                   metrics_to_extract="atom_plddts",
                                                   return_mean=True)

    binding_site_plddt = extract_af3_confidence_metrics(confidence_file_path=full_confidence_file_path,
                                                   atom_array=pred_aligned_atom_array,
                                                   mask=pred_binding_site_mask,
                                                   metrics_to_extract="atom_plddts",
                                                   return_mean=True)

    iptm = extract_af3_confidence_metrics(confidence_file_path=summary_confidence_file_path,
                                                   atom_array=pred_aligned_atom_array,
                                                   metrics_to_extract="iptm",
                                                   return_mean=True)

    interface_min_pae = extract_af3_confidence_metrics(confidence_file_path=summary_confidence_file_path,
                                                   atom_array=pred_aligned_atom_array,
                                                   metrics_to_extract="interface_min_pae",
                                                   return_mean=True)


    # Save pocket-aligned structure
    if save_aligned:
        # Create output path with "_pocket_aligned" suffix
        out_file = Path(pred_sample_path).parent / f"{Path(pred_sample_path).stem}_pocket_aligned.cif"
        try:
            save_cif_file(pred_aligned_atom_array, out_file)
        except Exception as e:
            print(f"Warning: Failed to save aligned structure: {e}")

    return {
        "ligand_rmsd": ligand_rmsd,
        "binding_site_rmsd": float(bs_rmsd),
        "num_bs_residues": int(num_bs_residues),
        "ligand_plddt": ligand_plddt,
        "binding_site_plddt": binding_site_plddt,
        "iptm": iptm,
        "interface_min_pae": interface_min_pae,
        "ligand_ccd_code": ligand_ccd_code,
    }

def extract_af3_confidence_metrics(confidence_file_path: str = None,
                                    atom_array: AtomArray = None,
                                    mask: TensorType["n", bool] = None,
                                    metrics_to_extract: str = "atom_plddts",
                                    return_mean: bool = True):
    """
    Extract confidence metrics from an AF3 confidence file.

    Note: aw_parse adds unresolved residues with NaN coordinates, so atom_array may have
    more atoms than the confidence file. We filter to only valid (non-NaN) coordinates.
    """
    with open(confidence_file_path, "r") as f:
        confidence_data = json.load(f)


    if metrics_to_extract == "atom_plddts":
        metric = torch.tensor(confidence_data["atom_plddts"], dtype=torch.float16)

        # Filter out NaN coordinate atoms: aw_parse with add_missing_atoms=True
        # adds unresolved atoms with NaN coordinates that don't exist in AF3 output.
        valid_coords_mask = ~np.isnan(atom_array.coord).any(axis=1)
        num_valid_atoms = int(valid_coords_mask.sum())

        assert len(metric) == num_valid_atoms, (
            f"Number of pLDDTs ({len(metric)}) != number of valid (non-NaN) atoms ({num_valid_atoms}). "
            f"Total atoms in atom_array: {len(atom_array)}, NaN atoms: {len(atom_array) - num_valid_atoms}"
        )

        # Filter mask to only valid (non-NaN) atoms so it aligns with metric
        if isinstance(mask, np.ndarray):
            mask_torch = torch.tensor(mask[valid_coords_mask], dtype=torch.bool)
        else:
            mask_torch = mask[torch.tensor(valid_coords_mask, dtype=torch.bool)].bool()

        # Apply mask to pLDDTs
        metric = metric[mask_torch]

        if return_mean:
            if len(metric) == 0:
                return None
            return metric.mean().item()
        else:
            return metric
    elif metrics_to_extract == "iptm":
        metric = confidence_data["iptm"]

    elif metrics_to_extract == "interface_min_pae":
        try:
            pae_01 = confidence_data["chain_pair_pae_min"][0][1]
            pae_10 = confidence_data["chain_pair_pae_min"][1][0]
            metric = min(pae_01, pae_10)
        except:
            print(f"Warning: Failed to extract interface_min_pae from confidence file: {confidence_file_path}")
            metric = None

    else:
        raise ValueError(f"Invalid metric to extract: {metrics_to_extract}")

    return metric

def calculate_ligand_rmsd_with_binding_site_superposition(
    pred_example: dict[str, Any] = None,
    sample_example: dict[str, Any] = None,
    receptor_pn_unit_iids: list = ["A_1"],
    ligand_pn_unit_iids: list = ["C_1"],
    pocket_distance: float = 8.0,
    save_aligned: bool = True,
    sample_path: str | Path = None,
    pred_path: str | Path = None,
) -> dict[str, float]:
    """
    Calculate ligand RMSD after superimposing structures based on binding site residues.
    Uses Atomworks framework for loading and processing.

    Parameters
    ----------
    ref_cif_path : Path
        Path to reference CIF file.
    pred_cif_path : Path
        Path to predicted CIF file.
    receptor_chain : str
        Chain ID for receptor.
    ligand_chain : str
        Chain ID for ligand.
    binding_site_radius : float
        Radius for defining binding site residues.
    save_aligned : bool
        If True, save the pocket-aligned predicted structure to the same directory
        with "_pocket_aligned" suffix.
    cif_parser_args: DictConfig = None,
        Additional keyword arguments to pass to atomworks.io.parser.parse.
        Useful for controlling hydrogen_policy, add_missing_atoms, etc.

    Returns
    -------
    dict
        Dictionary with RMSD values and other metrics.
    """

    sample_array = sample_example['atom_array']
    pred_array = pred_example['atom_array']

    print(f"pocket_distance: {pocket_distance}")
    # Annotate ligand pockets (binding site residues)
    sample_array = annotate_ligand_pockets(atom_array=sample_array,
                                           pocket_distance=pocket_distance,
                                           receptor_pn_unit_iids=receptor_pn_unit_iids,
                                           ligand_pn_unit_iids=ligand_pn_unit_iids)
    pred_array = annotate_ligand_pockets(atom_array=pred_array,
                                         pocket_distance=pocket_distance,
                                         receptor_pn_unit_iids=receptor_pn_unit_iids,
                                         ligand_pn_unit_iids=ligand_pn_unit_iids)

    # Get binding site CA atoms for superposition
    # Use sequential residue index (order in chain) instead of res_id for matching
    # because res_id may differ between structures (ref vs AF3 prediction)
    sample_receptor_mask = np.isin(sample_array.pn_unit_iid, receptor_pn_unit_iids)
    pred_receptor_mask = np.isin(pred_array.pn_unit_iid, receptor_pn_unit_iids)

    # Get all CA atoms from receptor chain
    sample_ca_mask = sample_receptor_mask & (sample_array.atom_name == "CA") & (sample_array.res_name != "UNK")

    # Delete UNK residues from pred_atom_array, it's from the sample sequence for the gaps between the actual residues.
    # Designed sequence don't output UNK residues, so we can safely delete them.
    pred_ca_mask = pred_receptor_mask & (pred_array.atom_name == "CA") & (pred_array.res_name != "UNK")

    sample_ca = sample_array[sample_ca_mask]
    pred_ca = pred_array[pred_ca_mask]

    assert len(sample_ca) == len(pred_ca), "Number of CA atoms in sample and pred must match"

    if len(sample_ca) == 0 or len(pred_ca) == 0:
        return {"error": "No CA atoms found", "ligand_rmsd": None}

    # Get binding site mask for CA atoms
    sample_bs_ca_mask = sample_array.is_ligand_pocket[sample_ca_mask]

    # Get binding site CA atoms by sequential index
    sample_bs_sorted = sample_ca[sample_bs_ca_mask]
    pred_bs_sorted = pred_ca[sample_bs_ca_mask]  # Use ref's BS mask for both

    assert (sample_bs_sorted.res_name == pred_bs_sorted.res_name).all(), "amino acid residues in sample and pred binding site must match"

    num_bs_residues = np.sum(sample_bs_ca_mask)

    if len(sample_bs_sorted) == 0:
        return {"error": "No binding site CA atoms found", "ligand_rmsd": None}

    # Align pred onto ref using binding site CA atoms
    # align_atom_arrays: aligns mbl_sele to tgt_sele, applies transform to mbl_full
    pred_aligned, bs_rmsd = align_atom_arrays(
        mbl_sele=pred_bs_sorted,  # pred binding site (to be aligned)
        tgt_sele=sample_bs_sorted,   # ref binding site (target)
        mbl_full=pred_array       # full pred structure (to be transformed)
    )

    # Get ligand atoms
    sample_lig_mask = np.isin(sample_array.pn_unit_iid, ligand_pn_unit_iids) & (sample_array.element != "H")
    pred_lig_mask = np.isin(pred_aligned.pn_unit_iid, ligand_pn_unit_iids) & (pred_aligned.element != "H")

    sample_lig = sample_array[sample_lig_mask]
    pred_lig = pred_aligned[pred_lig_mask]

    if len(sample_lig) == 0 or len(pred_lig) == 0:
        return {"error": "No ligand atoms found", "ligand_rmsd": None}

    # Match ligand atoms by name
    sample_atom_names = sample_lig.atom_name
    pred_atom_names = pred_lig.atom_name
    common_atom_names = np.intersect1d(sample_atom_names, pred_atom_names)

    if len(common_atom_names) == 0:
        return {"error": "No common ligand atoms", "ligand_rmsd": None}

    # Calculate symmetry-corrected RMSD using RDKit
    ligand_rmsd = None
    try:
        # Convert ligand atom arrays to RDKit molecules
        sample_lig_full = sample_array[np.isin(sample_array.pn_unit_iid, ligand_pn_unit_iids)]
        pred_lig_full = pred_aligned[np.isin(pred_aligned.pn_unit_iid, ligand_pn_unit_iids)]

        # Use atom_array_to_rdkit with sanitize fallback
        try:
            sample_mol = atom_array_to_rdkit(sample_lig_full, sanitize=True)
            print("Sample ligand sanitization successful")
        except Exception:
            sample_mol = atom_array_to_rdkit(sample_lig_full, sanitize=False)
            print("Sample ligand sanitization failed, not using sanitization fallback")

        try:
            pred_mol = atom_array_to_rdkit(pred_lig_full, sanitize=True)
            print("Pred ligand sanitization successful")
        except Exception:
            pred_mol = atom_array_to_rdkit(pred_lig_full, sanitize=False)
            print("Pred ligand sanitization failed, not using sanitization fallback")

        if sample_mol and pred_mol:
            # Remove hydrogens for RMSD calculation
            sample_mol = Chem.RemoveHs(sample_mol)
            pred_mol = Chem.RemoveHs(pred_mol)

            # Try substructure match first
            match = sample_mol.GetSubstructMatch(pred_mol)
            if match:
                ligand_rmsd = rdMolAlign.CalcRMS(sample_mol, pred_mol)
                print(f"Substructure match found, symmetry-corrected RMSD: {ligand_rmsd:.4f} Å")
            else:
                ligand_rmsd = AllChem.GetBestRMS(sample_mol, pred_mol)
                print(f"No substructure match found, using GetBestRMS: {ligand_rmsd:.4f} Å")

    except Exception:
        return {"error": "Failed to calculate ligand RMSD using RDKit", "ligand_rmsd": None}

    # Calculate best ligand RMSD
    # Save pocket-aligned structure if requested
    aligned_path = None
    if save_aligned:
        # Create output path with "_pocket_aligned" suffix
        aligned_path = Path(pred_path).parent / f"{Path(pred_path).stem}_pocket_aligned.cif"
        try:
            to_cif_file(
                pred_aligned,
                aligned_path,
                include_entity_poly=True,
                include_entity_nonpoly=True,
                include_nan_coords=False,
                include_bonds=True,
            )
        except Exception as e:
            print(f"Warning: Failed to save aligned structure: {e}")
            aligned_path = None

    #! return pred_array and masks for pLDDT extraction
    # Create ligand and binding site masks for the aligned pred structure
    pred_ligand_mask = np.isin(pred_aligned.pn_unit_iid, ligand_pn_unit_iids)
    pred_binding_site_mask = (pred_aligned.is_ligand_pocket == True) & (pred_aligned.res_name != "UNK")

    return {
        "ligand_rmsd": ligand_rmsd,
        "binding_site_rmsd": bs_rmsd,
        "num_bs_residues": int(num_bs_residues),
        "num_matched_atoms": len(common_atom_names),
        "aligned_path": str(aligned_path) if aligned_path else None,
        "aligned_pred_array": pred_aligned,
        "pred_ligand_mask": pred_ligand_mask,
        "pred_binding_site_mask": pred_binding_site_mask,
    }



def _compute_tmalign_score(pdb_1: str, pdb_2: str) -> tuple[float, float]:
    """
    Compute TM-score between two PDBs. This uses TM-align, so
    we don't need to check if the residue indices are aligned.

    Returns:
        - tmalign_score_1: TM-align score normalized by length of pdb_1
        - tmalign_score_2: TM-align score normalized by length of pdb_2
    """
    try:
        cmd = [f"{os.environ['SOFTWARE_PATH']}/tmalign/TMalign", pdb_1, pdb_2]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        tmalign_score_1 = None
        tmalign_score_2 = None

        # Parse TM-align output
        for line in output.splitlines():
            if line.startswith("TM-score="):
                parts = line.strip().split()
                score = float(parts[1])
                if "Chain_1" in line:
                    tmalign_score_1 = score
                elif "Chain_2" in line:
                    tmalign_score_2 = score

        tmalign_score_1 = tmalign_score_1 if tmalign_score_1 is not None else np.nan
        tmalign_score_2 = tmalign_score_2 if tmalign_score_2 is not None else np.nan
    except Exception as e:
        print(f"Error computing TM-align score: {e}")
        tmalign_score_1 = np.nan
        tmalign_score_2 = np.nan

    return tmalign_score_1, tmalign_score_2
