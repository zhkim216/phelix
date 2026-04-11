"""
Consolidated Potts energy analysis utilities.

Provides functions for:
  - Potts forward passes (with/without ligand conditioning)
  - Delta metric computation (h, J, combined)
  - Distance computation (all-atom and pseudo-CB to ligand)
  - Per-PDB normalization of delta metrics
  - Cutoff-based pocket/scaffold residue selection
  - Overlap and recall analysis for pocket selection evaluation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray, get_residue_starts
from scipy.spatial.distance import cdist

import atomworks.enums as aw_enums


###########################################################
# Token/residue mapping
###########################################################


def map_token_to_residue_info(
    atom_array: AtomArray,
) -> list[tuple[str, int, str]]:
    """Get (chain_id, res_id, res_name) for each token in the atom array.

    Tokens correspond to residue starts in the atom array.
    """
    starts = get_residue_starts(atom_array)
    return [
        (
            str(atom_array.chain_id[s]),
            int(atom_array.res_id[s]),
            str(atom_array.res_name[s]),
        )
        for s in starts
    ]


###########################################################
# Distance computation
###########################################################


def compute_per_residue_min_distance_to_ligand(
    atom_array: AtomArray,
) -> dict[tuple[str, int], float]:
    """Compute per-residue minimum all-atom distance to any ligand atom.

    Returns dict mapping (chain_id, res_id) -> min distance in angstroms.
    Only includes protein residues.
    """
    ligand_mask = (~atom_array.is_covalent_modification) & (~atom_array.is_polymer)
    ligand_coords = atom_array.coord[ligand_mask]

    valid_lig = ~np.isnan(ligand_coords).any(axis=1)
    ligand_coords = ligand_coords[valid_lig]
    if len(ligand_coords) == 0:
        return {}

    prot_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    valid_prot = prot_mask & ~np.isnan(atom_array.coord).any(axis=1)

    prot_indices = np.where(valid_prot)[0]
    if len(prot_indices) == 0:
        return {}

    prot_coords = atom_array.coord[prot_indices]
    prot_chain_ids = atom_array.chain_id[prot_indices]
    prot_res_ids = atom_array.res_id[prot_indices]

    dists = cdist(prot_coords, ligand_coords)
    min_dists_per_atom = dists.min(axis=1)

    residue_min_dists: dict[tuple[str, int], float] = {}
    for i, (cid, rid) in enumerate(zip(prot_chain_ids, prot_res_ids)):
        key = (str(cid), int(rid))
        d = float(min_dists_per_atom[i])
        if key not in residue_min_dists or d < residue_min_dists[key]:
            residue_min_dists[key] = d

    return residue_min_dists


def compute_pseudocb_distances(
    atom_array: AtomArray,
) -> dict[tuple[str, int], float]:
    """Compute per-residue pseudo-CB distance to nearest ligand atom.

    Uses pseudo-CB formula: -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
    where a = cross(b, c), b = CA - N, c = C - CA.

    Returns dict mapping (chain_id, res_id) -> min pseudo-CB distance in angstroms.
    """
    from atomworks.ml.utils.token import apply_and_spread_token_wise

    # Identify valid ligand atoms (>=5 atoms per ligand unit)
    ligand_mask = (~atom_array.is_covalent_modification) & (~atom_array.is_polymer)
    lig_iids, lig_counts = np.unique(
        atom_array.pn_unit_iid[ligand_mask], return_counts=True
    )
    valid_lig_iids = lig_iids[lig_counts >= 5]
    all_valid_lig_mask = np.isin(atom_array.pn_unit_iid, valid_lig_iids)
    ligand_coords = atom_array.coord[all_valid_lig_mask]

    valid_lig = ~np.isnan(ligand_coords).any(axis=1)
    ligand_coords = ligand_coords[valid_lig]
    if len(ligand_coords) == 0:
        return {}

    # Valid protein residues with resolved N, CA, C, O
    is_atomized = atom_array.atomize
    if hasattr(atom_array, "atom_is_protein_chain"):
        prot_mask = ~is_atomized & atom_array.atom_is_protein_chain
    else:
        prot_mask = ~is_atomized & (
            atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
        )

    is_ncaco_resolved = np.isin(atom_array.atom_name, ["N", "CA", "C", "O"]) & (
        atom_array.occupancy > 0
    )
    has_all_backbone = apply_and_spread_token_wise(
        atom_array, is_ncaco_resolved, lambda x: np.sum(x) == 4
    )
    valid_res = prot_mask & has_all_backbone

    ca_mask = valid_res & (atom_array.atom_name == "CA")
    n_mask = valid_res & (atom_array.atom_name == "N")
    c_mask = valid_res & (atom_array.atom_name == "C")
    if ca_mask.sum() == 0:
        return {}

    ca_coords = atom_array.coord[ca_mask]
    n_coords = atom_array.coord[n_mask]
    c_coords = atom_array.coord[c_mask]

    b = ca_coords - n_coords
    c_vec = c_coords - ca_coords
    a = np.cross(b, c_vec)
    pseudo_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca_coords

    valid_cb = ~np.isnan(pseudo_cb).any(axis=1)
    if valid_cb.sum() == 0:
        return {}

    dists = cdist(pseudo_cb[valid_cb], ligand_coords)
    min_dists = dists.min(axis=1)

    chain_ids = atom_array.chain_id[ca_mask][valid_cb]
    res_ids = atom_array.res_id[ca_mask][valid_cb]

    return {
        (str(cid), int(rid)): float(d)
        for cid, rid, d in zip(chain_ids, res_ids, min_dists)
    }


###########################################################
# Potts forward pass
###########################################################


def run_potts_forward(
    model: torch.nn.Module,
    batch: dict,
    protein_only: bool,
) -> dict[str, torch.Tensor]:
    """Run a single forward pass to get Potts parameters.

    Args:
        model: SeqDenoiser model.
        batch: Batch dict from get_sd_batch (modified in place).
        protein_only: If True, remove ligand conditioning.

    Returns:
        potts_decoder_aux dict with h, J, edge_idx, mask_i, mask_ij.
    """
    from allatom_design.eval.eval_utils.seq_des_utils import initialize_sampling_masks

    batch = initialize_sampling_masks(batch, protein_only=protein_only)
    batch["noise_labels"] = None
    batch["noise"] = None

    sampling_inputs = {"batch_size": 1, "add_noise": False}
    potts_decoder_aux, batch_out, _ = model.denoiser.compute_potts_params(
        batch, sampling_inputs=sampling_inputs
    )
    return potts_decoder_aux


###########################################################
# Delta metric computation
###########################################################


def compute_potts_deltas(
    potts_lig: dict[str, torch.Tensor],
    potts_nol: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute per-residue delta metrics between with-ligand and without-ligand Potts params.

    Returns dict with:
        delta_h: ||h_lig - h_nol||_2 per residue [N]
        delta_J: sum_j ||J_ij_lig - J_ij_nol||_F per residue [N]
        delta_combined: (delta_h + delta_J) / max(delta_h + delta_J), per-PDB normalized [N]
        mask_i: valid residue mask [N]
    """
    h_lig = potts_lig["h"][0]  # [N, 20]
    h_nol = potts_nol["h"][0]
    J_lig = potts_lig["J"][0]  # [N, K, 20, 20]
    J_nol = potts_nol["J"][0]
    mask_i = potts_lig["mask_i"][0]  # [N]
    mask_ij = potts_lig["mask_ij"][0]  # [N, K]

    delta_h = torch.norm(h_lig - h_nol, dim=-1)  # [N]

    delta_J_diff = (J_lig - J_nol).flatten(-2, -1)  # [N, K, 400]
    delta_J_per_edge = torch.norm(delta_J_diff, dim=-1)  # [N, K]
    delta_J = (delta_J_per_edge * mask_ij).sum(dim=-1)  # [N]

    valid = mask_i.bool()
    raw_sum = delta_h + delta_J
    delta_combined = raw_sum / raw_sum[valid].max().clamp(min=1e-8)  # [N]

    return {
        "delta_h": delta_h.cpu(),
        "delta_J": delta_J.cpu(),
        "delta_combined": delta_combined.cpu(),
        "mask_i": mask_i.cpu(),
    }


def normalize_deltas_per_pdb(
    delta: torch.Tensor,
    mask_i: torch.Tensor,
) -> torch.Tensor:
    """Per-PDB max normalization of a delta tensor.

    Returns delta / max(delta[valid]), clamped to avoid division by zero.
    """
    valid = mask_i.bool()
    delta_max = delta[valid].max().clamp(min=1e-8)
    return delta / delta_max


###########################################################
# Cutoff-based pocket/scaffold selection
###########################################################


def apply_cutoff(
    norm_values: torch.Tensor,
    token_info: list[tuple[str, int, str]],
    mask_i: torch.Tensor,
    cutoff: float,
    constraint_type: str = "pocket",
) -> tuple[list[str], list[int], int]:
    """Apply normalized value cutoff to select pocket or scaffold residues.

    Args:
        norm_values: Per-residue normalized values (e.g. norm_J) [N].
        token_info: List of (chain_id, res_id, res_name) per token.
        mask_i: Valid residue mask [N].
        cutoff: Threshold for pocket selection.
        constraint_type: "pocket" selects norm_values >= cutoff,
                         "scaffold" selects norm_values < cutoff (complement).

    Returns:
        (chain_ids, res_ids, num_selected)
    """
    valid = mask_i.bool()
    pocket_mask = (norm_values >= cutoff) & valid

    if constraint_type == "scaffold":
        select_mask = valid & ~pocket_mask
    else:
        select_mask = pocket_mask

    chain_ids = []
    res_ids = []
    for idx in range(len(mask_i)):
        if select_mask[idx] and idx < len(token_info):
            cid, rid, _ = token_info[idx]
            chain_ids.append(cid)
            res_ids.append(rid)

    return chain_ids, res_ids, len(chain_ids)


###########################################################
# Overlap and recall analysis
###########################################################


def compute_overlap(
    per_res_df: pd.DataFrame,
    potts_cutoffs: list[float],
    dist_cutoffs: list[float],
) -> pd.DataFrame:
    """Compute pocket overlap: |potts_pocket intersection cb_pocket| / |potts_pocket|.

    Args:
        per_res_df: DataFrame with columns pdb_key, norm_J, pseudo_cb_dist.
        potts_cutoffs: List of norm_J thresholds.
        dist_cutoffs: List of distance thresholds in angstroms.

    Returns:
        DataFrame with coverage metrics per pdb_key x potts_cutoff x dist_cutoff.
    """
    rows = []
    for pdb_key, grp in per_res_df.groupby("pdb_key"):
        valid = grp.dropna(subset=["pseudo_cb_dist"])
        for pc in potts_cutoffs:
            potts_pocket = valid["norm_J"] >= pc
            n_potts = int(potts_pocket.sum())
            for dc in dist_cutoffs:
                cb_pocket = valid["pseudo_cb_dist"] <= dc
                n_cb = int(cb_pocket.sum())
                n_overlap = int((potts_pocket & cb_pocket).sum())
                coverage = n_overlap / n_potts if n_potts > 0 else float("nan")
                rows.append({
                    "pdb_key": pdb_key,
                    "potts_cutoff": pc,
                    "dist_cutoff": dc,
                    "n_potts_pocket": n_potts,
                    "n_cb_pocket": n_cb,
                    "n_overlap": n_overlap,
                    "coverage": coverage,
                    "n_total": len(valid),
                })
    return pd.DataFrame(rows)


def compute_recall_per_pdb(
    df: pd.DataFrame,
    cutoffs: np.ndarray,
    pocket_distances: list[float],
) -> pd.DataFrame:
    """Compute recall and precision of norm_J-based pocket selection.

    For each PDB x cutoff x pocket_distance, computes recall and precision
    against distance-based ground truth.

    Args:
        df: DataFrame with columns pdb_id, norm_J, min_distance.
        cutoffs: Array of norm_J threshold values to sweep.
        pocket_distances: List of distance thresholds for ground truth pockets.

    Returns:
        DataFrame with recall, precision, frac_selected per combination.
    """
    rows = []
    for pdb_id, sub in df.groupby("pdb_id"):
        n_total = len(sub)
        for cutoff in cutoffs:
            predicted = sub["norm_J"] >= cutoff
            n_selected = predicted.sum()
            frac_selected = n_selected / n_total

            for pd_cutoff in pocket_distances:
                true_pocket = sub["min_distance"] < pd_cutoff
                n_true = true_pocket.sum()

                if n_true == 0:
                    recall = np.nan
                    precision = np.nan
                else:
                    tp = (predicted & true_pocket).sum()
                    recall = tp / n_true
                    precision = tp / n_selected if n_selected > 0 else 0.0

                rows.append({
                    "pdb_id": pdb_id,
                    "cutoff": round(cutoff, 2),
                    "pocket_dist": pd_cutoff,
                    "n_true_pocket": n_true,
                    "n_selected": n_selected,
                    "n_total": n_total,
                    "frac_selected": frac_selected,
                    "recall": recall,
                    "precision": precision,
                })
    return pd.DataFrame(rows)


def load_and_normalize_results(csv_path: str | Path) -> pd.DataFrame:
    """Load results CSV and add per-PDB normalized delta_J column (norm_J).

    Args:
        csv_path: Path to results.csv with columns pdb_id, delta_J, min_distance, etc.

    Returns:
        DataFrame with added norm_J column.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["min_distance"])
    df["norm_J"] = df.groupby("pdb_id")["delta_J"].transform(
        lambda x: x / x.max()
    )
    return df
