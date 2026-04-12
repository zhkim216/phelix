"""Parse Glide output files (CSV scores) and compute pose metrics."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from atomworks.io.tools.rdkit import atom_array_to_rdkit

logger = logging.getLogger(__name__)

# Glide CSV column name mapping: internal name -> human-readable name
GLIDE_SCORE_COLUMNS = {
    "r_i_docking_score": "docking_score",
    "r_i_glide_gscore": "glide_score",
    "r_i_glide_emodel": "emodel",
    "r_i_glide_energy": "glide_energy",
    "r_i_glide_ligand_efficiency": "ligand_efficiency",
    "r_i_glide_ligand_efficiency_sa": "ligand_efficiency_sa",
    "r_i_glide_ligand_efficiency_ln": "ligand_efficiency_ln",
    "r_i_glide_evdw": "evdw",
    "r_i_glide_ecoul": "ecoul",
    "r_i_glide_einternal": "einternal",
    "r_i_glide_erotb": "erotb",
    "r_i_glide_esite": "esite",
    "r_i_glide_lipo": "lipo",
    "r_i_glide_hbond": "hbond",
    "r_i_glide_metal": "metal",
    "r_i_glide_rewards": "rewards",
    "r_i_glide_RMSdev": "input_rmsd",
}


def parse_glide_csv(csv_path: str) -> pd.DataFrame:
    """Parse a Glide CSV output file.

    Renames internal Schrodinger property names to human-readable names.

    Args:
        csv_path: Path to Glide CSV file.

    Returns:
        DataFrame with one row per docked pose, columns renamed.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Glide CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Rename known columns
    rename_map = {}
    for old_name, new_name in GLIDE_SCORE_COLUMNS.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name
    if rename_map:
        df = df.rename(columns=rename_map)

    # Rename title column
    if "Title" in df.columns:
        df = df.rename(columns={"Title": "title"})

    return df


def extract_best_scores(results_df: pd.DataFrame) -> dict[str, float]:
    """Extract the best (lowest) scores from parsed Glide results.

    Returns:
        Dict with best scores (lowest docking_score / glide_score).
    """
    scores: dict[str, float] = {}

    if results_df.empty:
        return scores

    for col in ["docking_score", "glide_score", "emodel"]:
        if col in results_df.columns:
            scores[f"best_{col}"] = results_df[col].min()

    if "ligand_efficiency" in results_df.columns:
        scores["best_ligand_efficiency"] = results_df["ligand_efficiency"].min()

    return scores


def compute_redock_vs_reference_rmsd(
    redock_sdf_path: str,
    ref_ligand_array,
) -> dict[str, Any]:
    """Compute symmetry-corrected RMSD between redocked pose and reference ligand.

    Both inputs must be in the same coordinate frame (pocket-aligned).
    Uses RDKit CalcRMS for symmetry-aware atom mapping without re-alignment.

    Args:
        redock_sdf_path: Path to Glide redocked SDF file.
        ref_ligand_array: Reference ligand AtomArray (from original sample CIF).

    Returns:
        Dict with 'redock_vs_ref_ligand_rmsd' and optional 'error'.
    """
    # Load redocked pose (best pose = first molecule)
    suppl = Chem.SDMolSupplier(redock_sdf_path, removeHs=True)
    redock_mol = next((m for m in suppl if m is not None), None)
    if redock_mol is None:
        return {"redock_vs_ref_ligand_rmsd": None, "error": "no_valid_pose_in_sdf"}

    # Convert reference ligand AtomArray to RDKit mol
    try:
        ref_mol = atom_array_to_rdkit(ref_ligand_array, sanitize=True)
    except Exception:
        try:
            ref_mol = atom_array_to_rdkit(ref_ligand_array, sanitize=False)
        except Exception:
            return {"redock_vs_ref_ligand_rmsd": None, "error": "ref_rdkit_conversion_failed"}

    if ref_mol is None:
        return {"redock_vs_ref_ligand_rmsd": None, "error": "ref_rdkit_conversion_failed"}

    ref_mol = Chem.RemoveHs(ref_mol)
    redock_mol = Chem.RemoveHs(redock_mol)

    # Compute symmetry-corrected RMSD (no re-alignment)
    try:
        match = ref_mol.GetSubstructMatch(redock_mol)
        if match:
            rmsd = rdMolAlign.CalcRMS(ref_mol, redock_mol)
        else:
            rmsd = AllChem.GetBestRMS(ref_mol, redock_mol)
    except Exception as e:
        return {"redock_vs_ref_ligand_rmsd": None, "error": f"rmsd_calc_failed: {e}"}

    return {"redock_vs_ref_ligand_rmsd": rmsd}
