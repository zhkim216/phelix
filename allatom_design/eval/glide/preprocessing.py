"""Preprocessing AF3 predicted structures for Glide evaluation.

Reads CIF files, separates protein and ligand, writes to PDB/SDF
for Schrodinger tools.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import AtomArray
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem

import atomworks.enums as aw_enums
from atomworks.constants import METAL_ELEMENTS
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.io.utils.io_utils import to_pdb_string

from allatom_design.utils.sample_io_utils import load_example_with_parse

logger = logging.getLogger(__name__)


def preprocess_structure(
    cif_path: str,
    out_dir: str,
    sample_id: str | None = None,
    cif_parse_cfg: dict | None = None,
    receptor_pn_unit_iids: list[str] | None = None,
    ligand_pn_unit_iids: list[str] | None = None,
) -> dict[str, Any]:
    """Read an AF3 predicted CIF, separate protein and ligand, write to PDB/SDF.

    Args:
        cif_path: Path to AF3 predicted CIF file.
        out_dir: Directory to write output files.
        sample_id: Sample identifier. Defaults to CIF filename stem.
        cif_parse_cfg: Config for atomworks CIF parser.
        receptor_pn_unit_iids: Protein chain IDs. Auto-detected if None.
        ligand_pn_unit_iids: Ligand chain IDs. Auto-detected if None.

    Returns:
        Dict with paths, arrays, and metadata for downstream pipeline steps.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_id is None:
        sample_id = Path(cif_path).stem

    # Load structure using existing atomworks parser
    # Convert dict to OmegaConf if needed (load_example_with_parse expects DictConfig)
    if cif_parse_cfg is not None and not isinstance(cif_parse_cfg, DictConfig):
        cif_parse_cfg = OmegaConf.create(cif_parse_cfg)
    example = load_example_with_parse(cif_path, cif_parse_cfg=cif_parse_cfg)
    atom_array = example["atom_array"]

    # Auto-detect protein and ligand chains if not specified
    if receptor_pn_unit_iids is None:
        receptor_pn_unit_iids = get_protein_pn_unit_iids(atom_array)
    if ligand_pn_unit_iids is None:
        ligand_pn_unit_iids = get_ligand_pn_unit_iids(atom_array)

    if not receptor_pn_unit_iids:
        raise ValueError(f"No protein chains found in {cif_path}")
    if not ligand_pn_unit_iids:
        raise ValueError(f"No ligand chains found in {cif_path}")

    # Separate protein and ligand
    protein_mask = np.isin(atom_array.pn_unit_iid, receptor_pn_unit_iids)
    ligand_mask = np.isin(atom_array.pn_unit_iid, ligand_pn_unit_iids)

    protein_array = atom_array[protein_mask]
    ligand_array = atom_array[ligand_mask]

    # Write protein to PDB for PrepWizard
    protein_pdb_path = str(out_dir / f"{sample_id}_protein.pdb")
    pdb_string = to_pdb_string(protein_array)
    with open(protein_pdb_path, "w") as f:
        f.write(pdb_string)
    logger.info(f"Wrote protein PDB: {protein_pdb_path} ({len(protein_array)} atoms)")

    # Write ligand to SDF via RDKit
    ligand_sdf_path = str(out_dir / f"{sample_id}_ligand.sdf")
    write_ligand_sdf(ligand_array, ligand_sdf_path)
    logger.info(f"Wrote ligand SDF: {ligand_sdf_path} ({len(ligand_array)} atoms)")

    # Compute ligand centroid (heavy atoms only)
    ligand_centroid = compute_ligand_centroid(ligand_array)

    return {
        "sample_id": sample_id,
        "cif_path": cif_path,
        "protein_pdb_path": protein_pdb_path,
        "ligand_sdf_path": ligand_sdf_path,
        "ligand_centroid": ligand_centroid,
        "atom_array": atom_array,
        "protein_atom_array": protein_array,
        "ligand_atom_array": ligand_array,
        "receptor_pn_unit_iids": receptor_pn_unit_iids,
        "ligand_pn_unit_iids": ligand_pn_unit_iids,
    }


def get_protein_pn_unit_iids(atom_array: AtomArray) -> list[str]:
    """Get pn_unit_iids for all polypeptide chains."""
    prot_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    return sorted(set(atom_array[prot_mask].pn_unit_iid))


def _is_single_metal_ion(atom_array: AtomArray, pn_unit_iid: str) -> bool:
    """Check if a pn_unit is a single metal ion (1 atom, metal element)."""
    mask = atom_array.pn_unit_iid == pn_unit_iid
    sub = atom_array[mask]
    return len(sub) == 1 and sub.element[0].upper() in METAL_ELEMENTS


def get_ligand_pn_unit_iids(atom_array: AtomArray) -> list[str]:
    """Get pn_unit_iids for non-polymer chains, excluding single metal ions."""
    non_polymer_types = list(aw_enums.ChainTypeInfo.NON_POLYMERS)
    lig_mask = np.isin(atom_array.chain_type, non_polymer_types)
    candidates = sorted(set(atom_array[lig_mask].pn_unit_iid))
    return [pid for pid in candidates if not _is_single_metal_ion(atom_array, pid)]


def compute_dynamic_outerbox(
    ligand_array: AtomArray, padding: float = 20.0,
) -> list[float]:
    """Compute OUTERBOX from ligand coordinate range + padding per axis.

    Per PoseX protocol: (x_range + padding, y_range + padding, z_range + padding).
    """
    heavy_mask = ligand_array.element != "H"
    coords = ligand_array[heavy_mask].coord if heavy_mask.any() else ligand_array.coord
    ranges = coords.max(axis=0) - coords.min(axis=0)
    return (ranges + padding).tolist()


def compute_ligand_centroid(ligand_array: AtomArray) -> np.ndarray:
    """Compute centroid of ligand heavy atoms."""
    heavy_mask = ligand_array.element != "H"
    if not heavy_mask.any():
        return ligand_array.coord.mean(axis=0)
    return ligand_array[heavy_mask].coord.mean(axis=0)


def write_ligand_sdf(ligand_array: AtomArray, sdf_path: str) -> None:
    """Convert ligand AtomArray to SDF via RDKit."""
    try:
        mol = atom_array_to_rdkit(ligand_array, sanitize=True)
    except Exception:
        logger.warning("Ligand sanitization failed, retrying without sanitization")
        mol = atom_array_to_rdkit(ligand_array, sanitize=False)

    if mol is None:
        raise ValueError("Failed to convert ligand to RDKit molecule")

    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
