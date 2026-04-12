"""PoseBusters evaluation for AF3 predicted ligand poses.

Canonical PoseBusters library. Consumers:
``allatom_design.eval.glide.pipeline`` (integrated Glide + PB pipeline) and
``allatom_design.eval.posebusters.run_pb_eval`` (standalone PB CLI).

Modular functions for:
- Preparing PB inputs from CIF files (reuses glide/preprocessing.py)
- Running PB validity checks (single or batch)
- Computing pb_valid summary metric
- Discovering AF3 prediction CIF files
- Batch evaluation with multiprocessing

Requires posebusters >= 0.6.0.

Note:
    Applies a monkey-patch to ``posebusters.modules.flatness`` to work
    around an rdkit pip-wheel bug where ``GetSubstructMatches`` with
    aromatic-hybridization SMARTS corrupts the C++ Conformer object,
    causing a segfault on subsequent ``GetAtomPosition`` calls.
    The fix pre-extracts all 3D coordinates to a numpy array via
    ``GetPositions()`` *before* calling ``GetSubstructMatches``.
"""

import logging
import math
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from posebusters import PoseBusters

from allatom_design.eval.glide.preprocessing import preprocess_structure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monkey-patch: fix rdkit pip-wheel segfault in PoseBusters flatness check.
# See debug/260401_glide_pb_debug/ for reproduction scripts.
# ---------------------------------------------------------------------------
__patched = False


def _patch_posebusters_flatness():
    """Replace ``check_flatness`` to pre-extract coords before SMARTS match.

    The rdkit 2026.3.1 (and 2025.9.6) pip wheel has a bug where
    ``GetSubstructMatches`` with ``[ar5^2]``-style SMARTS corrupts the
    internal C++ Conformer object. Any subsequent ``GetAtomPosition()``
    call reads garbage memory and segfaults. The workaround is to copy
    all coordinates to a numpy array via ``GetPositions()`` *before*
    calling ``GetSubstructMatches``, then use numpy indexing.
    """
    global __patched
    if __patched:
        return
    try:
        import posebusters.modules.flatness as _flat
        from copy import deepcopy

        import numpy as np
        from rdkit.Chem.rdmolfiles import MolFromSmarts
        from rdkit.Chem.rdmolops import SanitizeMol

        _orig_empty = _flat._empty_results

        def _check_flatness_safe(
            mol_pred,
            threshold_flatness=0.1,
            flat_systems=_flat.flat,
            check_nonflat=False,
        ):
            mol = deepcopy(mol_pred)
            try:
                assert mol_pred.GetNumConformers() > 0, (
                    "Molecule does not have a conformer."
                )
                # Pre-extract ALL coords BEFORE SanitizeMol/SMARTS (workaround).
                all_coords = mol.GetConformer().GetPositions()
                SanitizeMol(mol)
            except Exception:
                return _orig_empty

            planar_groups = []
            types = []
            for flat_system, smarts in flat_systems.items():
                match = MolFromSmarts(smarts)
                atom_groups = list(mol.GetSubstructMatches(match))
                planar_groups += atom_groups
                types += [flat_system] * len(atom_groups)

            # Use pre-extracted numpy coords instead of GetAtomPosition.
            coords = [all_coords[list(g)] for g in planar_groups]
            max_distances = [
                float(_flat._get_distances_to_plane(X).max()) for X in coords
            ]
            if not check_nonflat:
                flatness_passes = [
                    bool(d <= threshold_flatness) for d in max_distances
                ]
                extreme_distance = (
                    max(max_distances) if max_distances else np.nan
                )
            else:
                flatness_passes = [
                    bool(d >= threshold_flatness) for d in max_distances
                ]
                extreme_distance = (
                    min(max_distances) if max_distances else np.nan
                )
            details = {
                "type": types,
                "planar_group": planar_groups,
                "max_distance": max_distances,
                "flatness_passes": flatness_passes,
            }
            results = {
                "num_systems_checked": len(planar_groups),
                "num_systems_passed": sum(flatness_passes),
                "max_distance": extreme_distance,
                "flatness_passes": (
                    all(flatness_passes) if len(flatness_passes) > 0 else True
                ),
            }
            return {"results": results, "details": details}

        # Patch ALL references — PoseBusters captures the function at import
        # time via `from .modules.flatness import check_flatness`, so
        # replacing the module attribute alone is not enough.
        _flat.check_flatness = _check_flatness_safe

        import posebusters.posebusters as _pb
        _pb.check_flatness = _check_flatness_safe
        if "flatness" in _pb.module_dict:
            _pb.module_dict["flatness"] = _check_flatness_safe

        import posebusters as _pb_init
        if hasattr(_pb_init, "check_flatness"):
            _pb_init.check_flatness = _check_flatness_safe

        __patched = True
        logger.debug("Patched posebusters flatness (module + module_dict + __init__)")
    except Exception as e:
        logger.warning(f"Failed to patch PoseBusters flatness: {e}")


_patch_posebusters_flatness()

# PB 0.6.0 validity test columns (full_report=False names).
# Excludes loading columns. Used by add_pb_valid() for the summary.
PB_VALIDITY_COLUMNS = [
    # Chemical validity
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    # Intramolecular validity
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    # Intermolecular validity
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
]

# Loading columns tracked separately for diagnostics.
PB_LOADING_COLUMNS = [
    "mol_pred_loaded",
    "mol_cond_loaded",
]


# ---------------------------------------------------------------------------
# 1. Input preparation (reuses glide/preprocessing.py)
# ---------------------------------------------------------------------------

def prepare_pb_inputs(
    cif_path: str,
    out_dir: str,
    sample_id: str | None = None,
    cif_parse_cfg: dict | None = None,
) -> dict[str, str]:
    """Convert an AF3 CIF to ligand SDF + receptor PDB for PoseBusters.

    Reuses ``preprocess_structure()`` from glide/preprocessing.py so that
    the CIF -> PDB/SDF conversion is shared across Glide and PB pipelines.

    Returns:
        Dict with keys ``mol_pred``, ``mol_cond``, ``sample_id``.
    """
    result = preprocess_structure(
        cif_path=cif_path,
        out_dir=out_dir,
        sample_id=sample_id,
        cif_parse_cfg=cif_parse_cfg,
    )
    return {
        "mol_pred": result["ligand_sdf_path"],
        "mol_cond": result["protein_pdb_path"],
        "sample_id": result["sample_id"],
    }


# ---------------------------------------------------------------------------
# 2. PoseBusters execution
# ---------------------------------------------------------------------------

def run_pb_single(
    mol_pred: str,
    mol_cond: str,
    mol_true: str | None = None,
    config: str = "dock",
    full_report: bool = False,
) -> pd.DataFrame:
    """Run PoseBusters bust() on one ligand-protein pair.

    Args:
        mol_pred: Predicted ligand SDF path.
        mol_cond: Receptor PDB path.
        mol_true: Reference ligand SDF (only for ``config="redock"``).
        config: ``"dock"`` (no reference) or ``"redock"``.
        full_report: If True, include detailed per-subtest columns.
            False (default) returns boolean pass/fail summary columns
            that match :data:`PB_VALIDITY_COLUMNS`.

    Returns:
        Single-row DataFrame of PB test results.
    """
    pb = PoseBusters(config=config)
    return pb.bust(
        mol_pred=mol_pred,
        mol_true=mol_true,
        mol_cond=mol_cond,
        full_report=full_report,
    )


def run_pb_batch(
    entries: pd.DataFrame,
    config: str = "dock",
    max_workers: int = 0,
    full_report: bool = False,
) -> pd.DataFrame:
    """Run PoseBusters bust_table() on a batch of entries.

    Args:
        entries: DataFrame with columns ``mol_pred``, ``mol_cond``,
                 and optionally ``mol_true``.
        config: ``"dock"`` or ``"redock"``.
        max_workers: PB-internal parallel workers.
        full_report: If True, include per-subtest columns.

    Returns:
        Combined DataFrame of PB results for all entries.
    """
    pb = PoseBusters(config=config, max_workers=max_workers)
    return pb.bust_table(entries, full_report=full_report)


# ---------------------------------------------------------------------------
# 3. Summary metric
# ---------------------------------------------------------------------------

def add_pb_valid(
    bust_results: pd.DataFrame,
    validity_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add a ``pb_valid`` column: True iff all validity checks pass.

    By default uses :data:`PB_VALIDITY_COLUMNS`, skipping any that are
    absent (e.g. cofactor/water columns when no cofactors exist).
    """
    if validity_columns is None:
        validity_columns = [
            c for c in PB_VALIDITY_COLUMNS if c in bust_results.columns
        ]

    if not validity_columns:
        logger.warning("No validity columns found in bust results")
        bust_results["pb_valid"] = pd.NA
        return bust_results

    bust_results = bust_results.copy()
    valid_df = bust_results[validity_columns]
    # Rows where all present checks pass AND at least one check is present.
    all_pass = valid_df.fillna(False).all(axis=1)
    has_any = valid_df.notna().any(axis=1)
    bust_results["pb_valid"] = all_pass & has_any
    return bust_results


# ---------------------------------------------------------------------------
# 4. CIF discovery
# ---------------------------------------------------------------------------

def discover_af3_cif_paths(
    af3_pred_dir: str,
    cif_pattern: str = "*_model_pocket_aligned.cif",
) -> list[dict[str, str]]:
    """Find AF3 prediction CIF files with parsed metadata.

    Expected directory layout::

        af3_pred_dir/
          {designed_sample_id}/
            seed-{seed}_sample-{n}/
              {name}_{cif_pattern}

    Returns:
        Sorted list of dicts with ``cif_path``, ``designed_sample_id``,
        ``diffusion_id``.
    """
    af3_dir = Path(af3_pred_dir)
    entries = []

    for cif_path in sorted(af3_dir.rglob(cif_pattern)):
        relative = cif_path.relative_to(af3_dir)
        parts = relative.parts

        if len(parts) >= 3:
            designed_sample_id = parts[0]
            diffusion_id = parts[1]  # e.g. "seed-42_sample-0"
        else:
            designed_sample_id = cif_path.stem
            diffusion_id = "unknown"

        entries.append({
            "cif_path": str(cif_path),
            "designed_sample_id": designed_sample_id,
            "diffusion_id": diffusion_id,
        })

    return entries


# ---------------------------------------------------------------------------
# 5. Per-entry evaluation (picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _evaluate_single_entry(
    entry: dict,
    out_dir: str,
    config: str = "dock",
    cif_parse_cfg: dict | None = None,
    full_report: bool = False,
) -> dict:
    """Evaluate one CIF through the full PB pipeline.

    Designed to be called via ``multiprocessing.Pool.map`` (all args are
    plain dicts/strings so they are picklable).
    """
    cif_path = entry["cif_path"]
    sample_id = f"{entry['designed_sample_id']}_{entry['diffusion_id']}"

    try:
        work_dir = str(Path(out_dir) / sample_id)
        pb_inputs = prepare_pb_inputs(
            cif_path=cif_path,
            out_dir=work_dir,
            sample_id=sample_id,
            cif_parse_cfg=cif_parse_cfg,
        )

        result_df = run_pb_single(
            mol_pred=pb_inputs["mol_pred"],
            mol_cond=pb_inputs["mol_cond"],
            config=config,
            full_report=full_report,
        )

        if len(result_df) > 0:
            result_dict = result_df.iloc[0].to_dict()
        else:
            result_dict = {"error": "empty_result"}

    except Exception as e:
        logger.error(f"PB evaluation failed for {cif_path}: {e}")
        result_dict = {"error": str(e)}

    # Always attach metadata so the caller can join back.
    result_dict["designed_sample_id"] = entry["designed_sample_id"]
    result_dict["diffusion_id"] = entry["diffusion_id"]
    result_dict["cif_path"] = cif_path
    return result_dict


# ---------------------------------------------------------------------------
# 6. Batch evaluation with multiprocessing + array-job splitting
# ---------------------------------------------------------------------------

def split_entries_for_array_job(
    entries: list[dict],
    array_id: int | None = None,
    num_arrays: int | None = None,
) -> list[dict]:
    """Slice *entries* for a SLURM array task.

    If *array_id* is None (no array job), returns all entries unchanged.
    Falls back to ``$SLURM_ARRAY_TASK_ID`` / ``$SLURM_ARRAY_TASK_COUNT``
    when the explicit arguments are None but the env vars are set.
    """
    if array_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is not None:
            array_id = int(env_id)
            num_arrays = int(
                os.environ.get("SLURM_ARRAY_TASK_COUNT", num_arrays or 1)
            )

    if array_id is None:
        return entries

    num_arrays = num_arrays or 1
    chunk_size = math.ceil(len(entries) / num_arrays)
    start = array_id * chunk_size
    end = min(start + chunk_size, len(entries))
    chunk = entries[start:end]
    logger.info(
        f"Array {array_id}/{num_arrays}: processing entries [{start}:{end}] "
        f"({len(chunk)} of {len(entries)} total)"
    )
    return chunk


def evaluate_batch(
    entries: list[dict],
    out_dir: str,
    config: str = "dock",
    cif_parse_cfg: dict | None = None,
    num_workers: int = 1,
    full_report: bool = False,
) -> pd.DataFrame:
    """Run PB evaluation on a list of CIF entries.

    Args:
        entries: List of dicts from :func:`discover_af3_cif_paths`.
        out_dir: Working directory for intermediate PDB/SDF files.
        config: ``"dock"`` or ``"redock"``.
        cif_parse_cfg: Passed to the CIF parser (plain dict, not DictConfig).
        num_workers: Parallel workers.  1 = sequential.
        full_report: False (default) for boolean summary columns,
            True for detailed per-subtest metrics.

    Returns:
        DataFrame with one row per entry, including ``pb_valid``.
    """
    if not entries:
        logger.warning("No entries to evaluate")
        return pd.DataFrame()

    eval_fn = partial(
        _evaluate_single_entry,
        out_dir=out_dir,
        config=config,
        cif_parse_cfg=cif_parse_cfg,
        full_report=full_report,
    )

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(pool.map(eval_fn, entries))
    else:
        results = [eval_fn(e) for e in entries]

    df = pd.DataFrame(results)
    df = add_pb_valid(df)
    return df
