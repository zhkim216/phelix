"""CCD -> heavy atom count cache, derived from the SMILES cache.

Heavy atom counts are read by the BMSM augmentation step to compute the
``q_pn_unit_resolution_ratio`` (``num_resolved_atoms`` / ``expected_heavy``)
column. We resolve them through RDKit on the cached SMILES rather than going
back to atomworks: the atomworks path occasionally aborts at the C level on
malformed CCDs, while ``Chem.MolFromSmiles(..., sanitize=False)`` always
returns cleanly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm


logger = logging.getLogger(__name__)


def heavy_atoms_from_smiles(smiles: str | None) -> int | None:
    """Count heavy atoms via RDKit (sanitize=False so malformed CCDs still parse)."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


def build_heavy_atom_cache(
    ccd_codes: set[str],
    smiles_map: dict[str, str],
    cache_path: Path,
) -> dict[str, int]:
    """Resolve heavy atom counts for every CCD in ``ccd_codes``, persisting to disk.

    Existing cache entries are preserved; missing codes are computed from
    ``smiles_map`` (codes whose SMILES is missing or unparseable are simply
    skipped — downstream reads must tolerate missing keys). The cache file is
    written once at the end of resolution.
    """
    cache_path = Path(cache_path)
    existing: dict[str, int] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        for code, value in raw.items():
            if value is None:
                continue
            existing[code] = int(value)
        logger.info("Loaded %d entries from cache %s", len(existing), cache_path)

    missing = sorted(code for code in ccd_codes if code and code not in existing)
    if not missing:
        return existing

    logger.info(
        "Computing heavy atom counts for %d new CCD codes from SMILES",
        len(missing),
    )
    resolved = 0
    for code in tqdm(missing, desc="Counting heavy atoms"):
        count = heavy_atoms_from_smiles(smiles_map.get(code))
        if count is not None:
            existing[code] = count
            resolved += 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(existing, f)
    logger.info(
        "Heavy atom cache updated: +%d resolved (%d still missing) -> %s",
        resolved,
        len(missing) - resolved,
        cache_path,
    )
    return existing
