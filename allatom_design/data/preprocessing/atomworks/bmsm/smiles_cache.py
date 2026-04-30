"""CCD -> SMILES cache helpers.

Two backends are stacked behind the same JSON cache:

1. **RCSB REST** (``fetch_smiles_from_rcsb`` / ``fetch_all_smiles``) — primary
   source. Uses the uppercase ``SMILES_stereo`` / ``SMILES`` fields of
   ``rcsb_chem_comp_descriptor`` (the lowercase spelling used by an older
   helper returns ``None`` for every code).
2. **atomworks fallback** (``run_atomworks_smiles_fallback``) — best-effort
   recovery for codes the RCSB API returned ``None`` for. atomworks parses
   the CCD CIF directly via RDKit; the fallback flushes its progress to disk
   every ``CACHE_FLUSH_EVERY`` codes so that a C-level abort inside RDKit
   never wipes a long fetch.

Both ``fetch_all_smiles`` and ``run_atomworks_smiles_fallback`` persist the
cache they receive — callers can treat them as "fetch and store" primitives
and never need to write the JSON themselves.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from rdkit import Chem
from tqdm import tqdm

from atomworks.io.tools.rdkit import ccd_code_to_rdkit


# atomworks RDKit calls occasionally die at the C level; flushing the cache
# every N codes bounds how much fetch work a single crash can erase.
CACHE_FLUSH_EVERY = 100


def fetch_smiles_from_rcsb(ccd_code: str) -> str | None:
    """Fetch a SMILES string from the RCSB Data API for ``ccd_code``.

    Prefers ``SMILES_stereo`` (with stereochemistry) over ``SMILES``. Returns
    ``None`` on any HTTP error, missing field, or network failure — callers
    are expected to fall back via :func:`run_atomworks_smiles_fallback`.
    """
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_code}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        desc = data.get("rcsb_chem_comp_descriptor", {})
        return desc.get("SMILES_stereo") or desc.get("SMILES")
    except Exception:
        return None


def generate_smiles_from_ccd(ccd_code: str) -> str | None:
    """Generate a SMILES via atomworks' CCD -> RDKit conversion.

    Used only as a fallback for CCDs the RCSB API has no SMILES for; returns
    ``None`` on any RDKit/atomworks failure.
    """
    try:
        mol = ccd_code_to_rdkit(
            ccd_code, fix_stereochemistry=False, return_atom_array=False
        )
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_smiles_cache(cache_path: Path) -> dict[str, str | None]:
    """Load the SMILES cache JSON, returning an empty dict if the file is missing."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_smiles_cache(cache: dict[str, str | None], cache_path: Path) -> None:
    """Persist the SMILES cache to ``cache_path`` (overwrites)."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def fetch_all_smiles(
    ccd_codes: list[str],
    cache_path: Path,
    num_workers: int = 32,
) -> dict[str, str | None]:
    """Fill ``cache_path`` with SMILES for every code in ``ccd_codes``.

    Codes already present in the cache (any value, including ``None``) are
    skipped; only the codes the cache has never seen are fetched. The cache
    is saved once at the end of the fetch.
    """
    cache_path = Path(cache_path)
    cache = load_smiles_cache(cache_path)

    uncached_codes = [c for c in ccd_codes if c not in cache]
    print(f"  - Cached: {len(ccd_codes) - len(uncached_codes)}")
    print(f"  - To fetch: {len(uncached_codes)}")

    if not uncached_codes:
        return cache

    def fetch_single(ccd_code: str) -> tuple[str, str | None]:
        return ccd_code, fetch_smiles_from_rcsb(ccd_code)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(fetch_single, uncached_codes),
                total=len(uncached_codes),
                desc="Fetching SMILES from RCSB",
            )
        )

    for ccd_code, smiles in results:
        cache[ccd_code] = smiles  # may be None if RCSB had no entry

    save_smiles_cache(cache, cache_path)
    print(f"  - Cache updated and saved to {cache_path}")
    return cache


def run_atomworks_smiles_fallback(
    ccd_codes: list[str],
    smiles_cache: dict[str, str | None],
    cache_path: Path,
    flush_every: int = CACHE_FLUSH_EVERY,
) -> dict[str, str | None]:
    """Fill SMILES entries that are ``None`` via atomworks' CCD parser.

    Designed to survive C-level crashes in RDKit by flushing ``smiles_cache``
    to ``cache_path`` every ``flush_every`` codes. Codes whose cache value is
    already non-``None`` are skipped. The dict is mutated in place and also
    returned for convenience.
    """
    cache_path = Path(cache_path)
    missing = [c for c in ccd_codes if smiles_cache.get(c) is None]
    if not missing:
        return smiles_cache

    for i, ccd_code in enumerate(tqdm(missing, desc="atomworks fallback"), start=1):
        smiles_cache[ccd_code] = generate_smiles_from_ccd(ccd_code)
        if i % flush_every == 0:
            save_smiles_cache(smiles_cache, cache_path)
    save_smiles_cache(smiles_cache, cache_path)
    return smiles_cache
