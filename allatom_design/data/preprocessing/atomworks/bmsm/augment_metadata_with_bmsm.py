#!/usr/bin/env python3
"""
Augment metadata parquet with BMSM (biologically meaningful small molecule) columns.

Heavy atom counts are resolved from an RDKit-parseable SMILES cache so
malformed CCDs do not trigger uncatchable C-level aborts; the upstream
``q_pn_unit_is_protein`` column (produced by ``cluster_sequences.py``) is
consumed directly. Missing SMILES are auto-fetched from the RCSB REST API
(with an atomworks fallback) via :mod:`.smiles_cache`, and heavy atom counts
are derived from those SMILES via :mod:`.heavy_atom_cache`. Both caches are
persisted alongside the dataset (``ccd_smiles_cache_metadata_v9.json`` and
``ccd_heavy_atom_counts_v9.json``).

The ``has_filtered_ccd`` whitelist (``passed_ccd_codes_metadata_v9.txt``) is
produced by :mod:`.ccd_filter` (Table A3 plinder non-artifact filter); this
module only consumes the resulting whitelist.

BMSM formula (no resolution_ratio gating):
    BMSM = (~is_polymer) & has_filtered_ccd & has_protein_contact_within(<= cutoff Å)

Added columns (per pn_unit row):
  - ``q_pn_unit_is_small_molecule`` (bool) — ``~polymer & ~metal & ~halide``.
    Stored eagerly so downstream filters can rely on a single canonical
    definition.
  - ``q_pn_unit_has_filtered_ccd`` (bool)
  - ``q_pn_unit_expected_heavy_atoms_non_polymer`` (int or NaN)
  - ``q_pn_unit_resolution_ratio`` (float, NaN for polymers) — kept for
    downstream analysis even though BMSM no longer ANDs against it.
  - ``q_pn_unit_is_biologically_meaningful_small_molecule`` (bool)
  - ``q_pn_unit_is_maybe_covalently_linked_to_protein`` (bool) — evaluated on
    every non-polymer row so the PMSM filter in ``atomworks_sd_dataset.py``
    (``pmsm_keep = ... | (is_sm & is_cov) | (is_sm & ~is_cov & heavy_pass)``)
    can trust this flag across the full small-molecule space.
  - ``num_contacting_protein_chains`` (int32) — distinct protein pn_units
    contacted within ``protein_contact_cutoff`` Å. Mirrors the runtime helper
    in ``atomworks_sd_dataset._add_num_contacting_protein_chains`` so
    precomputing is an idempotent overwrite for the loader, but lets offline
    analysis read the column directly from the parquet.
  - ``q_pn_unit_per_partner_contacts_to_protein_small_molecule`` (JSON str
    or NaN) — the upstream ``q_pn_unit_per_partner_contacts_small_molecule``
    partner list filtered to protein partners only (partners whose
    ``pn_unit_iid`` is a protein in the same ``(pdb_id, assembly_id)``).
    Null when the source is null (i.e. the row is not a small-molecule
    query); ``"[]"`` when the source is non-empty but contains no protein
    partner — preserves the "small-molecule query, no protein contact"
    semantic distinct from "non-small-molecule query".
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from allatom_design.data.preprocessing.atomworks.bmsm.heavy_atom_cache import (
    build_heavy_atom_cache,
)
from allatom_design.data.preprocessing.atomworks.bmsm.smiles_cache import (
    fetch_all_smiles,
    load_smiles_cache,
    run_atomworks_smiles_fallback,
)


logger = logging.getLogger(__name__)


V9_DATASET_DIR = "/home/possu/jinho/datasets/atomworks_pdb_full_v9"

DEFAULT_INPUT = f"{V9_DATASET_DIR}/metadata_ligval_seq_clustered_04.parquet"
DEFAULT_OUTPUT = f"{V9_DATASET_DIR}/metadata_ligval_seq_clustered_04_bmsm.parquet"
DEFAULT_SMILES_CACHE = f"{V9_DATASET_DIR}/ccd_smiles_cache_metadata_v9.json"
DEFAULT_HEAVY_ATOM_CACHE = f"{V9_DATASET_DIR}/ccd_heavy_atom_counts_v9.json"
DEFAULT_PASSED_CCDS = f"{V9_DATASET_DIR}/passed_ccd_codes_metadata_v9.txt"

MIN_ALLOWED_DISTANCE = 2.4
DISTANCE_CUTOFF = 5.0


def _parse_ccd_codes(res_names: object) -> list[str]:
    if res_names is None:
        return []
    if isinstance(res_names, float) and pd.isna(res_names):
        return []
    text = str(res_names)
    if not text:
        return []
    return [code.strip() for code in text.split(",") if code.strip()]


def _parse_contacts(raw: object) -> list[dict]:
    if raw is None:
        return []
    # pandas may already have decoded JSON-typed columns into Python lists.
    if isinstance(raw, list):
        return raw
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if not isinstance(raw, str) or not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _expected_heavy_atoms_composite(
    codes: list[str],
    smiles_map: dict[str, str],
    heavy_atom_counts: dict[str, int],
) -> int | None:
    """Heavy atom count for a (possibly multi-residue) CCD tuple.

    Multi-residue expectations go through an RDKit canonicalization of the
    joined SMILES so that bond-count artefacts (shared atoms, salt codes) are
    resolved consistently; single-residue codes fall back to the cached
    per-code counts.
    """
    if not codes:
        return None
    if len(codes) == 1:
        return heavy_atom_counts.get(codes[0])

    parts = [smiles_map.get(c) for c in codes]
    if any(p is None for p in parts):
        return None
    joined = ".".join(parts)
    mol = Chem.MolFromSmiles(joined)
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


def compute_has_filtered_ccd(res_names: object, passed_set: set[str]) -> bool:
    """Any constituent CCD passing the upstream curation qualifies the row.

    Mirrors v3 step2: a multi-residue pn_unit is accepted as long as at least
    one of its CCDs survived the filter — the stricter ``all()`` check dropped
    too many biologically valid compositions.
    """
    codes = _parse_ccd_codes(res_names)
    if not codes:
        return False
    return bool(set(codes) & passed_set)


def analyze_protein_contacts(
    row: pd.Series,
    protein_iid_set: set[str],
    contact_cutoff: float,
    covalent_threshold: float,
) -> tuple[bool, bool]:
    """Return ``(has_protein_contact_within_cutoff, is_maybe_covalently_linked)``.

    A row stays BMSM only if it touches a protein within ``contact_cutoff`` (Å);
    it is flagged "maybe covalent" if any such contact sits at or below
    ``covalent_threshold`` (matches v3 semantics, ``<=``).
    """
    contacts = _parse_contacts(row.get("q_pn_unit_contacting_pn_unit_iids"))
    if not contacts:
        return (False, False)

    has_valid_contact = False
    is_maybe_covalent = False

    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        distance = contact.get("min_distance")
        if distance is None:
            continue
        iid = contact.get("pn_unit_iid")
        if iid is None or iid not in protein_iid_set:
            continue
        dist_f = float(distance)
        if dist_f <= contact_cutoff:
            has_valid_contact = True
            if dist_f <= covalent_threshold:
                is_maybe_covalent = True
                break

    return (has_valid_contact, is_maybe_covalent)


def _build_protein_iid_sets(df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Group protein pn_unit iids by (pdb_id, assembly_id) for O(1) per-row lookup.

    Uses the upstream ``q_pn_unit_is_protein`` column verbatim; a tuple key
    keeps it safe against pdb/assembly ids that contain underscores.
    """
    is_protein = df["q_pn_unit_is_protein"].fillna(False).astype(bool)
    protein_rows = df[is_protein.values]
    result: dict[tuple[str, str], set[str]] = {}
    for (pdb_id, assembly_id), group in protein_rows.groupby(
        ["pdb_id", "assembly_id"], sort=False
    ):
        key = (str(pdb_id), str(assembly_id))
        result[key] = set(group["q_pn_unit_iid"].astype(str).tolist())
    return result


def _count_protein_contacts_in_row(
    row: pd.Series,
    is_protein_lookup: dict[tuple[str, str, str], bool],
    distance_cutoff: float,
) -> int:
    """Count distinct protein pn_units contacted by ``row`` within ``distance_cutoff`` Å.    
    """
    contacts = _parse_contacts(row.get("q_pn_unit_contacting_pn_unit_iids"))
    if not contacts:
        return 0
    pdb_id = row.get("pdb_id")
    assembly_id = str(row.get("assembly_id"))
    n_protein = 0
    for c in contacts:
        if not isinstance(c, dict):
            continue
        cid = c.get("pn_unit_iid")
        md = c.get("min_distance")
        if cid is None or md is None:
            continue
        if float(md) > distance_cutoff:
            continue
        if is_protein_lookup.get((pdb_id, assembly_id, cid), False):
            n_protein += 1
    return n_protein


def add_num_contacting_protein_chains(
    df: pd.DataFrame,
    distance_cutoff: float = DISTANCE_CUTOFF,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``num_contacting_protein_chains`` (int32) appended.

    Counts, per pn_unit row, the distinct contacted pn_units that are protein
    and sit within ``distance_cutoff`` Å. Applied to every row (not just
    non-polymers) — within a pdb_id all rows share resolution, so the loader
    and downstream analysis can apply their own filters without recomputing.
    """
    keys = list(
        zip(
            df["pdb_id"],
            df["assembly_id"].astype(str),
            df["q_pn_unit_iid"],
        )
    )
    is_protein_lookup = dict(
        zip(keys, df["q_pn_unit_is_protein"].fillna(False).astype(bool))
    )
    tqdm.pandas(desc="num_contacting_protein_chains")
    counts = df.progress_apply(
        lambda row: _count_protein_contacts_in_row(
            row, is_protein_lookup, distance_cutoff
        ),
        axis=1,
    )
    out = df.copy()
    out["num_contacting_protein_chains"] = counts.astype("int32")
    return out


def add_per_partner_contacts_to_protein_small_molecule(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy of ``df`` with
    ``q_pn_unit_per_partner_contacts_to_protein_small_molecule`` (JSON str) appended.

    Filters ``q_pn_unit_per_partner_contacts_small_molecule`` to the partner
    entries whose ``pn_unit_iid`` is a protein in the same
    ``(pdb_id, assembly_id)``. Source-null rows (non-small-molecule queries)
    stay null; rows that are small-molecule queries but contact no protein
    become ``"[]"`` so downstream code can distinguish "non-small-molecule
    query" from "small-molecule query with no protein partner".
    """
    protein_iid_lookup = _build_protein_iid_sets(df)

    raw_col = df["q_pn_unit_per_partner_contacts_small_molecule"]
    pdb_ids = df["pdb_id"].astype(str).tolist()
    assembly_ids = df["assembly_id"].astype(str).tolist()

    out_values: list[object] = []
    for raw, pdb_id, assembly_id in tqdm(
        zip(raw_col, pdb_ids, assembly_ids),
        total=len(df),
        desc="per_partner_contacts_to_protein_small_molecule",
    ):
        # Source null ⇒ preserve null (non-SM query rows).
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            out_values.append(None)
            continue
        entries = _parse_contacts(raw)
        protein_set = protein_iid_lookup.get((pdb_id, assembly_id), set())
        filtered = [
            e
            for e in entries
            if isinstance(e, dict) and e.get("pn_unit_iid") in protein_set
        ]
        out_values.append(json.dumps(filtered))

    out = df.copy()
    out["q_pn_unit_per_partner_contacts_to_protein_small_molecule"] = out_values
    return out


def augment_metadata_with_bmsm(
    input_df: pd.DataFrame,
    passed_ccd_codes: set[str],
    smiles_map: dict[str, str],
    heavy_atom_counts: dict[str, int],
    covalent_distance_threshold: float = MIN_ALLOWED_DISTANCE,
    protein_contact_cutoff: float = DISTANCE_CUTOFF,
) -> pd.DataFrame:
    """Add BMSM / cov / num_contacting columns to ``input_df`` and return the augmented copy.

    BMSM formula:
        BMSM = (~is_polymer)
             & has_filtered_ccd
             & has_protein_contact_within(<= protein_contact_cutoff Å)

    ``q_pn_unit_resolution_ratio`` is still computed and stored so downstream
    analysis can use it, but BMSM no longer ANDs against it (callers wanted
    cov/contact bookkeeping decoupled from completeness gating).

    ``q_pn_unit_is_small_molecule`` (= ``~polymer & ~metal & ~halide``) is
    written eagerly so downstream filters can rely on one canonical small-
    molecule definition that excludes both metals and halides.

    ``is_maybe_covalently_linked_to_protein`` and ``num_contacting_protein_chains``
    are evaluated for every non-polymer / every row respectively, so downstream
    consumers (PMSM filter in ``atomworks_sd_dataset.py``, offline analyses)
    can trust the flags across the full SM space.

    The dataset loader in ``atomworks_sd_dataset.py`` also recomputes
    ``num_contacting_protein_chains`` at runtime with the same
    ``protein_contact_cutoff``, so this precomputation is an idempotent
    overwrite — value-identical but available offline.
    """
    df = input_df.copy()

    df["q_pn_unit_has_filtered_ccd"] = df["q_pn_unit_non_polymer_res_names"].apply(
        lambda x: compute_has_filtered_ccd(x, passed_ccd_codes)
    )

    is_polymer = df["q_pn_unit_is_polymer"].fillna(False).astype(bool)
    is_metal = df["q_pn_unit_is_metal"].fillna(False).astype(bool)
    is_halide = df["q_pn_unit_is_halide"].fillna(False).astype(bool)
    df["q_pn_unit_is_small_molecule"] = (~is_polymer) & (~is_metal) & (~is_halide)

    def _expected(res_names: object, polymer: bool) -> float:
        if polymer:
            return float("nan")
        codes = _parse_ccd_codes(res_names)
        value = _expected_heavy_atoms_composite(codes, smiles_map, heavy_atom_counts)
        return float("nan") if value is None else float(value)

    df["q_pn_unit_expected_heavy_atoms_non_polymer"] = [
        _expected(res_names, polymer)
        for res_names, polymer in zip(
            df["q_pn_unit_non_polymer_res_names"].tolist(),
            is_polymer.tolist(),
        )
    ]

    resolved = df["q_pn_unit_num_resolved_atoms"].astype(float)
    expected = df["q_pn_unit_expected_heavy_atoms_non_polymer"].astype(float)
    # expected == 0 would produce inf; force NaN so the ratio stays well-defined.
    safe_expected = expected.where(expected > 0, other=float("nan"))
    df["q_pn_unit_resolution_ratio"] = resolved / safe_expected

    is_non_polymer = ~is_polymer
    bmsm_prefilter = is_non_polymer & df["q_pn_unit_has_filtered_ccd"].astype(bool)
    logger.info(
        "BMSM prefilter (~polymer & has_filtered_ccd): %d rows",
        int(bmsm_prefilter.sum()),
    )

    protein_iid_lookup = _build_protein_iid_sets(df)

    def _analyze(row: pd.Series) -> tuple[bool, bool]:
        key = (str(row.get("pdb_id")), str(row.get("assembly_id")))
        protein_set = protein_iid_lookup.get(key)
        if not protein_set:
            return (False, False)
        return analyze_protein_contacts(
            row,
            protein_iid_set=protein_set,
            contact_cutoff=protein_contact_cutoff,
            covalent_threshold=covalent_distance_threshold,
        )

    has_valid_contact = pd.Series(False, index=df.index, dtype=bool)
    is_maybe_covalent = pd.Series(False, index=df.index, dtype=bool)

    # Scan every non-polymer row so ``is_maybe_covalently_linked_to_protein``
    # is truthful across the full small-molecule space (required by the PMSM
    # filter in ``atomworks_sd_dataset.py``).
    if is_non_polymer.any():
        logger.info(
            "Evaluating protein contacts for %d non-polymer rows "
            "(contact_cutoff=%.1f A, covalent_threshold=%.1f A)",
            int(is_non_polymer.sum()),
            protein_contact_cutoff,
            covalent_distance_threshold,
        )
        tqdm.pandas(desc="Protein contact analysis")
        contact_results = df.loc[is_non_polymer].progress_apply(_analyze, axis=1)
        has_valid_contact.loc[is_non_polymer] = contact_results.apply(lambda t: t[0])
        is_maybe_covalent.loc[is_non_polymer] = contact_results.apply(lambda t: t[1])

    df["has_protein_contact_within_5A"] = has_valid_contact

    df["q_pn_unit_is_biologically_meaningful_small_molecule"] = (
        is_non_polymer
        & df["q_pn_unit_has_filtered_ccd"].astype(bool)
        & df["has_protein_contact_within_5A"].astype(bool)
    )
    df["q_pn_unit_is_maybe_covalently_linked_to_protein"] = is_maybe_covalent

    logger.info(
        "BMSM (+ %0.1f A protein-contact requirement): %d rows",
        protein_contact_cutoff,
        int(df["q_pn_unit_is_biologically_meaningful_small_molecule"].sum()),
    )

    df = add_num_contacting_protein_chains(df, distance_cutoff=protein_contact_cutoff)
    df = add_per_partner_contacts_to_protein_small_molecule(df)

    return df


def _load_passed_ccd_codes(path: Path) -> set[str]:
    with open(path) as f:
        codes = {line.strip() for line in f if line.strip()}
    logger.info("Loaded %d passed CCD codes from %s", len(codes), path)
    return codes


def _collect_ccd_codes(df: pd.DataFrame) -> set[str]:
    all_codes: set[str] = set()
    for value in df["q_pn_unit_non_polymer_res_names"].dropna().unique():
        all_codes.update(_parse_ccd_codes(value))
    return all_codes


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input-parquet", default=DEFAULT_INPUT, help=f"Input metadata parquet (default: {DEFAULT_INPUT})")
    ap.add_argument("--output-parquet", default=DEFAULT_OUTPUT, help=f"Output metadata parquet with BMSM columns (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--passed-ccd-codes-txt", default=DEFAULT_PASSED_CCDS, help=f"Newline-separated CCD codes that passed upstream filters (default: {DEFAULT_PASSED_CCDS})")
    ap.add_argument("--smiles-cache-json", default=DEFAULT_SMILES_CACHE, help=f"JSON cache of CCD -> SMILES (default: {DEFAULT_SMILES_CACHE})")
    ap.add_argument("--heavy-atom-cache-json", default=DEFAULT_HEAVY_ATOM_CACHE, help=f"JSON cache of CCD -> heavy atom count (default: {DEFAULT_HEAVY_ATOM_CACHE})")
    ap.add_argument("--smiles-fetch-workers", type=int, default=32, help="Parallel workers when fetching missing SMILES from RCSB")
    ap.add_argument("--covalent-distance-threshold", type=float, default=MIN_ALLOWED_DISTANCE)
    ap.add_argument("--protein-contact-cutoff", type=float, default=DISTANCE_CUTOFF)
    ap.add_argument("--max-pdb-ids", type=int, default=0, help="Debug: keep only the first N distinct pdb_ids before augmenting (0 = no limit)")
    args = ap.parse_args()

    input_path = Path(args.input_parquet)
    output_path = Path(args.output_parquet)
    passed_path = Path(args.passed_ccd_codes_txt)
    smiles_path = Path(args.smiles_cache_json)
    heavy_cache_path = Path(args.heavy_atom_cache_json)

    logger.info("Reading metadata from %s", input_path)
    metadata = pd.read_parquet(input_path)
    logger.info("Input rows: %d", len(metadata))

    if args.max_pdb_ids > 0:
        keep = metadata["pdb_id"].drop_duplicates().head(args.max_pdb_ids)
        metadata = metadata[metadata["pdb_id"].isin(keep)].copy()
        logger.info(
            "Smoke run: limited to %d pdb_ids (%d rows)",
            args.max_pdb_ids,
            len(metadata),
        )

    passed_ccd_codes = _load_passed_ccd_codes(passed_path)

    # SMILES cache: ``load_smiles_cache`` returns ``{}`` when the file does not
    # yet exist, so the very first run simply fetches every metadata CCD from
    # RCSB. ``fetch_all_smiles`` and ``run_atomworks_smiles_fallback`` both
    # persist to ``smiles_path``.
    ccd_codes_in_metadata = _collect_ccd_codes(metadata)
    logger.info("Unique non-polymer CCD codes in metadata: %d", len(ccd_codes_in_metadata))

    current_cache = load_smiles_cache(smiles_path)
    missing_ccds = sorted(
        code
        for code in ccd_codes_in_metadata
        if code and not (isinstance(current_cache.get(code), str) and current_cache.get(code))
    )
    if missing_ccds:
        logger.info(
            "Fetching SMILES for %d CCDs missing from cache via RCSB",
            len(missing_ccds),
        )
        current_cache = fetch_all_smiles(
            missing_ccds, cache_path=smiles_path, num_workers=args.smiles_fetch_workers
        )
        still_missing = [
            code
            for code in missing_ccds
            if not (isinstance(current_cache.get(code), str) and current_cache.get(code))
        ]
        if still_missing:
            logger.info(
                "Falling back to atomworks CCD lookup for %d unresolved CCDs",
                len(still_missing),
            )
            current_cache = run_atomworks_smiles_fallback(
                still_missing, current_cache, smiles_path
            )

    smiles_map = {
        code: smi
        for code, smi in current_cache.items()
        if isinstance(smi, str) and smi
    }
    logger.info("Active SMILES map size: %d", len(smiles_map))

    heavy_atom_counts = build_heavy_atom_cache(
        ccd_codes_in_metadata, smiles_map, heavy_cache_path
    )

    augmented = augment_metadata_with_bmsm(
        metadata,
        passed_ccd_codes=passed_ccd_codes,
        smiles_map=smiles_map,
        heavy_atom_counts=heavy_atom_counts,
        covalent_distance_threshold=args.covalent_distance_threshold,
        protein_contact_cutoff=args.protein_contact_cutoff,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(output_path)
    logger.info("Wrote augmented metadata -> %s (rows=%d)", output_path, len(augmented))

    is_sm_col = augmented["q_pn_unit_is_small_molecule"].astype(bool)
    is_bmsm_col = augmented["q_pn_unit_is_biologically_meaningful_small_molecule"].astype(bool)
    is_cov_col = augmented["q_pn_unit_is_maybe_covalently_linked_to_protein"].astype(bool)

    has_filtered = int(augmented["q_pn_unit_has_filtered_ccd"].sum())
    bmsm_count = int(is_bmsm_col.sum())
    covalent_count = int(is_cov_col.sum())
    sm_count = int(is_sm_col.sum())
    bmsm_cov_count = int((is_bmsm_col & is_cov_col).sum())
    nonbmsm_sm_cov_count = int((is_sm_col & ~is_bmsm_col & is_cov_col).sum())
    logger.info(
        "BMSM stats: has_filtered_ccd=%d, small_molecule=%d, biologically_meaningful=%d, "
        "maybe_covalent=%d (bmsm∩cov=%d, non-bmsm sm∩cov=%d)",
        has_filtered,
        sm_count,
        bmsm_count,
        covalent_count,
        bmsm_cov_count,
        nonbmsm_sm_cov_count,
    )

    num_contacting = augmented["num_contacting_protein_chains"]
    ncp_vc = num_contacting.value_counts().sort_index()
    ncp_head = {int(k): int(v) for k, v in list(ncp_vc.items())[:8]}
    logger.info(
        "num_contacting_protein_chains distribution (k=0..7): %s; rows>=1: %d",
        ncp_head,
        int((num_contacting >= 1).sum()),
    )

    protein_partner_col = augmented[
        "q_pn_unit_per_partner_contacts_to_protein_small_molecule"
    ]
    nonnull_partner = int(protein_partner_col.notna().sum())
    nonempty_partner = int(
        protein_partner_col.dropna().apply(lambda s: s != "[]").sum()
    )
    logger.info(
        "per_partner_contacts_to_protein_small_molecule: nonnull=%d, has_protein_partner=%d",
        nonnull_partner,
        nonempty_partner,
    )


if __name__ == "__main__":
    main()
