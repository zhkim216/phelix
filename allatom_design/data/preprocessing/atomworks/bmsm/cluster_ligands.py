#!/usr/bin/env python3
"""
Cluster biologically meaningful small molecule (BMSM) rows in a metadata parquet
using **complete-linkage hierarchical clustering** on Morgan/ECFP4 fingerprints
with Tanimoto distance.

Complete-linkage cuts the dendrogram at the **diameter cutoff**:
``fcluster(Z, t=1-tanimoto, criterion="distance")`` is equivalent to
"the maximum pairwise distance inside any cluster is <= 1 - tanimoto".
Therefore every pair of members of every cluster is **guaranteed** to share
Tanimoto similarity >= ``--tanimoto-cutoff``.

Multi-residue small molecules (comma-separated CCD codes in
``q_pn_unit_non_polymer_res_names``) are handled by dot-joining the per-CCD
SMILES and then canonicalising through RDKit. Canonical SMILES is
order-independent, so manual sorting of CCD codes is not required. Two
chemically-distinct multi-residue ligands that collapse to the same canonical
SMILES are intentionally merged at this canonicalisation step (before any
clustering).

Produces a new ``q_pn_unit_bmsm_ligand_cluster_id`` int32 column: non-BMSM rows
and rows whose composite SMILES cannot be built/parsed get the sentinel ``-1``.

Dependencies
    - rdkit (Chem, DataStructs, AllChem)
    - fastcluster (linkage)
    - scipy (fcluster)
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import fastcluster
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from allatom_design.data.preprocessing.atomworks.bmsm.smiles_cache import (
    load_smiles_cache,
)


logger = logging.getLogger(__name__)

BMSM_COLUMN = "q_pn_unit_is_biologically_meaningful_small_molecule"
CCD_CODES_COLUMN = "q_pn_unit_non_polymer_res_names"
CLUSTER_ID_COLUMN = "q_pn_unit_bmsm_ligand_cluster_id"

# Complete-linkage requires a condensed distance array of N*(N-1)/2 float64
# entries. Memory budget (assuming 32 GB job):
#   N = 43_000  ->  ~7.4 GB     (typical)
#   N = 60_000  ->  ~14 GB      (warn here — still fits)
#   N = 80_000  ->  ~25 GB      (abort here — risk of OOM with overhead)
UNIQUE_SMILES_WARN_THRESHOLD = 60_000
MAX_UNIQUE_SMILES = 80_000


def canonicalize_composite_smiles(
    ccd_codes: list[str],
    smiles_map: dict[str, str | None],
) -> Optional[str]:
    """
    Build a canonical composite SMILES for a (possibly multi-residue) small molecule.

    Each CCD code's SMILES is looked up in ``smiles_map``; missing entries or RDKit
    parse failures cause the whole composite to be dropped (return ``None``).
    Individual fragments are dot-joined and passed through
    ``Chem.MolFromSmiles``/``Chem.MolToSmiles`` so the output is independent of
    the CCD code ordering.
    """
    if not ccd_codes:
        return None

    fragments: list[str] = []
    for ccd_code in ccd_codes:
        smiles = smiles_map.get(ccd_code)
        # Any missing fragment invalidates the whole composite; we cannot cluster
        # a partial ligand.
        if not smiles:
            return None
        fragments.append(smiles)

    composite = ".".join(fragments)
    mol = Chem.MolFromSmiles(composite)
    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_bmsm_canonical_smiles(
    bmsm_df: pd.DataFrame,
    smiles_map: dict[str, str | None],
) -> tuple[pd.Series, list[str]]:
    """
    Canonicalise the composite SMILES for every BMSM row.

    Returns:
        per_row_canonical:
            ``pd.Series`` aligned to ``bmsm_df.index``; entries are canonical
            SMILES strings, or ``None`` when canonicalisation failed.
        unique_canonical:
            Deduplicated list of canonical SMILES (insertion order preserved
            for run-to-run determinism).
    """
    per_row: list[Optional[str]] = []
    for raw_codes in tqdm(
        bmsm_df[CCD_CODES_COLUMN].tolist(),
        desc="Canonicalising composite SMILES",
    ):
        if not isinstance(raw_codes, str) or not raw_codes.strip():
            per_row.append(None)
            continue
        ccd_codes = [c.strip() for c in raw_codes.split(",") if c.strip()]
        per_row.append(canonicalize_composite_smiles(ccd_codes, smiles_map))

    n_failed = sum(1 for s in per_row if s is None)
    logger.info(
        f"Composite SMILES: {len(per_row) - n_failed:,}/{len(per_row):,} "
        f"rows canonicalised, {n_failed:,} failures (will get -1)."
    )

    unique_canonical: list[str] = []
    seen: set[str] = set()
    for smi in per_row:
        if smi is None or smi in seen:
            continue
        seen.add(smi)
        unique_canonical.append(smi)

    per_row_canonical = pd.Series(per_row, index=bmsm_df.index, dtype=object)
    return per_row_canonical, unique_canonical


def generate_morgan_fingerprints(
    canonical_smiles_list: list[str],
    radius: int,
    n_bits: int,
) -> list:
    """
    Generate Morgan/ECFP fingerprints for each canonical SMILES.

    Returns a list aligned to ``canonical_smiles_list``; entries are RDKit
    ``ExplicitBitVect`` objects, or ``None`` when parsing/fingerprinting failed.
    Fingerprint failures should be rare here because every input SMILES has
    already round-tripped through RDKit during canonicalisation.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: list = []
    for smi in tqdm(canonical_smiles_list, desc="Computing Morgan fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            continue
        try:
            fps.append(generator.GetFingerprint(mol))
        except Exception:
            fps.append(None)
    return fps


def compute_condensed_tanimoto_distance(fps: list) -> np.ndarray:
    """
    Compute the upper-triangle (condensed) Tanimoto distance matrix.

    Output layout matches scipy's ``squareform`` convention: for ``n`` inputs,
    the returned array has length ``n * (n - 1) / 2`` and the entry for pair
    ``(i, j)`` with ``i < j`` is at flat index
    ``i * (n - 1) - i * (i + 1) / 2 + (j - i - 1)``.

    All entries in ``fps`` must be non-None (this is a precondition; the caller
    is expected to filter/abort upstream).
    """
    if any(fp is None for fp in fps):
        raise ValueError(
            "compute_condensed_tanimoto_distance received a None fingerprint; "
            "filter failed fingerprints before calling."
        )

    n = len(fps)
    if n < 2:
        return np.empty(0, dtype=np.float64)

    n_pairs = n * (n - 1) // 2
    dists = np.empty(n_pairs, dtype=np.float64)
    idx = 0
    for i in tqdm(range(n - 1), desc="Computing Tanimoto distances"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        sims_arr = np.asarray(sims, dtype=np.float64)
        dists[idx : idx + sims_arr.size] = 1.0 - sims_arr
        idx += sims_arr.size

    assert idx == n_pairs, f"distance fill mismatch: {idx} vs {n_pairs}"
    return dists


def run_complete_linkage_clustering(
    condensed_distance: np.ndarray,
    distance_cutoff: float,
    n: int,
) -> np.ndarray:
    """
    Complete-linkage hierarchical clustering with a diameter cutoff.

    ``condensed_distance`` is the upper-triangle distance vector for ``n`` points
    (length ``n * (n - 1) / 2``). ``n`` is passed explicitly so that the n<2
    case (where the condensed array is empty) can be handled unambiguously.
    The dendrogram is cut so that no resulting cluster has any pairwise distance
    exceeding ``distance_cutoff``.

    Returns a 1-d ``numpy.ndarray`` of length ``n`` with cluster labels
    (1-indexed, matching scipy's ``fcluster`` convention).

    For ``n < 2`` no linkage is performed (fastcluster cannot build a tree from
    fewer than 2 points): the empty case returns an empty label array, the
    singleton case returns ``[1]``.
    """
    expected_pairs = n * (n - 1) // 2
    if condensed_distance.size != expected_pairs:
        raise ValueError(
            f"condensed_distance length {condensed_distance.size} does not match "
            f"n*(n-1)/2 = {expected_pairs} for n={n}."
        )

    if n == 0:
        return np.empty(0, dtype=np.int32)
    if n == 1:
        return np.ones(1, dtype=np.int32)

    # preserve_input=False lets fastcluster reuse / overwrite the input buffer,
    # halving peak memory for large N.
    Z = fastcluster.linkage(condensed_distance, method="complete", preserve_input=False)
    labels = fcluster(Z, t=distance_cutoff, criterion="distance")
    return labels


def assign_cluster_ids(
    bmsm_df: pd.DataFrame,
    per_row_canonical: pd.Series,
    canonical_to_label: dict[str, int],
) -> pd.Series:
    """
    Map each BMSM row to its cluster id via vectorised SMILES->label lookup.

    Rows whose canonical SMILES is ``None`` (canonicalisation failed) or whose
    canonical SMILES has no entry in ``canonical_to_label`` (fingerprint failed
    upstream) get the ``-1`` sentinel. Returned dtype is ``np.int32``, aligned
    to ``bmsm_df.index``.
    """
    mapped = per_row_canonical.map(canonical_to_label)
    cluster_series = mapped.fillna(-1).astype(np.int32)
    cluster_series.index = bmsm_df.index
    return cluster_series


def augment_metadata_with_ligand_cluster(
    input_df: pd.DataFrame,
    smiles_map: dict[str, str | None],
    tanimoto_cutoff: float = 0.8,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
) -> pd.DataFrame:
    """
    Add ``q_pn_unit_bmsm_ligand_cluster_id`` to ``input_df``.

    Non-BMSM rows retain the ``-1`` sentinel. BMSM rows are clustered by
    chemical similarity on their composite canonical SMILES using complete-
    linkage hierarchical clustering with the diameter cutoff
    ``1 - tanimoto_cutoff``.
    """
    if BMSM_COLUMN not in input_df.columns:
        raise KeyError(
            f"Input parquet is missing required column '{BMSM_COLUMN}'. "
            f"Run the BMSM-augmentation step (Step 1) first."
        )
    if CCD_CODES_COLUMN not in input_df.columns:
        raise KeyError(f"Input parquet is missing required column '{CCD_CODES_COLUMN}'.")

    out_df = input_df.copy()
    out_df[CLUSTER_ID_COLUMN] = np.int32(-1)

    # BMSM_COLUMN is non-null per the schema, so we don't need .fillna defenses.
    bmsm_mask = out_df[BMSM_COLUMN].astype(bool)
    logger.info(
        f"Input rows: {len(out_df):,} total, {int(bmsm_mask.sum()):,} flagged as BMSM."
    )

    bmsm_df = out_df.loc[bmsm_mask]
    if bmsm_df.empty:
        logger.warning("No BMSM rows — output column will be all -1.")
        return out_df

    distance_cutoff = 1.0 - tanimoto_cutoff

    # ----------------------------------------------------------------- [1/5]
    t0 = time.perf_counter()
    logger.info(f"[1/5] Loading BMSM rows ({len(bmsm_df):,} rows)...")
    # (Already done above — this stage is the slice; report and move on.)
    logger.info(f"[1/5] Done in {time.perf_counter() - t0:.1f}s.")

    # ----------------------------------------------------------------- [2/5]
    t0 = time.perf_counter()
    logger.info("[2/5] Canonicalising SMILES...")
    per_row_canonical, unique_canonical = load_bmsm_canonical_smiles(
        bmsm_df, smiles_map
    )
    logger.info(
        f"[2/5] Unique canonical SMILES: {len(unique_canonical):,} "
        f"(elapsed {time.perf_counter() - t0:.1f}s)."
    )

    n_unique = len(unique_canonical)
    if n_unique > MAX_UNIQUE_SMILES:
        raise RuntimeError(
            f"Unique canonical SMILES count {n_unique:,} exceeds "
            f"MAX_UNIQUE_SMILES={MAX_UNIQUE_SMILES:,}. Complete-linkage requires "
            f"a condensed Tanimoto distance array of N*(N-1)/2 float64 entries; "
            f"at this N the buffer alone would consume "
            f"~{n_unique * (n_unique - 1) // 2 * 8 / 1e9:.1f} GB before linkage "
            f"overhead. Aborting to avoid OOM. Either bump MAX_UNIQUE_SMILES "
            f"and run on a larger memory job, or shard the input."
        )
    if n_unique > UNIQUE_SMILES_WARN_THRESHOLD:
        logger.warning(
            f"Unique SMILES count {n_unique:,} exceeds "
            f"{UNIQUE_SMILES_WARN_THRESHOLD:,}; the condensed distance array "
            f"will be ~{n_unique * (n_unique - 1) // 2 * 8 / 1e9:.1f} GB. "
            f"Ensure the job has enough RAM."
        )

    if n_unique == 0:
        logger.warning("No valid canonical SMILES — all BMSM rows will get -1.")
        return out_df

    # ----------------------------------------------------------------- [3/5]
    t0 = time.perf_counter()
    logger.info(
        f"[3/5] Generating Morgan FPs (radius={morgan_radius}, "
        f"nBits={morgan_bits})..."
    )
    fps = generate_morgan_fingerprints(
        unique_canonical, radius=morgan_radius, n_bits=morgan_bits
    )

    # Filter post-canonicalisation fingerprint failures (should be rare).
    valid_smiles: list[str] = []
    valid_fps: list = []
    for smi, fp in zip(unique_canonical, fps):
        if fp is None:
            continue
        valid_smiles.append(smi)
        valid_fps.append(fp)
    n_fp_failed = len(unique_canonical) - len(valid_fps)
    if n_fp_failed:
        logger.warning(
            f"Morgan fingerprinting failed for {n_fp_failed:,} unique SMILES "
            f"(rows referencing these SMILES will get -1)."
        )
    logger.info(
        f"[3/5] {len(valid_fps):,} valid fingerprints "
        f"(elapsed {time.perf_counter() - t0:.1f}s)."
    )

    if not valid_fps:
        logger.warning("No valid fingerprints — all BMSM rows will get -1.")
        return out_df

    n = len(valid_fps)

    # ----------------------------------------------------------------- [4/5]
    t0 = time.perf_counter()
    n_pairs = n * (n - 1) // 2
    logger.info(
        f"[4/5] Computing condensed Tanimoto distance "
        f"({n_pairs:,} pairs, ~{n_pairs * 8 / 1e9:.2f} GB float64)..."
    )
    condensed = compute_condensed_tanimoto_distance(valid_fps)
    # Free fingerprints once distances are materialised.
    del valid_fps, fps
    logger.info(f"[4/5] Done in {time.perf_counter() - t0:.1f}s.")

    # ----------------------------------------------------------------- [5/5]
    t0 = time.perf_counter()
    logger.info(
        f"[5/5] Complete-linkage + cluster assignment "
        f"(tanimoto >= {tanimoto_cutoff}, distance cutoff {distance_cutoff})..."
    )
    labels = run_complete_linkage_clustering(condensed, distance_cutoff, n=n)
    del condensed

    # Build canonical SMILES -> 1-indexed cluster label.
    canonical_to_label: dict[str, int] = {
        smi: int(lbl) for smi, lbl in zip(valid_smiles, labels)
    }

    cluster_ids = assign_cluster_ids(bmsm_df, per_row_canonical, canonical_to_label)
    out_df.loc[bmsm_mask, CLUSTER_ID_COLUMN] = cluster_ids.values
    out_df[CLUSTER_ID_COLUMN] = out_df[CLUSTER_ID_COLUMN].astype(np.int32)
    logger.info(f"[5/5] Done in {time.perf_counter() - t0:.1f}s.")

    # -------- final summary ----------------------------------------------
    cluster_sizes = Counter(int(lbl) for lbl in labels)
    n_clusters = len(cluster_sizes)
    n_singletons = sum(1 for c in cluster_sizes.values() if c == 1)
    sizes = np.fromiter(cluster_sizes.values(), dtype=np.int64)
    max_size = int(sizes.max()) if sizes.size else 0
    median_size = int(np.median(sizes)) if sizes.size else 0
    logger.info(
        f"Total: {n_clusters:,} clusters "
        f"({n_singletons:,} singletons, max size {max_size:,}, "
        f"median size {median_size:,})."
    )

    # Top-5 cluster sizes (largest first) for parity with prior log format.
    top_sizes = sorted(
        Counter(cluster_sizes.values()).items(), key=lambda kv: kv[0], reverse=True
    )[:5]
    logger.info("Top-5 cluster sizes (size -> count):")
    for size, count in top_sizes:
        logger.info(f"  size {size}: {count:,} clusters")

    n_unassigned = int((out_df.loc[bmsm_mask, CLUSTER_ID_COLUMN] == -1).sum())
    if n_unassigned:
        logger.info(
            f"BMSM rows without a cluster id (SMILES/fingerprint failures): "
            f"{n_unassigned:,}"
        )

    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster biologically meaningful small molecules in a metadata "
            "parquet using complete-linkage hierarchical clustering on Morgan/"
            "ECFP4 fingerprints with Tanimoto distance."
        )
    )
    parser.add_argument("--input-parquet", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument(
        "--smiles-cache-json",
        required=True,
        type=Path,
        help="CCD -> SMILES cache, e.g. ccd_smiles_cache_metadata_v9.json",
    )
    parser.add_argument("--tanimoto-cutoff", type=float, default=0.8)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    parser.add_argument(
        "--max-pdb-ids",
        type=int,
        default=0,
        help="Debug: keep only the first N distinct pdb_ids before clustering (0 = no limit)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input parquet: {args.input_parquet}")
    input_df = pd.read_parquet(args.input_parquet)

    if args.max_pdb_ids > 0:
        keep = input_df["pdb_id"].drop_duplicates().head(args.max_pdb_ids)
        input_df = input_df[input_df["pdb_id"].isin(keep)].copy()
        logger.info(
            f"Smoke run: limited to {args.max_pdb_ids} pdb_ids ({len(input_df):,} rows)."
        )

    # Early validation: surface schema problems before any heavy work.
    if BMSM_COLUMN not in input_df.columns:
        raise KeyError(
            f"Input parquet is missing required column '{BMSM_COLUMN}'. "
            f"Run the BMSM-augmentation step (Step 1) first."
        )
    if CCD_CODES_COLUMN not in input_df.columns:
        raise KeyError(
            f"Input parquet is missing required column '{CCD_CODES_COLUMN}'."
        )

    logger.info(f"Reading SMILES cache: {args.smiles_cache_json}")
    smiles_map = load_smiles_cache(args.smiles_cache_json)
    logger.info(f"SMILES cache entries: {len(smiles_map):,}")

    out_df = augment_metadata_with_ligand_cluster(
        input_df,
        smiles_map,
        tanimoto_cutoff=args.tanimoto_cutoff,
        morgan_radius=args.morgan_radius,
        morgan_bits=args.morgan_bits,
    )

    logger.info(f"Writing output parquet: {args.output_parquet}")
    out_df.to_parquet(args.output_parquet)
    logger.info("Done.")


if __name__ == "__main__":
    main()
