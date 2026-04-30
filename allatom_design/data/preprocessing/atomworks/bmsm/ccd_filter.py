"""Table A3 plinder non-artifact CCD filter.

Produces the ``passed_ccd_codes_metadata_<suffix>.txt`` whitelist consumed by
:mod:`augment_metadata_with_bmsm` as the ``q_pn_unit_has_filtered_ccd`` source
of truth.

Pipeline
--------
1. Read all non-polymer CCD codes referenced in the input metadata parquet.
2. Drop codes appearing in the plinder artifact list.
3. Resolve SMILES via the RCSB REST API (cached on disk).
4. Fall back to atomworks' CCD -> RDKit converter for codes RCSB has no SMILES for.
5. Apply the Table A3 non-artifact criteria (heavy atom count, carbon count,
   linear hydrocarbon linker length, unspecified atom count).

The absolute-charge criterion from Table A3 is intentionally omitted — ADP /
ATP and other biologically important multi-phosphate ligands carry |q| > 2.

CLI entry
---------
This module is CLI-invocable; the previous v3..v8 invocations remain valid
modulo the import path:

    python -m allatom_design.data.preprocessing.atomworks.bmsm.ccd_filter \\
        --metadata-parquet ... \\
        --suffix v9 \\
        --plinder-artifact-txt ... \\
        --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from allatom_design.data.preprocessing.atomworks.bmsm.smiles_cache import (
    fetch_all_smiles,
    run_atomworks_smiles_fallback,
)


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def get_longest_linear_hydrocarbon_linker(mol, max_count: int = 50) -> int:
    """Length of the longest unbranched hydrocarbon linker in ``mol``.

    Returns ``-1`` on any RDKit failure (sanitize, substructure match, etc.)
    so the caller can treat the molecule as failing the linker criterion.
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SYMMRINGS)
        link_unit = "[#6D2R0]"
        for i in range(max_count):
            chain_smarts = "~".join([link_unit] * (i + 1))
            if len(mol.GetSubstructMatches(Chem.MolFromSmarts(chain_smarts))) == 0:
                return i
        return -1
    except Exception:
        return -1


def is_non_artifact_ligand(smiles: str) -> bool:
    """Apply Table A3 non-artifact criteria (plinder).

    Criteria:
      - Heavy atom count > 5 (excludes single ions and tiny fragments).
      - Carbon atom count > 2.
      - Longest unbranched hydrocarbon linker <= 12 atoms.
      - No unspecified atoms (atomic number 0).

    The absolute-charge criterion in Table A3 is *not* applied here: ADP/ATP
    and related biologically essential ligands legitimately carry |q| > 2.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False

    if mol.GetNumHeavyAtoms() <= 5:
        return False

    carbon = Chem.MolFromSmarts("[#6]")
    if len(mol.GetSubstructMatches(carbon)) <= 2:
        return False

    if get_longest_linear_hydrocarbon_linker(mol) > 12:
        return False

    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return False

    return True


def filter_by_table_a3(
    ccd_codes: list[str],
    smiles_cache: dict[str, str | None],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split ``ccd_codes`` into ``(passed, failed)`` per Table A3.

    Failures are returned as ``(ccd_code, reason)`` pairs where reason is one
    of ``no_smiles``, ``table_a3_failed``, or the str() of any exception.
    """
    passed: list[str] = []
    failed: list[tuple[str, str]] = []

    for ccd_code in tqdm(ccd_codes, desc="Filtering by Table A3"):
        smiles = smiles_cache.get(ccd_code)
        if smiles is None:
            failed.append((ccd_code, "no_smiles"))
            continue
        try:
            if is_non_artifact_ligand(smiles):
                passed.append(ccd_code)
            else:
                failed.append((ccd_code, "table_a3_failed"))
        except Exception as e:
            failed.append((ccd_code, str(e)))

    return passed, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Table A3 CCD filter: fetch SMILES, apply plinder non-artifact "
            "criteria, emit passed/failed CCD lists."
        )
    )
    parser.add_argument(
        "--metadata-parquet",
        required=True,
        type=str,
        help=(
            "Input metadata parquet with q_pn_unit_non_polymer_res_names and "
            "q_pn_unit_is_polymer columns."
        ),
    )
    parser.add_argument(
        "--suffix",
        required=True,
        type=str,
        help="Suffix for output files (e.g. 'v9'); produces *_metadata_{suffix}.*.",
    )
    parser.add_argument(
        "--plinder-artifact-txt",
        required=True,
        type=str,
        help="Newline-separated CCD codes to exclude as artifacts (plinder origin).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR),
        help="Directory for output files (default: this script's directory).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Parallel HTTP workers for the RCSB API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = args.suffix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_cache_path = output_dir / f"ccd_smiles_cache_metadata_{suffix}.json"

    print(f"Loading metadata from {args.metadata_parquet} ...")
    metadata = pd.read_parquet(args.metadata_parquet)

    non_polymer = metadata[~metadata["q_pn_unit_is_polymer"]]
    raw_codes = non_polymer["q_pn_unit_non_polymer_res_names"].unique().tolist()

    expanded: list[str] = []
    for value in raw_codes:
        if value is None:
            continue
        for piece in value.split(","):
            stripped = piece.strip()
            if stripped:
                expanded.append(stripped)
    unique_codes = sorted(set(expanded))
    print(f"unique_ccd_codes: {len(unique_codes)}")

    with open(args.plinder_artifact_txt) as f:
        artifacts = {line.strip() for line in f if line.strip()}
    non_artifact_codes = [c for c in unique_codes if c not in artifacts]
    print(f"unique_ccd_codes_filtered1 (artifact removed): {len(non_artifact_codes)}")

    print(f"\n=== Step 1: Fetching SMILES from RCSB API (cache={smiles_cache_path}) ===")
    smiles_cache = fetch_all_smiles(
        non_artifact_codes,
        cache_path=smiles_cache_path,
        num_workers=args.num_workers,
    )

    has_smiles = sum(1 for c in non_artifact_codes if smiles_cache.get(c) is not None)
    no_smiles_codes = [c for c in non_artifact_codes if smiles_cache.get(c) is None]
    print(f"  - SMILES available: {has_smiles}/{len(non_artifact_codes)}")
    print(f"  - No SMILES: {len(no_smiles_codes)}")

    if no_smiles_codes:
        print(f"\n=== Step 1.5: atomworks fallback for {len(no_smiles_codes)} codes ===")
        before = sum(1 for c in no_smiles_codes if smiles_cache.get(c) is not None)
        smiles_cache = run_atomworks_smiles_fallback(
            no_smiles_codes, smiles_cache, smiles_cache_path
        )
        after = sum(1 for c in no_smiles_codes if smiles_cache.get(c) is not None)
        print(f"  - Generated: {after - before}/{len(no_smiles_codes)}")

        has_smiles = sum(
            1 for c in non_artifact_codes if smiles_cache.get(c) is not None
        )
        print(
            f"  - SMILES available (after fallback): "
            f"{has_smiles}/{len(non_artifact_codes)}"
        )

    print("\n=== Step 2: Filtering by Table A3 criteria ===")
    passed, failed = filter_by_table_a3(non_artifact_codes, smiles_cache)

    print("\n=== FINAL RESULTS ===")
    print(f"Passed: {len(passed)}, Failed: {len(failed)}")
    print(f"  - Failed (no_smiles): {len([f for f in failed if f[1] == 'no_smiles'])}")
    print(
        f"  - Failed (table_a3): "
        f"{len([f for f in failed if f[1] == 'table_a3_failed'])}"
    )
    print(
        f"  - Failed (other): "
        f"{len([f for f in failed if f[1] not in ('no_smiles', 'table_a3_failed')])}"
    )

    passed_path = output_dir / f"passed_ccd_codes_metadata_{suffix}.txt"
    with open(passed_path, "w") as f:
        for code in sorted(passed):
            f.write(f"{code}\n")
    print(f"Saved passed CCD codes to {passed_path}")

    failed_txt = output_dir / f"failed_ccd_codes_metadata_{suffix}.txt"
    with open(failed_txt, "w") as f:
        for code, _ in sorted(failed, key=lambda kv: kv[0]):
            f.write(f"{code}\n")
    print(f"Saved failed CCD codes to {failed_txt}")

    failed_json = output_dir / f"failed_ccd_codes_metadata_{suffix}.json"
    with open(failed_json, "w") as f:
        json.dump([{"ccd_code": c, "reason": r} for c, r in failed], f, indent=2)
    print(f"Saved failed CCD codes with reasons to {failed_json}")


if __name__ == "__main__":
    main()
