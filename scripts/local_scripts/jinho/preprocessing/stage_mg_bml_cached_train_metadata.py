#!/usr/bin/env python3
"""Stage cache-backed AtomWorks metadata for the MG BML prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_ROOT = Path("/home/yjhk/model-dev/datasets/atomworks_pdb_full_v9")
OUT_DIR = DATASET_ROOT / "mg_bml_prototype"
CACHE_DIR = DATASET_ROOT / "cached_examples"

POLICY_SOURCES = {
    "no_filter": DATASET_ROOT / "metadata_ligval_seq_clustered_03_with_mg_pubmed_evidence.parquet",
    "substring": DATASET_ROOT / "metadata_ligval_seq_clustered_03_with_mg_pubmed_evidence.parquet",
    "gpt": DATASET_ROOT / "metadata_ligval_seq_clustered_03_with_mg_pubmed_evidence.parquet",
}


def _split_ccd_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    return [tok.strip().upper() for tok in str(value).split(",") if tok.strip()]


def _has_exact_mg(value: Any) -> bool:
    return "MG" in _split_ccd_tokens(value)


def _cached_pdb_ids(cache_dir: Path) -> set[str]:
    return {path.stem.lower() for path in cache_dir.glob("*.pt")}


def _stage_policy(policy_name: str, source_path: Path, cached_ids: set[str]) -> dict[str, Any]:
    policy_out_dir = OUT_DIR / policy_name
    output_path = policy_out_dir / f"{source_path.stem}.cached_examples.parquet"

    record: dict[str, Any] = {
        "policy": policy_name,
        "source_metadata": str(source_path),
        "output_metadata": str(output_path),
        "source_exists": source_path.exists(),
    }
    if not source_path.exists():
        record["status"] = "missing_source"
        return record

    policy_out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(source_path)
    n_source_rows = len(df)
    n_source_pdbs = int(df["pdb_id"].nunique())

    pdb_ids = df["pdb_id"].astype(str).str.lower()
    out = df.loc[pdb_ids.isin(cached_ids)].copy()
    out.to_parquet(output_path, index=False)

    mg_mask = out["q_pn_unit_non_polymer_res_names"].apply(_has_exact_mg)
    dep_dates = pd.to_datetime(out["deposition_date"], errors="coerce")
    date_mask = dep_dates <= pd.Timestamp("2023-01-01")

    record.update(
        {
            "status": "written",
            "source_rows": int(n_source_rows),
            "source_unique_pdbs": n_source_pdbs,
            "cached_rows": int(len(out)),
            "cached_unique_pdbs": int(out["pdb_id"].nunique()),
            "cached_rows_with_exact_mg_token": int(mg_mask.sum()),
            "cached_pdbs_with_exact_mg_token": int(out.loc[mg_mask, "pdb_id"].nunique()),
            "cached_rows_deposition_date_le_2023_01_01": int(date_mask.sum()),
            "cached_pdbs_deposition_date_le_2023_01_01": int(out.loc[date_mask, "pdb_id"].nunique()),
        }
    )
    return record


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached_ids = _cached_pdb_ids(CACHE_DIR)

    policies = [
        _stage_policy(policy_name, source_path, cached_ids)
        for policy_name, source_path in POLICY_SOURCES.items()
    ]
    manifest = {
        "dataset_root": str(DATASET_ROOT),
        "cache_dir": str(CACHE_DIR),
        "out_dir": str(OUT_DIR),
        "cached_example_count": len(cached_ids),
        "policy_outputs": policies,
        "filter_policy": "keep all metadata rows whose pdb_id has cached_examples/{pdb_id}.pt",
        "purpose": "local runnable MG BML prototype training input; policy semantics remain in the source metadata parquet",
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
