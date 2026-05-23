#!/usr/bin/env python3
"""Stage local MG native-val artifacts for MG BML prototype evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from atomworks.ml.example_id import generate_example_id


SOURCE_DIR = Path("/home/yjhk/model-dev/datasets/val_cifs/mg_native_val_cifs_test")
OUT_DIR = SOURCE_DIR / "staged_local"
REMOTE_DATASET_ROOT = "/scratch/users/zhkim216/datasets"
LOCAL_DATASET_ROOT = "/home/yjhk/model-dev/datasets"


def _parse_query(value) -> list[str]:
    if isinstance(value, str):
        import ast

        return [str(v) for v in ast.literal_eval(value)]
    return [str(v) for v in list(value)]


def _local_cif_path(path: str) -> str:
    local = Path(str(path).replace(REMOTE_DATASET_ROOT, LOCAL_DATASET_ROOT))
    if local.exists():
        return str(local)
    if local.suffix == ".gz":
        ungzipped = Path(str(local)[:-3])
        if ungzipped.exists():
            return str(ungzipped)
    raise FileNotFoundError(f"could not resolve local CIF path for {path!r}")


def _with_example_index(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    out["query_pn_unit_iids"] = out["query_pn_unit_iids"].apply(_parse_query)
    out["path"] = out["path"].apply(_local_cif_path)
    out["example_id"] = out.apply(
        lambda row: generate_example_id(
            [dataset_name],
            row["pdb_id"],
            row["assembly_id"],
            row["query_pn_unit_iids"],
        ),
        axis=1,
    )
    return out.set_index("example_id", drop=False, verify_integrity=True)


def _site_sample_id(row: pd.Series) -> str:
    protein = str(row["protein_iid"]).replace("_", "-")
    mg = str(row["mg_iid"]).replace("_", "-")
    return f"{row['pdb_id']}--{protein}--{mg}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cifs_dir = OUT_DIR / "cifs"
    cifs_dir.mkdir(parents=True, exist_ok=True)

    metadata_all_in = SOURCE_DIR / "metadata_for_training_mg_nativeval_sm_all.parquet"
    metadata_site_in = SOURCE_DIR / "metadata_for_training_mg_nativeval_sm_site_rich.parquet"
    audit_in = SOURCE_DIR / "mg_nativeval_site_audit.tsv"

    all_df = _with_example_index(pd.read_parquet(metadata_all_in), "mg_nativeval_all")
    site_df = _with_example_index(pd.read_parquet(metadata_site_in), "mg_nativeval_site")

    all_out = OUT_DIR / "metadata_for_training_mg_nativeval_sm_all.local.parquet"
    site_out = OUT_DIR / "metadata_for_training_mg_nativeval_sm_site_rich.local.parquet"
    all_df.to_parquet(all_out)
    site_df.to_parquet(site_out)

    val_ids = sorted(all_df["pdb_id"].astype(str).str.lower().unique())
    val_ids_out = OUT_DIR / "mg_nativeval_ids.txt"
    val_ids_out.write_text("\n".join(val_ids) + "\n")

    sampling_rows = []
    for _, row in site_df.reset_index(drop=True).iterrows():
        sample_id = _site_sample_id(row)
        source_path = Path(row["path"])
        staged_path = cifs_dir / f"{sample_id}.cif"
        if staged_path.exists() or staged_path.is_symlink():
            staged_path.unlink()
        staged_path.symlink_to(source_path)
        sampling_rows.append(
            {
                "sample_id": sample_id,
                "pdb_id": sample_id,
                "native_pdb_id": row["pdb_id"],
                "assembly_id": row["assembly_id"],
                "protein_iid": row["protein_iid"],
                "mg_iid": row["mg_iid"],
                "query_pn_unit_iids": repr(_parse_query(row["query_pn_unit_iids"])),
                "path": str(staged_path),
                "source_path": str(source_path),
            }
        )

    sampling_out = OUT_DIR / "sampling_inputs_mg_nativeval_site_local.csv"
    pd.DataFrame(sampling_rows).to_csv(sampling_out, index=False)

    manifest = {
        "source_dir": str(SOURCE_DIR),
        "out_dir": str(OUT_DIR),
        "inputs": {
            "metadata_all": str(metadata_all_in),
            "metadata_site": str(metadata_site_in),
            "site_audit": str(audit_in),
        },
        "outputs": {
            "metadata_all_local": str(all_out),
            "metadata_site_local": str(site_out),
            "validation_ids": str(val_ids_out),
            "sampling_inputs_site_local": str(sampling_out),
            "cifs_dir": str(cifs_dir),
        },
        "counts": {
            "metadata_all_rows": int(len(all_df)),
            "metadata_site_rows": int(len(site_df)),
            "sampling_rows": int(len(sampling_rows)),
            "unique_pdbs": int(all_df["pdb_id"].nunique()),
        },
        "path_policy": {
            "remote_root": REMOTE_DATASET_ROOT,
            "local_root": LOCAL_DATASET_ROOT,
            "gz_fallback": "use .cif when .cif.gz path is absent",
            "site_symlink_names": "hyphen-only sample IDs so existing eval lookup can key by Path.stem",
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
