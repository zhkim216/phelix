#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_DIR = Path("/scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test")
DATASET_ID = "mg_denovo_rasa_le0p25_uniform_bins_N100"
PATHLIKE_COLUMNS_TO_DROP = {
    "sample_path",
    "staged_sample_path",
    "source_sha256",
    "staged_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_pdb_id(row: pd.Series) -> str:
    if "pdb_id" in row.index and pd.notna(row["pdb_id"]) and str(row["pdb_id"]).strip():
        return Path(str(row["pdb_id"]).strip()).stem
    if "staged_sample_id" in row.index and pd.notna(row["staged_sample_id"]) and str(row["staged_sample_id"]).strip():
        return Path(str(row["staged_sample_id"]).strip()).stem
    if "sample_file" in row.index and pd.notna(row["sample_file"]) and str(row["sample_file"]).strip():
        sample_file = str(row["sample_file"]).strip()
        if sample_file.endswith(".cif.gz"):
            return sample_file[:-7]
        return Path(sample_file).stem
    if "model_index" in row.index and pd.notna(row["model_index"]):
        return f"mg_len150_model_{int(row['model_index'])}"
    raise ValueError("Cannot infer pdb_id; expected one of pdb_id, staged_sample_id, sample_file, model_index")


def source_candidates(dataset_dir: Path, row: pd.Series, pdb_id: str) -> list[Path]:
    candidates: list[Path] = []
    if "sample_file" in row.index and pd.notna(row["sample_file"]) and str(row["sample_file"]).strip():
        sample_file = str(row["sample_file"]).strip()
        candidates.extend([
            dataset_dir / sample_file,
            dataset_dir / "samples" / sample_file,
            dataset_dir / "cifs" / sample_file,
        ])
    candidates.extend([
        dataset_dir / f"{pdb_id}.cif",
        dataset_dir / "samples" / f"{pdb_id}.cif",
        dataset_dir / "cifs" / f"{pdb_id}.cif",
        dataset_dir / f"{pdb_id}.cif.gz",
        dataset_dir / "samples" / f"{pdb_id}.cif.gz",
        dataset_dir / "cifs" / f"{pdb_id}.cif.gz",
    ])
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_source_path(dataset_dir: Path, row: pd.Series, pdb_id: str) -> Path:
    candidates = source_candidates(dataset_dir, row, pdb_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No source .cif or .cif.gz found for {pdb_id}. Tried:\n  {candidate_text}")


def is_cif_gz(path: Path) -> bool:
    return path.suffixes[-2:] == [".cif", ".gz"]


def is_cif(path: Path) -> bool:
    return path.suffix == ".cif"


def decompress_cif_gz(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with gzip.open(source_path, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.replace(target_path)


def same_file(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.samefile(path_b)
    except FileNotFoundError:
        return False


def stage_cif_source(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if is_cif_gz(source_path):
        decompress_cif_gz(source_path, target_path)
    elif is_cif(source_path):
        if same_file(source_path, target_path):
            return
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        shutil.copyfile(source_path, tmp_path)
        tmp_path.replace(target_path)
    else:
        raise ValueError(f"expected .cif or .cif.gz source file, got: {source_path}")


def discover_sampling_inputs_csv(dataset_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"sampling inputs CSV not found: {path}")
        return path

    matches = sorted(dataset_dir.glob(f"sampling_inputs_*{DATASET_ID}*.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No sampling_inputs CSV matching *{DATASET_ID}*.csv found in {dataset_dir}. "
            "Pass --sampling-inputs-csv explicitly."
        )
    match_text = "\n  ".join(str(path) for path in matches)
    raise ValueError(f"Multiple N100 sampling inputs CSVs found; pass --sampling-inputs-csv explicitly:\n  {match_text}")


def prepare(
    *,
    dataset_dir: Path,
    sampling_inputs_csv: Path | None,
    smoke_n: int,
    force: bool,
    dry_run: bool,
) -> dict:
    dataset_dir = dataset_dir.expanduser()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir does not exist: {dataset_dir}")

    input_csv = discover_sampling_inputs_csv(dataset_dir, sampling_inputs_csv)
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"sampling inputs CSV is empty: {input_csv}")

    rows = []
    source_paths = []
    for _, row in df.iterrows():
        pdb_id = infer_pdb_id(row)
        source_path = resolve_source_path(dataset_dir, row, pdb_id)
        if not (is_cif(source_path) or is_cif_gz(source_path)):
            raise ValueError(f"expected .cif or .cif.gz source file for {pdb_id}, got: {source_path}")
        source_paths.append(source_path)
        rows.append({**row.to_dict(), "pdb_id": pdb_id})

    staged_df = pd.DataFrame(rows)
    if staged_df["pdb_id"].duplicated().any():
        dupes = sorted(staged_df.loc[staged_df["pdb_id"].duplicated(), "pdb_id"].astype(str).unique().tolist())
        raise ValueError(f"duplicate pdb_id values in sampling inputs: {dupes[:10]}")

    cifs_dir = dataset_dir / "cifs"
    list_path = dataset_dir / f"{DATASET_ID}.txt"
    smoke_list_path = dataset_dir / f"{DATASET_ID}_smoke{smoke_n}.txt"
    sanitized_csv_path = dataset_dir / f"sampling_inputs_{DATASET_ID}.csv"
    manifest_path = dataset_dir / f"{DATASET_ID}_manifest.json"

    sample_ids = staged_df["pdb_id"].astype(str).tolist()
    staged_df["staged_sample_id"] = sample_ids
    staged_df["staged_sample_file"] = [f"{pdb_id}.cif" for pdb_id in sample_ids]
    for column in PATHLIKE_COLUMNS_TO_DROP:
        if column in staged_df.columns:
            staged_df = staged_df.drop(columns=[column])

    target_paths = [cifs_dir / f"{pdb_id}.cif" for pdb_id in sample_ids]
    existing_target_conflicts = [
        target_path
        for source_path, target_path in zip(source_paths, target_paths)
        if target_path.exists() and not same_file(source_path, target_path)
    ]
    if existing_target_conflicts and not force and not dry_run:
        preview = "\n  ".join(str(path) for path in existing_target_conflicts[:10])
        raise FileExistsError(
            f"{len(existing_target_conflicts)} staged CIFs already exist. Pass --force to replace. First paths:\n  {preview}"
        )

    manifest_inputs = []
    for source_path, target_path, pdb_id in zip(source_paths, target_paths, sample_ids):
        source_sha256 = sha256_file(source_path)
        if not dry_run:
            stage_cif_source(source_path, target_path)
            staged_sha256 = sha256_file(target_path)
        else:
            staged_sha256 = None
        manifest_inputs.append({
            "pdb_id": pdb_id,
            "source_file": source_path.name,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "staged_sample_file": target_path.name,
            "staged_sample_path": str(target_path),
            "staged_sha256": staged_sha256,
        })

    if not dry_run:
        list_path.write_text("\n".join(sample_ids) + "\n")
        smoke_list_path.write_text("\n".join(sample_ids[:smoke_n]) + "\n")
        staged_df.to_csv(sanitized_csv_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "dataset_id": DATASET_ID,
        "dataset_dir": str(dataset_dir),
        "input_sampling_inputs_csv": str(input_csv),
        "source_rows": int(len(df)),
        "staged_rows": int(len(staged_df)),
        "smoke_n": int(smoke_n),
        "outputs": {
            "cifs_dir": str(cifs_dir),
            "sample_id_list": str(list_path),
            "smoke_sample_id_list": str(smoke_list_path),
            "sampling_inputs_csv": str(sanitized_csv_path),
            "manifest": str(manifest_path),
        },
        "dropped_columns": sorted([column for column in PATHLIKE_COLUMNS_TO_DROP if column in df.columns]),
        "inputs": manifest_inputs,
    }

    if not dry_run:
        if len(list(cifs_dir.glob("*.cif"))) < len(sample_ids):
            raise RuntimeError(f"Expected at least {len(sample_ids)} staged CIFs in {cifs_dir}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Sherlock Mg denovo RASA N100 CIFs for lc_seq_des_multi evaluation."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sampling-inputs-csv", type=Path, default=None)
    parser.add_argument("--smoke-n", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Replace existing staged cifs/*.cif files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print manifest without writing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(
        dataset_dir=args.dataset_dir,
        sampling_inputs_csv=args.sampling_inputs_csv,
        smoke_n=args.smoke_n,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "dry_run": manifest["dry_run"],
        "dataset_id": manifest["dataset_id"],
        "source_rows": manifest["source_rows"],
        "staged_rows": manifest["staged_rows"],
        "outputs": manifest["outputs"],
        "dropped_columns": manifest["dropped_columns"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
