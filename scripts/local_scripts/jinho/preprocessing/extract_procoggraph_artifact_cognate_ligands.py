#!/usr/bin/env python3
"""Extract ProCogGraph CATH cognate-ligand rows for artifact CCD codes."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ARTIFACT_LIST_DIR = Path("/home/yjhk/model-dev/datasets/artifact_lists")
DEFAULT_CATH_COGNATE_LIGANDS = Path(
    "/home/yjhk/model-dev/datasets/procoggraph_database_flat_files_zenodo_15204472"
    "/extracted/procoggraph_coglig_domain_mappings_v1-0/cath_cognate_ligands.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/yjhk/model-dev/datasets/procoggraph_database_flat_files_zenodo_15204472"
    "/artifact_cognate_ligands"
)

FILTERED_FILENAME = "cath_cognate_ligands_artifact_hetcode_rows.csv"
UNMATCHED_FILENAME = "unmatched_artifact_codes.tsv"


@dataclass(frozen=True)
class ArtifactCode:
    ccd_code: str
    in_biolip2: bool
    in_buffer_text: bool
    in_openfold: bool
    in_plinder: bool

    @property
    def n_sources(self) -> int:
        return sum(
            [
                self.in_biolip2,
                self.in_buffer_text,
                self.in_openfold,
                self.in_plinder,
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter ProCogGraph cath_cognate_ligands.csv rows whose hetCode is in "
            "the artifact CCD code union, and write unmatched artifact codes."
        )
    )
    parser.add_argument(
        "--artifact-list-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_LIST_DIR,
        help=f"Artifact list directory. Default: {DEFAULT_ARTIFACT_LIST_DIR}",
    )
    parser.add_argument(
        "--cath-cognate-ligands",
        type=Path,
        default=DEFAULT_CATH_COGNATE_LIGANDS,
        help=f"Input cath_cognate_ligands.csv. Default: {DEFAULT_CATH_COGNATE_LIGANDS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def normalize_code(raw_code: str) -> str:
    return raw_code.strip().upper()


def read_first_token_codes(path: Path) -> set[str]:
    codes: set[str] = set()
    with path.open(errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            code = normalize_code(re.split(r"\s+", line)[0])
            if code and code not in {".", "?"}:
                codes.add(code)
    return codes


def read_tsv_codes(path: Path, column: str) -> set[str]:
    codes: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain required column {column!r}")
        for row in reader:
            code = normalize_code(row.get(column) or "")
            if code and code not in {".", "?"}:
                codes.add(code)
    return codes


def load_artifact_codes(artifact_list_dir: Path) -> dict[str, ArtifactCode]:
    required_files = {
        "biolip2": artifact_list_dir / "biolip2_ligand_list",
        "buffer_text": artifact_list_dir / "buffer_from_text_list.tsv",
        "openfold": artifact_list_dir / "openfold_canonicalized.tsv",
        "plinder": artifact_list_dir / "plinder_artifact_ccd_codes.txt",
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing artifact list file(s): " + ", ".join(missing))

    source_codes = {
        "biolip2": read_first_token_codes(required_files["biolip2"]),
        "buffer_text": read_tsv_codes(required_files["buffer_text"], "ccd_code_current"),
        "openfold": read_tsv_codes(required_files["openfold"], "ccd_code_current"),
        "plinder": read_first_token_codes(required_files["plinder"]),
    }
    all_codes = sorted(set().union(*source_codes.values()))
    return {
        code: ArtifactCode(
            ccd_code=code,
            in_biolip2=code in source_codes["biolip2"],
            in_buffer_text=code in source_codes["buffer_text"],
            in_openfold=code in source_codes["openfold"],
            in_plinder=code in source_codes["plinder"],
        )
        for code in all_codes
    }


def ensure_writable_outputs(output_dir: Path, output_paths: list[Path], force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Output file(s) already exist; rerun with --force to overwrite: "
            + ", ".join(existing)
        )


def filter_cath_rows(
    input_csv: Path,
    filtered_csv: Path,
    artifact_codes: dict[str, ArtifactCode],
) -> dict[str, object]:
    artifact_code_set = set(artifact_codes)
    matched_counts: Counter[str] = Counter()
    input_rows = 0
    output_rows = 0

    tmp_path = filtered_csv.with_suffix(filtered_csv.suffix + ".tmp")
    with input_csv.open(newline="") as src, tmp_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames or "hetCode" not in reader.fieldnames:
            raise ValueError(f"{input_csv} does not contain required column 'hetCode'")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            input_rows += 1
            het_code = normalize_code(row.get("hetCode") or "")
            if het_code in artifact_code_set:
                writer.writerow(row)
                matched_counts[het_code] += 1
                output_rows += 1
    tmp_path.replace(filtered_csv)
    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "matched_unique_hetcodes": len(matched_counts),
        "matched_counts": matched_counts,
    }


def write_unmatched_artifact_codes(
    path: Path,
    artifact_codes: dict[str, ArtifactCode],
    matched_counts: Counter[str],
) -> int:
    fieldnames = [
        "ccd_code",
        "in_biolip2",
        "in_buffer_text",
        "in_openfold",
        "in_plinder",
        "n_sources",
    ]
    unmatched_codes = sorted(set(artifact_codes) - set(matched_counts))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for code in unmatched_codes:
            entry = artifact_codes[code]
            writer.writerow(
                {
                    "ccd_code": entry.ccd_code,
                    "in_biolip2": int(entry.in_biolip2),
                    "in_buffer_text": int(entry.in_buffer_text),
                    "in_openfold": int(entry.in_openfold),
                    "in_plinder": int(entry.in_plinder),
                    "n_sources": entry.n_sources,
                }
            )
    tmp_path.replace(path)
    return len(unmatched_codes)


def main() -> int:
    args = parse_args()
    if not args.cath_cognate_ligands.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.cath_cognate_ligands}")

    filtered_csv = args.output_dir / FILTERED_FILENAME
    unmatched_tsv = args.output_dir / UNMATCHED_FILENAME
    ensure_writable_outputs(args.output_dir, [filtered_csv, unmatched_tsv], args.force)

    artifact_codes = load_artifact_codes(args.artifact_list_dir)
    filter_stats = filter_cath_rows(
        args.cath_cognate_ligands,
        filtered_csv,
        artifact_codes,
    )
    unmatched_count = write_unmatched_artifact_codes(
        unmatched_tsv,
        artifact_codes,
        filter_stats["matched_counts"],
    )

    summary = {
        "artifact_code_count": len(artifact_codes),
        "input_rows": filter_stats["input_rows"],
        "output_rows": filter_stats["output_rows"],
        "matched_unique_hetcodes": filter_stats["matched_unique_hetcodes"],
        "unmatched_artifact_codes": unmatched_count,
        "filtered_csv": str(filtered_csv),
        "unmatched_tsv": str(unmatched_tsv),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
