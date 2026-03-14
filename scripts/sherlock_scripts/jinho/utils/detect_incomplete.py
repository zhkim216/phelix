"""Detect incomplete evaluation jobs by finding missing CCD codes per eval config and step."""

import argparse
import csv
import os
from pathlib import Path


CSV_FILES = {
    "seq_recovery": "seq_recovery_metrics.csv",
    "sc_metrics": "all_sc_metrics_per_designed_sample.csv",
    "docking_metrics": "all_docking_metrics_per_designed_sample.csv",
}

# Column that contains the sample ID (to extract CCD code from)
# seq_recovery uses "example_id", sc/docking use "input_sample_id"
ID_COLUMNS = {
    "seq_recovery": "example_id",
    "sc_metrics": "input_sample_id",
    "docking_metrics": "input_sample_id",
}


def extract_ccd_code(sample_id: str) -> str:
    """Extract CCD code from sample ID like '0H7_len_150_0_model_0'."""
    return sample_id.split("_len_")[0]


def get_ccd_codes_from_csv(csv_path: Path, id_column: str) -> set[str]:
    """Read a CSV and return the set of unique CCD codes."""
    ccd_codes = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row[id_column]
            ccd_codes.add(extract_ccd_code(sample_id))
    return ccd_codes


def detect_incomplete(experiment_path: str, output_dir: str | None = None):
    experiment_path = Path(experiment_path)
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    # Find eval config directories
    eval_configs = sorted([
        d for d in experiment_path.iterdir()
        if d.is_dir() and d.name.startswith("eval_")
    ])

    if not eval_configs:
        print(f"No eval_* directories found in {experiment_path}")
        return

    total_incomplete_steps = 0
    total_missing_entries = 0

    for eval_config_dir in eval_configs:
        eval_config_name = eval_config_dir.name

        # Find step directories
        step_dirs = sorted([
            d for d in eval_config_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        ])

        if not step_dirs:
            continue

        # Build reference CCD set: union across all steps and all CSV files
        reference_ccd_codes = set()
        for step_dir in step_dirs:
            for csv_key, csv_filename in CSV_FILES.items():
                csv_path = step_dir / csv_filename
                if csv_path.exists():
                    ccd_codes = get_ccd_codes_from_csv(csv_path, ID_COLUMNS[csv_key])
                    reference_ccd_codes.update(ccd_codes)

        # Detect missing CCD codes per step
        rows = []
        for step_dir in step_dirs:
            step_name = step_dir.name
            for csv_key, csv_filename in CSV_FILES.items():
                csv_path = step_dir / csv_filename
                if not csv_path.exists():
                    # Entire file missing — all CCD codes are missing
                    for ccd in sorted(reference_ccd_codes):
                        rows.append((eval_config_name, step_name, csv_key, ccd))
                    continue

                present_ccd_codes = get_ccd_codes_from_csv(csv_path, ID_COLUMNS[csv_key])
                missing = reference_ccd_codes - present_ccd_codes
                for ccd in sorted(missing):
                    rows.append((eval_config_name, step_name, csv_key, ccd))

        # Write output CSV
        if rows:
            out_path = output_dir / f"incomplete_{eval_config_name}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["eval_config", "step", "missing_from", "ccd_code"])
                writer.writerows(rows)

            # Summary
            incomplete_steps = len(set(r[1] for r in rows))
            total_incomplete_steps += incomplete_steps
            total_missing_entries += len(rows)
            print(f"[{eval_config_name}] {incomplete_steps} incomplete steps, {len(rows)} missing entries -> {out_path}")
        else:
            print(f"[{eval_config_name}] all steps complete")

    print(f"\nTotal: {total_incomplete_steps} incomplete steps, {total_missing_entries} missing entries across {len(eval_configs)} eval configs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect incomplete evaluation jobs")
    parser.add_argument("experiment_path", help="Path to experiment directory containing eval_* subdirectories")
    parser.add_argument("--output-dir", default=None, help="Output directory for result CSVs (default: script directory)")
    args = parser.parse_args()
    detect_incomplete(args.experiment_path, args.output_dir)
