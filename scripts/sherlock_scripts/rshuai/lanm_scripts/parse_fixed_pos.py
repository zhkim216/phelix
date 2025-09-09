# ~/code/allatom-design/scripts/generate_manifest.py
import os
import csv
from pathlib import Path
import sys

def format_fix_pos(pos_string):
    """
    Converts a space-separated string of residue numbers '11 12 13'
    into the required comma-separated format 'A11,A12,A13'.
    """
    if not pos_string.strip():
        return "null"  # Or handle as an error if positions are always required

    # Prepend 'A' to each number and join with a comma
    return ",".join([f"A{num}" for num in pos_string.split()])

def main():
    # --- Configuration ---
    # The top-level directory containing all the experimental runs
    base_search_dir = Path("/scratch/users/rshuai/datasets/lanm/lanm_pd_scaffolded")

    # The output manifest file. The sbatch script will read from this.
    manifest_file = Path("/scratch/users/rshuai/datasets/lanm/lanm_pd_scaffolded/lanm_scaffolded_fixed_pos_manifest.txt")

    # --- End Configuration ---

    if not base_search_dir.is_dir():
        print(f"Error: Base search directory not found at {base_search_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for 'design_input_fixpos.csv' in: {base_search_dir}")

    # Use a dictionary to store unique jobs. Key: pdb_directory, Value: formatted_fix_pos
    # This automatically handles the fact that many PDBs in the CSV share the same directory and fix_pos.
    unique_jobs = {}

    # Find all the relevant CSV files
    csv_files = list(base_search_dir.rglob("design_input_fixpos.csv"))

    if not csv_files:
        print("Error: No 'design_input_fixpos.csv' files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files to process.")

    for csv_path in csv_files:
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pdb_full_path = Path(row['pdb_path'])
                    # The PDB directory is the first two parents of the path
                    pdb_dir = "/".join(pdb_full_path.parts[-3:-1])

                    # Process the fix_pos string
                    formatted_pos = format_fix_pos(row['fix_pos'])

                    # Store the unique job
                    unique_jobs[pdb_dir] = formatted_pos
        except Exception as e:
            print(f"Warning: Could not process {csv_path}. Error: {e}", file=sys.stderr)

    if not unique_jobs:
        print("Error: No valid jobs were found after processing CSV files.", file=sys.stderr)
        sys.exit(1)

    # Write the results to the manifest file. We'll use a tab separator.
    with open(manifest_file, 'w') as f:
        for pdb_dir, fix_pos in sorted(unique_jobs.items()):
            f.write(f"{pdb_dir}\t{fix_pos}\n")

    print(f"\nSuccessfully generated manifest file with {len(unique_jobs)} unique jobs.")
    print(f"Manifest file located at: {manifest_file}")

if __name__ == "__main__":
    main()