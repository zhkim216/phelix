"""
Expand a `_numsample1.csv` pocket-constraint file to cover multiple
`_sample{s}` keys per base, reusing the same `fixed_pos_seq`. No forward
passes are run — this is a pure pandas row-duplication utility.

Use case: run `make_pocket_pos_constraint_mad.py` once with `num_samples=1`
(cheap, one row per base), then call this offline in an interactive shell to
match downstream stage-2 sample counts (`_sample0`, `_sample1`, …).

Usage:
    python -m allatom_design.eval.sampling.expand_pocket_constraint_csv \
        --input  pocket_constraint_mad_k02_numsample1.csv \
        --num-samples 5
    # → pocket_constraint_mad_k02_numsample5.csv

    # Programmatic use:
    from allatom_design.eval.sampling.expand_pocket_constraint_csv import (
        expand_single_csv,
    )
    expand_single_csv("pocket_constraint_mad_k02_numsample1.csv", num_samples=5)
"""

import argparse
import re
from pathlib import Path

import pandas as pd


_SAMPLE_RE = re.compile(r"^(.+)_sample(\d+)$")


_NUMSAMPLE_SUFFIX_RE = re.compile(r"_numsample\d+\.csv$")


def expand_single_csv(
    input_csv: str | Path,
    num_samples: int,
    output_csv: str | Path | None = None,
) -> Path:
    """Expand a `_numsample1.csv` to `_numsample{num_samples}.csv`.

    For each input row, parses `pdb_key` with `^(.+)_sample(\\d+)$` and emits
    `num_samples` new rows with `pdb_key = f"{base}_sample{s}"` for
    `s = 0 .. num_samples - 1`, reusing all other columns unchanged.

    Args:
        input_csv: Path to the source CSV (typically ending in
            `_numsample1.csv`).
        num_samples: Number of sample variants per base to emit.
        output_csv: Optional explicit output path. If omitted, swaps any
            existing `_numsample{N}.csv` suffix for the new count, or
            appends `_numsample{N}` to the stem otherwise.

    Returns:
        The output Path.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    input_path = Path(input_csv)
    df = pd.read_csv(input_path)
    if "pdb_key" not in df.columns:
        raise ValueError(f"{input_path} is missing required column 'pdb_key'")

    expanded_rows: list[dict] = []
    unmatched: list[str] = []
    for _, row in df.iterrows():
        stem = row["pdb_key"]
        m = _SAMPLE_RE.match(str(stem))
        if m is None:
            unmatched.append(str(stem))
            continue
        base = m.group(1)
        for s in range(num_samples):
            new_row = row.to_dict()
            new_row["pdb_key"] = f"{base}_sample{s}"
            expanded_rows.append(new_row)

    if unmatched:
        print(
            f"WARNING: {len(unmatched)} rows did not match "
            f"`{{base}}_sample{{idx}}` and were skipped. "
            f"First: {unmatched[:3]}"
        )

    out_df = pd.DataFrame(expanded_rows, columns=df.columns)

    if output_csv is None:
        new_suffix = f"_numsample{num_samples}.csv"
        if _NUMSAMPLE_SUFFIX_RE.search(input_path.name):
            out_name = _NUMSAMPLE_SUFFIX_RE.sub(new_suffix, input_path.name)
        else:
            out_name = f"{input_path.stem}_numsample{num_samples}.csv"
        output_path = input_path.with_name(out_name)
    else:
        output_path = Path(output_csv)

    out_df.to_csv(output_path, index=False)
    print(
        f"Wrote {len(out_df)} rows "
        f"({len(df) - len(unmatched)} unique bases × {num_samples}) "
        f"→ {output_path}"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Expand a _numsample1.csv pocket-constraint file to "
                    "multiple _sample{s} keys per base."
    )
    parser.add_argument("--input", required=True,
                        help="Path to _numsample1.csv input")
    parser.add_argument("--num-samples", type=int, required=True,
                        help="Target number of sample variants per base")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: swap _numsample{N}.csv suffix)")
    args = parser.parse_args()
    expand_single_csv(args.input, args.num_samples, args.output)


if __name__ == "__main__":
    main()
