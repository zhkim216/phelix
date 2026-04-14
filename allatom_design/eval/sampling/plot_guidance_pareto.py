"""CLI entrypoint for plotting guidance Pareto fronts.

Usage:
    python -m allatom_design.eval.sampling.plot_guidance_pareto \
        --csv "<run_dir>/step_*/guidance_metrics*.csv" \
        --out_dir <run_dir>/pareto

Notes:
    * ``--csv`` accepts one or more glob patterns; all matching files are
      concatenated before plotting.
    * ``--mode`` selects which figures to emit: ``both`` (default),
      ``per_example``, or ``aggregated``.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from allatom_design.eval.eval_utils.guidance_pareto import (
    load_guidance_metrics,
    plot_guidance_pareto,
    plot_guidance_pareto_all_modes,
)


def _expand_glob_patterns(patterns: list[str]) -> list[Path]:
    matched: list[Path] = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if not hits:
            # Support direct paths too.
            if Path(pattern).exists():
                matched.append(Path(pattern))
            else:
                print(f"[warn] No files matched pattern: {pattern}")
            continue
        matched.extend(Path(h) for h in hits)
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the Pareto front of (U_uncond, U_cond) across a gamma sweep.",
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more paths or glob patterns pointing at guidance_metrics*.csv files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write plots into (created if missing).",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "per_example", "aggregated"],
        default="both",
        help="Which plots to emit.",
    )
    parser.add_argument(
        "--x",
        type=str,
        default="U_uncond",
        help="Column to plot on the x-axis (default: U_uncond).",
    )
    parser.add_argument(
        "--y",
        type=str,
        default="U_cond",
        help="Column to plot on the y-axis (default: U_cond).",
    )
    parser.add_argument(
        "--all",
        dest="all_modes",
        action="store_true",
        help=(
            "Emit the full bundle of Pareto plots (total, per-residue, "
            "pocket, pocket-per-residue) into subdirs under --out_dir. "
            "Overrides --x/--y/--mode when set."
        ),
    )
    args = parser.parse_args()

    csv_paths = _expand_glob_patterns(args.csv)
    if not csv_paths:
        raise SystemExit("No csv files matched the provided --csv patterns.")

    print(f"Loading {len(csv_paths)} csv file(s):")
    for p in csv_paths:
        print(f"  - {p}")

    df = load_guidance_metrics(csv_paths)
    print(
        f"Loaded {len(df)} rows across "
        f"{df['example_id'].nunique()} example_id(s), "
        f"{df['gamma'].nunique()} gamma value(s)."
    )

    if args.all_modes:
        written_all = plot_guidance_pareto_all_modes(df, args.out_dir)
        print("Wrote plots:")
        for subdir, paths in written_all.items():
            print(f"  [{subdir}]")
            for p in paths:
                print(f"    - {p}")
    else:
        written = plot_guidance_pareto(df, args.out_dir, x=args.x, y=args.y, mode=args.mode)
        print("Wrote plots:")
        for p in written:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
