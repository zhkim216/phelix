#!/usr/bin/env python3
"""
CLI for extracting Egret embeddings from structure files.

Examples:
  Single file (invariant descriptor):
    python -m allatom_design.data.embeddings.extract_embeddings \
      --model /path/to/EGRET_1.model \
      --input /path/to/example.pdb \
      --out_dir /tmp/egret_out

  Batch from a directory (equivariant descriptor):
    python -m allatom_design.data.embeddings.extract_embeddings \
      --model /path/to/EGRET_1.model \
      --input_dir /path/to/structures \
      --pattern "*.pdb" \
      --descriptor equivariant \
      --out_dir /tmp/egret_out
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

from .egret_embedder import EgretEmbedder, MissingDependencyError


def gather_inputs(args: argparse.Namespace) -> List[str]:
    """Collect input file paths according to CLI flags."""
    if args.input and args.input_dir:
        raise SystemExit("Provide either --input or --input_dir, not both.")
    if not args.input and not args.input_dir:
        raise SystemExit("One of --input or --input_dir is required.")

    paths: list[str] = []
    if args.input:
        paths = [args.input]
    else:
        pattern = args.pattern or "*"
        root = Path(args.input_dir)
        paths = sorted(glob.glob(str(root / "**" / pattern), recursive=True))
        if not paths:
            raise SystemExit(f"No files matched in {root} with pattern {pattern}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to Egret model file (e.g., EGRET_1.model)")
    ap.add_argument("--input", help="Single structure file path")
    ap.add_argument("--input_dir", help="Directory containing structure files")
    ap.add_argument("--pattern", default="*", help="Glob pattern under --input_dir, e.g., *.pdb or *.cif")
    ap.add_argument("--descriptor", default="invariant", choices=["invariant", "equivariant"], help="Descriptor type to export")
    ap.add_argument("--frame_index", type=int, default=0, help="Frame index to read from multi-frame files")
    ap.add_argument("--out_dir", required=True, help="Output directory for Parquet/NPZ files")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device preference")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"], help="Default dtype for calculator")
    args = ap.parse_args()

    try:
        embedder = EgretEmbedder(args.model, default_dtype=args.dtype, device=args.device)
    except MissingDependencyError as e:
        raise SystemExit(str(e))

    inputs = gather_inputs(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    written = embedder.embed_many(inputs, descriptor=args.descriptor, frame_index=args.frame_index, out_dir=args.out_dir)
    print(f"Wrote {len(written)} files to {args.out_dir}")


if __name__ == "__main__":
    main()



