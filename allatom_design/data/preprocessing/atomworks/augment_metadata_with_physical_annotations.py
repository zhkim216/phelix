"""
Augment metadata parquet with physical meaningfulness annotations.

Derives from raw counts already computed during preprocessing:
  - q_pn_unit_is_physically_meaningful_metal
  - q_pn_unit_is_physically_meaningful_small_molecule
  - q_pn_unit_context_group_iids  (from contacting_pn_unit_iids at configurable distance)
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def derive_context_groups(contacts_json: str, context_distance: float = 5.0) -> str:
    """Extract pn_unit_iids within context_distance from the contacts JSON column."""
    if pd.isna(contacts_json) or not contacts_json:
        return json.dumps([])
    try:
        contacts = json.loads(contacts_json) if isinstance(contacts_json, str) else contacts_json
    except (json.JSONDecodeError, TypeError):
        return json.dumps([])
    return json.dumps([
        c["pn_unit_iid"]
        for c in contacts
        if c.get("min_distance") is not None and c["min_distance"] <= context_distance
    ])


def augment(
    df: pd.DataFrame,
    *,
    min_metal_occupancy: float = 0.5,
    min_coordination_partners: int = 3,
    min_neighboring_heavy_atoms: int = 3,
    context_distance: float = 5.0,
) -> pd.DataFrame:
    """Add derived physical annotation columns to the dataframe."""
    out = df.copy()

    # --- Physically meaningful metals ---
    has_coord = out["q_pn_unit_n_coordination_partners_metal"].notna()
    out["q_pn_unit_is_physically_meaningful_metal"] = (
        out["q_pn_unit_is_metal"]
        & (out["q_pn_unit_avg_occupancy_nonpolymer"] >= min_metal_occupancy)
        & has_coord
        & (out["q_pn_unit_n_coordination_partners_metal"] >= min_coordination_partners)
    )

    # --- Physically meaningful small molecules ---
    has_neigh = out["q_pn_unit_n_neighboring_heavy_atoms_small_molecule"].notna()
    out["q_pn_unit_is_physically_meaningful_small_molecule"] = (
        has_neigh
        & (out["q_pn_unit_n_neighboring_heavy_atoms_small_molecule"] >= min_neighboring_heavy_atoms)
        & (~out["q_pn_unit_is_polymer"])
        & (~out["q_pn_unit_is_metal"])
    )

    # --- Context group iids (from contacting at configurable distance) ---
    if "q_pn_unit_contacting_pn_unit_iids" in out.columns:
        out["q_pn_unit_context_group_iids"] = out["q_pn_unit_contacting_pn_unit_iids"].apply(
            lambda x: derive_context_groups(x, context_distance=context_distance)
        )

    return out


def main():
    parser = argparse.ArgumentParser(description="Augment metadata with physical meaningfulness annotations.")
    parser.add_argument("--input-parquet", required=True, help="Input metadata parquet path.")
    parser.add_argument("--output-parquet", default=None, help="Output parquet path (default: overwrites input).")
    parser.add_argument("--min-metal-occupancy", type=float, default=0.5)
    parser.add_argument("--min-metal-coordination-partners", type=int, default=3)
    parser.add_argument("--min-small-molecule-neighboring-heavy-atoms", type=int, default=3)
    parser.add_argument("--context-distance", type=float, default=5.0, help="Distance threshold for context group derivation.")
    args = parser.parse_args()

    input_path = Path(args.input_parquet)
    output_path = Path(args.output_parquet) if args.output_parquet else input_path

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Thresholds: metal_occ>={args.min_metal_occupancy}, "
          f"coord_partners>={args.min_metal_coordination_partners}, "
          f"neigh_heavy>={args.min_small_molecule_neighboring_heavy_atoms}, "
          f"context_dist<={args.context_distance}")

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows.")

    # Check required columns
    required = [
        "q_pn_unit_is_metal", "q_pn_unit_is_polymer",
        "q_pn_unit_n_coordination_partners_metal",
        "q_pn_unit_n_neighboring_heavy_atoms_small_molecule",
        "q_pn_unit_avg_occupancy_nonpolymer",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns (run preprocessing with updated DataPreprocessor first): {missing}")

    result = augment(
        df,
        min_metal_occupancy=args.min_metal_occupancy,
        min_coordination_partners=args.min_metal_coordination_partners,
        min_neighboring_heavy_atoms=args.min_small_molecule_neighboring_heavy_atoms,
        context_distance=args.context_distance,
    )

    # Summary
    n_metals = result["q_pn_unit_is_metal"].sum()
    n_phys_metals = result["q_pn_unit_is_physically_meaningful_metal"].sum()
    n_sm = (~result["q_pn_unit_is_polymer"] & ~result["q_pn_unit_is_metal"]).sum()
    n_phys_sm = result["q_pn_unit_is_physically_meaningful_small_molecule"].sum()
    print(f"Metals: {n_phys_metals:,}/{n_metals:,} physically meaningful")
    print(f"Small molecules: {n_phys_sm:,}/{n_sm:,} physically meaningful")

    if "q_pn_unit_context_group_iids" in result.columns:
        n_with_ctx = result["q_pn_unit_context_group_iids"].apply(
            lambda x: len(json.loads(x)) > 0 if pd.notna(x) else False
        ).sum()
        print(f"Rows with context groups: {n_with_ctx:,}/{len(result):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
