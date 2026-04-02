"""Sample selection for Glide batch evaluation.

Parses AF3 evaluation metrics (docking + SC), applies quality cutoffs,
and selects the best diffusion sample per designed sequence.
"""

import ast
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_af3_metrics(
    docking_csv_path: str,
    sc_csv_path: str,
) -> pd.DataFrame:
    """Load and flatten AF3 metrics into a per-diffusion DataFrame.

    Each row in the input CSVs has dict-string columns (diffusion_0..4).
    This function explodes them into one row per (designed_sample, diffusion_idx).
    """
    dock_df = pd.read_csv(docking_csv_path)
    sc_df = pd.read_csv(sc_csv_path)

    # Index SC by designed_sample_id for fast lookup
    sc_indexed = sc_df.set_index("designed_sample_id")

    records = []
    n_diffusions = sum(1 for c in dock_df.columns if c.startswith("diffusion_"))

    for _, row in dock_df.iterrows():
        designed_id = row["designed_sample_id"]
        input_id = row["input_sample_id"]

        sc_row = sc_indexed.loc[designed_id]

        for diff_idx in range(n_diffusions):
            col = f"diffusion_{diff_idx}"
            dock_metrics = ast.literal_eval(row[col])
            sc_metrics = ast.literal_eval(sc_row[col])

            records.append({
                "designed_sample_id": designed_id,
                "input_sample_id": input_id,
                "diffusion_idx": diff_idx,
                **dock_metrics,
                **sc_metrics,
            })

    return pd.DataFrame(records)


def select_best_diffusion(
    flat_df: pd.DataFrame,
    ligand_rmsd_cutoff: float = 2.0,
    ligand_plddt_cutoff: float = 70.0,
) -> pd.DataFrame:
    """Select best diffusion sample per designed_sample_id.

    Filters by cutoffs, then picks the diffusion with highest ligand_plddt.
    """
    mask = (
        (flat_df["ligand_rmsd"] <= ligand_rmsd_cutoff)
        & (flat_df["ligand_plddt"] >= ligand_plddt_cutoff)
    )
    filtered = flat_df[mask]

    if filtered.empty:
        logger.warning(
            f"No samples pass cutoffs (rmsd <= {ligand_rmsd_cutoff}, "
            f"plddt >= {ligand_plddt_cutoff})"
        )
        return pd.DataFrame()

    n_before = flat_df["designed_sample_id"].nunique()
    n_after = filtered["designed_sample_id"].nunique()
    logger.info(
        f"Selection: {n_after}/{n_before} designed samples pass cutoffs "
        f"(rmsd <= {ligand_rmsd_cutoff}, plddt >= {ligand_plddt_cutoff})"
    )

    idx = filtered.groupby("designed_sample_id")["ligand_plddt"].idxmax()
    return filtered.loc[idx].reset_index(drop=True)


def find_af3_prediction_path(
    af3_preds_dir: str,
    designed_sample_id: str,
    diffusion_idx: int,
    seed: int = 42,
) -> str | None:
    """Find the AF3 prediction CIF path for a given sample and diffusion.

    Handles naming inconsistency: folder may use 'len150' instead of 'len_150'.
    """
    seed_dir = f"seed-{seed}_sample-{diffusion_idx}"
    preds_dir = Path(af3_preds_dir)

    # Try exact match
    exact = preds_dir / designed_sample_id / seed_dir
    if exact.is_dir():
        cifs = list(exact.glob("*_model_pocket_aligned.cif"))
        if cifs:
            return str(cifs[0])

    # Try with 'len150' variant (folder naming inconsistency)
    import re
    alt_id = re.sub(r"_len_(\d+)_", r"_len\1_", designed_sample_id)
    alt = preds_dir / alt_id / seed_dir
    if alt.is_dir():
        cifs = list(alt.glob("*_model_pocket_aligned.cif"))
        if cifs:
            return str(cifs[0])

    # Glob fallback: search all subdirectories
    pattern = f"*{designed_sample_id.split('_')[0]}*/{seed_dir}/*_model_pocket_aligned.cif"
    matches = list(preds_dir.glob(pattern))
    if matches:
        return str(matches[0])

    logger.warning(f"AF3 prediction not found: {designed_sample_id} diffusion_{diffusion_idx}")
    return None
