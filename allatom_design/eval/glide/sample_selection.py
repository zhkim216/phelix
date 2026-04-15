"""Sample selection for Glide batch evaluation.

Parses AF3 evaluation metrics (docking + SC), applies length-dependent
protein-quality filtering, and picks the best (designed_sample, diffusion)
per input scaffold matching fig4 Pipeline B (``debug/plot_scripts``).
"""

import ast
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fig4 Pipeline B constants + helpers (duplicated from debug/plot_scripts)
# ---------------------------------------------------------------------------
# debug/ is not part of the importable allatom_design package, so these are
# copied verbatim. Keep in sync with:
#   debug/plot_scripts/success_utils.py:15 DESIGNABLE_LENGTH_THRESHOLDS
#   debug/plot_scripts/data_loaders.py:57  extract_length
#   debug/plot_scripts/data_loaders.py:73  normalize_input_id
#   debug/plot_scripts/success_utils.py:21 build_protein_quality_mask

DESIGNABLE_LENGTH_THRESHOLDS = {
    (150, 250): {"ca_plddt_threshold": 80, "sc_rmsd_threshold": 2.0},
    (350, 450): {"ca_plddt_threshold": 70, "sc_rmsd_threshold": 3.0},
}

_LENGTH_PATTERN = re.compile(r"_len_(\d+)_")
_BASELINE_SEQ_SUFFIX = re.compile(r"_\d+$")
_SEQ_SAMPLE_SUFFIX = re.compile(r"_sample\d+$")


def _extract_length(sample_id: str) -> float:
    """Return integer length parsed from ``_len_<N>_`` in the sample id."""
    if not isinstance(sample_id, str):
        return np.nan
    match = _LENGTH_PATTERN.search(sample_id)
    return int(match.group(1)) if match else np.nan


def _normalize_input_id(sample_id: str, is_baseline: bool) -> str:
    """Collapse a per-sequence-sample id to its backbone id.

    Always strips trailing ``_sample<N>``: the seq denoiser names each
    sequence sample ``{backbone}_sample{i}``, and in the twostage pipeline
    the stage-1 output becomes ``input_sample_id`` for stage-2 eval — so
    different ``i`` values belong to the same backbone and must collapse
    before the per-scaffold reduction (otherwise they double-count).

    Baseline runs additionally strip trailing ``_<seqN>`` (LigandMPNN /
    ProteinMPNN use that suffix to index sequence variants).
    """
    if not isinstance(sample_id, str):
        return sample_id
    out = _SEQ_SAMPLE_SUFFIX.sub("", sample_id)
    if is_baseline:
        out = _BASELINE_SEQ_SUFFIX.sub("", out)
    return out


def _build_protein_quality_mask(
    df: pd.DataFrame,
    length_thresholds: dict,
) -> pd.Series:
    """Length-dependent ``(avg_ca_plddt, sc_ca_rmsd)`` mask.

    Rows whose ``length`` is not covered by any key tuple in
    ``length_thresholds`` are excluded (False).
    """
    mask = pd.Series(False, index=df.index)
    for lengths, thresholds in length_thresholds.items():
        length_mask = df["length"].isin(lengths)
        quality_mask = (
            (df["avg_ca_plddt"] >= thresholds["ca_plddt_threshold"])
            & (df["sc_ca_rmsd"] <= thresholds["sc_rmsd_threshold"])
        )
        mask |= (length_mask & quality_mask)
    return mask


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
    ligand_rmsd_cutoff: float | None = 2.0,
    ligand_plddt_cutoff: float | None = 80.0,
    apply_protein_filter: bool = True,
    is_baseline: bool = False,
    length_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Fig4 Pipeline B sample selection.

    Input ``flat_df`` has one row per (designed_sample_id, diffusion_idx)
    with both SC and docking metric columns merged in by
    :func:`load_af3_metrics`.

    Steps (match ``debug/plot_scripts/data_loaders.load_merged_designable_csv``
    + ``success_utils.summarize_success``):

    1. Derive ``length`` from ``input_sample_id``.
    2. If ``apply_protein_filter``: drop rows whose length is not covered
       by ``length_thresholds`` (default :data:`DESIGNABLE_LENGTH_THRESHOLDS`).
    3. If ``apply_protein_filter``: keep only rows passing the
       length-dependent ``(avg_ca_plddt, sc_ca_rmsd)`` quality gate.
    4. Per ``designed_sample_id``: keep the surviving diffusion with max
       ``ligand_plddt``.
    5. Per (normalized) ``input_sample_id``: keep the designed-sample row
       with max ``ligand_plddt``. ``is_baseline=True`` strips trailing
       ``_<seqN>`` from ids so LigandMPNN/ProteinMPNN seq variants collapse
       under one scaffold.
    6. Apply ligand cutoffs (``ligand_rmsd <= cutoff``,
       ``ligand_plddt >= cutoff``) to the collapsed rows.

    Applying cutoffs before the per-input collapse would diverge when an
    input's best-lplddt row fails the rmsd cutoff while a second-best row
    passes — fig4 discards that input, so we must too.
    """
    if length_thresholds is None:
        length_thresholds = DESIGNABLE_LENGTH_THRESHOLDS

    n_total = len(flat_df)
    if n_total == 0:
        return pd.DataFrame()

    df = flat_df.copy()
    df["length"] = df["input_sample_id"].map(_extract_length)

    if apply_protein_filter:
        covered_lengths = {l for lengths in length_thresholds for l in lengths}
        in_length = df["length"].isin(covered_lengths)
        n_length = int(in_length.sum())
        logger.info(
            f"Length filter: {n_length}/{n_total} rows in lengths "
            f"{sorted(covered_lengths)}"
        )
        df = df[in_length]
        if df.empty:
            logger.warning("No rows survive length filter")
            return pd.DataFrame()

        protein_mask = _build_protein_quality_mask(df, length_thresholds)
        n_protein = int(protein_mask.sum())
        logger.info(
            f"Protein quality filter: {n_protein}/{len(df)} rows pass "
            f"length-dependent (avg_ca_plddt, sc_ca_rmsd) thresholds"
        )
        df = df[protein_mask]
        if df.empty:
            logger.warning("No rows survive protein quality filter")
            return pd.DataFrame()

    # Step 4: best diffusion per designed sample by ligand_plddt.
    if "ligand_plddt" not in df.columns:
        raise KeyError(
            "flat_df missing 'ligand_plddt' column required for selection"
        )
    best_diff_idx = df.groupby("designed_sample_id")["ligand_plddt"].idxmax()
    per_design = df.loc[best_diff_idx]
    logger.info(
        f"Per-designed-sample collapse: {len(per_design)} designed samples"
    )

    # Step 5: best designed sample per (normalized) input by ligand_plddt.
    per_design = per_design.copy()
    per_design["_norm_input_id"] = per_design["input_sample_id"].map(
        lambda s: _normalize_input_id(s, is_baseline)
    )
    n_raw_inputs = per_design["input_sample_id"].nunique()
    best_input_idx = per_design.groupby("_norm_input_id")["ligand_plddt"].idxmax()
    per_input = per_design.loc[best_input_idx].drop(columns="_norm_input_id")
    logger.info(
        f"Per-input collapse (is_baseline={is_baseline}): "
        f"{len(per_input)} unique input scaffolds"
    )
    n_dedup = n_raw_inputs - len(per_input)
    if n_dedup > 0:
        logger.info(
            f"Sequence-sample collapse: {n_dedup} duplicate backbones merged "
            f"({n_raw_inputs} raw input_sample_ids → {len(per_input)} backbones)"
        )

    # Step 6: ligand cutoffs on the collapsed rows.
    mask = pd.Series(True, index=per_input.index)
    if ligand_rmsd_cutoff is not None and "ligand_rmsd" in per_input.columns:
        mask &= per_input["ligand_rmsd"] <= ligand_rmsd_cutoff
    if ligand_plddt_cutoff is not None:
        mask &= per_input["ligand_plddt"] >= ligand_plddt_cutoff
    selected = per_input[mask]

    logger.info(
        f"Ligand cutoffs (rmsd <= {ligand_rmsd_cutoff}, "
        f"plddt >= {ligand_plddt_cutoff}): "
        f"{len(selected)}/{len(per_input)} inputs survive"
    )

    return selected.reset_index(drop=True)


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
