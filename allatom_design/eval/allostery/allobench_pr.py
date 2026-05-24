"""
AlloBench Potts allosteric pocket detection: precision-recall evaluation.

Two-pass pipeline per AlloBench monomer entry:
  Pass 0 (CPU): extract PDB chain sequences, batch-align against UniProt seqs with
                MMseqs2, pre-compute UniProt -> author_res_id maps.
  Pass 1 (GPU): run the sequence denoiser twice per structure -- once with all
                non-polymer conditioning intact (orthosteric + allosteric),
                once with the allosteric modulator atoms masked out -- and
                compute per-residue delta_J / delta_h / delta_combined.

Evaluation targets:
  (a) Active site residues from AlloBench.csv's ``active_site_residue`` column,
      after UniProt-to-PDB mapping.
  (b) Orthosteric pocket residues (<=6A from any orthosteric ligand atom), where
      the orthosteric ligand is inferred at runtime as any non-allosteric ligand
      entity with an atom <=5A from any active-site atom.

Outputs:
  results.csv              -- per-residue delta_* + is_active_site + is_ortho_6A
  per_entry_stats.csv      -- per-PDB bookkeeping
  skipped.csv              -- reason log for skipped entries
  pr_active_site.png       -- global + per-PDB-normalised precision-recall
  pr_ortho_6A.png          -- same, ortho pocket target
  mad_active_site.png      -- per-PDB MAD thresholds overlayed on PR curve
  mad_ortho_6A.png         -- same, ortho pocket target

Usage:
  python -m allatom_design.eval.allostery.allobench_pr debug=true num_debug_samples=5
"""

from __future__ import annotations

import traceback
from pathlib import Path

import atomworks.enums as aw_enums
import hydra
import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm

from allatom_design.eval.allostery.allobench_utils import (
    build_allosteric_atom_mask,
    compute_ortho_pocket_labels,
    compute_per_residue_min_dist_to_atoms,
    extract_protein_chain_sequences,
    identify_orthosteric_ligand_atoms,
    is_single_protein_chain,
    light_load_cif,
    load_allobench,
    prepare_forward_passes,
    run_mmseqs_alignment_batch,
    run_potts_forward_prepared,
    uniprot_idx_to_pdb_residue,
)
from allatom_design.eval.eval_utils.potts_utils import (
    compute_potts_deltas,
    map_token_to_residue_info,
)
from allatom_design.eval.eval_utils.sd_data_utils import get_sd_batch
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


###########################################################
# Pass 0 helpers
###########################################################


def run_pass0_alignment(
    meta: pd.DataFrame,
    cfg: DictConfig,
) -> tuple[dict, dict]:
    """Collect PDB chain sequences and run MMseqs2 alignment once.

    Returns:
        pre_alignment: dict[(pdb_id, chain_id)] -> alignment dict.
        chain_residues_cache: dict[(pdb_id, chain_id)] -> [(author_res_id, aa1), ...]
    """
    print("[pass 0] Extracting chain sequences for alignment...")
    entries: list[dict] = []
    chain_residues_cache: dict[tuple[str, str], list[tuple[int, str]]] = {}

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="extract seqs"):
        pdb_id = row["pdb_id"]
        cif_path = Path(cfg.pdb_structures_dir) / f"{pdb_id}.cif"
        if not cif_path.exists():
            continue
        try:
            aa = light_load_cif(cif_path)
        except Exception as e:
            print(f"  [pass0] {pdb_id}: light_load_cif failed: {e}")
            continue

        chain_seqs = extract_protein_chain_sequences(aa)
        if not chain_seqs:
            continue
        for chain_id, residues in chain_seqs.items():
            if len(residues) < cfg.min_chain_length:
                continue
            pdb_seq = "".join(a for _, a in residues)
            chain_residues_cache[(pdb_id, chain_id)] = residues
            entries.append(
                {
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "pdb_seq": pdb_seq,
                    "uniprot_seq": str(row["sequence"]),
                }
            )

    if not entries:
        print("[pass 0] No entries to align.")
        return {}, chain_residues_cache

    print(f"[pass 0] Running mmseqs easy-search on {len(entries)} queries...")
    tmp_dir = Path(cfg.output_dir) / "mmseqs_tmp"
    pre_alignment = run_mmseqs_alignment_batch(
        entries=entries,
        mmseqs_bin=cfg.mmseqs_bin,
        tmp_dir=tmp_dir,
        min_identity=cfg.min_alignment_identity,
        threads=int(cfg.mmseqs_threads),
    )
    print(
        f"[pass 0] Aligned {len(pre_alignment)} / {len(entries)} chains "
        f"(threshold >= {cfg.min_alignment_identity})"
    )
    return pre_alignment, chain_residues_cache


###########################################################
# Pass 1 helpers
###########################################################


def process_single_pdb(
    row: pd.Series,
    model: torch.nn.Module,
    cfg: DictConfig,
    pre_alignment: dict,
    chain_residues_cache: dict,
    device: str,
) -> tuple[list[dict], dict]:
    """Process one AlloBench entry. Returns (per-residue rows, entry stats dict).

    Both may be empty if the entry is skipped for any reason; the caller logs
    the status via ``entry_stats['status']``.
    """
    pdb_id = row["pdb_id"]
    cif_path = Path(cfg.pdb_structures_dir) / f"{pdb_id}.cif"
    direction = str(cfg.get("direction", "allo_to_active"))
    entry_stats: dict = {
        "pdb_id": pdb_id,
        "status": "ok",
        "direction": direction,
        "n_active_mapped": 0,
        "n_ortho_residues": 0,
        "n_allo_site_keys": 0,
        "n_allo_site_in_pdb": 0,
        "alignment_identity": np.nan,
        "n_residues_scored": 0,
    }

    if not cif_path.exists():
        entry_stats["status"] = "no_cif"
        return [], entry_stats

    try:
        batch = get_sd_batch(
            pdb_paths=[str(cif_path)],
            sample_is_designed=cfg.input_sample_is_designed,
            cif_parse_cfg=cfg.cif_parse_cfg,
            preprocess_cfg=cfg.preprocess_cfg,
            featurizer_cfg=cfg.featurizer_cfg,
            device=device,
        )
    except Exception as e:
        entry_stats["status"] = f"batch_build_failed: {e.__class__.__name__}"
        return [], entry_stats

    atom_array = batch["atom_array"][0]
    if not is_single_protein_chain(atom_array):
        entry_stats["status"] = "not_single_chain_runtime"
        return [], entry_stats

    n_tokens = int(batch["token_pad_mask"].sum().item())
    max_tokens = cfg.get("max_tokens_for_forward", None)
    if max_tokens is not None and n_tokens > int(max_tokens):
        entry_stats["status"] = f"too_many_tokens:{n_tokens}"
        return [], entry_stats

    allo_atom_mask = build_allosteric_atom_mask(
        atom_array,
        modulator_alias=row["modulator_alias"],
        modulator_chain=row["modulator_chain"],
        modulator_resi=row.get("modulator_resi", None),
    )
    if allo_atom_mask is None or allo_atom_mask.sum() == 0:
        entry_stats["status"] = "no_modulator_atoms"
        return [], entry_stats

    prot_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    protein_chain_ids = {str(c) for c in np.unique(atom_array.chain_id[prot_mask])}

    active_keys: set[tuple[str, int]] = set()
    identities: list[float] = []
    for chain_id in protein_chain_ids:
        key = (pdb_id, chain_id)
        if key not in pre_alignment:
            continue
        alignment = pre_alignment[key]
        identities.append(alignment["pident"])
        chain_residues = chain_residues_cache.get(key, [])
        if not chain_residues:
            continue
        for u_idx in row["active_site_uniprot"]:
            r = uniprot_idx_to_pdb_residue(u_idx, alignment, chain_residues)
            if r is None:
                continue
            active_keys.add((chain_id, int(r[0])))
    if identities:
        entry_stats["alignment_identity"] = float(max(identities))
    entry_stats["n_active_mapped"] = len(active_keys)

    ortho_atom_mask = identify_orthosteric_ligand_atoms(
        atom_array,
        allo_atom_mask=allo_atom_mask,
        active_site_pdb_keys=active_keys,
        active_site_proximity=float(cfg.active_site_proximity),
    )
    if ortho_atom_mask.sum() > 0:
        ortho_labels = compute_ortho_pocket_labels(
            atom_array,
            ortho_atom_mask=ortho_atom_mask,
            cutoff=float(cfg.ortho_cutoff),
        )
        ortho_dists = compute_per_residue_min_dist_to_atoms(atom_array, ortho_atom_mask)
        has_ortho_label = True
        entry_stats["n_ortho_residues"] = int(sum(ortho_labels.values()))
    else:
        ortho_labels = {}
        ortho_dists = {}
        has_ortho_label = False
    allo_dists = compute_per_residue_min_dist_to_atoms(atom_array, allo_atom_mask)

    if direction == "allo_to_active":
        pass_A_zero: np.ndarray | None = None
        pass_B_zero: np.ndarray | None = allo_atom_mask
    elif direction == "ortho_to_allo":
        if ortho_atom_mask.sum() == 0:
            entry_stats["status"] = "no_ortho_ligand"
            return [], entry_stats
        pass_A_zero = allo_atom_mask
        pass_B_zero = allo_atom_mask | ortho_atom_mask
    else:
        raise ValueError(f"unknown direction: {direction}")

    try:
        batch_A, batch_B = prepare_forward_passes(
            batch,
            pass_A_zero_atoms=pass_A_zero,
            pass_B_zero_atoms=pass_B_zero,
        )
    except ValueError as e:
        entry_stats["status"] = f"prepare_failed: {e.__class__.__name__}"
        return [], entry_stats

    try:
        potts_A = run_potts_forward_prepared(model, batch_A)
        potts_B = run_potts_forward_prepared(model, batch_B)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        entry_stats["status"] = "oom"
        return [], entry_stats
    except Exception as e:
        entry_stats["status"] = f"forward_failed: {e.__class__.__name__}"
        return [], entry_stats

    if not torch.equal(potts_A["edge_idx"], potts_B["edge_idx"]):
        entry_stats["status"] = "edge_idx_mismatch"
        return [], entry_stats

    deltas = compute_potts_deltas(potts_A, potts_B)
    mask_i = deltas["mask_i"]
    token_info = map_token_to_residue_info(atom_array)

    allo_site_keys = row.get("allo_site_pdb_keys", set()) or set()
    if not isinstance(allo_site_keys, set):
        allo_site_keys = set(allo_site_keys)
    has_allo_site_label = len(allo_site_keys) > 0
    entry_stats["n_allo_site_keys"] = len(allo_site_keys)

    pdb_chain_res_set = {
        (str(atom_array.chain_id[i]), int(atom_array.res_id[i]))
        for i in np.where(prot_mask)[0]
    }
    entry_stats["n_allo_site_in_pdb"] = len(allo_site_keys & pdb_chain_res_set)

    rows: list[dict] = []
    for t_idx in range(len(mask_i)):
        if not mask_i[t_idx].bool().item():
            continue
        if t_idx >= len(token_info):
            continue
        cid, rid, rname = token_info[t_idx]
        rows.append(
            {
                "pdb_id": pdb_id,
                "token_idx": t_idx,
                "chain_id": cid,
                "res_id": rid,
                "res_name": rname,
                "delta_h": float(deltas["delta_h"][t_idx].item()),
                "delta_J": float(deltas["delta_J"][t_idx].item()),
                "delta_combined": float(deltas["delta_combined"][t_idx].item()),
                "dist_to_ortho": float(ortho_dists.get((cid, rid), np.nan)),
                "dist_to_allo": float(allo_dists.get((cid, rid), np.nan)),
                "is_active_site": bool((cid, rid) in active_keys),
                "has_active_site_label": bool(len(active_keys) > 0),
                "is_ortho_6A": bool(ortho_labels.get((cid, rid), False)),
                "has_ortho_label": bool(has_ortho_label),
                "is_allo_site": bool((cid, rid) in allo_site_keys),
                "has_allo_site_label": bool(has_allo_site_label),
            }
        )

    entry_stats["n_residues_scored"] = len(rows)
    if not rows:
        entry_stats["status"] = "no_valid_residues"
    return rows, entry_stats


###########################################################
# Plotting
###########################################################


def _f1_iso_curves(ax, vals=(0.2, 0.4, 0.6, 0.8)):
    for f1_val in vals:
        r = np.linspace(0.01, 1, 100)
        denom = 2 * r - f1_val
        with np.errstate(divide="ignore", invalid="ignore"):
            p = f1_val * r / denom
        valid = np.isfinite(p) & (p > 0) & (p <= 1)
        ax.plot(r[valid], p[valid], "--", color="gray", alpha=0.25, lw=0.8)


def _per_pdb_normalize(df: pd.DataFrame, score_col: str) -> np.ndarray:
    return df.groupby("pdb_id")[score_col].transform(
        lambda x: x / max(float(x.max()), 1e-8)
    ).to_numpy()


def plot_pr_curves(
    df: pd.DataFrame,
    target_col: str,
    mask_col: str,
    out_path: Path,
    title: str,
) -> dict[str, float]:
    """Plot PR for delta_h / delta_J / delta_combined, raw and per-PDB normalised."""
    sub = df[df[mask_col]].copy()
    if len(sub) == 0 or sub[target_col].sum() == 0:
        print(f"  [plot_pr_curves] {out_path.name}: no positives, skipping")
        return {}

    y = sub[target_col].astype(int).to_numpy()
    base_rate = y.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ["delta_h", "delta_J", "delta_combined"]
    colors = {"delta_h": "#1f77b4", "delta_J": "#d62728", "delta_combined": "#2ca02c"}
    aucs: dict[str, float] = {}

    for score_col in metrics:
        scores = sub[score_col].to_numpy()
        p, r, _ = precision_recall_curve(y, scores)
        pr_auc = auc(r, p)
        aucs[f"raw_{score_col}"] = float(pr_auc)
        axes[0].plot(r, p, label=f"{score_col} (AUC={pr_auc:.3f})", color=colors[score_col], lw=2)

        norm_scores = _per_pdb_normalize(sub, score_col)
        p2, r2, _ = precision_recall_curve(y, norm_scores)
        pr_auc2 = auc(r2, p2)
        aucs[f"norm_{score_col}"] = float(pr_auc2)
        axes[1].plot(r2, p2, label=f"{score_col} (AUC={pr_auc2:.3f})", color=colors[score_col], lw=2)

    for ax, label in zip(axes, ["Global", "Per-PDB max-normalised"]):
        ax.axhline(base_rate, color="gray", ls=":", lw=1, label=f"base={base_rate:.3f}")
        _f1_iso_curves(ax)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_title(label)
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")
    return aucs


def _mad_threshold(
    df: pd.DataFrame,
    group_cols: list[str],
    score_col: str,
    k: float,
    sigma_factor: float = 1.4826,
) -> pd.Series:
    """For each ``group_cols`` group, return threshold = median + k * sigma_factor * MAD over ``score_col``."""
    def _thr(x: pd.Series) -> float:
        v = x.to_numpy()
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        return med + k * sigma_factor * mad
    return df.groupby(group_cols)[score_col].transform(lambda x: _thr(x))


def plot_mad_operating_points(
    df: pd.DataFrame,
    target_col: str,
    mask_col: str,
    out_path: Path,
    title: str,
    k_values: list[float],
    sigma_factor: float = 1.4826,
    score_col: str = "delta_J",
) -> pd.DataFrame:
    """For each k in ``k_values``, classify per-PDB residues as positive if
    ``score_col`` >= median + k * sigma_factor * MAD, then aggregate TP/FP/FN
    across all PDBs and plot precision & recall vs k alongside the full PR curve.
    """
    sub = df[df[mask_col]].copy()
    if len(sub) == 0 or sub[target_col].sum() == 0:
        print(f"  [plot_mad] {out_path.name}: no positives, skipping")
        return pd.DataFrame()

    y = sub[target_col].astype(int).to_numpy()
    scores = sub[score_col].to_numpy()
    p_curve, r_curve, _ = precision_recall_curve(y, scores)
    pr_auc = auc(r_curve, p_curve)

    rows = []
    for k in k_values:
        thr = _mad_threshold(sub, ["pdb_id"], score_col, k=k, sigma_factor=sigma_factor).to_numpy()
        predicted = sub[score_col].to_numpy() >= thr
        tp = int(((predicted == 1) & (y == 1)).sum())
        fp = int(((predicted == 1) & (y == 0)).sum())
        fn = int(((predicted == 0) & (y == 1)).sum())
        n_pred = int(predicted.sum())
        precision_k = tp / max(n_pred, 1) if n_pred > 0 else np.nan
        recall_k = tp / max(tp + fn, 1)
        f1_k = (2 * precision_k * recall_k / (precision_k + recall_k)) if (precision_k and recall_k) else np.nan
        rows.append(
            {
                "k": float(k),
                "n_pred": n_pred,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision_k,
                "recall": recall_k,
                "f1": f1_k,
            }
        )
    mad_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(r_curve, p_curve, color="steelblue", lw=2, label=f"{score_col} (AUC={pr_auc:.3f})")
    axes[0].axhline(float(y.mean()), color="gray", ls=":", lw=1, label=f"base={y.mean():.3f}")
    _f1_iso_curves(axes[0])
    for _, r in mad_df.iterrows():
        if np.isnan(r["precision"]):
            continue
        axes[0].scatter(r["recall"], r["precision"], s=30, zorder=5, color="red")
        axes[0].annotate(f"k={r['k']:g}", (r["recall"], r["precision"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("PR curve with MAD operating points")
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].plot(mad_df["k"], mad_df["precision"], "o-", label="precision", color="tab:blue")
    axes[1].plot(mad_df["k"], mad_df["recall"], "o-", label="recall", color="tab:orange")
    axes[1].plot(mad_df["k"], mad_df["f1"], "o-", label="F1", color="tab:green")
    axes[1].set_xlabel("k (MAD units above median)")
    axes[1].set_ylabel("value")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_title("MAD threshold sweep")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")
    return mad_df


###########################################################
# Distance-binned MAD
###########################################################


def binned_mad_analysis(
    df: pd.DataFrame,
    *,
    target_col: str,
    mask_col: str,
    dist_col: str,
    k_values: list[float],
    n_bins: int = 8,
    sigma_factor: float = 1.4826,
    score_col: str = "delta_J",
) -> pd.DataFrame:
    """Per-PDB distance-quantile bins, per-bin MAD threshold, aggregate TP/FP/FN per bin per k.

    The locality envelope of ``score_col`` (decay with ``dist_col``) is suppressed by
    computing the MAD baseline within each distance shell. The resulting (bin_idx, k)
    cells expose residuals: cells with ``lift > 1`` indicate that the score is
    anomalously high *for its distance shell*, which is the long-range coupling test.
    """
    sub = df[df[mask_col] & df[dist_col].notna()].copy()
    if sub.empty or sub[target_col].sum() == 0:
        return pd.DataFrame()

    def _bin(x: pd.Series) -> pd.Series:
        return pd.qcut(x.rank(method="first"), q=n_bins, labels=False, duplicates="drop")

    sub["dist_bin"] = sub.groupby("pdb_id")[dist_col].transform(_bin)
    sub = sub[sub["dist_bin"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["dist_bin"] = sub["dist_bin"].astype(int)

    rows = []
    for k in k_values:
        thr = _mad_threshold(
            sub, ["pdb_id", "dist_bin"], score_col, k=k, sigma_factor=sigma_factor
        ).to_numpy()
        pred_all = (sub[score_col].to_numpy() >= thr).astype(int)
        sub["_pred_k"] = pred_all
        for b, bin_df in sub.groupby("dist_bin"):
            y = bin_df[target_col].to_numpy().astype(int)
            p = bin_df["_pred_k"].to_numpy().astype(int)
            n_res = int(len(bin_df))
            n_pos = int(y.sum())
            n_pred = int(p.sum())
            tp = int(((p == 1) & (y == 1)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            base = float(n_pos / n_res) if n_res > 0 else np.nan
            precision = float(tp / n_pred) if n_pred > 0 else np.nan
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
            if (
                not np.isnan(precision)
                and not np.isnan(recall)
                and (precision + recall) > 0
            ):
                f1 = float(2 * precision * recall / (precision + recall))
            else:
                f1 = np.nan
            lift = (
                float(precision / base)
                if (base and base > 0 and not np.isnan(precision))
                else np.nan
            )
            d_med = bin_df.groupby("pdb_id")[dist_col].agg(["min", "max"]).median()
            rows.append(
                {
                    "bin_idx": int(b),
                    "k": float(k),
                    "n_residues": n_res,
                    "n_pos": n_pos,
                    "n_pred": n_pred,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "base_rate_in_bin": base,
                    "lift": lift,
                    "dist_min_med": float(d_med["min"]),
                    "dist_max_med": float(d_med["max"]),
                }
            )
    sub.drop(columns=["_pred_k"], inplace=True, errors="ignore")
    return pd.DataFrame(rows).sort_values(["k", "bin_idx"]).reset_index(drop=True)


def plot_binned_mad(
    binned: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    """Heatmap of binned MAD lift over (bin_idx × k)."""
    import matplotlib.colors as mcolors

    if binned.empty:
        print(f"  [plot_binned_mad] {out_path.name}: empty, skipping")
        return

    pivot_lift = binned.pivot(index="bin_idx", columns="k", values="lift")
    pivot_prec = binned.pivot(index="bin_idx", columns="k", values="precision")
    pivot_npred = binned.pivot(index="bin_idx", columns="k", values="n_pred")

    fig_w = max(6.0, 1.0 * len(pivot_lift.columns) + 4.0)
    fig_h = max(4.0, 0.7 * len(pivot_lift.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    arr = pivot_lift.values.astype(float)
    masked = np.ma.masked_invalid(arr)
    cmap = matplotlib.colormaps.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgrey")
    norm = mcolors.LogNorm(vmin=0.5, vmax=4.0)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(pivot_lift.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot_lift.columns])
    ax.set_xlabel("k (MAD units above per-bin median)")

    bin_dist = (
        binned.drop_duplicates("bin_idx").set_index("bin_idx").sort_index()
    )
    yticklabels = []
    for b in pivot_lift.index:
        d_min = bin_dist.loc[b, "dist_min_med"]
        d_max = bin_dist.loc[b, "dist_max_med"]
        yticklabels.append(f"bin {b}\n[{d_min:.0f}–{d_max:.0f} Å]")
    ax.set_yticks(range(len(pivot_lift.index)))
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_ylabel("distance bin (closest → farthest)")

    for i, b in enumerate(pivot_lift.index):
        for j, k in enumerate(pivot_lift.columns):
            val = pivot_lift.loc[b, k]
            n_pred = pivot_npred.loc[b, k]
            prec = pivot_prec.loc[b, k]
            if pd.isna(val) or pd.isna(prec) or n_pred == 0:
                txt = "–"
                color = "black"
            else:
                txt = f"{prec * 100:.0f}%\n(n={int(n_pred)})"
                color = "black" if val < 2.0 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("lift = precision / base_rate_in_bin")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


###########################################################
# Main
###########################################################


@hydra.main(
    config_path="../../configs_local/eval/allostery",
    config_name="allobench_pr",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    direction = str(cfg.get("direction", "allo_to_active"))
    if direction not in ("allo_to_active", "ortho_to_allo"):
        raise ValueError(f"unknown direction: {direction}")

    output_dir = Path(cfg.output_dir) / direction
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Direction: {direction}")
    print(f"Output dir: {output_dir}")

    print("Loading AlloBench metadata...")
    meta = load_allobench(
        allobench_csv=cfg.allobench_csv,
        asd_csv=cfg.asd_csv,
        monomer_only=bool(cfg.monomer_only),
    )
    if cfg.debug and cfg.num_debug_samples:
        meta = meta.head(int(cfg.num_debug_samples)).reset_index(drop=True)
    print(f"  {len(meta)} entries (debug={cfg.debug})")

    pre_alignment, chain_residues_cache = run_pass0_alignment(meta, cfg)

    print("Loading model...")
    lit_model = LitSeqDenoiser.load_from_checkpoint(cfg.ckpt_path).eval()
    model = lit_model.model.to(device)

    all_rows: list[dict] = []
    all_stats: list[dict] = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="pass 1"):
        try:
            rows, stats = process_single_pdb(
                row=row,
                model=model,
                cfg=cfg,
                pre_alignment=pre_alignment,
                chain_residues_cache=chain_residues_cache,
                device=device,
            )
        except Exception as e:
            print(f"  [{row['pdb_id']}] unhandled error: {e}")
            traceback.print_exc()
            stats = {
                "pdb_id": row["pdb_id"],
                "status": f"unhandled:{e.__class__.__name__}",
                "direction": direction,
                "n_active_mapped": 0,
                "n_ortho_residues": 0,
                "n_allo_site_keys": 0,
                "n_allo_site_in_pdb": 0,
                "alignment_identity": np.nan,
                "n_residues_scored": 0,
            }
            rows = []
        all_stats.append(stats)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(output_dir / "per_entry_stats.csv", index=False)

    skipped = stats_df[stats_df["status"] != "ok"]
    if not skipped.empty:
        skipped.to_csv(output_dir / "skipped.csv", index=False)
    status_counts = stats_df["status"].value_counts().to_dict()
    print(f"Entry status: {status_counts}")

    if df.empty:
        print("No per-residue rows produced. Exiting before plots.")
        return

    df.to_csv(output_dir / "results.csv", index=False)
    print(
        f"Saved results.csv: {len(df)} residues from {df['pdb_id'].nunique()} PDBs "
        f"(active_label: {int(df['has_active_site_label'].sum())}, "
        f"ortho_label: {int(df['has_ortho_label'].sum())}, "
        f"allo_label: {int(df['has_allo_site_label'].sum())})"
    )

    print("Plotting...")
    k_values = list(cfg.mad_k_values)

    pr_auc_active = plot_pr_curves(
        df, target_col="is_active_site", mask_col="has_active_site_label",
        out_path=output_dir / "pr_active_site.png",
        title=f"AlloBench active site detection [{direction}]",
    )
    pr_auc_ortho = plot_pr_curves(
        df, target_col="is_ortho_6A", mask_col="has_ortho_label",
        out_path=output_dir / "pr_ortho_6A.png",
        title=f"AlloBench orthosteric pocket (<6A) detection [{direction}]",
    )
    pr_auc_allo = plot_pr_curves(
        df, target_col="is_allo_site", mask_col="has_allo_site_label",
        out_path=output_dir / "pr_allo_site.png",
        title=f"AlloBench allosteric site detection [{direction}]",
    )

    mad_active = plot_mad_operating_points(
        df, target_col="is_active_site", mask_col="has_active_site_label",
        out_path=output_dir / "mad_active_site.png",
        title=f"Active site: per-PDB MAD threshold [{direction}]",
        k_values=k_values,
        sigma_factor=float(cfg.mad_sigma_factor),
    )
    mad_ortho = plot_mad_operating_points(
        df, target_col="is_ortho_6A", mask_col="has_ortho_label",
        out_path=output_dir / "mad_ortho_6A.png",
        title=f"Orthosteric pocket (<6A): per-PDB MAD threshold [{direction}]",
        k_values=k_values,
        sigma_factor=float(cfg.mad_sigma_factor),
    )
    mad_allo = plot_mad_operating_points(
        df, target_col="is_allo_site", mask_col="has_allo_site_label",
        out_path=output_dir / "mad_allo_site.png",
        title=f"Allosteric site: per-PDB MAD threshold [{direction}]",
        k_values=k_values,
        sigma_factor=float(cfg.mad_sigma_factor),
    )

    if not mad_active.empty:
        mad_active.to_csv(output_dir / "mad_active_site.csv", index=False)
    if not mad_ortho.empty:
        mad_ortho.to_csv(output_dir / "mad_ortho_6A.csv", index=False)
    if not mad_allo.empty:
        mad_allo.to_csv(output_dir / "mad_allo_site.csv", index=False)

    print("Distance-binned MAD...")
    dist_col_for_dir = "dist_to_ortho" if direction == "ortho_to_allo" else "dist_to_allo"
    n_bins = int(cfg.get("n_distance_bins", 8))
    binned_results: dict[str, pd.DataFrame] = {}
    for target_col, mask_col, name in [
        ("is_active_site", "has_active_site_label", "active_site"),
        ("is_ortho_6A", "has_ortho_label", "ortho_6A"),
        ("is_allo_site", "has_allo_site_label", "allo_site"),
    ]:
        if dist_col_for_dir not in df.columns:
            continue
        binned = binned_mad_analysis(
            df,
            target_col=target_col,
            mask_col=mask_col,
            dist_col=dist_col_for_dir,
            k_values=k_values,
            n_bins=n_bins,
            sigma_factor=float(cfg.mad_sigma_factor),
        )
        if binned.empty:
            continue
        binned.to_csv(output_dir / f"mad_{name}_binned.csv", index=False)
        plot_binned_mad(
            binned,
            out_path=output_dir / f"mad_{name}_binned.png",
            title=(
                f"{name}: binned MAD lift "
                f"[{direction} | bin by {dist_col_for_dir} | K={n_bins}]"
            ),
        )
        binned_results[name] = binned

    far_half_lift = np.nan
    if "allo_site" in binned_results:
        allo_binned = binned_results["allo_site"]
        far_half = allo_binned[allo_binned["bin_idx"] >= (n_bins // 2)]
        if not far_half.empty and far_half["lift"].notna().any():
            far_half_lift = float(far_half["lift"].max())

    summary = {
        "direction": direction,
        "n_entries": int(len(stats_df)),
        "n_entries_ok": int((stats_df["status"] == "ok").sum()),
        "n_residues": int(len(df)),
        "n_active_positive": int(df["is_active_site"].sum()),
        "n_ortho_positive": int(df["is_ortho_6A"].sum()),
        "n_allo_site_positive": int(df["is_allo_site"].sum()),
        "pr_auc_active": pr_auc_active,
        "pr_auc_ortho": pr_auc_ortho,
        "pr_auc_allo": pr_auc_allo,
        "binned_mad_n_bins": n_bins,
        "binned_mad_dist_col": dist_col_for_dir,
        "binned_mad_allo_max_lift_far_half": far_half_lift,
    }
    pd.Series(summary, name="value").to_csv(output_dir / "summary.csv")
    print(f"Summary: {summary}")
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
