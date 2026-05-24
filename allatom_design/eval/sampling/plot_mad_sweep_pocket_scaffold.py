"""
MAD k-sweep plots for pocket/scaffold selection using Potts delta metrics.

For each metric in {delta_h, delta_J, delta_h + delta_J}:
  - pooled global MAD statistics over all valid residues
  - threshold(k) = median + k * 1.4826 * MAD
  - pocket view:    pred = delta >= threshold, GT = min_distance <  d_plus
  - scaffold view:  pred = delta <  threshold, GT = min_distance >  d_minus

Outputs (under cfg.output_dir):
  mad_sweep_aggregate.csv                         — one row per (metric, view, k, distance_cutoff)
  {h,J,hJ}/mad_sweep_prcurve_pocket.png           — PR curve (6 pocket cutoffs, F1 iso-contours, k labels)
  {h,J,hJ}/mad_sweep_prcurve_scaffold.png         — PR curve (3 scaffold cutoffs, F1 iso-contours, k labels)
  {h,J,hJ}/mad_sweep_ksweep_pocket.png            — 3-panel P/R/F1 vs MAD k (pocket cutoffs)
  {h,J,hJ}/mad_sweep_ksweep_scaffold.png          — 3-panel P/R/F1 vs MAD k (scaffold cutoffs)

Usage:
    python -m allatom_design.eval.sampling.plot_mad_sweep_pocket_scaffold \
        --config-name plot_mad_sweep_260410
"""

from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig


METRICS = {
    "h":  ("delta_h",  "Δh"),
    "J":  ("delta_J",  "ΔJ"),
    "hJ": ("delta_hJ", "Δ(h+j)"),
}

POCKET_COLORS = {
    4.0:  "#d62728",  # red
    5.0:  "#ff7f0e",  # orange
    6.0:  "#2ca02c",  # green
    8.0:  "#1f77b4",  # blue
    10.0: "#9467bd",  # purple
    12.0: "#8c564b",  # brown
}

SCAFFOLD_COLORS = {
    10.0: "#d62728",  # red
    12.0: "#ff7f0e",  # orange
    14.0: "#2ca02c",  # green
}

POCKET_LABELS = {
    4.0:  "< 4Å",
    5.0:  "< 5Å",
    6.0:  "< 6Å",
    8.0:  "< 8Å (2nd shell)",
    10.0: "< 10Å",
    12.0: "< 12Å (extended)",
}

SCAFFOLD_LABELS = {
    10.0: "> 10Å",
    12.0: "> 12Å",
    14.0: "> 14Å",
}


def pooled_mad_stats(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma_est = 1.4826 * mad
    return median, sigma_est


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 1e-12 else 0.0
    return precision, recall, f1


def compute_sweep(
    df: pd.DataFrame,
    metric_col: str,
    k_grid: np.ndarray,
    pocket_distances: list[float],
    scaffold_distances: list[float],
) -> pd.DataFrame:
    """Run pooled-MAD sweep and return long-form DataFrame of metrics."""
    values = df[metric_col].to_numpy()
    min_dist = df["min_distance"].to_numpy()

    median, sigma = pooled_mad_stats(values)

    rows = []
    for k in k_grid:
        threshold = median + float(k) * sigma

        pred_pocket = values >= threshold
        pred_scaffold = values < threshold
        n_pred_pocket = int(pred_pocket.sum())
        n_pred_scaffold = int(pred_scaffold.sum())

        for d in pocket_distances:
            gt_pocket = min_dist < d
            tp = int((pred_pocket & gt_pocket).sum())
            fp = int((pred_pocket & ~gt_pocket).sum())
            fn = int((~pred_pocket & gt_pocket).sum())
            p, r, f = prf1(tp, fp, fn)
            rows.append({
                "view": "pocket",
                "k": float(k),
                "threshold": threshold,
                "distance_cutoff": float(d),
                "n_pred_pos": n_pred_pocket,
                "n_gt_pos": int(gt_pocket.sum()),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": p, "recall": r, "f1": f,
            })

        for d in scaffold_distances:
            gt_scaffold = min_dist > d
            tp = int((pred_scaffold & gt_scaffold).sum())
            fp = int((pred_scaffold & ~gt_scaffold).sum())
            fn = int((~pred_scaffold & gt_scaffold).sum())
            p, r, f = prf1(tp, fp, fn)
            rows.append({
                "view": "scaffold",
                "k": float(k),
                "threshold": threshold,
                "distance_cutoff": float(d),
                "n_pred_pos": n_pred_scaffold,
                "n_gt_pos": int(gt_scaffold.sum()),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": p, "recall": r, "f1": f,
            })
    return pd.DataFrame(rows)


def _draw_f1_isocurves(ax, f1_values=(0.2, 0.4, 0.6, 0.8)):
    for f1 in f1_values:
        r = np.linspace(f1 / 2 + 1e-3, 1.0, 200)
        denom = 2 * r - f1
        with np.errstate(divide="ignore", invalid="ignore"):
            p = f1 * r / denom
        valid = np.isfinite(p) & (p > 0) & (p <= 1.05)
        ax.plot(r[valid], p[valid], linestyle=":", color="lightgray", lw=1.0, zorder=1)
        # Label at right edge
        if valid.any():
            r_end = r[valid][-1]
            p_end = p[valid][-1]
            ax.text(min(r_end + 0.005, 1.0), p_end, f"F1={f1}",
                    color="lightgray", fontsize=8, va="center", ha="left")


def _plot_pr_curve(
    sweep: pd.DataFrame,
    view: str,
    metric_label: str,
    distances: list[float],
    colors: dict,
    labels: dict,
    annotation_ks: set[float],
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 8))

    _draw_f1_isocurves(ax)

    sub = sweep[sweep["view"] == view].sort_values("k")
    for d in distances:
        curve = sub[sub["distance_cutoff"] == d].sort_values("k")
        ax.plot(curve["recall"], curve["precision"],
                marker="o", markersize=6, lw=2,
                color=colors[d], label=labels[d], zorder=3)
        for _, row in curve.iterrows():
            k_val = round(float(row["k"]), 2)
            # Match annotation set (tolerate floating-point)
            if any(abs(k_val - ak) < 1e-6 for ak in annotation_ks):
                ax.annotate(f"k={k_val}",
                            xy=(row["recall"], row["precision"]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color="#333333")

    title_view = "Pocket" if view == "pocket" else "Scaffold"
    ax.set_title(f"PR Curve: {metric_label} MAD Sweep with Different {title_view} Definitions",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ksweep(
    sweep: pd.DataFrame,
    view: str,
    metric_label: str,
    distances: list[float],
    colors: dict,
    labels: dict,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panel_specs = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]

    sub = sweep[sweep["view"] == view].sort_values("k")
    for ax, (col, panel_title) in zip(axes, panel_specs):
        for d in distances:
            curve = sub[sub["distance_cutoff"] == d].sort_values("k")
            ax.plot(curve["k"], curve[col],
                    marker="o", markersize=4, lw=1.5,
                    color=colors[d], label=labels[d])
        ax.set_title(f"{panel_title} vs MAD k", fontsize=11)
        ax.set_xlabel("MAD k", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc="best", fontsize=8)

    if view == "pocket":
        suptitle = (f"MAD k Sweep ({metric_label}): Pocket Selection with Different Pocket Definitions\n"
                    f"(predict pocket = residues with {metric_label} ≥ median + k·MAD_σ)")
    else:
        suptitle = (f"MAD k Sweep ({metric_label}): Scaffold Selection with Different Scaffold Definitions\n"
                    f"(predict scaffold = residues with {metric_label} < median + k·MAD_σ)")
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@hydra.main(
    config_path="../../configs_local/eval/sampling",
    config_name="plot_mad_sweep_260410",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    per_residue_csv = Path(cfg.per_residue_csv)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-residue CSV: {per_residue_csv}")
    df = pd.read_csv(per_residue_csv)
    df = df.dropna(subset=["min_distance", "delta_h", "delta_J"]).reset_index(drop=True)
    df["delta_hJ"] = df["delta_h"] + df["delta_J"]
    print(f"  Loaded {len(df)} residues from {df['pdb_id'].nunique()} PDBs")

    k_grid = np.array(list(cfg.k_grid), dtype=float)
    pocket_distances = [float(d) for d in cfg.pocket_distances]
    scaffold_distances = [float(d) for d in cfg.scaffold_distances]
    annotation_ks = set(float(k) for k in cfg.annotation_ks)

    all_rows = []
    for metric_key, (col, metric_label) in METRICS.items():
        print(f"\n=== {metric_key} ({col}) ===")
        median, sigma = pooled_mad_stats(df[col].to_numpy())
        print(f"  pooled median={median:.4f}, MAD_σ={sigma:.4f}")

        sweep = compute_sweep(
            df=df,
            metric_col=col,
            k_grid=k_grid,
            pocket_distances=pocket_distances,
            scaffold_distances=scaffold_distances,
        )
        sweep["metric"] = metric_key
        sweep["metric_col"] = col
        all_rows.append(sweep)

        metric_dir = output_dir / metric_key
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Pocket PR curve
        _plot_pr_curve(
            sweep, view="pocket", metric_label=metric_label,
            distances=pocket_distances, colors=POCKET_COLORS, labels=POCKET_LABELS,
            annotation_ks=annotation_ks,
            output_path=metric_dir / "mad_sweep_prcurve_pocket.png",
        )
        print(f"  saved {metric_dir / 'mad_sweep_prcurve_pocket.png'}")

        # Pocket k-sweep
        _plot_ksweep(
            sweep, view="pocket", metric_label=metric_label,
            distances=pocket_distances, colors=POCKET_COLORS, labels=POCKET_LABELS,
            output_path=metric_dir / "mad_sweep_ksweep_pocket.png",
        )
        print(f"  saved {metric_dir / 'mad_sweep_ksweep_pocket.png'}")

        # Scaffold PR curve
        _plot_pr_curve(
            sweep, view="scaffold", metric_label=metric_label,
            distances=scaffold_distances, colors=SCAFFOLD_COLORS, labels=SCAFFOLD_LABELS,
            annotation_ks=annotation_ks,
            output_path=metric_dir / "mad_sweep_prcurve_scaffold.png",
        )
        print(f"  saved {metric_dir / 'mad_sweep_prcurve_scaffold.png'}")

        # Scaffold k-sweep
        _plot_ksweep(
            sweep, view="scaffold", metric_label=metric_label,
            distances=scaffold_distances, colors=SCAFFOLD_COLORS, labels=SCAFFOLD_LABELS,
            output_path=metric_dir / "mad_sweep_ksweep_scaffold.png",
        )
        print(f"  saved {metric_dir / 'mad_sweep_ksweep_scaffold.png'}")

    # Aggregate CSV
    agg = pd.concat(all_rows, ignore_index=True)
    agg = agg[[
        "metric", "metric_col", "view", "k", "threshold", "distance_cutoff",
        "n_pred_pos", "n_gt_pos", "tp", "fp", "fn", "precision", "recall", "f1",
    ]]
    agg_path = output_dir / "mad_sweep_aggregate.csv"
    agg.to_csv(agg_path, index=False)
    print(f"\nSaved aggregate sweep: {agg_path} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
