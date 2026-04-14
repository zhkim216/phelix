"""
Pareto-front plotting utilities for classifier-free guidance sweeps.

Given a `guidance_metrics*.csv` produced by `redesign_with_lcaliby` with
columns ``example_id, designed_sample_id, gamma, U_cond, U_uncond, U_mixed``,
this module computes and plots the (U_uncond, U_cond) Pareto front traced
out by the gamma sweep:

- ``U_cond`` — Potts energy of the designed sequence evaluated with the
  ligand-conditioned parameters ("ligand fit").
- ``U_uncond`` — Potts energy of the same sequence evaluated with the
  ligand-masked parameters ("ligand-free stability").

Both objectives are minimised; the Pareto front is the set of samples that
no other sample strictly dominates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def load_guidance_metrics(csv_paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Concatenate one or more ``guidance_metrics*.csv`` files.

    Args:
        csv_paths: a single path or an iterable of paths. Paths may be
            strings or :class:`pathlib.Path`.

    Returns:
        A single :class:`pandas.DataFrame` with the union of rows. An
        extra ``source_csv`` column is added so users can track which
        file each row came from.
    """
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    csv_paths = [Path(p) for p in csv_paths]

    frames = []
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"guidance metrics csv not found: {p}")
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)
    if not frames:
        raise ValueError("No csv paths were provided.")
    combined = pd.concat(frames, ignore_index=True)

    required = {"example_id", "gamma", "U_cond", "U_uncond"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(
            f"guidance metrics csv is missing required columns: {missing}"
        )
    return combined


def compute_pareto_front(
    df: pd.DataFrame, x: str = "U_uncond", y: str = "U_cond"
) -> pd.DataFrame:
    """Return the Pareto-optimal subset (minimise both ``x`` and ``y``).

    Uses an O(n^2) dominance filter, which is fast enough for the
    expected sweep sizes (~10^4 rows or fewer).
    """
    if len(df) == 0:
        return df.copy()
    arr = df[[x, y]].to_numpy()
    n = arr.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        xi, yi = arr[i]
        # j dominates i iff both coordinates are <= i's AND at least one is strictly <.
        le = (arr[:, 0] <= xi) & (arr[:, 1] <= yi)
        lt = (arr[:, 0] < xi) | (arr[:, 1] < yi)
        dominators = le & lt
        dominators[i] = False
        if dominators.any():
            dominated[i] = True
    front = df.loc[~dominated].copy()
    return front.sort_values(by=x).reset_index(drop=True)


def _scatter_with_front(
    ax: plt.Axes,
    sub_df: pd.DataFrame,
    *,
    x: str = "U_uncond",
    y: str = "U_cond",
    title: str | None = None,
    cmap: str = "viridis",
) -> None:
    """Draw a (U_uncond, U_cond) scatter coloured by gamma + Pareto front overlay."""
    gamma = sub_df["gamma"].to_numpy(dtype=float)
    xs = sub_df[x].to_numpy(dtype=float)
    ys = sub_df[y].to_numpy(dtype=float)

    sc = ax.scatter(
        xs, ys, c=gamma, cmap=cmap,
        vmin=float(np.nanmin(gamma)) if np.isfinite(gamma).any() else 0.0,
        vmax=float(np.nanmax(gamma)) if np.isfinite(gamma).any() else 1.0,
        s=18, edgecolors="none", alpha=0.85,
    )

    front = compute_pareto_front(sub_df, x=x, y=y)
    if len(front) > 0:
        ax.plot(
            front[x].to_numpy(), front[y].to_numpy(),
            color="crimson", linewidth=1.2, marker="o", markersize=4,
            label="Pareto front",
        )

    ax.set_xlabel(f"{x} (ligand-free energy)")
    ax.set_ylabel(f"{y} (ligand-conditioned energy)")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return sc


def plot_guidance_pareto_per_example(
    df: pd.DataFrame,
    out_png: str | Path,
    *,
    x: str = "U_uncond",
    y: str = "U_cond",
    max_cols: int = 4,
) -> Path:
    """One subplot per ``example_id``; each subplot shows its own Pareto front.

    Returns the output PNG path.
    """
    example_ids = sorted(df["example_id"].unique())
    n_examples = len(example_ids)
    if n_examples == 0:
        raise ValueError("No example_ids found in guidance metrics dataframe.")

    n_cols = min(max_cols, n_examples)
    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False,
    )
    last_sc = None
    for idx, example_id in enumerate(example_ids):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        sub = df.loc[df["example_id"] == example_id]
        last_sc = _scatter_with_front(ax, sub, x=x, y=y, title=str(example_id))

    # Blank any unused axes.
    for idx in range(n_examples, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    if last_sc is not None:
        cbar = fig.colorbar(
            last_sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02
        )
        cbar.set_label("gamma")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_guidance_pareto_aggregated(
    df: pd.DataFrame,
    out_png: str | Path,
    *,
    x: str = "U_uncond",
    y: str = "U_cond",
) -> Path:
    """Single scatter of all samples across all examples, coloured by gamma.

    Also draws the global Pareto front.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = _scatter_with_front(ax, df, x=x, y=y, title="All examples (aggregated)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("gamma")

    # Add a legend entry for the Pareto front.
    handles = [
        Line2D([0], [0], color="crimson", linewidth=1.2, marker="o",
               markersize=4, label="Pareto front"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_guidance_pareto(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    x: str = "U_uncond",
    y: str = "U_cond",
    mode: str = "both",
) -> list[Path]:
    """Convenience wrapper that emits per-example + aggregated plots.

    Args:
        df: guidance metrics dataframe (from :func:`load_guidance_metrics`).
        out_dir: directory to write plots into; created if missing.
        mode: one of ``"both"``, ``"per_example"``, ``"aggregated"``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if mode in ("both", "per_example"):
        written.append(
            plot_guidance_pareto_per_example(
                df, out_dir / "pareto_per_example.png", x=x, y=y,
            )
        )
    if mode in ("both", "aggregated"):
        written.append(
            plot_guidance_pareto_aggregated(
                df, out_dir / "pareto_aggregated.png", x=x, y=y,
            )
        )
    if mode not in ("both", "per_example", "aggregated"):
        raise ValueError(f"Unknown mode={mode!r}")
    return written


def plot_guidance_median_curve(
    df: pd.DataFrame,
    out_png: str | Path,
    *,
    x: str = "U_uncond",
    y: str = "U_cond",
) -> Path:
    """Two-panel summary of how ``x`` / ``y`` move as ``gamma`` is swept.

    Left panel: median ± IQR of ``x`` and ``y`` as a function of gamma,
    aggregated over all example_ids in ``df``.

    Right panel: the median (x, y) trajectory, one point per gamma value,
    coloured by gamma and annotated with the gamma value next to each point.
    This is the same construction as the ad-hoc median_curve plot from the
    prior guidance sweep, promoted here to a first-class helper.
    """
    if len(df) == 0:
        raise ValueError("Cannot plot median curve on an empty dataframe.")

    g = df.groupby("gamma")
    stat = pd.DataFrame(
        {
            f"{y}_med": g[y].median(),
            f"{y}_q25": g[y].quantile(0.25),
            f"{y}_q75": g[y].quantile(0.75),
            f"{x}_med": g[x].median(),
            f"{x}_q25": g[x].quantile(0.25),
            f"{x}_q75": g[x].quantile(0.75),
        }
    ).reset_index().sort_values("gamma")

    gammas = stat["gamma"].to_numpy(dtype=float)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: median ± IQR vs gamma, twin axis for x and y.
    ax_l.plot(gammas, stat[f"{y}_med"], color="tab:blue", marker="o", label=f"{y} median")
    ax_l.fill_between(
        gammas, stat[f"{y}_q25"], stat[f"{y}_q75"],
        color="tab:blue", alpha=0.2, label=f"{y} IQR",
    )
    ax_l.set_xlabel("gamma")
    ax_l.set_ylabel(y, color="tab:blue")
    ax_l.tick_params(axis="y", labelcolor="tab:blue")
    ax_l.grid(True, alpha=0.3)

    ax_l_twin = ax_l.twinx()
    ax_l_twin.plot(
        gammas, stat[f"{x}_med"], color="tab:red", marker="s", label=f"{x} median",
    )
    ax_l_twin.fill_between(
        gammas, stat[f"{x}_q25"], stat[f"{x}_q75"],
        color="tab:red", alpha=0.2, label=f"{x} IQR",
    )
    ax_l_twin.set_ylabel(x, color="tab:red")
    ax_l_twin.tick_params(axis="y", labelcolor="tab:red")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_l_twin.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best", frameon=True)
    ax_l.set_title("Median ± IQR vs gamma")

    # Right: median (x, y) trajectory coloured by gamma.
    xs_med = stat[f"{x}_med"].to_numpy(dtype=float)
    ys_med = stat[f"{y}_med"].to_numpy(dtype=float)
    ax_r.plot(xs_med, ys_med, color="0.5", linewidth=1.0, alpha=0.6, zorder=1)
    sc = ax_r.scatter(
        xs_med, ys_med, c=gammas, cmap="viridis",
        vmin=float(np.nanmin(gammas)) if np.isfinite(gammas).any() else 0.0,
        vmax=float(np.nanmax(gammas)) if np.isfinite(gammas).any() else 1.0,
        s=80, edgecolors="black", linewidths=0.6, zorder=2,
    )
    for gi, (xi, yi) in enumerate(zip(xs_med, ys_med)):
        ax_r.annotate(
            f"{gammas[gi]:.2f}", (xi, yi),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8,
        )
    ax_r.set_xlabel(f"{x} (median across examples)")
    ax_r.set_ylabel(f"{y} (median across examples)")
    ax_r.set_title("Median Pareto trajectory")
    ax_r.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax_r, shrink=0.85, pad=0.02)
    cbar.set_label("gamma")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# (subdir, x_column, y_column) triples emitted by plot_guidance_pareto_all_modes.
PARETO_MODES: list[tuple[str, str, str]] = [
    ("pareto", "U_uncond", "U_cond"),
    ("pareto_per_res", "U_uncond_per_res", "U_cond_per_res"),
    ("pareto_pocket", "U_uncond_pocket", "U_cond_pocket"),
    ("pareto_pocket_per_res", "U_uncond_pocket_per_res", "U_cond_pocket_per_res"),
]


def plot_guidance_pareto_all_modes(
    df: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, list[Path]]:
    """Emit the full Pareto bundle for each (x, y) pair in :data:`PARETO_MODES`.

    For each mode we write ``pareto_per_example.png``,
    ``pareto_aggregated.png``, and ``median_curve.png`` into its own subdir.
    Modes whose required columns are not present in ``df`` are skipped with
    a printed warning (so older CSVs still produce the total-energy plots).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path]] = {}
    for subdir, x_col, y_col in PARETO_MODES:
        if x_col not in df.columns or y_col not in df.columns:
            print(
                f"[warn] Skipping {subdir}: columns "
                f"{x_col!r}/{y_col!r} not present in dataframe."
            )
            continue
        sub_out = out_dir / subdir
        sub_out.mkdir(parents=True, exist_ok=True)

        mode_written: list[Path] = []
        mode_written.extend(plot_guidance_pareto(df, sub_out, x=x_col, y=y_col, mode="both"))
        mode_written.append(
            plot_guidance_median_curve(df, sub_out / "median_curve.png", x=x_col, y=y_col)
        )
        written[subdir] = mode_written
    return written
