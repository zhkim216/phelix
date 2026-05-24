"""
Analyze Potts weight differences (with vs without ligand conditioning)
and their relationship to ligand pocket proximity.

Computes three per-residue delta metrics:
  1. ||Δh_i||_2: L2 norm of site field difference
  2. Σ_j ||ΔJ_ij||_F: sum of Frobenius norms of coupling differences
  3. Combined: normalized sum of (1) and (2)

Plots metrics vs residue-ligand distance and overlap with distance-based pockets.
"""

from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm

from allatom_design.eval.eval_utils.eval_setup_utils import get_pdb_files
from allatom_design.eval.eval_utils.sd_data_utils import get_sd_batch
from allatom_design.eval.eval_utils.potts_utils import (
    run_potts_forward,
    compute_potts_deltas,
    map_token_to_residue_info,
    compute_per_residue_min_distance_to_ligand,
)
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


###########################################################
# Plotting
###########################################################

def plot_metric_vs_distance(df: pd.DataFrame, output_dir: Path):
    """Plot 1: scatter + box plot for each metric vs distance."""
    metrics = ["delta_h", "delta_J", "delta_combined"]
    metric_labels = ["||Δh||₂", "Σⱼ||ΔJ_ij||_F", "Combined"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        # Filter to valid distances
        mask = df["min_distance"] <= df["min_distance"].quantile(0.99)
        sub = df[mask]

        # Scatter
        ax.scatter(sub["min_distance"], sub[metric], alpha=0.15, s=8, c="steelblue")

        # Box plot overlay: bin by distance
        bins = np.arange(0, 21, 2)
        sub = sub.copy()
        sub["dist_bin"] = pd.cut(sub["min_distance"], bins=bins)
        groups = sub.groupby("dist_bin", observed=True)[metric]

        positions = []
        data_to_plot = []
        for name, group in groups:
            if len(group) > 0:
                positions.append(name.mid)
                data_to_plot.append(group.values)

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, positions=positions, widths=1.2,
                            patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor("orange")
                patch.set_alpha(0.5)

        ax.set_xlabel("Min distance to ligand (Å)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Distance")
        ax.set_xlim(0, 20)

    plt.tight_layout()
    fig.savefig(output_dir / "metric_vs_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric_vs_distance.png")


def plot_pocket_overlap(df: pd.DataFrame, pocket_distances: list[float], output_dir: Path):
    """Plot 2: precision-recall curves for each metric × pocket distance."""
    metrics = ["delta_h", "delta_J", "delta_combined"]
    metric_labels = ["||Δh||₂", "Σⱼ||ΔJ_ij||_F", "Combined"]

    fig, axes = plt.subplots(len(metrics), len(pocket_distances),
                             figsize=(5 * len(pocket_distances), 5 * len(metrics)))

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        for j, pd_cutoff in enumerate(pocket_distances):
            ax = axes[i, j]

            # Ground truth: residue within pd_cutoff of ligand
            y_true = (df["min_distance"] < pd_cutoff).astype(int).values
            scores = df[metric].values

            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                ax.text(0.5, 0.5, "No positive/negative\nsamples",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{mlabel}\npocket < {pd_cutoff}Å")
                continue

            precision, recall, thresholds = precision_recall_curve(y_true, scores)
            pr_auc = auc(recall, precision)

            ax.plot(recall, precision, color="steelblue", lw=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{mlabel}\npocket < {pd_cutoff}Å (AUC={pr_auc:.3f})")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)

            # Add F1 iso-curves
            for f1_val in [0.2, 0.4, 0.6, 0.8]:
                r_vals = np.linspace(0.01, 1, 100)
                denom = 2 * r_vals - f1_val
                with np.errstate(divide="ignore", invalid="ignore"):
                    p_vals = f1_val * r_vals / denom
                valid = np.isfinite(p_vals) & (p_vals > 0) & (p_vals <= 1)
                ax.plot(r_vals[valid], p_vals[valid], "--", color="gray", alpha=0.3, lw=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "pocket_overlap_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pocket_overlap_pr.png")


def plot_per_pdb_metric_vs_distance(df: pd.DataFrame, output_dir: Path):
    """Plot 3: per-PDB scatter of delta_combined vs distance."""
    pdb_ids = df["pdb_id"].unique()
    n = len(pdb_ids)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, pdb_id in enumerate(pdb_ids):
        ax = axes[idx // ncols, idx % ncols]
        sub = df[df["pdb_id"] == pdb_id]
        ax.scatter(sub["min_distance"], sub["delta_combined"], alpha=0.5, s=15, c="steelblue")
        ax.axvline(x=6.0, color="red", linestyle="--", alpha=0.5, label="6Å")
        ax.set_xlabel("Min dist to ligand (Å)")
        ax.set_ylabel("Delta combined")
        ax.set_title(f"{pdb_id} (n={len(sub)})")
        ax.set_xlim(0, 20)
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "per_pdb_metric_vs_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per_pdb_metric_vs_distance.png")


###########################################################
# Main
###########################################################

@hydra.main(
    config_path="../../configs_local/eval/sampling",
    config_name="analyze_potts_pocket",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 1. Load model
    # -------------------------------------------------------
    print("Loading model...")
    lit_model = LitSeqDenoiser.load_from_checkpoint(cfg.ckpt_path).eval()
    model = lit_model.model.to(device)

    # Load sampling inputs CSV
    sampling_inputs_csv = cfg.get("sampling_inputs_csv", None)
    if sampling_inputs_csv is not None:
        sampling_inputs_df = pd.read_csv(sampling_inputs_csv)
    else:
        sampling_inputs_df = None

    # -------------------------------------------------------
    # 2. Collect native_val PDB paths
    # -------------------------------------------------------
    print("Collecting PDB paths...")
    selected_pdb_ids = set()
    with open(cfg.native_selected_list) as f:
        for line in f:
            pid = line.strip()
            if pid:
                # Remove extension if present (e.g., "2pog.cif" -> "2pog")
                pid = Path(pid).stem
                selected_pdb_ids.add(pid.lower())
    print(f"  Selected PDB IDs: {len(selected_pdb_ids)}")

    all_cifs = sorted(Path(cfg.native_cif_dir).glob("*.cif"))
    pdb_paths = [
        str(p) for p in all_cifs
        if p.stem.split("_")[0].lower() in selected_pdb_ids
    ]
    print(f"  Found {len(pdb_paths)} CIF files matching selected PDB IDs")

    if cfg.debug and cfg.num_debug_samples:
        pdb_paths = pdb_paths[: cfg.num_debug_samples]
        print(f"  Debug mode: using {len(pdb_paths)} samples")

    # -------------------------------------------------------
    # 3. Process each sample
    # -------------------------------------------------------
    all_results = []

    for pdb_path in tqdm(pdb_paths, desc="Processing PDBs"):
        pdb_id = Path(pdb_path).stem
        try:
            result = process_single_pdb(
                pdb_path=pdb_path,
                pdb_id=pdb_id,
                model=model,
                cfg=cfg,
                sampling_inputs_df=sampling_inputs_df,
                device=device,
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"  Error processing {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("No results collected. Exiting.")
        return

    # -------------------------------------------------------
    # 4. Aggregate and save results
    # -------------------------------------------------------
    df = pd.concat(all_results, ignore_index=True)
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path} ({len(df)} residues from {df['pdb_id'].nunique()} PDBs)")

    # -------------------------------------------------------
    # 5. Generate plots
    # -------------------------------------------------------
    print("Generating plots...")
    plot_metric_vs_distance(df, output_dir)
    plot_pocket_overlap(df, cfg.pocket_distances, output_dir)
    if df["pdb_id"].nunique() > 1:
        plot_per_pdb_metric_vs_distance(df, output_dir)

    print(f"Done. All outputs saved to {output_dir}")


def process_single_pdb(
    pdb_path: str,
    pdb_id: str,
    model,
    cfg: DictConfig,
    sampling_inputs_df: pd.DataFrame | None,
    device: str,
) -> pd.DataFrame | None:
    """Process a single PDB: two forward passes + distance computation."""

    # Build batch (WITH ligand)
    batch_lig = get_sd_batch(
        pdb_paths=[pdb_path],
        sample_is_designed=cfg.input_sample_is_designed,
        cif_parse_cfg=cfg.cif_parse_cfg,
        preprocess_cfg=cfg.preprocess_cfg,
        featurizer_cfg=cfg.featurizer_cfg,
        device=device,
        sampling_inputs_df=sampling_inputs_df,
    )

    # Build batch (WITHOUT ligand) — separate build to avoid state sharing
    batch_nol = get_sd_batch(
        pdb_paths=[pdb_path],
        sample_is_designed=cfg.input_sample_is_designed,
        cif_parse_cfg=cfg.cif_parse_cfg,
        preprocess_cfg=cfg.preprocess_cfg,
        featurizer_cfg=cfg.featurizer_cfg,
        device=device,
        sampling_inputs_df=sampling_inputs_df,
    )

    # Forward pass: WITH ligand conditioning
    potts_lig = run_potts_forward(model, batch_lig, protein_only=False)

    # Forward pass: WITHOUT ligand conditioning
    potts_nol = run_potts_forward(model, batch_nol, protein_only=True)

    # Sanity check: edge_idx should be identical
    assert torch.equal(potts_lig["edge_idx"], potts_nol["edge_idx"]), \
        f"{pdb_id}: edge_idx differs between passes!"

    # Compute delta metrics
    deltas = compute_potts_deltas(potts_lig, potts_nol)

    # Compute per-residue distance from atom_array
    atom_array = batch_lig["atom_array"][0]
    residue_dists = compute_per_residue_min_distance_to_ligand(atom_array)

    if not residue_dists:
        print(f"  {pdb_id}: no ligand found, skipping")
        return None

    # Map tokens to residues
    token_residue_info = map_token_to_residue_info(atom_array)
    mask_i = deltas["mask_i"]

    # Build per-residue results
    rows = []
    for token_idx in range(len(mask_i)):
        if not mask_i[token_idx].bool().item():
            continue
        if token_idx >= len(token_residue_info):
            continue

        chain_id, res_id, res_name = token_residue_info[token_idx]
        key = (chain_id, res_id)
        min_dist = residue_dists.get(key, float("nan"))

        rows.append({
            "pdb_id": pdb_id,
            "token_idx": token_idx,
            "chain_id": chain_id,
            "res_id": res_id,
            "res_name": res_name,
            "min_distance": min_dist,
            "delta_h": deltas["delta_h"][token_idx].item(),
            "delta_J": deltas["delta_J"][token_idx].item(),
            "delta_combined": deltas["delta_combined"][token_idx].item(),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
