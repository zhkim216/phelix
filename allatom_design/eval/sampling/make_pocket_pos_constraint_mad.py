"""
Make pocket positional constraint CSVs using per-PDB MAD thresholding.

Input: a directory of stage-1 sample CIFs named `{base}_sample{s}.cif`,
e.g. `0H7_len_150_0_model_0_sample0.cif`. `num_samples` selects how many of
those per-base variants to forward-pass — `num_samples=1` keeps only
`_sample0`, `num_samples=N` keeps `_sample0` … `_sample{N-1}`. Each selected
file is forward-passed independently, so distinct stage-1 samples can yield
distinct pocket constraints.

For each selected CIF:
1. Two forward passes (with/without ligand) → delta_J per residue
2. Per-PDB MAD: threshold = median + k × 1.4826 × MAD
3. Classification:
     delta_J <  threshold → scaffold (resampled: NOT in fixed_pos_seq)
     delta_J >= threshold → pocket   (fixed:     IN  fixed_pos_seq)

One row per CIF; `pdb_key = Path(pdb_path).stem` (already contains
`_sample{s}`) so downstream `pos_constraint_df.loc[example_id]` matches.

Output naming (per shard):
    pocket_constraint_mad_k{K}_numsample{N}_array_{C}.csv
After merging shards with `merge_array_csvs.py`:
    pocket_constraint_mad_k{K}_numsample{N}.csv

Use `expand_pocket_constraint_csv.py` to duplicate a `_numsample1.csv` across
multiple `_sample{s}` keys offline without re-running forward passes.

Usage:
    python -m allatom_design.eval.sampling.make_pocket_pos_constraint_mad \
        num_samples=1
"""

import re
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import lightning as L
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.eval.eval_utils.eval_setup_utils import get_pdb_files
from allatom_design.eval.eval_utils.sd_data_utils import get_sd_batch
from allatom_design.eval.eval_utils.seq_des_utils import _indices_to_pos_string
from allatom_design.eval.eval_utils.potts_utils import (
    run_potts_forward,
    compute_potts_deltas,
    map_token_to_residue_info,
)
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


def process_single_pdb(
    pdb_path: str,
    model: torch.nn.Module,
    cfg: DictConfig,
    device: str,
    k_values: list[float],
    max_tokens: int,
) -> dict[float, dict] | None:
    """
    Run two-pass Potts forward, compute per-PDB MAD threshold, select pocket
    residues (delta_J >= threshold) as the fixed set; scaffold residues
    (delta_J < threshold) are left to be resampled.

    Returns:
        dict mapping k -> single row dict for this CIF.
        Returns None on skip (OOM guard, etc.).
    """
    pdb_stem = Path(pdb_path).stem

    # Build first batch (with ligand)
    batch_lig = get_sd_batch(
        pdb_paths=[pdb_path],
        sample_is_designed=cfg.input_sample_is_designed,
        cif_parse_cfg=cfg.cif_parse_cfg,
        preprocess_cfg=cfg.preprocess_cfg,
        featurizer_cfg=cfg.featurizer_cfg,
        device=device,
    )

    # OOM guard
    n_tokens = int(batch_lig["token_pad_mask"].sum().item())
    if n_tokens > max_tokens:
        print(f"  {pdb_stem}: {n_tokens} tokens > {max_tokens}, skipping")
        return None

    # Build second batch (separate to avoid state sharing)
    batch_nol = get_sd_batch(
        pdb_paths=[pdb_path],
        sample_is_designed=cfg.input_sample_is_designed,
        cif_parse_cfg=cfg.cif_parse_cfg,
        preprocess_cfg=cfg.preprocess_cfg,
        featurizer_cfg=cfg.featurizer_cfg,
        device=device,
    )

    # Two forward passes
    potts_lig = run_potts_forward(model, batch_lig, protein_only=False)
    potts_nol = run_potts_forward(model, batch_nol, protein_only=True)

    # Compute deltas
    deltas = compute_potts_deltas(potts_lig, potts_nol)
    delta_J = deltas["delta_J"]  # [N]
    mask_i = deltas["mask_i"]    # [N]
    valid = mask_i.bool()

    n_valid = int(valid.sum().item())
    if n_valid == 0:
        print(f"  {pdb_stem}: no valid residues, skipping")
        return None

    # Per-PDB MAD statistics
    valid_J = delta_J[valid]
    median_J = valid_J.median().item()
    mad_J = (valid_J - median_J).abs().median().item()
    sigma_est = 1.4826 * mad_J

    # Map tokens to residue info
    atom_array = batch_lig["atom_array"][0]
    token_info = map_token_to_residue_info(atom_array)

    # For each k: threshold → classify → constraint string
    # scaffold_mask is kept for diagnostics; pocket_mask (its valid complement)
    # is what actually goes into fixed_pos_seq to be conditioned on.
    results: dict[float, dict] = {}
    for k in k_values:
        threshold = median_J + k * sigma_est
        scaffold_mask = (delta_J < threshold) & valid   # resampled (NOT in fixed_pos_seq)
        pocket_mask = (delta_J >= threshold) & valid    # fixed (IN fixed_pos_seq)

        chain_ids = []
        res_ids = []
        for idx in range(len(mask_i)):
            if pocket_mask[idx] and idx < len(token_info):
                cid, rid, _ = token_info[idx]
                chain_ids.append(cid)
                res_ids.append(rid)

        if chain_ids:
            fixed_pos_seq = _indices_to_pos_string(
                np.array(chain_ids), np.array(res_ids)
            )
        else:
            fixed_pos_seq = ""

        n_pocket = len(chain_ids)
        n_scaffold = n_valid - n_pocket

        results[k] = {
            "pdb_key": pdb_stem,
            "fixed_pos_seq": fixed_pos_seq,
            "fixed_pos_scn": np.nan,
            "num_scaffold_residues": n_scaffold,
            "num_pocket_residues": n_pocket,
            "num_tokens": n_tokens,
        }

    return results


@hydra.main(
    config_path="../../configs_local/eval/sampling",
    config_name="make_pocket_pos_constraint_mad",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = list(cfg.k_values)
    num_samples = int(cfg.num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    # Load model
    print("Loading model...")
    lit_model = LitSeqDenoiser.load_from_checkpoint(cfg.ckpt_path).eval()
    model = lit_model.model.to(device)

    # Get PDB paths
    print("Collecting PDB paths...")
    pdb_paths = get_pdb_files(**cfg.pdb_cfg)
    print(f"  Found {len(pdb_paths)} CIF files")

    # Filter to stage-1 samples with sample_idx < num_samples.
    # Each base `{lig}_len_{L}_{idx}_model_{M}` has variants `_sample0`,
    # `_sample1`, ... ; num_samples picks the first N per base.
    sample_re = re.compile(r"^(.+)_sample(\d+)$")
    filtered: list[str] = []
    unmatched: list[str] = []
    for pdb_path in pdb_paths:
        stem = Path(pdb_path).stem
        m = sample_re.match(stem)
        if m is None:
            unmatched.append(stem)
            continue
        if int(m.group(2)) < num_samples:
            filtered.append(pdb_path)
    print(f"  Kept {len(filtered)}/{len(pdb_paths)} files with sample_idx < {num_samples}")
    if unmatched:
        print(
            f"  WARNING: {len(unmatched)} files did not match "
            f"`{{base}}_sample{{idx}}` pattern and were skipped. "
            f"First few: {unmatched[:5]}"
        )
    pdb_paths = filtered

    if cfg.debug and cfg.num_debug_samples:
        pdb_paths = pdb_paths[: cfg.num_debug_samples]
        print(f"  Debug mode: using {len(pdb_paths)} samples")

    # Accumulate results per k
    all_rows: dict[float, list[dict]] = {k: [] for k in k_values}
    failed = []

    for pdb_path in tqdm(pdb_paths, desc="Processing PDBs"):
        pdb_stem = Path(pdb_path).stem
        try:
            result = process_single_pdb(
                pdb_path=pdb_path,
                model=model,
                cfg=cfg,
                device=device,
                k_values=k_values,
                max_tokens=cfg.max_tokens_for_forward,
            )
            if result is not None:
                for k in k_values:
                    all_rows[k].append(result[k])
            else:
                failed.append(pdb_stem)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM for {pdb_stem}, skipping")
            failed.append(pdb_stem)
        except Exception as e:
            print(f"  Error processing {pdb_stem}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(pdb_stem)

    # Trailing `_array_{id}` shard suffix so `merge_array_csvs.py` can
    # concatenate across array tasks with its `^(.+)_array_\d+\.csv$` regex.
    array_id = cfg.pdb_cfg.get("array_id", None)
    array_suffix = f"_array_{array_id}" if array_id is not None else ""
    n_suffix = f"_numsample{num_samples}"

    # Save CSVs for each k
    for k in k_values:
        rows = all_rows[k]
        if not rows:
            print(f"  k={k}: no results, skipping")
            continue

        df = pd.DataFrame(rows)
        k_str = f"{k:.1f}".replace(".", "")  # 0.2 -> "02", 1.0 -> "10"

        # Minimal CSV (consumable by lc_seq_des_multi.py).
        # Layout: pocket_constraint_mad_k{K}_numsample{N}_array_{C}.csv
        minimal_cols = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn"]
        minimal_path = output_dir / f"pocket_constraint_mad_k{k_str}{n_suffix}{array_suffix}.csv"
        df[minimal_cols].to_csv(minimal_path, index=False)

        # Full CSV (with metadata).
        # Layout: pocket_constraint_mad_k{K}_numsample{N}_full_array_{C}.csv
        full_path = output_dir / f"pocket_constraint_mad_k{k_str}{n_suffix}_full{array_suffix}.csv"
        df.to_csv(full_path, index=False)

        print(f"  k={k}: saved {len(df)} rows -> {minimal_path.name}")

    # Summary
    n_processed = len(all_rows[k_values[0]]) if all_rows[k_values[0]] else 0
    print(f"\nSummary:")
    print(f"  Processed: {n_processed}, Failed: {len(failed)}")
    print(f"  k values: {k_values}")
    print(f"  Samples per base forwarded: {num_samples}")

    for k in k_values:
        rows = all_rows[k]
        if rows:
            df = pd.DataFrame(rows)
            mean_scaffold = df["num_scaffold_residues"].mean()
            mean_pocket = df["num_pocket_residues"].mean()
            print(f"  k={k}: mean scaffold={mean_scaffold:.1f}, mean pocket={mean_pocket:.1f}")

    if failed:
        print(f"  Failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
