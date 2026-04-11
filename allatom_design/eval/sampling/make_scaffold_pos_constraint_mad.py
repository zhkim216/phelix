"""
Make scaffold positional constraint CSVs using per-PDB MAD thresholding.

For each CIF:
1. Two forward passes (with/without ligand) → delta_J per residue
2. Per-PDB MAD: threshold = median + k × 1.4826 × MAD
3. delta_J < threshold → scaffold (constrained)

Outputs one CSV per k value, with _sample0/_sample1 rows per CIF.

Usage:
    python -m allatom_design.eval.sampling.make_scaffold_pos_constraint_mad
"""

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
    num_samples: int,
) -> dict[float, list[dict]] | None:
    """
    Run two-pass Potts forward, compute per-PDB MAD threshold, select scaffold.

    Returns:
        dict mapping k -> list of row dicts (one per sample).
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
    results: dict[float, list[dict]] = {}
    for k in k_values:
        threshold = median_J + k * sigma_est
        scaffold_mask = (delta_J < threshold) & valid

        chain_ids = []
        res_ids = []
        for idx in range(len(mask_i)):
            if scaffold_mask[idx] and idx < len(token_info):
                cid, rid, _ = token_info[idx]
                chain_ids.append(cid)
                res_ids.append(rid)

        if chain_ids:
            fixed_pos_seq = _indices_to_pos_string(
                np.array(chain_ids), np.array(res_ids)
            )
        else:
            fixed_pos_seq = ""

        n_scaffold = len(chain_ids)
        n_pocket = n_valid - n_scaffold

        # Duplicate for num_samples
        rows = []
        for s in range(num_samples):
            pdb_key = f"{pdb_stem}_sample{s}"
            rows.append({
                "pdb_key": pdb_key,
                "fixed_pos_seq": fixed_pos_seq,
                "fixed_pos_scn": np.nan,
                "num_scaffold_residues": n_scaffold,
                "num_pocket_residues": n_pocket,
                "num_tokens": n_tokens,
            })
        results[k] = rows

    return results


@hydra.main(
    config_path="../../configs_local/eval/sampling",
    config_name="make_scaffold_pos_constraint_mad",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = list(cfg.k_values)
    num_samples = cfg.num_samples

    # Load model
    print("Loading model...")
    lit_model = LitSeqDenoiser.load_from_checkpoint(cfg.ckpt_path).eval()
    model = lit_model.model.to(device)

    # Get PDB paths
    print("Collecting PDB paths...")
    pdb_paths = get_pdb_files(**cfg.pdb_cfg)
    print(f"  Found {len(pdb_paths)} CIF files")

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
                num_samples=num_samples,
            )
            if result is not None:
                for k in k_values:
                    all_rows[k].extend(result[k])
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

    # Shard suffix for array jobs
    array_id = cfg.pdb_cfg.get("array_id", None)
    shard_suffix = f"_shard{array_id}" if array_id is not None else ""

    # Save CSVs for each k
    for k in k_values:
        rows = all_rows[k]
        if not rows:
            print(f"  k={k}: no results, skipping")
            continue

        df = pd.DataFrame(rows)
        k_str = f"{k:.1f}".replace(".", "")  # 0.2 -> "02", 1.0 -> "10"

        # Minimal CSV (consumable by lc_seq_des_multi.py)
        minimal_cols = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn"]
        minimal_path = output_dir / f"scaffold_constraint_mad_k{k_str}{shard_suffix}.csv"
        df[minimal_cols].to_csv(minimal_path, index=False)

        # Full CSV (with metadata)
        full_path = output_dir / f"scaffold_constraint_mad_k{k_str}{shard_suffix}_full.csv"
        df.to_csv(full_path, index=False)

        print(f"  k={k}: saved {len(df)} rows -> {minimal_path.name}")

    # Summary
    n_processed = len(all_rows[k_values[0]]) // num_samples if all_rows[k_values[0]] else 0
    print(f"\nSummary:")
    print(f"  Processed: {n_processed}, Failed: {len(failed)}")
    print(f"  k values: {k_values}")
    print(f"  Samples per CIF: {num_samples}")

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
