"""
Smoke test: is it safe to replace `copy.deepcopy(batch)` with `dict(batch)`
(shallow copy) inside `SeqDenoiser.forward` (sd_model.py:56)?

This script:
  1. Loads `LitSeqDenoiser` with randomly-initialized weights.
  2. Pulls N mini-batches from the real train dataloader.
  3. For each batch, runs forward twice:
       - once with the stock deepcopy implementation (baseline)
       - once with a shallow-copy subclass (`SeqDenoiserShallow`)
     using the same RNG state so mask_selector draws match.
  4. Compares:
       - Output tensor equality between the two modes.
       - Whether the caller's original batch dict was mutated (new keys
         added, tensors replaced, tensors modified in place).
  5. Prints a per-batch report + summary verdict.

Weights are randomly initialized — we only need a deterministic forward to
compare outputs; no checkpoint is required.

Usage
-----
    python debug/260417_training_speed_debug/smoke_test_deepcopy.py \
        --pdb-path /scratch/users/zhkim216/datasets/atomworks_pdb_full_v3 \
        --metadata-path /scratch/users/zhkim216/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_grouped_250205.parquet \
        --residue-cache-dir /scratch/users/zhkim216/datasets/atomworks_cached_residue_data \
        --n-batches 5

CPU-only works (device auto-selected).  Config follows exp43.sbatch's overrides
with `compile_model=false`, `batch_size=2`, and max_tokens/max_atoms at the
production values (exp43 uses 512/4608 — the featurizer pads here).
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Make repo importable when run as a script
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from allatom_design.data.datasets.atomworks_sd_dataset import AtomworksSDDataModule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


# ----------------------------------------------------------------------------
# Shallow-copy variant of SeqDenoiser.forward
# ----------------------------------------------------------------------------
class SeqDenoiserShallow(SeqDenoiser):
    """Same as SeqDenoiser but replaces the `copy.deepcopy(batch)` with `dict(batch)`."""

    def forward(self, batch, t=None):
        outputs = {}
        batch = dict(batch)  # shallow copy — only copies the top-level dict

        with torch.no_grad():
            batch["seq_cond_mask"] = self.mask_selector.sample_seq_cond_mask(batch, t)
            batch["atom_cond_mask"], scn_token_mask, expanded_bb_mask = self.mask_selector.sample_atom_cond_mask(batch)
            batch["pseudo_context_mask"] = (scn_token_mask + expanded_bb_mask).clamp(max=1.0)

        _, aux_preds = self.denoiser(batch)
        outputs.update(aux_preds)
        return outputs


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------
def build_cfg(args) -> DictConfig:
    """Compose the seq_denoiser config with exp43-style overrides (reduced)."""
    config_dir = str(REPO_ROOT / "allatom_design" / "configs" / "seq_denoiser")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3.2"):
        cfg = compose(
            config_name="seq_denoiser.yaml",
            overrides=[
                "denoiser=lc_atom_mpnn",
                f"train.batch_size={args.batch_size}",
                "train.compile.compile_model=false",
                "train.debug=false",
                "num_workers=0",
                "wandb.no_wandb=true",
                "loss.main_seq_loss_pocket_only=false",
                "loss.main_potts_loss_pocket_only=false",
                "loss.seq_loss.per_token_avg=false",
                "loss.potts.per_token_avg=false",
                "data/dataset@data=ligandmpnn",
                f"data.pdb_path={args.pdb_path}",
                f"data.train_metadata_path={args.metadata_path}",
                f"data.residue_cache_dir={args.residue_cache_dir}",
                f"data.validation_ids_file={args.validation_ids_file}",
                f"data.val_metadata_path={args.val_metadata_path}",
                "data.sampling_weights.ligand_cluster_col=null",
                "+data.max_interface_distance=5.0",
                "data.exclude_val_cluster=true",
                "data.grouping_scheme=neighbor",
                "data.pocket_only_training=false",
                "data.interface_only_training=false",
                "data.featurizer_cfg.is_inference=false",
                "data.featurizer_cfg.training_structure_noise=0.1",
                "data.featurizer_cfg.occupancy_threshold_protein_backbone=0.0",
                "data.featurizer_cfg.remove_unresolved_tokens=true",
                f"data.featurizer_cfg.max_tokens={args.max_tokens}",
                f"data.featurizer_cfg.max_atoms={args.max_atoms}",
                "data.featurizer_cfg.crop_center_cutoff_distance=30.0",
                "data.featurizer_cfg.crop_spatial_p_protein_monomer_chain=0.25",
                "data.featurizer_cfg.pocket_distance=6.0",
                "data.featurizer_cfg.asymmetric_noise=false",
                "data.featurizer_cfg.drop_prob_non_protein_chains=0.0",
                "data.sampling_weights.alphas_interface.a_protein_protein=0.0",
                "data.sampling_weights.alphas_interface.a_protein_nuc=0.0",
                "data.sampling_weights.alphas_interface.a_protein_peptide=0.0",
                "data.sampling_weights.alphas_interface.a_protein_small_molecule=1.0",
                "data.sampling_weights.alphas_interface.a_protein_metal=0.0",
                "data.sampling_weights.alphas_interface.a_protein_loi=1.0",
                "data.use_biologically_meaningful_small_molecule=true",
                "data.exclude_small_molecules_covalently_linked_to_protein=true",
                "data.samples_per_epoch=1000",
                "data.sampling_weights.k_percentile=90.0",
                f"data.batch_size={args.batch_size}",
                "data.num_workers=0",
                "denoiser.mpnn.k_neighbors=24",
                "denoiser.mpnn.ligand_conditioning=true",
                "denoiser.mpnn.token_features.use_pocket_rbf=false",
                "denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb",
                "denoiser.mpnn.token_features.ligand_atom_context_num=36",
                "denoiser.mpnn.potts.k_neighbors_potts=24",
                "mask_selector.restype_masking_schedule=uniform_cosine_t",
                "mask_selector.pseudo_ligand_backbone_mask_radius=0",
            ],
        )
    return cfg


# ----------------------------------------------------------------------------
# Batch snapshot / diff helpers
# ----------------------------------------------------------------------------
def snapshot_batch(batch: dict) -> dict:
    """Capture keys, tensor data_ptr, and a clone of every tensor value."""
    snap = {"keys": set(batch.keys()), "tensors": {}, "data_ptrs": {}}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            snap["tensors"][k] = v.detach().clone()
            snap["data_ptrs"][k] = v.data_ptr()
    return snap


def diff_snapshots(before: dict, after_batch: dict) -> dict:
    """Detect added keys, changed data_ptr, in-place tensor mutations."""
    added = set(after_batch.keys()) - before["keys"]
    removed = before["keys"] - set(after_batch.keys())

    tensors_modified_inplace = []
    tensors_replaced = []
    for k, old_tensor in before["tensors"].items():
        if k not in after_batch:
            continue
        new_val = after_batch[k]
        if not isinstance(new_val, torch.Tensor):
            tensors_replaced.append(k)
            continue
        if new_val.data_ptr() != before["data_ptrs"][k]:
            tensors_replaced.append(k)
            continue
        # Same data_ptr — check values
        if old_tensor.shape != new_val.shape or not torch.equal(old_tensor, new_val.detach().cpu() if new_val.is_cuda else new_val.detach()):
            # Re-check on the tensor's device to avoid spurious d_type mismatch
            try:
                equal = torch.equal(old_tensor.to(new_val.device), new_val.detach())
            except Exception:
                equal = False
            if not equal:
                tensors_modified_inplace.append(k)
    return {
        "added_keys": sorted(added),
        "removed_keys": sorted(removed),
        "tensors_replaced": sorted(tensors_replaced),
        "tensors_modified_inplace": sorted(tensors_modified_inplace),
    }


def outputs_allclose(a, b, rtol: float = 0.0, atol: float = 0.0,
                     prefix: str = "") -> tuple[bool, list[str]]:
    """Strict equality for every tensor in nested dicts/tuples/lists.

    Handles the case where an output value is itself a dict (e.g.
    ``potts_decoder_aux`` returned by the denoiser).
    """
    # Dict: recurse key-by-key.
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False, [f"{prefix}key_mismatch: a={sorted(a)} b={sorted(b)}"]
        mismatches = []
        for k in a:
            sub_prefix = f"{prefix}{k}." if prefix else f"{k}."
            _, sub_mis = outputs_allclose(a[k], b[k], rtol=rtol, atol=atol, prefix=sub_prefix)
            mismatches.extend(sub_mis)
        return len(mismatches) == 0, mismatches

    # Tensor: compare with torch.equal/allclose.
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        key = prefix.rstrip(".") or "<tensor>"
        if a.shape != b.shape:
            return False, [f"{key}: shape {tuple(a.shape)} vs {tuple(b.shape)}"]
        if rtol == 0.0 and atol == 0.0:
            if not torch.equal(a, b):
                max_abs = (a.float() - b.float()).abs().max().item()
                return False, [f"{key}: not torch.equal (max abs diff = {max_abs:.3e})"]
        else:
            if not torch.allclose(a, b, rtol=rtol, atol=atol):
                return False, [f"{key}: not allclose"]
        return True, []

    # Tuple/list: recurse element-wise.
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, [f"{prefix}length mismatch: {len(a)} vs {len(b)}"]
        mismatches = []
        for i, (va, vb) in enumerate(zip(a, b)):
            _, sub_mis = outputs_allclose(va, vb, rtol=rtol, atol=atol, prefix=f"{prefix}[{i}].")
            mismatches.extend(sub_mis)
        return len(mismatches) == 0, mismatches

    # None or scalar: direct comparison.
    if a is None and b is None:
        return True, []
    if a == b:
        return True, []
    return False, [f"{prefix.rstrip('.')}: non-tensor inequality ({type(a).__name__} vs {type(b).__name__})"]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_forward_with_seed(model, batch, seed: int):
    """Run model(batch) with a deterministic RNG state and return outputs + mutated batch."""
    set_all_seeds(seed)
    outputs = model(batch)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--residue-cache-dir", type=str, required=True)
    parser.add_argument("--validation-ids-file", type=str,
                        default="/scratch/users/zhkim216/datasets/splits/lmpnn_validation_ids.txt",
                        help="Path to validation IDs text file (override for local runs)")
    parser.add_argument("--val-metadata-path", type=str,
                        default="/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/metadata_lmpnnval_small_molecule_filtered_for_training.parquet",
                        help="Path to val metadata parquet (override for local runs)")
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Local test: drop to e.g. 128 for fast CPU smoke")
    parser.add_argument("--max-atoms", type=int, default=4608,
                        help="Local test: drop to e.g. 1024 for fast CPU smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    cfg = build_cfg(args)
    print("=== Config composed ===")
    print(f"  batch_size = {cfg.train.batch_size}")
    print(f"  max_tokens = {cfg.data.featurizer_cfg.max_tokens}")
    print(f"  max_atoms  = {cfg.data.featurizer_cfg.max_atoms}")

    set_all_seeds(args.seed)

    # --- Build model (random init) ---
    lit_model = LitSeqDenoiser(cfg)
    lit_model.model.set_scale_factors({"bb": (0.0, cfg.model.sigma_data[0]),
                                       "scn": (0.0, cfg.model.sigma_data[1])})
    stock_model = lit_model.model.to(device).eval()

    # Build shallow variant on the same weights
    shallow_model = SeqDenoiserShallow(cfg.model).to(device).eval()
    shallow_model.load_state_dict(stock_model.state_dict(), strict=True)

    # --- Build datamodule + loader ---
    dm = AtomworksSDDataModule(cfg.data)
    loader = dm.train_dataloader()
    it = iter(loader)

    summary = {"equal": [], "original_mutated": [], "tensors_modified_inplace_any": []}

    print("\n=== Deepcopy Smoke Test ===")
    for i in range(args.n_batches):
        try:
            batch = next(it)
        except StopIteration:
            print(f"Dataloader exhausted after {i} batches")
            break

        batch = move_batch_to_device(batch, device)

        # ---- Baseline (deepcopy) ----
        batch_stock = copy.deepcopy(batch)  # outer copy so we can detect mutation independently
        snap_stock = snapshot_batch(batch_stock)
        out_stock = run_forward_with_seed(stock_model, batch_stock, seed=args.seed + 1000 + i)
        diff_stock = diff_snapshots(snap_stock, batch_stock)

        # ---- Shallow variant ----
        batch_shallow = copy.deepcopy(batch)
        snap_shallow = snapshot_batch(batch_shallow)
        out_shallow = run_forward_with_seed(shallow_model, batch_shallow, seed=args.seed + 1000 + i)
        diff_shallow = diff_snapshots(snap_shallow, batch_shallow)

        # ---- Compare outputs ----
        equal, mismatches = outputs_allclose(out_stock, out_shallow)

        any_mutation = bool(
            diff_shallow["added_keys"]
            or diff_shallow["removed_keys"]
            or diff_shallow["tensors_replaced"]
            or diff_shallow["tensors_modified_inplace"]
        )

        print(f"\nBatch {i+1}/{args.n_batches}:")
        print(f"  outputs equal (torch.equal all tensors): {equal}")
        if not equal:
            for m in mismatches[:5]:
                print(f"    - {m}")
        print(f"  [deepcopy baseline] original batch post-forward:")
        print(f"      added keys: {diff_stock['added_keys']}")
        print(f"      tensors replaced: {diff_stock['tensors_replaced']}")
        print(f"      tensors modified in place: {diff_stock['tensors_modified_inplace']}")
        print(f"  [shallow variant ] original batch post-forward:")
        print(f"      added keys: {diff_shallow['added_keys']}")
        print(f"      tensors replaced: {diff_shallow['tensors_replaced']}")
        print(f"      tensors modified in place: {diff_shallow['tensors_modified_inplace']}")

        summary["equal"].append(equal)
        summary["original_mutated"].append(any_mutation)
        summary["tensors_modified_inplace_any"].append(bool(diff_shallow["tensors_modified_inplace"]))

    # ---- Summary verdict ----
    print("\n=== Summary ===")
    all_equal = all(summary["equal"]) and len(summary["equal"]) > 0
    any_mutation = any(summary["original_mutated"])
    any_inplace = any(summary["tensors_modified_inplace_any"])

    print(f"  batches tested: {len(summary['equal'])}")
    print(f"  outputs equal across all batches: {all_equal}")
    print(f"  original batch mutated in shallow mode: {any_mutation}")
    print(f"  tensors modified in place in shallow mode: {any_inplace}")

    if all_equal and not any_inplace:
        verdict = "YES"
        rationale = (
            "All output tensors are bit-identical, and no tensor in the original batch "
            "is modified in place by the shallow-copy variant. New keys added to the "
            "shallow copy (seq_cond_mask/atom_cond_mask/pseudo_context_mask) leak back "
            "into the caller's batch, but the caller (LitSeqDenoiser.training_step) does "
            "not read those keys after forward, so this is harmless."
            if any_mutation else
            "All output tensors are bit-identical and the original batch is untouched."
        )
    elif all_equal and any_inplace:
        verdict = "CONDITIONAL"
        rationale = (
            "Outputs match, but some tensors in the caller's batch are modified in place. "
            "If any downstream code re-reads those tensors after forward (e.g. loss computation "
            "uses batch['xyz']), replacing deepcopy with dict() changes behavior. Inspect the "
            "flagged keys before committing the change."
        )
    else:
        verdict = "NO"
        rationale = "Outputs differ between deepcopy and shallow modes — shallow replacement is unsafe."

    print(f"  safe to replace deepcopy with dict(): {verdict}")
    print(f"  rationale: {rationale}")


if __name__ == "__main__":
    main()
