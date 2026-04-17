"""
Training step profiler for sequence denoiser.

Measures per-step wall-clock breakdown of the training inner loop:
  - t_dataload: `next(iter(loader))` (CPU timer)
  - t_forward : model(batch)          (cuda.Event)
  - t_backward: loss.backward()       (cuda.Event)
  - t_optim   : optimizer.step()      (cuda.Event)
  - t_total   : whole step
  - peak_memory_GB / allocated_memory_GB

Supports bf16-mixed (autocast), torch.compile, optional torch.profiler Chrome
trace, and two config presets: `reduced` (fast local iteration) and `full`
(exp43_cfg0 training config).

Usage (local, reduced config, no compile):
    python debug/260417_training_speed_debug/profile_training.py \
        --config reduced \
        --pdb-path /scratch/users/zhkim216/datasets/atomworks_pdb_full_v3 \
        --metadata-path /scratch/users/zhkim216/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_grouped_250205.parquet \
        --residue-cache-dir /scratch/users/zhkim216/datasets/atomworks_cached_residue_data \
        --output-csv local_profile.csv

Usage (Sherlock A100, full exp43_cfg0, with compile):
    python debug/260417_training_speed_debug/profile_training.py \
        --config full --compile \
        --pdb-path /scratch/users/zhkim216/datasets/atomworks_pdb_full_v3 \
        --metadata-path /scratch/users/zhkim216/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_grouped_250205.parquet \
        --residue-cache-dir /scratch/users/zhkim216/datasets/atomworks_cached_residue_data \
        --output-csv sherlock_profile.csv

With Chrome trace (view at https://ui.perfetto.dev/):
    python debug/260417_training_speed_debug/profile_training.py \
        --config full --compile \
        --pdb-path ... --metadata-path ... --residue-cache-dir ... \
        --output-csv profile.csv \
        --trace-path profile_trace.json \
        --n-steps 50 --n-warmup 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from allatom_design.data.datasets.atomworks_sd_dataset import AtomworksSDDataModule
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# Overrides shared by reduced + full
_EXP43_COMMON_OVERRIDES = [
    "denoiser=lc_atom_mpnn",
    "train.debug=false",
    "wandb.no_wandb=true",
    "loss.main_seq_loss_pocket_only=false",
    "loss.main_potts_loss_pocket_only=false",
    "loss.seq_loss.per_token_avg=false",
    "loss.potts.per_token_avg=false",
    "data/dataset@data=ligandmpnn",
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
    "data.samples_per_epoch=100000",
    "data.sampling_weights.k_percentile=90.0",
    "denoiser.mpnn.k_neighbors=24",
    "denoiser.mpnn.ligand_conditioning=true",
    "denoiser.mpnn.token_features.use_pocket_rbf=false",
    "denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb",
    "denoiser.mpnn.token_features.ligand_atom_context_num=36",
    "denoiser.mpnn.potts.k_neighbors_potts=24",
    "mask_selector.restype_masking_schedule=uniform_cosine_t",
    "mask_selector.pseudo_ligand_backbone_mask_radius=0",
]

_REDUCED_OVERRIDES = [
    "train.batch_size=4",
    "data.batch_size=4",
    "data.num_workers=2",
    "num_workers=2",
    "data.featurizer_cfg.max_tokens=256",
    "data.featurizer_cfg.max_atoms=2304",
]

_FULL_OVERRIDES = [
    "train.batch_size=8",
    "data.batch_size=8",
    "data.num_workers=8",
    "num_workers=8",
    "data.featurizer_cfg.max_tokens=512",
    "data.featurizer_cfg.max_atoms=4608",
]


def build_cfg(args) -> DictConfig:
    """Compose seq_denoiser cfg with exp43 overrides + reduced/full preset."""
    preset_overrides = _REDUCED_OVERRIDES if args.config == "reduced" else _FULL_OVERRIDES
    overrides = [
        *_EXP43_COMMON_OVERRIDES,
        *preset_overrides,
        f"data.pdb_path={args.pdb_path}",
        f"data.train_metadata_path={args.metadata_path}",
        f"data.residue_cache_dir={args.residue_cache_dir}",
        f"data.validation_ids_file={args.validation_ids_file}",
        f"data.val_metadata_path={args.val_metadata_path}",
        f"train.compile.compile_model={'true' if args.compile else 'false'}",
    ]
    if args.batch_size is not None:
        overrides.extend([
            f"train.batch_size={args.batch_size}",
            f"data.batch_size={args.batch_size}",
        ])
    if args.precision is not None:
        overrides.append(f"trainer.precision={args.precision}")
    config_dir = str(REPO_ROOT / "allatom_design" / "configs" / "seq_denoiser")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3.2"):
        cfg = compose(config_name="seq_denoiser.yaml", overrides=overrides)
    return cfg


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def setup_data(cfg: DictConfig):
    """Build AtomworksSDDataModule and return a train dataloader iterator."""
    dm = AtomworksSDDataModule(cfg.data)
    loader = dm.train_dataloader()
    return dm, loader


def setup_model(cfg: DictConfig, device: torch.device):
    """Build LitSeqDenoiser, set scale factors, move to device, return (lit_model, optimizer)."""
    lit_model = LitSeqDenoiser(cfg)
    bb_std, scn_std = cfg.model.sigma_data
    lit_model.model.set_scale_factors({"bb": (0.0, bb_std), "scn": (0.0, scn_std)})
    lit_model = lit_model.to(device)
    lit_model.train()

    # configure_optimizers returns a dict; just pull the optimizer out
    opt_cfg = lit_model.configure_optimizers()
    optimizer = opt_cfg["optimizer"]
    return lit_model, optimizer


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------
# Profiling
# ----------------------------------------------------------------------------
class StepTimer:
    """Wraps `torch.cuda.Event(enable_timing=True)` with a CPU fallback."""

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        if use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._stop = torch.cuda.Event(enable_timing=True)
        self._t0 = None
        self._t1 = None

    def start(self):
        if self.use_cuda:
            self._start.record()
        else:
            self._t0 = time.perf_counter()

    def stop(self):
        if self.use_cuda:
            self._stop.record()
        else:
            self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self.use_cuda:
            # Caller must have cuda.synchronize()'d before reading
            return self._start.elapsed_time(self._stop)
        return (self._t1 - self._t0) * 1000.0


def profile_step(lit_model, optimizer, batch_iter, device, use_cuda: bool,
                 use_bf16: bool) -> dict:
    """Run one training step and return per-phase timing + memory."""
    t_total_start = time.perf_counter()

    # --- Dataload ---
    dl_t0 = time.perf_counter()
    batch = next(batch_iter)
    batch = move_batch_to_device(batch, device)
    if use_cuda:
        torch.cuda.synchronize()
    t_dataload_ms = (time.perf_counter() - dl_t0) * 1000.0

    # --- Forward ---
    fwd_timer = StepTimer(use_cuda)
    fwd_timer.start()
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16 and use_cuda else _nullcontext())
    meta_fields = lit_model._pop_non_tensor_fields(batch)
    with autocast_ctx:
        outputs = lit_model(batch)
        batch.update(meta_fields)
        loss, _ = lit_model.loss(outputs, batch, return_aux=True)
    fwd_timer.stop()

    # --- Backward ---
    bwd_timer = StepTimer(use_cuda)
    bwd_timer.start()
    loss.backward()
    bwd_timer.stop()

    # --- Optimizer ---
    opt_timer = StepTimer(use_cuda)
    opt_timer.start()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    opt_timer.stop()

    if use_cuda:
        torch.cuda.synchronize()

    t_forward_ms = fwd_timer.elapsed_ms()
    t_backward_ms = bwd_timer.elapsed_ms()
    t_optim_ms = opt_timer.elapsed_ms()
    t_total_ms = (time.perf_counter() - t_total_start) * 1000.0

    if use_cuda:
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    else:
        peak_gb = 0.0
        alloc_gb = 0.0

    return {
        "t_dataload_ms": t_dataload_ms,
        "t_forward_ms": t_forward_ms,
        "t_backward_ms": t_backward_ms,
        "t_optim_ms": t_optim_ms,
        "t_total_ms": t_total_ms,
        "peak_memory_GB": peak_gb,
        "allocated_memory_GB": alloc_gb,
        "loss": float(loss.detach().item()),
    }


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def summarize(rows: list[dict], accumulate_grad_batches: int, total_steps_target: int = 200_000):
    """Print a mean/p50/p95/max table + throughput estimate. Skips empty input."""
    if not rows:
        print("No rows to summarize.")
        return

    def stats(values):
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }

    cols = ["t_dataload_ms", "t_forward_ms", "t_backward_ms", "t_optim_ms", "t_total_ms"]
    col_names = {
        "t_dataload_ms": "dataload",
        "t_forward_ms":  "forward ",
        "t_backward_ms": "backward",
        "t_optim_ms":    "optim   ",
        "t_total_ms":    "total   ",
    }

    print(f"\n=== Profile Summary ({len(rows)} steps, warmup excluded) ===")
    print(f"{'phase':<14}{'mean':>10}{'p50':>10}{'p95':>10}{'max':>10}")
    for c in cols:
        s = stats([r[c] for r in rows])
        print(f"{col_names[c]:<14}{s['mean']:>10.2f}{s['p50']:>10.2f}{s['p95']:>10.2f}{s['max']:>10.2f}")

    peak = max(r["peak_memory_GB"] for r in rows)
    print(f"\nGPU peak memory: {peak:.2f} GB")

    mean_total_s = mean(r["t_total_ms"] for r in rows) / 1000.0
    if mean_total_s > 0:
        steps_per_day = int(86400 / mean_total_s)
        opt_steps_per_day = int(steps_per_day / max(accumulate_grad_batches, 1))
        print(f"Throughput: ~{steps_per_day:,} fwd-bwd-opt steps/day "
              f"(~{opt_steps_per_day:,} optimizer steps/day at accum={accumulate_grad_batches})")
        if total_steps_target:
            days_to_target = total_steps_target / max(opt_steps_per_day, 1)
            print(f"Days to reach {total_steps_target:,} optimizer steps: {days_to_target:.2f}")


def write_csv(path: str, rows: list[dict]):
    """Write one row per step to CSV (step_idx + timing columns)."""
    if not rows:
        return
    fieldnames = ["step_idx"] + list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"step_idx": i, **r})
    print(f"Wrote per-step CSV: {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=["reduced", "full"], default="reduced")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.set_defaults(compile=False)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from preset")
    parser.add_argument("--precision", type=str, default=None,
                        choices=["32-true", "bf16-mixed", "16-mixed"],
                        help="Override trainer.precision (e.g. 32-true for local GPUs without bf16 support)")
    parser.add_argument("--pdb-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--residue-cache-dir", type=str, required=True)
    parser.add_argument("--validation-ids-file", type=str,
                        default="/scratch/users/zhkim216/datasets/splits/lmpnn_validation_ids.txt",
                        help="Path to validation IDs text file (override for local runs)")
    parser.add_argument("--val-metadata-path", type=str,
                        default="/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/metadata_lmpnnval_small_molecule_filtered_for_training.parquet",
                        help="Path to val metadata parquet (override for local runs)")
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--trace-path", type=str, default=None,
                        help="If set, emit a Chrome trace JSON via torch.profiler for the measured window")
    parser.add_argument("--torch-logs", action="store_true",
                        help='Set TORCH_LOGS="recompiles,graph_breaks" before running')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accum-grad-batches", type=int, default=4,
                        help="Reported for throughput calc only; does not gate optimizer.step here")
    args = parser.parse_args()

    if args.torch_logs:
        os.environ["TORCH_LOGS"] = "recompiles,graph_breaks"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Device: {device}  use_cuda={use_cuda}")
    print(f"Config preset: {args.config}  compile: {args.compile}")

    cfg = build_cfg(args)
    print(f"  train.batch_size         = {cfg.train.batch_size}")
    print(f"  data.featurizer_cfg.max_tokens = {cfg.data.featurizer_cfg.max_tokens}")
    print(f"  data.featurizer_cfg.max_atoms  = {cfg.data.featurizer_cfg.max_atoms}")
    print(f"  num_workers              = {cfg.data.num_workers}")
    print(f"  precision (reported)     = {cfg.trainer.precision}")

    # Set up
    dm, loader = setup_data(cfg)
    lit_model, optimizer = setup_model(cfg, device)

    use_bf16 = str(cfg.trainer.precision) in ("bf16-mixed", "bf16", "bfloat16-mixed")

    batch_iter = iter(loader)

    # Warmup
    print(f"\n--- Warmup: {args.n_warmup} steps ---")
    for i in range(args.n_warmup):
        try:
            _ = profile_step(lit_model, optimizer, batch_iter, device, use_cuda, use_bf16)
        except StopIteration:
            batch_iter = iter(loader)
            _ = profile_step(lit_model, optimizer, batch_iter, device, use_cuda, use_bf16)
        if (i + 1) % 5 == 0:
            print(f"  warmup {i+1}/{args.n_warmup}")

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Measured window (optionally under torch.profiler)
    rows: list[dict] = []

    def measured_loop():
        nonlocal batch_iter
        print(f"\n--- Measuring: {args.n_steps} steps ---")
        for i in range(args.n_steps):
            try:
                row = profile_step(lit_model, optimizer, batch_iter, device, use_cuda, use_bf16)
            except StopIteration:
                batch_iter = iter(loader)
                row = profile_step(lit_model, optimizer, batch_iter, device, use_cuda, use_bf16)
            rows.append(row)
            if (i + 1) % max(args.n_steps // 10, 1) == 0:
                print(f"  step {i+1}/{args.n_steps}  "
                      f"total={row['t_total_ms']:.1f}ms  "
                      f"fwd={row['t_forward_ms']:.1f}ms  "
                      f"bwd={row['t_backward_ms']:.1f}ms  "
                      f"dl={row['t_dataload_ms']:.1f}ms  "
                      f"peak={row['peak_memory_GB']:.1f}GB")

    if args.trace_path:
        print(f"Tracing to {args.trace_path} via torch.profiler")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            measured_loop()
        prof.export_chrome_trace(args.trace_path)
        print(f"Wrote Chrome trace: {args.trace_path}")
    else:
        measured_loop()

    summarize(rows, accumulate_grad_batches=args.accum_grad_batches)

    if args.output_csv:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
