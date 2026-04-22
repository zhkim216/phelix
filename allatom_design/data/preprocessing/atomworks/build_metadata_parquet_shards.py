#!/usr/bin/env python3
"""
Build metadata parquet shards from mmCIF files using AtomWorks.

Per-file timeout is applied via `signal.alarm()` inside worker processes
to avoid hanging on pathological structures.

Features:
- Batch processing with intermediate saves for OOM resilience.
- Atomic writes (tmp + os.replace) for batch parquets, progress JSON,
  summary log, and final shard parquet — safe against SLURM wall-time kills.
- Resume support: on restart, rebuilds the completed-batch set from the
  *union* of the progress JSON (`completed_batches`) and on-disk batch
  parquets (glob `batch_*.parquet`). Either source alone is insufficient.
- "Attempted-empty" tracking: if every file in a batch fails/times out
  the batch is recorded in `attempted_empty_batches` so we don't retry
  a dead batch forever. Override with `retry_empty_batches: true`.
- Progress JSON schema is a backward-compatible superset — legacy files
  containing only `completed_batches` still work.
"""

import glob
import itertools
import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

import hydra
import pandas as pd
import yaml
from atomworks.ml.preprocessing.constants import ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING
from atomworks.ml.example_id import generate_example_id
from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import \
    DataPreprocessor
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.data.preprocessing.atomworks.sharding_utils import \
    take_shard, use_sharding

# Globals for worker processes (used in parallel mode)
PROCESSOR = None
TIMEOUT_SECS = 300
FETCH_LIGAND_VALIDITY_SCORES = True
DEFAULT_LIGAND_SCORES = [
    "RSCC",
    "RSR",
    "completeness",
    "intermolecular_clashes",
    "is_best_instance",
    "ranking_model_fit",
    "ranking_model_geometry",
]

# Upstream atomworks raises this when _entity_poly/_struct_asym declares a polymer chain
# with no observed atoms. Treat as a data-quality skip, not a pipeline error.
_MISMATCH_MARKER = "Mismatch between `atom_array` and `chain_to_sequence`"


def _atomic_write_parquet(df: pd.DataFrame, final_path: Path) -> None:
    """Write a parquet atomically via tmp + os.replace to avoid torn files on crashes."""
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, final_path)


def _atomic_write_json(final_path: Path, obj: dict) -> None:
    """Write a JSON atomically, fsync'd, to avoid torn progress files."""
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final_path)


def _atomic_write_text(final_path: Path, text: str) -> None:
    """Write a text file atomically via tmp + os.replace."""
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final_path)


def _load_progress(progress_file: Path) -> dict:
    """Load progress JSON; return {} if missing or unreadable.

    The progress JSON schema is a *superset*: legacy files contain only
    `completed_batches`; new keys (`attempted_empty_batches`, `batch_size`,
    `total_files`, `total_batches`, `updated_at`) are read via `.get(..., default)`
    so legacy progress files remain readable.
    """
    if not progress_file.exists():
        return {}
    try:
        with open(progress_file) as f:
            return json.load(f)
    except Exception as e:
        print(
            f"WARNING: progress file {progress_file} is unreadable ({e!r}); "
            "treating as empty. On-disk batch parquets will still be picked up."
        )
        return {}


def _worker_timeout_handler(signum, frame):
    """Signal handler used inside worker processes for timeouts."""
    raise TimeoutError("Processing timeout")


def _init_worker(preprocessor_args: dict, timeout: int, fetch_ligand_validity_scores: bool):
    """Initializer for worker processes: build a DataPreprocessor once per worker."""
    global PROCESSOR, TIMEOUT_SECS, FETCH_LIGAND_VALIDITY_SCORES
    PROCESSOR = DataPreprocessor(**preprocessor_args)
    TIMEOUT_SECS = timeout
    FETCH_LIGAND_VALIDITY_SCORES = fetch_ligand_validity_scores


def _process_cif_worker(cif_path: str):
    """Worker entrypoint used in parallel mode with per-file timeout.

    Returns a tuple (status, payload):
      - ('ok', rows)
      - ('timeout', 'Processing timeout')
      - ('error', repr(exception))
    """
    old_handler = signal.signal(signal.SIGALRM, _worker_timeout_handler)
    signal.alarm(TIMEOUT_SECS)
    try:
        # IMPORTANT: ligand validity score retrieval hits RCSB network.
        # For large-scale preprocessing on clusters this often causes timeouts/hangs.
        # We therefore allow disabling it explicitly via config.
        if FETCH_LIGAND_VALIDITY_SCORES:
            result = PROCESSOR.get_rows(cif_path, ligand_scores=DEFAULT_LIGAND_SCORES)
        else:
            result = PROCESSOR.get_rows(cif_path, ligand_scores=[])
        signal.alarm(0)
        return ('ok', result)
    except TimeoutError:
        return ('timeout', 'Processing timeout')
    except ValueError as e:
        if _MISMATCH_MARKER in str(e):
            return ('skip_data_quality', str(e))
        return ('error', repr(e))
    except Exception as e:
        return ('error', repr(e))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@hydra.main(config_path="../../../configs/data/preprocessing/atomworks", config_name="build_metadata_parquet_shards", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Process a set of mmCIFs using AtomWorks.
    
    This script supports sharding by setting:
      - num_shards (int) and shard_id (int)
    """
    # Create dataset directory + shard dir
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Only one shard writes the canonical config.yaml to avoid races
    if not use_sharding(cfg.shard_id, cfg.num_shards) or (cfg.shard_id == 0):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg_dict, f)

    # Setup
    use_parallel = cfg.num_workers > 1
    dataset_name = Path(cfg.out_dir).stem
    fetch_ligand_validity_scores = getattr(cfg, "fetch_ligand_validity_scores", True)

    # Get all CIF paths, then take this shard's slice
    cif_list_path = getattr(cfg, "cif_list_path", None)
    if cif_list_path:
        cif_paths_all = _read_cif_list(cif_list_path)
        if cfg.max_file_size is not None:
            original_len = len(cif_paths_all)
            cif_paths_all = [p for p in cif_paths_all if Path(p).stat().st_size <= cfg.max_file_size]
            print(f"Excluded {original_len - len(cif_paths_all)} files due to size.")
        print(f"Loaded {len(cif_paths_all)} mmCIFs from {cif_list_path}.")
    else:
        cif_paths_all = get_cif_paths(cfg.mmcif_dir, cfg.max_file_size)
    cif_paths_all = [path for path in cif_paths_all if Path(path).stem not in ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING] # exclude entries to exclude for preprocessing
    cif_paths = take_shard(cif_paths_all, shard_id=cfg.shard_id, num_shards=cfg.num_shards)
    print(f"Shard {cfg.shard_id}/{cfg.num_shards}: {len(cif_paths)} mmCIFs.")

    ### Process each CIF and save to parquet ###
    if len(cif_paths) == 0:
        print(f"Shard {cfg.shard_id}: no files to process, exiting.")
        return

    # Prepare paths for intermediate saves and progress tracking
    log_file = shard_dir / f"metadata_shard_{cfg.shard_id:05d}.log"
    progress_file = shard_dir / f"metadata_shard_{cfg.shard_id:05d}_progress.json"
    batch_dir = shard_dir / f"shard_{cfg.shard_id:05d}_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale .tmp files left by crashed atomic writes
    for stale in batch_dir.glob("batch_*.parquet.tmp"):
        try:
            stale.unlink()
            print(f"Removed stale tmp file: {stale}")
        except OSError as e:
            print(f"WARNING: could not remove stale tmp file {stale}: {e!r}")
    for stale_name in (
        f"metadata_shard_{cfg.shard_id:05d}.parquet.tmp",
        f"metadata_shard_{cfg.shard_id:05d}_progress.json.tmp",
        f"metadata_shard_{cfg.shard_id:05d}.log.tmp",
    ):
        stale = shard_dir / stale_name
        if stale.exists():
            try:
                stale.unlink()
                print(f"Removed stale tmp file: {stale}")
            except OSError as e:
                print(f"WARNING: could not remove stale tmp file {stale}: {e!r}")

    # Batch processing settings (defined up front so resume checks can use them)
    batch_size = getattr(cfg, "batch_size", 100)
    max_tasks_per_child = getattr(cfg, "max_tasks_per_child", 50)
    timeout = getattr(cfg, "processing_timeout", 300)
    retry_empty_batches = bool(getattr(cfg, "retry_empty_batches", False))
    total_batches = (len(cif_paths) + batch_size - 1) // batch_size

    # Resume: union of (a) progress JSON and (b) on-disk batch parquet glob.
    # Either source alone is insufficient: the JSON might be corrupt/stale, or
    # the process may have died between writing a batch and updating the JSON.
    progress = _load_progress(progress_file)
    completed_batches: set[int] = set(progress.get("completed_batches", []))
    attempted_empty: set[int] = set(progress.get("attempted_empty_batches", []))
    prev_batch_size = progress.get("batch_size")
    prev_total_files = progress.get("total_files")

    on_disk_batches: set[int] = set()
    for f in batch_dir.glob("batch_*.parquet"):
        # filename format: batch_00001.parquet
        try:
            on_disk_batches.add(int(f.stem.split("_")[1]))
        except Exception:
            continue
    completed_batches |= on_disk_batches
    # If a batch is now on disk, it is no longer "attempted empty".
    attempted_empty -= on_disk_batches

    if completed_batches or attempted_empty:
        print(
            f"Resuming shard {cfg.shard_id}: "
            f"{len(completed_batches)} completed batches, "
            f"{len(attempted_empty)} attempted-empty batches "
            f"(from progress.json + {len(on_disk_batches)} on-disk batch parquets)."
        )

    # Consistency checks: bail if batch_size changed, warn if input size changed.
    if prev_batch_size is not None and prev_batch_size != batch_size:
        raise SystemExit(
            f"batch_size mismatch for shard {cfg.shard_id}: previous run used "
            f"batch_size={prev_batch_size}, current run uses batch_size={batch_size}. "
            f"Mixing batch sizes would cause batch_num->file_range drift. "
            f"Restore batch_size={prev_batch_size} or delete {batch_dir} and {progress_file} "
            f"to restart this shard from scratch."
        )
    if prev_total_files is not None and prev_total_files != len(cif_paths):
        print(
            f"WARNING: shard {cfg.shard_id} input file count changed since last run: "
            f"prev={prev_total_files}, now={len(cif_paths)}. "
            "Already-saved batches will still be reused, but new-vs-old file ordering "
            "may have shifted — re-run from scratch if input set meaningfully changed."
        )

    # Track skipped details with reasons
    skipped_with_reason: list[tuple[str, str, str]] = []  # (path, reason, message)

    preprocessor_args = {**cfg.cif_parser_args, **cfg.data_preprocessor_cfg}

    # Process in batches (parallel or sequential), saving each batch parquet immediately
    for batch_idx in range(total_batches):
        batch_num = batch_idx + 1

        # Skip already completed batches
        if batch_num in completed_batches:
            print(f"Skipping already completed batch {batch_num}/{total_batches}")
            continue
        if (batch_num in attempted_empty) and not retry_empty_batches:
            print(
                f"Skipping previously-empty batch {batch_num}/{total_batches} "
                "(all files in this batch failed before). "
                "Set `retry_empty_batches: true` in config to retry."
            )
            continue

        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(cif_paths))
        batch_paths = cif_paths[batch_start:batch_end]
        batch_results: list[list[dict]] = []

        fallback_to_sequential = not use_parallel

        if use_parallel:
            try:
                with ProcessPoolExecutor(
                    max_workers=cfg.num_workers,
                    initializer=_init_worker,
                    initargs=(preprocessor_args, timeout, fetch_ligand_validity_scores),
                    max_tasks_per_child=max_tasks_per_child,
                ) as executor:
                    futures = [executor.submit(_process_cif_worker, cif_path) for cif_path in batch_paths]
                    for idx, fut in enumerate(tqdm(
                        futures,
                        total=len(batch_paths),
                        desc=f"Processing mmCIFs (shard {cfg.shard_id}, batch {batch_num}/{total_batches})",
                    )):
                        try:
                            status, payload = fut.result(timeout=timeout + 30)
                        except FuturesTimeoutError:
                            skipped_with_reason.append(
                                (batch_paths[idx], 'parent_timeout', f'Parent timed out after {timeout + 30}s')
                            )
                            batch_results.append([])
                            continue
                        if status == 'ok':
                            batch_results.append(payload)
                        else:
                            message = payload if isinstance(payload, str) else repr(payload)
                            skipped_with_reason.append((batch_paths[idx], status, message))
                            batch_results.append([])
            except Exception as e:
                print(f"WARNING: Batch {batch_num} failed with error: {repr(e)}")
                print("Falling back to sequential processing for this batch...")
                fallback_to_sequential = True

        if fallback_to_sequential:
            def _fallback_timeout_handler(signum, frame):
                raise TimeoutError("Fallback processing timeout")

            fallback_processor = DataPreprocessor(**preprocessor_args)
            for cif_path in tqdm(batch_paths, desc=f"Fallback sequential (shard {cfg.shard_id}, batch {batch_num}/{total_batches})"):
                old_handler = signal.signal(signal.SIGALRM, _fallback_timeout_handler)
                signal.alarm(timeout)
                try:
                    if fetch_ligand_validity_scores:
                        res = fallback_processor.get_rows(cif_path, ligand_scores=DEFAULT_LIGAND_SCORES)
                    else:
                        res = fallback_processor.get_rows(cif_path, ligand_scores=[])
                    batch_results.append(res)
                except TimeoutError:
                    skipped_with_reason.append((cif_path, 'fallback_timeout', 'Fallback processing timeout'))
                    batch_results.append([])
                except ValueError as fallback_e:
                    if _MISMATCH_MARKER in str(fallback_e):
                        skipped_with_reason.append((cif_path, 'skip_data_quality', str(fallback_e)))
                    else:
                        skipped_with_reason.append((cif_path, 'fallback_error', repr(fallback_e)))
                    batch_results.append([])
                except Exception as fallback_e:
                    skipped_with_reason.append((cif_path, 'fallback_error', repr(fallback_e)))
                    batch_results.append([])
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

        # Save batch parquet immediately (atomic: tmp -> rename).
        # An empty batch means every file in this slice timed out or errored.
        # Track it separately so we don't retry it on every resume.
        batch_df = pd.DataFrame(itertools.chain(*batch_results))
        batch_parquet = batch_dir / f"batch_{batch_num:05d}.parquet"
        if not batch_df.empty:
            _add_parquet_columns(batch_df, dataset_name, cfg.mmcif_dir)
            try:
                _atomic_write_parquet(batch_df, batch_parquet)
                completed_batches.add(batch_num)
                attempted_empty.discard(batch_num)
                print(f"Saved batch {batch_num} with {len(batch_df)} rows to {batch_parquet}")
            except Exception as e:
                # Write failure is a hard error: we lose this batch's work for this run,
                # but we leave the slot available for retry on the next run.
                print(f"ERROR: failed to write {batch_parquet}: {e!r}")
        else:
            attempted_empty.add(batch_num)
            print(
                f"Batch {batch_num}/{total_batches}: produced 0 rows "
                f"(all {len(batch_paths)} files failed/timed out); marked attempted_empty."
            )

        # Update progress JSON (atomic). Legacy key `completed_batches` is preserved;
        # new keys are additive so older readers still work via .get().
        try:
            _atomic_write_json(
                progress_file,
                {
                    "completed_batches": sorted(completed_batches),
                    "attempted_empty_batches": sorted(attempted_empty),
                    "batch_size": batch_size,
                    "total_files": len(cif_paths),
                    "total_batches": total_batches,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
        except Exception as e:
            print(f"WARNING: failed to write progress file {progress_file}: {e!r}")

    # Write summary log for skipped files (atomic).
    log_lines = [
        f"Shard {cfg.shard_id}: total_files={len(cif_paths)}, "
        f"completed_batches={len(completed_batches)}, "
        f"attempted_empty_batches={len(attempted_empty)}",
        f"Skipped {len(skipped_with_reason)} files:",
    ]
    for p, reason, message in skipped_with_reason:
        log_lines.append(f"SKIPPED\t{reason}\t{message}\t{p}")
    try:
        _atomic_write_text(log_file, "\n".join(log_lines) + "\n")
    except Exception as e:
        print(f"WARNING: failed to write log file {log_file}: {e!r}")

    # Merge batch parquets into final shard parquet
    _merge_batch_parquets(batch_dir, shard_dir, cfg.shard_id)


def _add_parquet_columns(df: pd.DataFrame, dataset_name: str, pdb_in_dir: str):
    """Add example_id and rel_path columns to dataframe (in-place)."""
    # Numeric-nullable columns: cast explicitly to float (None → NaN) BEFORE the blanket
    # object→str conversion below. Without this, a batch where every row has ``None`` for
    # one of these fields (e.g. a shard with no halide or no metal PN units) gets its
    # object column stringified to ``"None"``, which then fails type inference at merge time
    # against other batches where the same column ended up as float64.
    numeric_nullable_cols = (
        "q_pn_unit_n_coordination_partners_metal",
        "q_pn_unit_n_coordination_partners_halide",
        "q_pn_unit_n_neighboring_heavy_atoms_small_molecule",
        "q_pn_unit_avg_occupancy_nonpolymer",
    )
    for col in numeric_nullable_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert all remaining object columns to string (e.g. JSON-stringified list/dict fields).
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    # Add example_id
    df["example_id"] = df.apply(
        lambda x: generate_example_id(
            [str(dataset_name)],
            str(x["pdb_id"]),
            str(x["assembly_id"]),
            [str(x["q_pn_unit_iid"])],
        ),
        axis=1,
    )
    
    # Add relative path
    df["rel_path"] = df["path"].apply(lambda x: str(Path(x).relative_to(pdb_in_dir)))


def _merge_batch_parquets(batch_dir: Path, shard_dir: Path, shard_id: int):
    """Merge all batch parquets into a single shard parquet (atomic)."""
    batch_files = sorted(batch_dir.glob("batch_*.parquet"))
    if not batch_files:
        print(f"Shard {shard_id}: no batch files to merge.")
        return

    dfs = [pd.read_parquet(f) for f in batch_files]
    df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        print(f"Shard {shard_id}: produced 0 rows after merging.")
        return

    shard_out = shard_dir / f"metadata_shard_{shard_id:05d}.parquet"
    _atomic_write_parquet(df, shard_out)
    print(f"Shard {shard_id}: merged {len(batch_files)} batches, wrote {len(df)} rows to {shard_out}")


def get_cif_paths(mmcif_dir: str, max_file_size: int | None = None) -> list[str]:
    """
    Get list of CIF file paths with optional size filtering.
    
    Args:
        mmcif_dir: Directory containing mmCIF files
        max_file_size: Maximum file size in bytes (None to disable)
        
    Returns:
        List of CIF file paths after filtering
    """
    cif_paths = []
    cif_paths.extend(glob.glob(f"{mmcif_dir}/**/*.cif", recursive=True))
    cif_paths.extend(glob.glob(f"{mmcif_dir}/**/*.cif.gz", recursive=True))
    cif_paths.extend(glob.glob(f"{mmcif_dir}/**/*.mmcif", recursive=True))
    cif_paths.extend(glob.glob(f"{mmcif_dir}/**/*.mmcif.gz", recursive=True))
    # De-dup while keeping order-ish
    cif_paths = list(dict.fromkeys(cif_paths))
    
    # Filter by file size only
    if max_file_size is not None:
        original_len = len(cif_paths)
        cif_paths = [path for path in cif_paths if Path(path).stat().st_size <= max_file_size]
        print(f"Excluded {original_len - len(cif_paths)} files due to size.")
    
    print(f"Found {len(cif_paths)} mmCIFs.")
    return cif_paths


def _read_cif_list(cif_list_path: str) -> list[str]:
    """Read a newline-delimited list of absolute/relative mmCIF paths."""
    paths: list[str] = []
    with open(cif_list_path, "r") as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            paths.append(p)
    return paths




if __name__ == "__main__":
    main()
