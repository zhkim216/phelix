#!/usr/bin/env python3
"""
Build metadata parquet shards from mmCIF files using AtomWorks.

This script adds a minimal per-file timeout to avoid hanging on
pathological structures. Other behavior is kept identical to the
original implementation.

Features:
- Batch processing with intermediate saves for OOM resilience
- Resume support: skips already completed batch parquet files on restart
"""

import glob
import itertools
import json
import signal
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

    # Resume: detect completed batches from existing parquet files
    completed_batches: set[int] = set()
    for f in batch_dir.glob("batch_*.parquet"):
        # filename format: batch_00001.parquet
        try:
            batch_num = int(f.stem.split("_")[1])
            completed_batches.add(batch_num)
        except Exception:
            continue
    if completed_batches:
        print(f"Resuming: found {len(completed_batches)} completed batch parquet files in {batch_dir}")
    
    # Track skipped details with reasons
    skipped_with_reason: list[tuple[str, str, str]] = []  # (path, reason, message)
    
    # Batch processing settings
    batch_size = getattr(cfg, 'batch_size', 100)
    max_tasks_per_child = getattr(cfg, 'max_tasks_per_child', 50)
    timeout = getattr(cfg, 'processing_timeout', 300)
    total_batches = (len(cif_paths) + batch_size - 1) // batch_size

    preprocessor_args = {**cfg.cif_parser_args, **cfg.data_preprocessor_cfg}

    # Process in batches (parallel or sequential), saving each batch parquet immediately
    for batch_idx in range(total_batches):
        batch_num = batch_idx + 1

        # Skip already completed batches
        if batch_num in completed_batches:
            print(f"Skipping already completed batch {batch_num}/{total_batches}")
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
                except Exception as fallback_e:
                    skipped_with_reason.append((cif_path, 'fallback_error', repr(fallback_e)))
                    batch_results.append([])
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

        # Save batch parquet immediately
        batch_df = pd.DataFrame(itertools.chain(*batch_results))
        if not batch_df.empty:
            _add_parquet_columns(batch_df, dataset_name, cfg.mmcif_dir)
            batch_parquet = batch_dir / f"batch_{batch_num:05d}.parquet"
            batch_df.to_parquet(batch_parquet)
            print(f"Saved batch {batch_num} with {len(batch_df)} rows to {batch_parquet}")

        # Update progress (only completed batches)
        completed_batches.add(batch_num)
        try:
            with open(progress_file, 'w') as f:
                json.dump({'completed_batches': sorted(completed_batches)}, f)
        except Exception:
            pass

    # Write summary log for skipped files
    try:
        with open(log_file, 'w') as f:
            f.write(f"Shard {cfg.shard_id}: total_files={len(cif_paths)}, completed_batches={len(completed_batches)}\n")
            f.write(f"Skipped {len(skipped_with_reason)} files:\n")
            for p, reason, message in skipped_with_reason:
                f.write(f"SKIPPED\t{reason}\t{message}\t{p}\n")
    except Exception:
        pass

    # Merge batch parquets into final shard parquet
    _merge_batch_parquets(batch_dir, shard_dir, cfg.shard_id)


def _add_parquet_columns(df: pd.DataFrame, dataset_name: str, pdb_in_dir: str):
    """Add example_id and rel_path columns to dataframe (in-place)."""
    # Convert all object columns to string
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
    """Merge all batch parquets into a single shard parquet."""
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
    df.to_parquet(shard_out)
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
