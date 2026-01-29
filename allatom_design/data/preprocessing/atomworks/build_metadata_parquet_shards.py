#!/usr/bin/env python3
"""
Build metadata parquet shards from mmCIF files using AtomWorks.

This script adds a minimal per-file timeout to avoid hanging on
pathological structures. Other behavior is kept identical to the
original implementation.

Features:
- Batch processing with intermediate saves for OOM resilience
- Resume support: skips already processed files on restart
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

def _worker_timeout_handler(signum, frame):
    """Signal handler used inside worker processes for timeouts."""
    raise TimeoutError("Processing timeout")


def _init_worker(preprocessor_args: dict, timeout: int):
    """Initializer for worker processes: build a DataPreprocessor once per worker."""
    global PROCESSOR, TIMEOUT_SECS
    PROCESSOR = DataPreprocessor(**preprocessor_args)
    TIMEOUT_SECS = timeout


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
        result = PROCESSOR.get_rows(cif_path)
        signal.alarm(0)
        return ('ok', result)
    except TimeoutError:
        return ('timeout', 'Processing timeout')
    except Exception as e:
        return ('error', repr(e))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@hydra.main(config_path="../../../configs_local/data/preprocessing/atomworks", config_name="build_metadata_parquet_shards", version_base="1.3.2")
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

    # Get all CIF paths, then take this shard's slice
    cif_paths_all = get_cif_paths(cfg.mmcif_dir, cfg.max_file_size)
    cif_paths_all = [path for path in cif_paths_all if Path(path).stem not in ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING] # exclude entries to exclude for preprocessing
    cif_paths = take_shard(cif_paths_all, shard_id=cfg.shard_id, num_shards=cfg.num_shards)
    print(f"Shard {cfg.shard_id}/{cfg.num_shards}: {len(cif_paths)} mmCIFs.")

    # Initialize data preprocessor
    processor = DataPreprocessor(**{
        **cfg.cif_parser_args,
        **cfg.data_preprocessor_cfg
    })

    def _timeout_handler(signum, frame):
        """Signal handler for timeout."""
        raise TimeoutError("Processing timeout")
    
    def _process_cif_with_timeout(cif_path: str, timeout: int = 300):
        """
        Process a CIF file with timeout to avoid hanging on complex structures.
        
        Args:
            cif_path: Path to the CIF file to process
            timeout: Maximum processing time in seconds (default: 5 minutes)
            
        Returns:
            List of processed rows, or empty list if timeout/error occurs
        """
        # Set up signal-based timeout
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = processor.get_rows(cif_path)
            signal.alarm(0)  # Cancel the alarm
            return result
        except TimeoutError:
            # Timeout occurred - this will be logged by the caller
            return []
        except Exception as e:
            # Other error occurred - this will be logged by the caller
            return []
        finally:
            signal.alarm(0)  # Make sure alarm is cancelled
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler

    def _process_cif(cif_path: str):
        """
        Wrapper around the data preprocessor with timeout.
        
        Args:
            cif_path: Path to the CIF file to process
            
        Returns:
            List of processed rows, or empty list if timeout/failed
        """
        # Use timeout for processing (configurable via config)
        timeout = getattr(cfg, 'processing_timeout', 300)  # 5 minutes default
        return _process_cif_with_timeout(cif_path, timeout)

    ### Process each CIF and save to parquet ###
    if len(cif_paths) == 0:
        print(f"Shard {cfg.shard_id}: no files to process, exiting.")
        return

    # Prepare paths for intermediate saves and progress tracking
    log_file = shard_dir / f"metadata_shard_{cfg.shard_id:05d}.log"
    progress_file = shard_dir / f"metadata_shard_{cfg.shard_id:05d}_progress.json"
    batch_dir = shard_dir / f"shard_{cfg.shard_id:05d}_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Load progress if resuming
    processed_paths: set[str] = set()
    completed_batches: set[int] = set()
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_paths = set(progress_data.get('processed_paths', []))
                completed_batches = set(progress_data.get('completed_batches', []))
            print(f"Resuming: found {len(processed_paths)} already processed files, {len(completed_batches)} completed batches.")
        except Exception as e:
            print(f"Warning: could not load progress file: {e}")
    
    # Track skipped details with reasons
    skipped_with_reason: list[tuple[str, str, str]] = []  # (path, reason, message)
    
    # Batch processing settings
    batch_size = getattr(cfg, 'batch_size', 100)
    max_tasks_per_child = getattr(cfg, 'max_tasks_per_child', 50)
    total_batches = (len(cif_paths) + batch_size - 1) // batch_size

    # Process files (sequential or parallel with per-file timeouts)
    if use_parallel:
        preprocessor_args = {**cfg.cif_parser_args, **cfg.data_preprocessor_cfg}
        timeout = getattr(cfg, 'processing_timeout', 300)
        
        for batch_idx in range(total_batches):
            batch_num = batch_idx + 1
            
            # Skip already completed batches
            if batch_num in completed_batches:
                print(f"Skipping already completed batch {batch_num}/{total_batches}")
                continue
            
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(cif_paths))
            batch_paths = cif_paths[batch_start:batch_end]
            batch_results = []
            
            try:
                with ProcessPoolExecutor(
                    max_workers=cfg.num_workers, 
                    initializer=_init_worker,
                    initargs=(preprocessor_args, timeout),
                    max_tasks_per_child=max_tasks_per_child
                ) as executor:
                    futures = []
                    for cif_path in batch_paths:
                        futures.append(executor.submit(_process_cif_worker, cif_path))
                    for idx, fut in enumerate(tqdm(
                        futures,
                        total=len(batch_paths), 
                        desc=f"Processing mmCIFs (shard {cfg.shard_id}, batch {batch_num}/{total_batches})"
                    )):
                        try:
                            status, payload = fut.result(timeout=timeout + 30)
                        except FuturesTimeoutError:
                            skipped_with_reason.append((batch_paths[idx], 'parent_timeout', f'Parent timed out after {timeout + 30}s'))
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
                print(f"Falling back to sequential processing for this batch...")
                
                def _fallback_timeout_handler(signum, frame):
                    raise TimeoutError("Fallback processing timeout")
                
                for idx, cif_path in enumerate(batch_paths):
                    if idx < len(batch_results):
                        continue
                    
                    # Apply timeout to fallback processing
                    old_handler = signal.signal(signal.SIGALRM, _fallback_timeout_handler)
                    signal.alarm(timeout)
                    try:
                        fallback_processor = DataPreprocessor(**preprocessor_args)
                        res = fallback_processor.get_rows(cif_path)
                        signal.alarm(0)
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
                # Add columns before saving
                _add_parquet_columns(batch_df, dataset_name, cfg.mmcif_dir)
                batch_parquet = batch_dir / f"batch_{batch_num:05d}.parquet"
                batch_df.to_parquet(batch_parquet)
                print(f"Saved batch {batch_num} with {len(batch_df)} rows to {batch_parquet}")
            
            # Update progress
            processed_paths.update(batch_paths)
            completed_batches.add(batch_num)
            try:
                with open(progress_file, 'w') as f:
                    json.dump({
                        'processed_paths': list(processed_paths),
                        'completed_batches': list(completed_batches)
                    }, f)
            except Exception:
                pass
                
    else:
        # Sequential mode - process all at once, save as single batch
        results_list = []
        for cif_path in tqdm(cif_paths, desc=f"Processing mmCIFs (shard {cfg.shard_id})"):
            try:
                res = _process_cif(cif_path)
                if not res:
                    skipped_with_reason.append((cif_path, 'timeout_or_error', '-'))
                results_list.append(res)
            except TimeoutError:
                skipped_with_reason.append((cif_path, 'timeout', 'Processing timeout'))
                results_list.append([])
            except Exception as e:
                skipped_with_reason.append((cif_path, 'error', repr(e)))
                results_list.append([])
        
        batch_df = pd.DataFrame(itertools.chain(*results_list))
        if not batch_df.empty:
            _add_parquet_columns(batch_df, dataset_name, cfg.mmcif_dir)
            batch_parquet = batch_dir / "batch_00001.parquet"
            batch_df.to_parquet(batch_parquet)

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
    cif_paths = glob.glob(f"{mmcif_dir}/**/*.cif", recursive=True)
    
    # Filter by file size only
    if max_file_size is not None:
        original_len = len(cif_paths)
        cif_paths = [path for path in cif_paths if Path(path).stat().st_size <= max_file_size]
        print(f"Excluded {original_len - len(cif_paths)} files due to size.")
    
    print(f"Found {len(cif_paths)} mmCIFs.")
    return cif_paths




if __name__ == "__main__":
    main()
