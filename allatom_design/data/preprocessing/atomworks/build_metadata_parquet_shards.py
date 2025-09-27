#!/usr/bin/env python3
"""
Build metadata parquet shards from mmCIF files using AtomWorks.

This script adds a minimal per-file timeout to avoid hanging on
pathological structures. Other behavior is kept identical to the
original implementation.
"""

import glob
import itertools
import signal
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import hydra
import pandas as pd
import yaml
from atomworks.ml.example_id import generate_example_id
from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import \
    DataPreprocessor
from natsort import natsorted
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

    # Get all CIF paths, then take this shard's slice
    cif_paths_all = get_cif_paths(cfg.mmcif_dir, cfg.max_file_size)
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

    # Prepare logging file path (summary of skipped files)
    log_file = shard_dir / f"metadata_shard_{cfg.shard_id:05d}.log"
    
    # Track skipped details with reasons
    skipped_with_reason: list[tuple[str, str, str]] = []  # (path, reason, message)

    # Process files (sequential or parallel with per-file timeouts)
    if use_parallel:
        # Build args for workers once
        preprocessor_args = {**cfg.cif_parser_args, **cfg.data_preprocessor_cfg}
        timeout = getattr(cfg, 'processing_timeout', 300)
        # Pre-allocate results in input order to preserve deterministic ordering
        results_list = [None] * len(cif_paths)
        # Use ProcessPoolExecutor to get true parallelism and isolate timeouts
        with ProcessPoolExecutor(max_workers=cfg.num_workers, initializer=_init_worker,
                                 initargs=(preprocessor_args, timeout)) as executor:
            for idx, res in enumerate(tqdm(executor.map(_process_cif_worker, cif_paths), total=len(cif_paths), desc=f"Processing mmCIFs (shard {cfg.shard_id})")):
                status, payload = res
                if status == 'ok':
                    results_list[idx] = payload
                else:
                    message = payload if isinstance(payload, str) else repr(payload)
                    skipped_with_reason.append((cif_paths[idx], status, message))
                    results_list[idx] = []
    else:
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

    # Flatten list of lists and create DataFrame
    df = pd.DataFrame(itertools.chain(*results_list))

    # Write summary log for skipped files
    try:
        with open(log_file, 'w') as f:
            f.write(f"Shard {cfg.shard_id}: processed={len(results_list)}\n")
            f.write(f"Skipped {len(skipped_with_reason)} files:\n")
            for p, reason, message in skipped_with_reason:
                f.write(f"SKIPPED\t{reason}\t{message}\t{p}\n")
    except Exception:
        # Logging should never crash the run
        pass

    # If nothing produced, skip writing
    if df.empty:
        print(f"Shard {cfg.shard_id}: produced 0 rows, skipping parquet write.")
        return

    # Write one parquet per shard
    shard_out = shard_dir / f"metadata_shard_{cfg.shard_id:05d}.parquet"
    save_to_parquet(df, dataset_name, cfg.mmcif_dir, str(shard_out))
    print(f"Shard {cfg.shard_id}: wrote {len(df)} rows to {shard_out}")


def get_cif_paths(mmcif_dir: str, max_file_size: int | None = None) -> list[str]:
    """
    Get list of CIF file paths with optional size filtering.
    
    Args:
        mmcif_dir: Directory containing mmCIF files
        max_file_size: Maximum file size in bytes (None to disable)
        
    Returns:
        List of CIF file paths after filtering
    """
    cif_paths = glob.glob(f"{mmcif_dir}/**/*.cif.gz", recursive=True)
    
    # Filter by file size only
    if max_file_size is not None:
        original_len = len(cif_paths)
        cif_paths = [path for path in cif_paths if Path(path).stat().st_size <= max_file_size]
        print(f"Excluded {original_len - len(cif_paths)} files due to size.")
    
    print(f"Found {len(cif_paths)} mmCIFs.")
    return cif_paths


def save_to_parquet(df: pd.DataFrame,
                    dataset_name: str,
                    pdb_in_dir: str,
                    out_path: str):
    """
    Save a dataframe to parquet.
    Also adds an example_id column based on the name of the dataset.
    """
    # Convert all object columns to string to save to parquet
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    # Add example_id based on the name of this dataset
    df["example_id"] = df.apply(lambda x: generate_example_id(
        [dataset_name],
        x["pdb_id"],
        x["assembly_id"],
        [x["q_pn_unit_iid"]]), axis=1)

    # Add in relative path
    df["rel_path"] = df["path"].apply(lambda x: str(Path(x).relative_to(pdb_in_dir)))

    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
