#!/usr/bin/env python3
"""
Build metadata parquet shards from mmCIF files using AtomWorks.

This script processes mmCIF files with timeout mechanisms to prevent hanging
on problematic structures. Features:
- Timeout mechanism to skip files that take too long to process
- Robust error handling and progress tracking
- Detailed logging for each shard
- Sharding support for distributed processing
"""

import glob
import itertools
import logging
import signal
import time
from pathlib import Path
from multiprocessing import Process, Queue
from contextlib import contextmanager

import hydra
import pandas as pd
import yaml
from atomworks.ml.example_id import generate_example_id
from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import \
    DataPreprocessor
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from p_tqdm import p_umap
from tqdm import tqdm

from allatom_design.data.preprocessing.atomworks.sharding_utils import \
    take_shard, use_sharding


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

    # Set up logging for this shard
    log_file = shard_dir / f"metadata_shard_{cfg.shard_id:05d}.log"
    setup_shard_logging(log_file, cfg.shard_id)
    logger = logging.getLogger(f"shard_{cfg.shard_id}")
    
    # Get all CIF paths, then take this shard's slice
    cif_paths_all = get_cif_paths(cfg.mmcif_dir, cfg.max_file_size)
    cif_paths = take_shard(cif_paths_all, shard_id=cfg.shard_id, num_shards=cfg.num_shards)
    print(f"Shard {cfg.shard_id}/{cfg.num_shards}: {len(cif_paths)} mmCIFs.")

    # Initialize data preprocessor
    processor = DataPreprocessor(**{
        **cfg.cif_parser_args,
        **cfg.data_preprocessor_cfg
    })

    def _process_cif_with_timeout(cif_path: str, timeout: int = 300):
        """
        Process a CIF file with timeout to avoid hanging on complex structures.
        
        Args:
            cif_path: Path to the CIF file to process
            timeout: Maximum processing time in seconds (default: 5 minutes)
            
        Returns:
            List of processed rows, or empty list if timeout/error occurs
        """
        def _target(cif_path: str, result_queue: Queue):
            try:
                result = processor.get_rows(cif_path)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        result_queue = Queue()
        process = Process(target=_target, args=(cif_path, result_queue))
        process.start()
        process.join(timeout=timeout)
        
        if process.is_alive():
            # Process timed out - this will be logged by the caller
            process.terminate()
            process.join()
            return []
        
        if result_queue.empty():
            # No result returned - this will be logged as an error by the caller
            return []
        
        status, result = result_queue.get()
        if status == 'error':
            # Error occurred - this will be logged by the caller
            return []
        
        return result

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

    logger.info(f"Starting processing of {len(cif_paths)} files...")
    print(f"Shard {cfg.shard_id}: Starting processing of {len(cif_paths)} files...")
    start_time = time.time()
    
    # Track processing statistics
    timeout_files = []
    error_files = []
    processed_results = []
    
    # Note: p_umap doesn't work well with our timeout mechanism due to multiprocessing conflicts
    # Use sequential processing with timeout for robustness
    if use_parallel:
        logger.warning("Disabling parallel processing due to timeout mechanism compatibility")
        print("Warning: Disabling parallel processing due to timeout mechanism compatibility")
        use_parallel = False
    
    # Process files sequentially with progress tracking
    for i, cif_path in enumerate(tqdm(cif_paths, desc=f"Processing mmCIFs (shard {cfg.shard_id})")):
        file_start = time.time()
        result = _process_cif(cif_path)
        file_elapsed = time.time() - file_start
        
        if not result:  # Empty result (timeout or failed)
            if file_elapsed >= getattr(cfg, 'processing_timeout', 300) - 5:  # Within 5s of timeout
                timeout_files.append(cif_path)
                logger.warning(f"Timeout processing {cif_path} ({file_elapsed:.1f}s)")
            else:
                error_files.append(cif_path)
                logger.error(f"Error processing {cif_path} ({file_elapsed:.1f}s)")
        else:
            processed_results.append(result)
            logger.debug(f"Successfully processed {cif_path} ({file_elapsed:.1f}s, {len(result)} rows)")
        
        # Log progress every 100 files
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            status_msg = (f"Processed {i + 1}/{len(cif_paths)} files "
                         f"({rate:.2f} files/sec, {len(timeout_files)} timeouts, {len(error_files)} errors)")
            logger.info(status_msg)
            print(f"Shard {cfg.shard_id}: {status_msg}")

    # Flatten list of lists and create DataFrame
    df = pd.DataFrame(itertools.chain(*processed_results))
    
    # Report final processing statistics
    elapsed = time.time() - start_time
    final_msg = (f"Processing completed in {elapsed:.1f}s. "
                f"Successfully processed {len(processed_results)} files, "
                f"{len(timeout_files)} timeouts, {len(error_files)} errors")
    logger.info(final_msg)
    print(f"Shard {cfg.shard_id}: {final_msg}")

    # If nothing produced, skip writing
    if df.empty:
        print(f"Shard {cfg.shard_id}: produced 0 rows, skipping parquet write.")
        return

    # Write one parquet per shard
    shard_out = shard_dir / f"metadata_shard_{cfg.shard_id:05d}.parquet"
    save_to_parquet(df, dataset_name, cfg.mmcif_dir, str(shard_out))
    print(f"Shard {cfg.shard_id}: wrote {len(df)} rows to {shard_out}")


def setup_shard_logging(log_file: Path, shard_id: int):
    """
    Set up logging for a specific shard.
    
    Args:
        log_file: Path to the log file for this shard
        shard_id: ID of the shard for logger naming
    """
    logger = logging.getLogger(f"shard_{shard_id}")
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False


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
