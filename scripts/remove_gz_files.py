#!/usr/bin/env python3
"""
Script to remove original .cif.gz files after decompression is complete

This script safely removes .cif.gz files only if corresponding .cif files exist.
Useful when decompression was done without the --remove_gz option.

Usage:
    python remove_gz_files.py --pdb_mirror_path /path/to/pdb_mirror
"""

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import tqdm


def setup_logging(log_file: str = "remove_gz.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def remove_gz_file_safe(gz_file_path: str) -> Tuple[str, bool, str]:
    """
    Safely remove a .gz file only if corresponding .cif file exists
    
    Args:
        gz_file_path: Path to .gz file
        
    Returns:
        (file_path, success_status, message)
    """
    try:
        gz_path = Path(gz_file_path)
        cif_path = gz_path.with_suffix('')  # .cif.gz -> .cif
        
        # Check if corresponding .cif file exists
        if not cif_path.exists():
            return str(gz_file_path), False, "corresponding_cif_not_found"
        
        # Check if .cif file is not empty (basic safety check)
        if cif_path.stat().st_size == 0:
            return str(gz_file_path), False, "corresponding_cif_is_empty"
        
        # Remove the .gz file
        gz_path.unlink()
        return str(gz_file_path), True, "removed"
        
    except Exception as e:
        return str(gz_file_path), False, str(e)


def find_gz_files_with_cif(pdb_mirror_path: str) -> Tuple[List[str], List[str]]:
    """
    Find .gz files that have corresponding .cif files
    
    Returns:
        (removable_gz_files, orphaned_gz_files)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning for .cif.gz files in {pdb_mirror_path}...")
    
    removable_gz = []
    orphaned_gz = []
    pdb_path = Path(pdb_mirror_path)
    
    for gz_file in pdb_path.rglob("*.cif.gz"):
        cif_file = gz_file.with_suffix('')  # .cif.gz -> .cif
        
        if cif_file.exists() and cif_file.stat().st_size > 0:
            removable_gz.append(str(gz_file))
        else:
            orphaned_gz.append(str(gz_file))
    
    logger.info(f"Found {len(removable_gz)} .gz files with corresponding .cif files")
    logger.info(f"Found {len(orphaned_gz)} .gz files WITHOUT corresponding .cif files")
    
    return removable_gz, orphaned_gz


def process_removals(gz_files: List[str], num_workers: int = 8) -> Tuple[int, int, List[str]]:
    """
    Process file removals in parallel
    
    Returns:
        (success_count, error_count, error_list)
    """
    logger = logging.getLogger(__name__)
    
    success_count = 0
    error_count = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(remove_gz_file_safe, gz_file): gz_file 
            for gz_file in gz_files
        }
        
        # Display progress
        with tqdm.tqdm(total=len(gz_files), desc="Removing .gz files", unit="files") as pbar:
            for future in as_completed(future_to_file):
                file_path, success, message = future.result()
                
                if success:
                    success_count += 1
                    pbar.set_postfix({"Status": "Removed"})
                else:
                    error_count += 1
                    if message == "corresponding_cif_not_found":
                        error_msg = f"Skipped {file_path}: No corresponding .cif file"
                    elif message == "corresponding_cif_is_empty":
                        error_msg = f"Skipped {file_path}: Corresponding .cif file is empty"
                    else:
                        error_msg = f"Failed to remove {file_path}: {message}"
                    
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    pbar.set_postfix({"Status": "Skipped/Error"})
                
                pbar.update(1)
    
    return success_count, error_count, errors


def calculate_space_savings(gz_files: List[str]) -> float:
    """Calculate disk space that will be saved (in GB)"""
    total_size = 0
    for gz_file in gz_files:
        try:
            total_size += Path(gz_file).stat().st_size
        except:
            continue
    
    return total_size / (1024**3)


def main():
    parser = argparse.ArgumentParser(description="Remove .cif.gz files after decompression")
    parser.add_argument(
        "--pdb_mirror_path", 
        default="/home/possu/jinho/datasets/pdb_mirror",
        help="Path to PDB mirror directory"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=8, 
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true", 
        help="Only show what would be removed, don't actually remove files"
    )
    parser.add_argument(
        "--log_file",
        default="remove_gz.log",
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Check PDB mirror path
    if not os.path.exists(args.pdb_mirror_path):
        logger.error(f"PDB mirror path does not exist: {args.pdb_mirror_path}")
        return 1
    
    logger.info(f"Starting .gz file cleanup")
    logger.info(f"PDB mirror path: {args.pdb_mirror_path}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Dry run: {args.dry_run}")
    
    start_time = time.time()
    
    # Find removable and orphaned .gz files
    removable_gz, orphaned_gz = find_gz_files_with_cif(args.pdb_mirror_path)
    
    if not removable_gz and not orphaned_gz:
        logger.info("No .cif.gz files found!")
        return 0
    
    # Calculate space savings
    space_savings = calculate_space_savings(removable_gz)
    logger.info(f"Disk space that will be freed: {space_savings:.2f} GB")
    
    if orphaned_gz:
        logger.warning(f"\nFound {len(orphaned_gz)} .gz files without corresponding .cif files:")
        for orphan in orphaned_gz[:10]:  # Show first 10
            logger.warning(f"  - {orphan}")
        if len(orphaned_gz) > 10:
            logger.warning(f"  ... and {len(orphaned_gz) - 10} more")
        logger.warning("These files will NOT be removed (decompression may be incomplete)")
    
    if not removable_gz:
        logger.info("No .gz files can be safely removed (no corresponding .cif files found)")
        return 0
    
    if args.dry_run:
        logger.info(f"\nDry run completed. Would remove {len(removable_gz)} .gz files ({space_savings:.2f} GB)")
        return 0
    
    # User confirmation
    response = input(f"\nProceed with removing {len(removable_gz)} .gz files? "
                    f"(Will free {space_savings:.2f} GB of disk space) [y/N]: ")
    
    if response.lower() not in ['y', 'yes']:
        logger.info("Operation cancelled by user.")
        return 0
    
    # Process removals
    logger.info(f"Removing {len(removable_gz)} .gz files...")
    success, errors, error_list = process_removals(removable_gz, args.num_workers)
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"\n=== CLEANUP COMPLETED ===")
    logger.info(f"Total .gz files processed: {len(removable_gz)}")
    logger.info(f"Successfully removed: {success}")
    logger.info(f"Errors/Skipped: {errors}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Estimated space freed: {space_savings:.2f} GB")
    
    if error_list:
        logger.error(f"\nIssues encountered:")
        for error in error_list[:10]:  # Show first 10 errors
            logger.error(error)
        if len(error_list) > 10:
            logger.error(f"... and {len(error_list) - 10} more issues")
    
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())
