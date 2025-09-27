#!/usr/bin/env python3
"""
Script to decompress PDB mirror gz files in parallel

Usage:
    python decompress_pdb_mirror_parallel.py --pdb_mirror_path /path/to/pdb_mirror --num_workers 8
    
Features:
    - Parallel processing with multiprocessing
    - Progress display
    - Error logging
    - Skip already decompressed files
    - Batch processing for memory efficiency
"""

import argparse
import gzip
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import tqdm


def setup_logging(log_file: str = "decompress_pdb.log"):
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


def decompress_single_file(gz_file_path: str) -> Tuple[str, bool, str]:
    """
    Decompress a single gz file
    
    Args:
        gz_file_path: Path to compressed file
        
    Returns:
        (file_path, success_status, error_message)
    """
    try:
        gz_path = Path(gz_file_path)
        cif_path = gz_path.with_suffix('')  # .cif.gz -> .cif
        
        # Skip if already decompressed file exists
        if cif_path.exists():
            return str(gz_file_path), True, "already_exists"
        
        # Decompress the file
        with gzip.open(gz_path, 'rb') as f_in:
            with open(cif_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return str(gz_file_path), True, "success"
        
    except Exception as e:
        return str(gz_file_path), False, str(e)


def find_gz_files(pdb_mirror_path: str) -> List[str]:
    """Find all .cif.gz files in PDB mirror directory"""
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning for .cif.gz files in {pdb_mirror_path}...")
    
    gz_files = []
    pdb_path = Path(pdb_mirror_path)
    
    for gz_file in pdb_path.rglob("*.cif.gz"):
        gz_files.append(str(gz_file))
    
    logger.info(f"Found {len(gz_files)} .cif.gz files")
    return gz_files


def process_batch(gz_files: List[str], num_workers: int = 8) -> Tuple[int, int, List[str]]:
    """
    Process files in parallel batch
    
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
            executor.submit(decompress_single_file, gz_file): gz_file 
            for gz_file in gz_files
        }
        
        # Display progress
        with tqdm.tqdm(total=len(gz_files), desc="Decompressing", unit="files") as pbar:
            for future in as_completed(future_to_file):
                file_path, success, message = future.result()
                
                if success:
                    success_count += 1
                    if message == "already_exists":
                        pbar.set_postfix({"Status": "Skipped (exists)"})
                    else:
                        pbar.set_postfix({"Status": "Decompressed"})
                else:
                    error_count += 1
                    error_msg = f"Failed to decompress {file_path}: {message}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    pbar.set_postfix({"Status": "Error"})
                
                pbar.update(1)
    
    return success_count, error_count, errors


def estimate_disk_space(gz_files: List[str], sample_size: int = 100) -> float:
    """Estimate required disk space after decompression (in GB)"""
    logger = logging.getLogger(__name__)
    
    if len(gz_files) == 0:
        return 0.0
    
    # 샘플링으로 압축률 추정
    sample_files = gz_files[:min(sample_size, len(gz_files))]
    total_compressed = 0
    total_uncompressed = 0
    
    logger.info(f"Estimating disk space using {len(sample_files)} sample files...")
    
    for gz_file in sample_files:
        try:
            gz_path = Path(gz_file)
            total_compressed += gz_path.stat().st_size
            
            with gzip.open(gz_path, 'rb') as f:
                # 압축 해제된 크기 계산
                f.seek(0, 2)  # 파일 끝으로 이동
                total_uncompressed += f.tell()
                
        except Exception as e:
            logger.warning(f"Could not process sample file {gz_file}: {e}")
            continue
    
    if total_compressed == 0:
        return 0.0
    
    # 압축률 계산
    compression_ratio = total_uncompressed / total_compressed
    
    # 전체 압축된 파일들의 크기 계산
    total_gz_size = sum(Path(gz_file).stat().st_size for gz_file in gz_files)
    
    # 예상 압축 해제 크기 (GB)
    estimated_size_gb = (total_gz_size * compression_ratio) / (1024**3)
    
    logger.info(f"Estimated compression ratio: {compression_ratio:.2f}x")
    logger.info(f"Current compressed size: {total_gz_size / (1024**3):.2f} GB")
    logger.info(f"Estimated uncompressed size: {estimated_size_gb:.2f} GB")
    
    return estimated_size_gb


def main():
    parser = argparse.ArgumentParser(description="Decompress PDB mirror gz files in parallel")
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
        "--batch_size", 
        type=int, 
        default=10000, 
        help="Batch size for processing"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true", 
        help="Only estimate disk space, don't decompress"
    )
    parser.add_argument(
        "--log_file",
        default="decompress_pdb.log",
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging(args.log_file)
    
    # PDB mirror 경로 확인
    if not os.path.exists(args.pdb_mirror_path):
        logger.error(f"PDB mirror path does not exist: {args.pdb_mirror_path}")
        return 1
    
    logger.info(f"Starting PDB mirror decompression")
    logger.info(f"PDB mirror path: {args.pdb_mirror_path}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Batch size: {args.batch_size}")
    
    start_time = time.time()
    
    # gz 파일들 찾기
    gz_files = find_gz_files(args.pdb_mirror_path)
    
    if not gz_files:
        logger.warning("No .cif.gz files found!")
        return 0
    
    # 디스크 공간 추정
    estimated_size = estimate_disk_space(gz_files)
    logger.info(f"Estimated additional disk space needed: {estimated_size:.2f} GB")
    
    if args.dry_run:
        logger.info("Dry run completed. No files were decompressed.")
        return 0
    
    # 사용자 확인
    response = input(f"\nProceed with decompressing {len(gz_files)} files? "
                    f"(Estimated {estimated_size:.2f} GB additional space needed) [y/N]: ")
    
    if response.lower() not in ['y', 'yes']:
        logger.info("Operation cancelled by user.")
        return 0
    
    # 배치별로 처리
    total_success = 0
    total_errors = 0
    all_errors = []
    
    for i in range(0, len(gz_files), args.batch_size):
        batch = gz_files[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(gz_files) + args.batch_size - 1) // args.batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        success, errors, error_list = process_batch(batch, args.num_workers)
        total_success += success
        total_errors += errors
        all_errors.extend(error_list)
        
        logger.info(f"Batch {batch_num} completed: {success} success, {errors} errors")
    
    # 최종 결과
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"\n=== DECOMPRESSION COMPLETED ===")
    logger.info(f"Total files processed: {len(gz_files)}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Average speed: {len(gz_files)/duration:.2f} files/second")
    
    if all_errors:
        logger.error(f"\nErrors encountered:")
        for error in all_errors[:10]:  # 처음 10개 에러만 표시
            logger.error(error)
        if len(all_errors) > 10:
            logger.error(f"... and {len(all_errors) - 10} more errors")
    
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    exit(main())
