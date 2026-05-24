"""
Download the SAIR (Structurally Augmented IC50 Repository) dataset from HuggingFace.

Source: HuggingFace - SandboxAQ/SAIR
Contents: 5.2M protein-ligand 3D structures with IC50 labels

Usage:
    python download_sair_dataset.py --download-dir /scratch/users/zhkim216/sair
    python download_sair_dataset.py --download-dir /scratch/users/zhkim216/sair --structures-only
    python download_sair_dataset.py --download-dir /scratch/users/zhkim216/sair --parquet-only
"""
import argparse
import logging
import os
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "SandboxAQ/SAIR"

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path | None = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def list_repo_contents() -> tuple[list[str], list[str]]:
    """List parquet and structure archive files in the SAIR repo."""
    logger.info(f"Listing all files in {REPO_ID}...")
    all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    structure_files = [
        f
        for f in all_files
        if f.startswith("structures_compressed/") and f.endswith(".tar.gz")
    ]

    logger.info(f"Found {len(parquet_files)} parquet file(s)")
    logger.info(f"Found {len(structure_files)} structure archive(s)")
    return parquet_files, structure_files


def download_parquet(parquet_files: list[str], parquet_dir: Path):
    """Download parquet metadata files."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for pf in parquet_files:
        logger.info(f"Downloading parquet: {pf}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=pf,
            repo_type="dataset",
            local_dir=str(parquet_dir),
            local_dir_use_symlinks=False,
        )
    logger.info("Parquet download complete.")


def download_and_extract_structures(
    structure_files: list[str],
    download_dir: Path,
    structures_dir: Path,
):
    """Download and extract structure tar.gz files with resume support."""
    structures_dir.mkdir(parents=True, exist_ok=True)
    total = len(structure_files)
    skipped = 0
    extracted = 0
    failed = 0

    for i, sf in enumerate(structure_files, 1):
        archive_name = os.path.basename(sf)
        marker = structures_dir / f".done_{archive_name}"

        # Skip if already extracted
        if marker.exists():
            skipped += 1
            if i % 50 == 0:
                logger.info(f"  [{i}/{total}] Already extracted: {archive_name}, skipping")
            continue

        logger.info(f"  [{i}/{total}] Downloading: {archive_name}")
        tar_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=sf,
            repo_type="dataset",
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
        )

        logger.info(f"  [{i}/{total}] Extracting: {archive_name}")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(structures_dir)
            marker.touch()
            os.remove(tar_path)
            extracted += 1
        except Exception as e:
            logger.error(f"  Failed to extract {archive_name}: {e}")
            failed += 1
            continue

    logger.info(
        f"Structures done: {extracted} extracted, {skipped} skipped, {failed} failed"
    )


def print_summary(download_dir: Path, structures_dir: Path, parquet_dir: Path):
    """Print summary of downloaded data."""
    cif_count = sum(1 for _ in structures_dir.rglob("*.cif"))
    cif_gz_count = sum(1 for _ in structures_dir.rglob("*.cif.gz"))
    pdb_count = sum(1 for _ in structures_dir.rglob("*.pdb"))
    total_structures = cif_count + cif_gz_count + pdb_count

    parquet_count = sum(1 for _ in parquet_dir.rglob("*.parquet"))

    logger.info("=" * 44)
    logger.info("Download Summary")
    logger.info(f"  Parquet files: {parquet_count}")
    logger.info(f"  Structure files: {total_structures}")
    logger.info(f"    .cif: {cif_count}")
    logger.info(f"    .cif.gz: {cif_gz_count}")
    logger.info(f"    .pdb: {pdb_count}")
    logger.info(f"  Data directory: {parquet_dir}")
    logger.info(f"  Structures directory: {structures_dir}")
    logger.info("=" * 44)


def main():
    parser = argparse.ArgumentParser(
        description="Download SAIR dataset from HuggingFace"
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        required=True,
        help="Root directory for downloaded data",
    )
    parser.add_argument(
        "--structures-only",
        action="store_true",
        help="Only download structure archives (skip parquet)",
    )
    parser.add_argument(
        "--parquet-only",
        action="store_true",
        help="Only download parquet metadata (skip structures)",
    )
    args = parser.parse_args()

    download_dir = args.download_dir.resolve()
    structures_dir = download_dir / "structures"
    parquet_dir = download_dir / "data"
    log_file = download_dir / "download.log"

    setup_logging(log_file)

    logger.info("=" * 44)
    logger.info("SAIR Dataset Download")
    logger.info(f"  Download dir: {download_dir}")
    logger.info(f"  Structures only: {args.structures_only}")
    logger.info(f"  Parquet only: {args.parquet_only}")
    logger.info("=" * 44)

    parquet_files, structure_files = list_repo_contents()

    if not args.structures_only:
        download_parquet(parquet_files, parquet_dir)

    if not args.parquet_only:
        download_and_extract_structures(structure_files, download_dir, structures_dir)

    print_summary(download_dir, structures_dir, parquet_dir)


if __name__ == "__main__":
    main()
