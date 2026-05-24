#!/usr/bin/env python3
"""Download and set up a CCD mirror via HTTP (no rsync required).

Downloads the full components.cif.gz from wwPDB, then splits it into
the divided layout that atomworks expects: {first_letter}/{CODE}/{CODE}.cif

Usage:
    python setup_ccd_mirror_http.py /path/to/ccd_mirror

    # Keep the downloaded components.cif.gz for future use
    python setup_ccd_mirror_http.py /path/to/ccd_mirror --keep-archive

    # Use a pre-downloaded components.cif.gz
    python setup_ccd_mirror_http.py /path/to/ccd_mirror --from-archive /path/to/components.cif.gz
"""

import argparse
import gzip
import logging
import shutil
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"


def download_ccd_archive(dest: Path) -> Path:
    """Download components.cif.gz from wwPDB via HTTP."""
    logger.info(f"Downloading CCD archive from {CCD_URL} ...")
    logger.info("This is ~300 MB and may take a few minutes.")

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / 1e6
            mb_total = total_size / 1e6
            print(f"\r  {mb_down:.1f} / {mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(CCD_URL, str(dest), reporthook=_reporthook)
    print()
    logger.info(f"Download complete: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def split_ccd_archive(archive_path: Path, output_dir: Path) -> int:
    """Split components.cif.gz into individual CIF files in divided layout.

    Layout: output_dir/{first_letter}/{CODE}/{CODE}.cif
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    current_code = None
    current_lines: list[str] = []

    def _flush():
        nonlocal count
        if current_code is None or not current_lines:
            return
        subdir = current_code[0]
        out_path = output_dir / subdir / current_code / f"{current_code}.cif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(current_lines))
        count += 1
        if count % 5000 == 0:
            logger.info(f"  ... {count} components written")

    logger.info(f"Splitting CCD archive into {output_dir} ...")

    open_fn = gzip.open if str(archive_path).endswith(".gz") else open
    with open_fn(archive_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("data_"):
                _flush()
                current_code = line.strip().split("_", 1)[1].upper()
                current_lines = [line]
            else:
                current_lines.append(line)
    _flush()

    logger.info(f"Done: {count} components written to {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Set up CCD mirror via HTTP (no rsync needed)")
    parser.add_argument("destination", type=Path, help="Destination directory for the CCD mirror")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded components.cif.gz")
    parser.add_argument(
        "--from-archive",
        type=Path,
        default=None,
        help="Use an existing components.cif.gz instead of downloading",
    )
    args = parser.parse_args()

    dest = args.destination.resolve()

    if args.from_archive:
        archive_path = args.from_archive.resolve()
        if not archive_path.exists():
            logger.error(f"Archive not found: {archive_path}")
            sys.exit(1)
        logger.info(f"Using existing archive: {archive_path}")
    else:
        if args.keep_archive:
            archive_path = dest / "components.cif.gz"
            dest.mkdir(parents=True, exist_ok=True)
            download_ccd_archive(archive_path)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".cif.gz", delete=False)
            tmp.close()
            archive_path = Path(tmp.name)
            try:
                download_ccd_archive(archive_path)
            except Exception:
                archive_path.unlink(missing_ok=True)
                raise

    try:
        count = split_ccd_archive(archive_path, dest)
    finally:
        if not args.keep_archive and not args.from_archive:
            archive_path.unlink(missing_ok=True)

    readme_path = dest / "README"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("# CCD Mirror Information\n\n")
        fh.write(f"Source: {CCD_URL}\n")
        fh.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Total components: {count}\n")
        fh.write(f"Layout: {{first_letter}}/{{CODE}}/{{CODE}}.cif\n")

    logger.info("")
    logger.info(f"CCD mirror ready at: {dest}")
    logger.info(f"Set your environment variable:")
    logger.info(f"  export CCD_MIRROR_PATH={dest}")


if __name__ == "__main__":
    main()
