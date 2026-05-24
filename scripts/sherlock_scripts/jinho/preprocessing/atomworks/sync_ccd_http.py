#!/usr/bin/env python3
"""Sync specific CCD entries into a local mirror over HTTPS.

Drop-in replacement for `atomworks ccd sync --ccd-code ...` when rsync is
unavailable (e.g. inside a Sherlock container). Uses only the Python
standard library. Layout matches `_get_ccd_path`:

    {destination}/{CODE[0]}/{CODE}/{CODE}.cif

Primary source is EBI's PDBeChem mirror (same data as the rsync endpoint);
falls back to RCSB's per-ligand CIF endpoint if EBI returns 404/5xx.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

EBI_BASE = "https://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/ccd"
RCSB_BASE = "https://files.rcsb.org/ligands/download"
USER_AGENT = "atomworks-sync-ccd-http/1.0 (+https://github.com/baker-laboratory/atomworks)"


def ccd_relpath(code: str) -> Path:
    return Path(code[0]) / code / f"{code}.cif"


def ebi_url(code: str, base: str = EBI_BASE) -> str:
    return f"{base}/{code[0]}/{code}/{code}.cif"


def rcsb_url(code: str, base: str = RCSB_BASE) -> str:
    return f"{base}/{code}.cif"


@dataclass
class Result:
    code: str
    ok: bool
    source: str | None
    message: str


def _fetch(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, dest)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def fetch_one(
    code: str,
    dest_root: Path,
    *,
    ebi_base: str,
    rcsb_base: str,
    timeout: float,
    retries: int,
    force: bool,
) -> Result:
    target = dest_root / ccd_relpath(code)
    if target.exists() and not force:
        return Result(code, True, "cache", f"already present at {target}")

    sources: list[tuple[str, str]] = [
        ("ebi", ebi_url(code, ebi_base)),
        ("rcsb", rcsb_url(code, rcsb_base)),
    ]

    last_error = "no source attempted"
    for source_name, url in sources:
        for attempt in range(1, retries + 1):
            try:
                data = _fetch(url, timeout=timeout)
            except urllib.error.HTTPError as e:
                last_error = f"{source_name} HTTP {e.code} on attempt {attempt}"
                if e.code == 404:
                    break  # try next source
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = f"{source_name} network error on attempt {attempt}: {e}"
                time.sleep(min(2 ** (attempt - 1), 8))
                continue

            if not data or b"data_" not in data[:200]:
                last_error = f"{source_name} returned suspicious payload ({len(data)} bytes)"
                break

            try:
                _atomic_write(target, data)
            except OSError as e:
                return Result(code, False, source_name, f"write failed: {e}")
            return Result(code, True, source_name, f"{len(data)} bytes -> {target}")

    return Result(code, False, None, last_error)


def collect_codes(code_args: list[str] | None, codes_file: Path | None) -> list[str]:
    collected: list[str] = []
    if code_args:
        collected.extend(code_args)
    if codes_file:
        with codes_file.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                collected.append(line)

    seen: set[str] = set()
    unique: list[str] = []
    for c in collected:
        norm = c.strip().upper()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(norm)
    return unique


def invalidate_cache(dest_root: Path) -> None:
    """Drop the cached code list so atomworks rescans after we added files."""
    cache = dest_root / ".ccd_codes_cache"
    cache.unlink(missing_ok=True)
    try:
        os.utime(dest_root, None)
    except OSError:
        pass


def write_readme(dest_root: Path, codes: list[str], ok_codes: list[str], failed: list[Result]) -> None:
    readme = dest_root / "README"
    mode = "a" if readme.exists() else "w"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with readme.open(mode, encoding="utf-8") as fh:
        if mode == "w":
            fh.write("# CCD Mirror Information\n\n")
        fh.write(f"[{ts}] HTTPS sync (sync_ccd_http.py)\n")
        fh.write(f"  requested: {len(codes)} codes\n")
        fh.write(f"  succeeded: {len(ok_codes)}\n")
        fh.write(f"  failed:    {len(failed)}\n")
        if failed:
            fh.write("  failed codes: " + ", ".join(r.code for r in failed) + "\n")
        fh.write(f"  user: {os.getenv('USER') or 'unknown'}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sync specific CCD CIFs into a local mirror over HTTPS (no rsync required).",
    )
    p.add_argument("destination", type=Path, help="CCD mirror root directory (will be created if missing).")
    p.add_argument(
        "--ccd-code",
        action="append",
        default=[],
        metavar="CODE",
        help="CCD code to fetch (repeatable).",
    )
    p.add_argument(
        "--ccd-codes-file",
        type=Path,
        default=None,
        help="File with one CCD code per line (# for comments).",
    )
    p.add_argument("--ebi-base", default=EBI_BASE, help=f"EBI base URL (default: {EBI_BASE}).")
    p.add_argument("--rcsb-base", default=RCSB_BASE, help=f"RCSB fallback base URL (default: {RCSB_BASE}).")
    p.add_argument("--workers", type=int, default=8, help="Parallel download workers (default: 8).")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    p.add_argument("--retries", type=int, default=3, help="Retries per source before falling back.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even if the target file already exists in the mirror.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    codes = collect_codes(args.ccd_code, args.ccd_codes_file)
    if not codes:
        print("error: no CCD codes provided (use --ccd-code or --ccd-codes-file).", file=sys.stderr)
        return 2

    # Sanity-check the destination — bail early if it's clearly bogus.
    dest_root: Path = args.destination.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    if not os.access(dest_root, os.W_OK):
        print(f"error: {dest_root} is not writable.", file=sys.stderr)
        return 2
    if shutil.disk_usage(dest_root).free < 50 * 1024 * 1024:
        print(f"warning: less than 50MB free at {dest_root}", file=sys.stderr)

    print(f"Target mirror: {dest_root}")
    print(f"Fetching {len(codes)} CCD codes with {args.workers} workers")
    print(f"  primary:  {args.ebi_base}")
    print(f"  fallback: {args.rcsb_base}")

    results: list[Result] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                fetch_one,
                code,
                dest_root,
                ebi_base=args.ebi_base,
                rcsb_base=args.rcsb_base,
                timeout=args.timeout,
                retries=args.retries,
                force=args.force,
            ): code
            for code in codes
        }
        for fut in cf.as_completed(futures):
            res = fut.result()
            results.append(res)
            tag = "OK  " if res.ok else "FAIL"
            src = f"[{res.source or '-'}]"
            print(f"  {tag} {res.code:<6} {src:<8} {res.message}")

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    if ok:
        invalidate_cache(dest_root)
        write_readme(dest_root, codes, [r.code for r in ok], failed)

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s — {len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed codes: " + ", ".join(r.code for r in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
