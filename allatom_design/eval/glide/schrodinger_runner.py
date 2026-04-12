"""Wrappers for Schrodinger command-line tools.

All Schrodinger tools run as subprocesses. The lullaby_local conda
environment does NOT have Schrodinger Python packages installed.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 3600  # 1 hour


def find_schrodinger(schrodinger_path: str | None = None) -> str:
    """Resolve path to Schrodinger installation.

    Checks (in order):
    1. Explicit schrodinger_path argument
    2. $SCHRODINGER environment variable

    Raises:
        FileNotFoundError: If no valid Schrodinger installation is found.
    """
    if schrodinger_path and Path(schrodinger_path).is_dir():
        return str(Path(schrodinger_path).resolve())

    env_path = os.environ.get("SCHRODINGER")
    if env_path and Path(env_path).is_dir():
        return str(Path(env_path).resolve())

    raise FileNotFoundError(
        "Schrodinger installation not found. "
        "Set schrodinger_path in config or $SCHRODINGER environment variable."
    )


# Default license server for Stanford Sherlock.
_DEFAULT_LICENSE = "53001@srcc-license-srcf.stanford.edu"


def _ensure_schrodinger_env(schrodinger_path: str) -> dict[str, str]:
    """Build env dict with Schrodinger paths for subprocess calls.

    Ensures SCHRODINGER and SCHROD_LICENSE_FILE are set, falling back
    to defaults when the calling shell doesn't have them.

    Injects Schrodinger-bundled libs into LD_LIBRARY_PATH for this subprocess
    only. The parent Python process must keep a clean LD_LIBRARY_PATH so that
    AF3's C++ extension can resolve GLIBCXX_3.4.32 from the container's system
    libstdc++ — Schrodinger bundles an older libstdc++ that lacks this symbol
    and would otherwise shadow the system one.
    """
    env = os.environ.copy()

    env.setdefault("SCHRODINGER", schrodinger_path)

    if "SCHROD_LICENSE_FILE" not in env:
        logger.warning(
            f"SCHROD_LICENSE_FILE not set — using default: {_DEFAULT_LICENSE}"
        )
        env["SCHROD_LICENSE_FILE"] = _DEFAULT_LICENSE

    schrod_libs = env.get("SCHRODINGER_LD_LIBS")
    if not schrod_libs:
        schrod_libs = ":".join([
            f"{schrodinger_path}/internal/lib",
            f"{schrodinger_path}/mmshare-v7.1/lib/Linux-x86_64",
        ])
    parent_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{schrod_libs}:{parent_ld}" if parent_ld else schrod_libs

    return env


def _run_command(
    cmd: list[str],
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with logging and error handling."""
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    if result.returncode != 0:
        logger.error(f"Command failed (rc={result.returncode}): {cmd_str}")
        if result.stdout:
            logger.error(f"stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd_str, result.stdout, result.stderr
        )

    if result.stderr:
        logger.debug(f"stderr: {result.stderr}")

    return result


# ============================================================================
# PrepWizard - Protein Preparation
# ============================================================================

def run_prepwizard(
    input_file: str,
    output_file: str,
    schrodinger_path: str,
    options: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    log_dir: str | None = None,
) -> str:
    """Run Schrodinger PrepWizard for protein preparation.

    Args:
        input_file: Input PDB/MAE file.
        output_file: Output MAE file.
        schrodinger_path: Path to Schrodinger installation.
        options: PrepWizard options (e.g. {"noimpref": True, "nohtreat": False}).
        timeout: Timeout in seconds.
        log_dir: Directory for Schrodinger log files. Defaults to output_file's parent.

    Returns:
        Path to the output MAE file.
    """
    prepwizard = str(Path(schrodinger_path) / "utilities" / "prepwizard")
    if not Path(prepwizard).exists():
        raise FileNotFoundError(f"prepwizard not found: {prepwizard}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    cmd = [prepwizard]

    if options:
        for key, value in options.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"-{key}")
            else:
                cmd.extend([f"-{key}", str(value)])

    cmd.extend(["-NOJOBID"])
    cmd.extend([input_file, output_file])

    cwd = log_dir if log_dir else str(Path(output_file).parent)
    Path(cwd).mkdir(parents=True, exist_ok=True)
    env = _ensure_schrodinger_env(schrodinger_path)

    try:
        _run_command(cmd, cwd=cwd, timeout=timeout, env=env)
    except subprocess.CalledProcessError:
        # PrepWizard writes detailed errors to .log files in cwd
        for log_file in sorted(Path(cwd).glob("*.log")):
            try:
                log_content = log_file.read_text()
                if log_content.strip():
                    logger.error(f"PrepWizard log ({log_file.name}):\n{log_content[-2000:]}")
            except Exception:
                pass
        raise

    if not Path(output_file).exists():
        raise FileNotFoundError(f"PrepWizard output not found: {output_file}")

    logger.info(f"PrepWizard complete: {output_file}")
    return output_file


# ============================================================================
# Grid Generation
# ============================================================================

def write_gridgen_input(
    receptor_mae: str,
    grid_center: list[float] | tuple[float, ...],
    out_dir: str,
    jobname: str = "gridgen",
    inner_box: list[int] | None = None,
    outer_box: list[float] | None = None,
    forcefield: str = "OPLS4",
) -> str:
    """Write a Glide grid generation input file.

    Returns:
        Path to the written .in file.
    """
    if inner_box is None:
        inner_box = [10, 10, 10]
    if outer_box is None:
        outer_box = [30.0, 30.0, 30.0]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = str(out_dir / f"{jobname}.in")

    lines = [
        f"FORCEFIELD   {forcefield}",
        f"GRID_CENTER   {grid_center[0]:.4f}, {grid_center[1]:.4f}, {grid_center[2]:.4f}",
        f"INNERBOX   {inner_box[0]}, {inner_box[1]}, {inner_box[2]}",
        f"OUTERBOX   {outer_box[0]:.1f}, {outer_box[1]:.1f}, {outer_box[2]:.1f}",
        f"RECEP_FILE   {receptor_mae}",
    ]

    with open(input_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Wrote grid generation input: {input_file}")
    return input_file


def run_grid_generation(
    input_file: str,
    schrodinger_path: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Run Glide grid generation.

    Returns:
        Path to the generated grid .zip file.
    """
    glide = str(Path(schrodinger_path) / "glide")
    cwd = str(Path(input_file).parent)
    jobname = Path(input_file).stem

    cmd = [glide, input_file, "-NOJOBID", "-OVERWRITE"]
    env = _ensure_schrodinger_env(schrodinger_path)
    _run_command(cmd, cwd=cwd, timeout=timeout, env=env)

    grid_file = str(Path(cwd) / f"{jobname}.zip")
    if not Path(grid_file).exists():
        raise FileNotFoundError(f"Grid file not found: {grid_file}")

    logger.info(f"Grid generation complete: {grid_file}")
    return grid_file


# ============================================================================
# LigPrep - Ligand Preparation
# ============================================================================

def run_ligprep(
    input_sdf: str,
    output_file: str,
    schrodinger_path: str,
    options: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    log_dir: str | None = None,
) -> str:
    """Run Schrodinger LigPrep for ligand preparation.

    Args:
        input_sdf: Input SDF file.
        output_file: Output file (.maegz for MAE, .sdf for SDF).
        schrodinger_path: Path to Schrodinger installation.
        options: LigPrep options (e.g. {"epik": True, "bff": 16}).
        timeout: Timeout in seconds.
        log_dir: Directory for Schrodinger log files. Defaults to output_file's parent.

    Returns:
        Path to the output file.
    """
    ligprep = str(Path(schrodinger_path) / "ligprep")
    if not Path(ligprep).exists():
        raise FileNotFoundError(f"ligprep not found: {ligprep}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Determine output format from extension
    if output_file.endswith((".mae", ".maegz")):
        out_flag = "-omae"
    else:
        out_flag = "-osd"

    cmd = [ligprep, "-isd", input_sdf, out_flag, output_file]

    if options:
        for key, value in options.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"-{key}")
            else:
                cmd.extend([f"-{key}", str(value)])

    cmd.extend(["-NOJOBID"])

    cwd = log_dir if log_dir else str(Path(output_file).parent)
    Path(cwd).mkdir(parents=True, exist_ok=True)
    env = _ensure_schrodinger_env(schrodinger_path)
    _run_command(cmd, cwd=cwd, timeout=timeout, env=env)

    if not Path(output_file).exists():
        raise FileNotFoundError(f"LigPrep output not found: {output_file}")

    logger.info(f"LigPrep complete: {output_file}")
    return output_file


# ============================================================================
# Glide Docking / Scoring
# ============================================================================

def write_docking_input(
    gridfile: str,
    ligandfile: str,
    out_dir: str,
    jobname: str = "dock",
    docking_method: str = "confgen",
    precision: str = "SP",
    num_poses: int = 5,
    write_csv: bool = True,
    pose_outtype: str = "ligandlib_sd",
    compress_poses: bool = False,
    forcefield: str = "OPLS4",
    extra_keywords: dict[str, Any] | None = None,
) -> str:
    """Write a Glide docking/scoring input file.

    Args:
        gridfile: Path to grid .zip file.
        ligandfile: Path to ligand file (SDF/MAE).
        out_dir: Output directory.
        jobname: Job name (determines output filenames).
        docking_method: One of 'confgen', 'rigid', 'inplace', 'mininplace'.
        precision: 'SP' or 'HTVS'.
        num_poses: Max poses to report.
        write_csv: Write CSV output with scores.
        pose_outtype: Output format for poses.
        compress_poses: Compress output pose files.
        forcefield: Force field to use.
        extra_keywords: Additional Glide keywords.

    Returns:
        Path to the written .in file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = str(out_dir / f"{jobname}.in")

    lines = [
        f"FORCEFIELD   {forcefield}",
        f"GRIDFILE   {gridfile}",
        f"LIGANDFILE   {ligandfile}",
        f"PRECISION   {precision}",
        f"DOCKING_METHOD   {docking_method}",
        f"NREPORT   {num_poses}",
        f"POSE_OUTTYPE   {pose_outtype}",
        f"COMPRESS_POSES   {str(compress_poses).upper()}",
        f"WRITE_CSV   {str(write_csv).upper()}",
    ]

    if extra_keywords:
        for key, value in extra_keywords.items():
            if isinstance(value, bool):
                lines.append(f"{key}   {str(value).upper()}")
            else:
                lines.append(f"{key}   {value}")

    with open(input_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Wrote docking input: {input_file}")
    return input_file


def run_glide(
    input_file: str,
    schrodinger_path: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, str | None]:
    """Run Glide docking/scoring.

    Returns:
        Dict with paths to output files:
        - csv_path: Path to CSV output (if WRITE_CSV was True).
        - sdf_path: Path to SDF pose output (if ligandlib_sd).
        - pv_path: Path to poseviewer output (if poseviewer).
    """
    glide = str(Path(schrodinger_path) / "glide")
    cwd = str(Path(input_file).parent)
    jobname = Path(input_file).stem

    cmd = [glide, input_file, "-NOJOBID", "-OVERWRITE"]
    env = _ensure_schrodinger_env(schrodinger_path)
    _run_command(cmd, cwd=cwd, timeout=timeout, env=env)

    # Locate output files
    outputs: dict[str, str | None] = {
        "csv_path": None,
        "sdf_path": None,
        "pv_path": None,
    }

    csv_path = Path(cwd) / f"{jobname}.csv"
    if csv_path.exists():
        outputs["csv_path"] = str(csv_path)

    # SDF output: {jobname}_lib.sdf or {jobname}_lib.sdfgz
    for ext in [".sdf", ".sdfgz"]:
        sdf_path = Path(cwd) / f"{jobname}_lib{ext}"
        if sdf_path.exists():
            outputs["sdf_path"] = str(sdf_path)
            break

    # Poseviewer output
    for ext in [".maegz", ".mae"]:
        pv_path = Path(cwd) / f"{jobname}_pv{ext}"
        if pv_path.exists():
            outputs["pv_path"] = str(pv_path)
            break

    logger.info(f"Glide complete: {outputs}")
    return outputs
