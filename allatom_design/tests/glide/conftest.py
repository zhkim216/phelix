"""Shared fixtures for Glide pipeline tests."""

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Path to the example CIF in the debug directory
EXAMPLE_CIF = (
    "/home/possu/jinho/allatom-design/debug/260401_glide_debug/example_data/"
    "0H7_len_150_0_sample0_seed-42_sample-0_model_pocket_aligned.cif"
)

SCHRODINGER_PATH = "/home/possu/jinho/software/schrodinger"


def has_example_data() -> bool:
    return Path(EXAMPLE_CIF).exists()


def has_schrodinger() -> bool:
    return Path(SCHRODINGER_PATH).is_dir() and Path(SCHRODINGER_PATH, "glide").exists()


requires_example_data = pytest.mark.skipif(
    not has_example_data(), reason="Example CIF not found"
)
requires_schrodinger = pytest.mark.skipif(
    not has_schrodinger(), reason="Schrodinger not installed"
)


@pytest.fixture
def tmp_work_dir(tmp_path):
    """Temporary working directory for tests."""
    return str(tmp_path / "work")


@pytest.fixture
def sample_glide_csv(tmp_path):
    """Create a sample Glide CSV output file."""
    csv_content = textwrap.dedent("""\
        Title,r_i_docking_score,r_i_glide_gscore,r_i_glide_emodel,r_i_glide_energy,r_i_glide_ligand_efficiency
        ligand_pose1,-7.532,-7.123,-55.234,-45.123,-0.456
        ligand_pose2,-6.891,-6.543,-48.765,-40.234,-0.412
        ligand_pose3,-5.234,-5.012,-35.678,-32.456,-0.321
    """)
    csv_path = tmp_path / "dock_test.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def mock_schrodinger_path(tmp_path):
    """Create a mock Schrodinger installation directory."""
    schrodinger = tmp_path / "schrodinger"
    schrodinger.mkdir()

    # Create mock executables
    for tool in ["glide", "ligprep"]:
        tool_path = schrodinger / tool
        tool_path.write_text("#!/bin/bash\nexit 0\n")
        tool_path.chmod(0o755)

    utilities = schrodinger / "utilities"
    utilities.mkdir()
    prepwizard = utilities / "prepwizard"
    prepwizard.write_text("#!/bin/bash\nexit 0\n")
    prepwizard.chmod(0o755)

    return str(schrodinger)


@pytest.fixture
def sample_gridgen_input():
    """Expected content patterns for a grid generation input file."""
    return {
        "required_keywords": [
            "FORCEFIELD",
            "GRID_CENTER",
            "INNERBOX",
            "OUTERBOX",
            "RECEP_FILE",
        ],
    }


@pytest.fixture
def sample_docking_input():
    """Expected content patterns for a docking input file."""
    return {
        "required_keywords": [
            "FORCEFIELD",
            "GRIDFILE",
            "LIGANDFILE",
            "PRECISION",
            "DOCKING_METHOD",
            "WRITE_CSV",
        ],
    }
