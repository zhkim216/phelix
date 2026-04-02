"""Tests for Schrodinger runner module."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from allatom_design.eval.glide.schrodinger_runner import (
    _run_command,
    find_schrodinger,
    run_glide,
    run_grid_generation,
    run_ligprep,
    run_prepwizard,
    write_docking_input,
    write_gridgen_input,
)

from allatom_design.tests.glide.conftest import (
    SCHRODINGER_PATH,
    requires_schrodinger,
)


# ============================================================================
# find_schrodinger
# ============================================================================


class TestFindSchrodinger:
    def test_explicit_path(self, mock_schrodinger_path):
        path = find_schrodinger(mock_schrodinger_path)
        assert Path(path).is_dir()

    def test_env_variable(self, mock_schrodinger_path, monkeypatch):
        monkeypatch.setenv("SCHRODINGER", mock_schrodinger_path)
        path = find_schrodinger(None)
        assert Path(path).is_dir()

    def test_not_found(self, monkeypatch):
        monkeypatch.delenv("SCHRODINGER", raising=False)
        with pytest.raises(FileNotFoundError, match="Schrodinger installation not found"):
            find_schrodinger("/nonexistent/path")

    def test_none_path_no_env(self, monkeypatch):
        monkeypatch.delenv("SCHRODINGER", raising=False)
        with pytest.raises(FileNotFoundError):
            find_schrodinger(None)


# ============================================================================
# Input file generation (no Schrodinger needed)
# ============================================================================


class TestWriteGridgenInput:
    def test_basic(self, tmp_path, sample_gridgen_input):
        input_file = write_gridgen_input(
            receptor_mae="/fake/receptor.mae",
            grid_center=[1.0, 2.0, 3.0],
            out_dir=str(tmp_path),
            jobname="test_gridgen",
        )

        assert Path(input_file).exists()
        content = Path(input_file).read_text()

        for kw in sample_gridgen_input["required_keywords"]:
            assert kw in content, f"Missing keyword: {kw}"

        assert "1.0000, 2.0000, 3.0000" in content
        assert "/fake/receptor.mae" in content

    def test_custom_box_sizes(self, tmp_path):
        input_file = write_gridgen_input(
            receptor_mae="/fake/receptor.mae",
            grid_center=[0, 0, 0],
            out_dir=str(tmp_path),
            inner_box=[15, 15, 15],
            outer_box=[40.0, 40.0, 40.0],
        )
        content = Path(input_file).read_text()
        assert "15, 15, 15" in content
        assert "40.0, 40.0, 40.0" in content


class TestWriteDockingInput:
    def test_inplace(self, tmp_path, sample_docking_input):
        input_file = write_docking_input(
            gridfile="/fake/grid.zip",
            ligandfile="/fake/ligand.sdf",
            out_dir=str(tmp_path),
            jobname="test_inplace",
            docking_method="inplace",
            precision="SP",
        )

        assert Path(input_file).exists()
        content = Path(input_file).read_text()

        for kw in sample_docking_input["required_keywords"]:
            assert kw in content, f"Missing keyword: {kw}"

        assert "inplace" in content
        assert "SP" in content
        assert "/fake/grid.zip" in content
        assert "/fake/ligand.sdf" in content

    def test_confgen_redocking(self, tmp_path):
        input_file = write_docking_input(
            gridfile="/fake/grid.zip",
            ligandfile="/fake/ligand.sdf",
            out_dir=str(tmp_path),
            jobname="test_redock",
            docking_method="confgen",
            precision="SP",
            num_poses=10,
        )
        content = Path(input_file).read_text()
        assert "confgen" in content
        assert "NREPORT   10" in content

    def test_extra_keywords(self, tmp_path):
        input_file = write_docking_input(
            gridfile="/fake/grid.zip",
            ligandfile="/fake/ligand.sdf",
            out_dir=str(tmp_path),
            jobname="test_extra",
            docking_method="confgen",
            extra_keywords={"EXPANDED_SAMPLING": True, "CV_CUTOFF": 0.0},
        )
        content = Path(input_file).read_text()
        assert "EXPANDED_SAMPLING   TRUE" in content
        assert "CV_CUTOFF   0.0" in content

    def test_compress_poses_false(self, tmp_path):
        input_file = write_docking_input(
            gridfile="/fake/grid.zip",
            ligandfile="/fake/ligand.sdf",
            out_dir=str(tmp_path),
            jobname="test_nocompress",
            docking_method="confgen",
            compress_poses=False,
        )
        content = Path(input_file).read_text()
        assert "COMPRESS_POSES   FALSE" in content


# ============================================================================
# Command execution (mocked)
# ============================================================================


class TestRunCommand:
    def test_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["echo", "hello"],
                returncode=0,
                stdout="hello\n",
                stderr="",
            )
            result = _run_command(["echo", "hello"])
            assert result.returncode == 0

    def test_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["false"],
                returncode=1,
                stdout="",
                stderr="error message",
            )
            with pytest.raises(subprocess.CalledProcessError):
                _run_command(["false"])


class TestRunPrepwizard:
    def test_success(self, mock_schrodinger_path, tmp_path):
        input_pdb = str(tmp_path / "input.pdb")
        output_mae = str(tmp_path / "output.mae")
        Path(input_pdb).write_text("ATOM      1  N   ALA A   1\n")

        with patch(
            "allatom_design.eval.glide.schrodinger_runner._run_command"
        ) as mock_cmd:
            # Simulate prepwizard creating the output file
            def side_effect(*args, **kwargs):
                Path(output_mae).write_text("fake mae content")
                return subprocess.CompletedProcess(args[0], 0, "", "")

            mock_cmd.side_effect = side_effect
            result = run_prepwizard(
                input_pdb, output_mae, mock_schrodinger_path
            )
            assert result == output_mae

    def test_with_options(self, mock_schrodinger_path, tmp_path):
        input_pdb = str(tmp_path / "input.pdb")
        output_mae = str(tmp_path / "output.mae")
        Path(input_pdb).write_text("ATOM      1  N   ALA A   1\n")

        with patch(
            "allatom_design.eval.glide.schrodinger_runner._run_command"
        ) as mock_cmd:
            def side_effect(*args, **kwargs):
                Path(output_mae).write_text("fake mae content")
                return subprocess.CompletedProcess(args[0], 0, "", "")

            mock_cmd.side_effect = side_effect
            run_prepwizard(
                input_pdb,
                output_mae,
                mock_schrodinger_path,
                options={"noimpref": True, "rehtreat": False},
            )

            call_args = mock_cmd.call_args[0][0]
            assert "-noimpref" in call_args
            assert "-rehtreat" not in call_args


class TestRunGlide:
    def test_with_csv_output(self, tmp_path):
        """Test that run_glide finds CSV output."""
        input_file = str(tmp_path / "dock.in")
        Path(input_file).write_text("GRIDFILE   /fake/grid.zip\n")

        with patch(
            "allatom_design.eval.glide.schrodinger_runner._run_command"
        ) as mock_cmd:
            def side_effect(*args, **kwargs):
                # Simulate Glide creating output files
                (tmp_path / "dock.csv").write_text("Title,r_i_docking_score\nlig,-5.0\n")
                (tmp_path / "dock_lib.sdf").write_text("fake sdf")
                return subprocess.CompletedProcess(args[0], 0, "", "")

            mock_cmd.side_effect = side_effect
            outputs = run_glide(input_file, "/fake/schrodinger")
            assert outputs["csv_path"] is not None
            assert outputs["sdf_path"] is not None
