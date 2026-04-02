"""Tests for Glide pipeline orchestration."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from allatom_design.eval.glide.pipeline import (
    _compute_rmsd_vs_reference,
    _run_inplace_scoring,
    _run_redocking,
    evaluate_single_sample,
    run_glide_evaluation,
)

from allatom_design.tests.glide.conftest import (
    EXAMPLE_CIF,
    SCHRODINGER_PATH,
    requires_example_data,
    requires_schrodinger,
)


# ============================================================================
# Unit tests (mocked Schrodinger)
# ============================================================================


class TestRunInplaceScoring:
    """Test in-place scoring with mocked Glide."""

    def test_success(self, tmp_path):
        work_dir = str(tmp_path / "work")

        # Create mock CSV output
        def mock_run_glide(input_file, schrodinger_path, timeout=3600):
            csv_path = str(Path(input_file).parent / "dock_inplace.csv")
            Path(csv_path).write_text(
                "Title,r_i_docking_score,r_i_glide_gscore\nlig,-7.5,-7.0\n"
            )
            return {"csv_path": csv_path, "sdf_path": None, "pv_path": None}

        with patch(
            "allatom_design.eval.glide.pipeline.run_glide", side_effect=mock_run_glide
        ), patch(
            "allatom_design.eval.glide.pipeline.write_docking_input",
            return_value=str(tmp_path / "work" / "dock_inplace.in"),
        ):
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            metrics = _run_inplace_scoring(
                grid_file="/fake/grid.zip",
                ligand_file="/fake/ligand.sdf",
                work_dir=work_dir,
                schrodinger_path="/fake/schrodinger",
                glide_cfg={"inplace": {"precision": "SP"}},
            )

        assert "best_docking_score" in metrics
        assert metrics["best_docking_score"] == pytest.approx(-7.5)

    def test_no_csv_output(self, tmp_path):
        work_dir = str(tmp_path / "work")

        with patch(
            "allatom_design.eval.glide.pipeline.run_glide",
            return_value={"csv_path": None, "sdf_path": None, "pv_path": None},
        ), patch(
            "allatom_design.eval.glide.pipeline.write_docking_input",
            return_value=str(tmp_path / "work" / "dock_inplace.in"),
        ):
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            metrics = _run_inplace_scoring(
                grid_file="/fake/grid.zip",
                ligand_file="/fake/ligand.sdf",
                work_dir=work_dir,
                schrodinger_path="/fake/schrodinger",
                glide_cfg={},
            )

        assert metrics["error"] == "no_csv_output"


class TestRunRedocking:
    """Test re-docking with mocked Glide."""

    def test_success(self, tmp_path):
        work_dir = str(tmp_path / "work")

        def mock_run_glide(input_file, schrodinger_path, timeout=3600):
            csv_path = str(Path(input_file).parent / "dock_redock.csv")
            sdf_path = str(Path(input_file).parent / "dock_redock_lib.sdf")
            Path(csv_path).write_text(
                "Title,r_i_docking_score,r_i_glide_gscore\n"
                "pose1,-8.2,-7.8\n"
                "pose2,-7.1,-6.5\n"
            )
            Path(sdf_path).write_text("fake sdf")
            return {"csv_path": csv_path, "sdf_path": sdf_path, "pv_path": None}

        with patch(
            "allatom_design.eval.glide.pipeline.run_glide", side_effect=mock_run_glide
        ), patch(
            "allatom_design.eval.glide.pipeline.write_docking_input",
            return_value=str(tmp_path / "work" / "dock_redock.in"),
        ):
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            metrics = _run_redocking(
                grid_file="/fake/grid.zip",
                ligand_file="/fake/ligand.sdf",
                work_dir=work_dir,
                schrodinger_path="/fake/schrodinger",
                glide_cfg={"redocking": {"precision": "SP", "num_poses": 5}},
            )

        assert metrics["best_docking_score"] == pytest.approx(-8.2)
        assert metrics["num_poses"] == 2


class TestEvaluateSingleSample:
    """Test full single-sample evaluation with all steps mocked."""

    def test_inplace_only(self, tmp_path):
        """Test pipeline with only in-place scoring enabled."""
        work_dir = str(tmp_path / "work")

        # Mock all Schrodinger-dependent functions
        mock_preprocess_result = {
            "sample_id": "test_sample",
            "protein_pdb_path": "/fake/protein.pdb",
            "ligand_sdf_path": "/fake/ligand.sdf",
            "ligand_centroid": np.array([1.0, 2.0, 3.0]),
            "atom_array": MagicMock(),
            "protein_atom_array": MagicMock(),
            "ligand_atom_array": MagicMock(),
            "receptor_pn_unit_iids": ["A_1"],
            "ligand_pn_unit_iids": ["B_1"],
        }

        def mock_run_glide(input_file, schrodinger_path, timeout=3600):
            csv_dir = Path(input_file).parent
            csv_path = str(csv_dir / "dock_inplace.csv")
            Path(csv_path).write_text(
                "Title,r_i_docking_score,r_i_glide_gscore\nlig,-6.0,-5.5\n"
            )
            return {"csv_path": csv_path, "sdf_path": None, "pv_path": None}

        with patch(
            "allatom_design.eval.glide.pipeline.preprocess_structure",
            return_value=mock_preprocess_result,
        ), patch(
            "allatom_design.eval.glide.pipeline.find_schrodinger",
            return_value="/fake/schrodinger",
        ), patch(
            "allatom_design.eval.glide.pipeline.run_prepwizard",
            return_value="/fake/prepared.mae",
        ), patch(
            "allatom_design.eval.glide.pipeline.write_gridgen_input",
            return_value="/fake/gridgen.in",
        ), patch(
            "allatom_design.eval.glide.pipeline.compute_dynamic_outerbox",
            return_value=[30.0, 30.0, 30.0],
        ), patch(
            "allatom_design.eval.glide.pipeline.run_grid_generation",
            return_value="/fake/grid.zip",
        ), patch(
            "allatom_design.eval.glide.pipeline.run_glide",
            side_effect=mock_run_glide,
        ), patch(
            "allatom_design.eval.glide.pipeline.write_docking_input",
        ) as mock_write_dock:
            mock_write_dock.return_value = str(
                Path(work_dir) / "test_sample" / "dock_inplace.in"
            )
            Path(work_dir, "test_sample").mkdir(parents=True, exist_ok=True)

            metrics = evaluate_single_sample(
                cif_path="/fake/test.cif",
                work_dir=work_dir,
                schrodinger_cfg={"schrodinger_path": "/fake/schrodinger"},
                glide_cfg={
                    "modes": {
                        "inplace_scoring": True,
                        "redocking": False,
                        "rmsd_comparison": False,
                    },
                    "grid": {},
                },
            )

        assert metrics["sample_id"] == "test"  # derived from cif_path stem "/fake/test.cif"
        assert "inplace_best_docking_score" in metrics
        assert metrics["inplace_best_docking_score"] == pytest.approx(-6.0)


class TestRunGlideEvaluation:
    """Test batch evaluation."""

    def test_handles_failures(self, tmp_path):
        """Test that batch evaluation continues past failures."""
        log_dir = str(tmp_path / "output")

        with patch(
            "allatom_design.eval.glide.pipeline.evaluate_single_sample"
        ) as mock_eval:
            # First sample succeeds, second fails
            mock_eval.side_effect = [
                {"sample_id": "ok_sample", "inplace_best_docking_score": -5.0},
                Exception("PrepWizard failed"),
            ]

            cfg = {
                "schrodinger": {"schrodinger_path": "/fake"},
                "glide": {"modes": {"inplace_scoring": True}},
            }

            df = run_glide_evaluation(
                sample_paths=["/fake/ok.cif", "/fake/fail.cif"],
                cfg=cfg,
                log_dir=log_dir,
            )

        assert len(df) == 2
        assert df.iloc[0]["sample_id"] == "ok_sample"
        assert "error" in df.iloc[1]

        # Check results CSV was saved
        assert Path(log_dir, "glide_results.csv").exists()

        # Check failed samples list was saved
        assert Path(log_dir, "glide_failed_samples.txt").exists()
