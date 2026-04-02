"""Tests for Glide result parser module."""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from allatom_design.eval.glide.result_parser import (
    GLIDE_SCORE_COLUMNS,
    extract_best_scores,
    get_pose_coordinates,
    parse_glide_csv,
    parse_glide_sdf,
)


# ============================================================================
# CSV parsing
# ============================================================================


class TestParseGlideCsv:
    def test_basic(self, sample_glide_csv):
        df = parse_glide_csv(sample_glide_csv)
        assert len(df) == 3
        assert "docking_score" in df.columns
        assert "glide_score" in df.columns
        assert "emodel" in df.columns
        assert "title" in df.columns

    def test_scores_correct(self, sample_glide_csv):
        df = parse_glide_csv(sample_glide_csv)
        assert df["docking_score"].iloc[0] == pytest.approx(-7.532)
        assert df["glide_score"].iloc[0] == pytest.approx(-7.123)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_glide_csv("/nonexistent/file.csv")

    def test_unknown_columns_preserved(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("Title,r_i_docking_score,custom_prop\nlig,-5.0,42.0\n")
        df = parse_glide_csv(str(csv))
        assert "custom_prop" in df.columns
        assert "docking_score" in df.columns

    def test_empty_csv(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("Title,r_i_docking_score\n")
        df = parse_glide_csv(str(csv))
        assert len(df) == 0


# ============================================================================
# SDF parsing
# ============================================================================


class TestParseGlideSdf:
    def test_basic(self, sample_glide_sdf):
        poses = parse_glide_sdf(sample_glide_sdf)
        assert len(poses) == 1
        assert poses[0]["mol"] is not None
        assert poses[0]["mol"].GetNumAtoms() > 0

    def test_gzipped_sdf(self, tmp_path):
        """Test reading compressed .sdfgz files."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)

        # Write plain SDF first
        plain_path = str(tmp_path / "test.sdf")
        writer = Chem.SDWriter(plain_path)
        writer.write(mol)
        writer.close()

        # Compress it
        gz_path = str(tmp_path / "test.sdfgz")
        with open(plain_path, "r") as f_in:
            with gzip.open(gz_path, "wt") as f_out:
                f_out.write(f_in.read())

        poses = parse_glide_sdf(gz_path)
        assert len(poses) == 1

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_glide_sdf("/nonexistent/file.sdf")

    def test_multiple_poses(self, tmp_path):
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)

        sdf_path = str(tmp_path / "multi.sdf")
        writer = Chem.SDWriter(sdf_path)
        for i in range(3):
            writer.write(mol)
        writer.close()

        poses = parse_glide_sdf(sdf_path)
        assert len(poses) == 3


# ============================================================================
# Score extraction
# ============================================================================


class TestExtractBestScores:
    def test_basic(self):
        df = pd.DataFrame({
            "docking_score": [-7.5, -6.0, -5.0],
            "glide_score": [-7.0, -5.5, -4.5],
            "emodel": [-55.0, -45.0, -35.0],
        })
        scores = extract_best_scores(df)
        assert scores["best_docking_score"] == pytest.approx(-7.5)
        assert scores["best_glide_score"] == pytest.approx(-7.0)
        assert scores["best_emodel"] == pytest.approx(-55.0)

    def test_empty_df(self):
        df = pd.DataFrame()
        scores = extract_best_scores(df)
        assert scores == {}

    def test_missing_columns(self):
        df = pd.DataFrame({"docking_score": [-5.0]})
        scores = extract_best_scores(df)
        assert "best_docking_score" in scores
        assert "best_glide_score" not in scores

    def test_with_nan(self):
        df = pd.DataFrame({
            "docking_score": [np.nan, -5.0, -3.0],
            "glide_score": [-4.0, np.nan, -2.0],
        })
        scores = extract_best_scores(df)
        assert scores["best_docking_score"] == pytest.approx(-5.0)
        assert scores["best_glide_score"] == pytest.approx(-4.0)


# ============================================================================
# Pose coordinate extraction
# ============================================================================


class TestGetPoseCoordinates:
    def test_basic(self, sample_glide_sdf):
        coords = get_pose_coordinates(sample_glide_sdf, pose_index=0)
        assert coords is not None
        assert coords.ndim == 2
        assert coords.shape[1] == 3

    def test_out_of_range(self, sample_glide_sdf):
        coords = get_pose_coordinates(sample_glide_sdf, pose_index=999)
        assert coords is None
