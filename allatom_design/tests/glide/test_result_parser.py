"""Tests for Glide result parser module."""

import numpy as np
import pandas as pd
import pytest

from allatom_design.eval.glide.result_parser import (
    GLIDE_SCORE_COLUMNS,
    extract_best_scores,
    parse_glide_csv,
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
