"""Tests for evaluation utilities module.

Tests verify:
1. compute_bleu() returns float in [0, 100] range
2. compute_chrf() returns float in [0, 100] range
3. compute_genre_metrics() returns dict with per-genre BLEU scores
4. load_predictions() reads CSV with id, prediction columns
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluation_utils import (
    compute_bleu,
    compute_chrf,
    compute_genre_metrics,
    extract_genre_tag,
    load_predictions,
    load_validation_data,
)


class TestComputeBleu(unittest.TestCase):
    """Tests for compute_bleu function."""

    def test_bleu_returns_float_in_valid_range(self):
        """Test 1: compute_bleu returns float in [0, 100] range."""
        predictions = ["the cat sat on the mat", "hello world"]
        references = ["the cat sat on the mat", "hello world"]

        bleu = compute_bleu(predictions, references)

        self.assertIsInstance(bleu, float)
        self.assertGreaterEqual(bleu, 0.0)
        # Allow small floating-point tolerance above 100
        self.assertLessEqual(bleu, 100.1)

    def test_bleu_perfect_score(self):
        """Perfect match should give high BLEU score."""
        predictions = ["the quick brown fox", "jumps over lazy dog"]
        references = ["the quick brown fox", "jumps over lazy dog"]

        bleu = compute_bleu(predictions, references)

        # Perfect match should be 100 or very close
        self.assertGreater(bleu, 90.0)

    def test_bleu_low_score_for_mismatch(self):
        """Completely different text should give low BLEU score."""
        predictions = ["apple banana cherry", "dog elephant frog"]
        references = ["xyz abc def", "123 456 789"]

        bleu = compute_bleu(predictions, references)

        # Mismatch should give low score
        self.assertLess(bleu, 10.0)

    def test_bleu_empty_input(self):
        """Empty input should return 0."""
        bleu = compute_bleu([], [])
        self.assertEqual(bleu, 0.0)

    def test_bleu_raises_on_length_mismatch(self):
        """Mismatched lengths should raise ValueError."""
        predictions = ["a", "b"]
        references = ["a"]

        with self.assertRaises(ValueError) as cm:
            compute_bleu(predictions, references)

        self.assertIn("length", str(cm.exception).lower())


class TestComputeChrf(unittest.TestCase):
    """Tests for compute_chrf function."""

    def test_chrf_returns_float_in_valid_range(self):
        """Test 2: compute_chrf returns float in [0, 100] range."""
        predictions = ["the cat sat on the mat", "hello world"]
        references = ["the cat sat on the mat", "hello world"]

        chrf = compute_chrf(predictions, references)

        self.assertIsInstance(chrf, float)
        self.assertGreaterEqual(chrf, 0.0)
        self.assertLessEqual(chrf, 100.0)

    def test_chrf_perfect_score(self):
        """Perfect match should give high chrF score."""
        predictions = ["the quick brown fox", "jumps over lazy dog"]
        references = ["the quick brown fox", "jumps over lazy dog"]

        chrf = compute_chrf(predictions, references)

        # Perfect match should be 100 or very close
        self.assertGreater(chrf, 90.0)

    def test_chrf_empty_input(self):
        """Empty input should return 0."""
        chrf = compute_chrf([], [])
        self.assertEqual(chrf, 0.0)

    def test_chrf_raises_on_length_mismatch(self):
        """Mismatched lengths should raise ValueError."""
        predictions = ["a", "b", "c"]
        references = ["a"]

        with self.assertRaises(ValueError) as cm:
            compute_chrf(predictions, references)

        self.assertIn("length", str(cm.exception).lower())


class TestExtractGenreTag(unittest.TestCase):
    """Tests for extract_genre_tag function."""

    def test_extracts_letter_tag(self):
        """Extract [LETTER] tag."""
        result = extract_genre_tag("[LETTER] Some text here")
        self.assertEqual(result, "LETTER")

    def test_extracts_admin_tag(self):
        """Extract [ADMIN] tag."""
        result = extract_genre_tag("[ADMIN] Administrative document")
        self.assertEqual(result, "ADMIN")

    def test_returns_unknown_for_no_tag(self):
        """Return UNKNOWN when no tag present."""
        result = extract_genre_tag("Plain text without tag")
        self.assertEqual(result, "UNKNOWN")

    def test_returns_unknown_for_empty_string(self):
        """Return UNKNOWN for empty string."""
        result = extract_genre_tag("")
        self.assertEqual(result, "UNKNOWN")

    def test_returns_unknown_for_none(self):
        """Return UNKNOWN for None input."""
        result = extract_genre_tag(None)
        self.assertEqual(result, "UNKNOWN")


class TestComputeGenreMetrics(unittest.TestCase):
    """Tests for compute_genre_metrics function."""

    def test_genre_metrics_returns_dict_with_per_genre_scores(self):
        """Test 3: compute_genre_metrics returns dict with per-genre BLEU scores."""
        predictions = [
            "letter content one",
            "letter content two",
            "admin content one",
        ]
        references = [
            "letter content one",
            "letter content two",
            "admin content one",
        ]
        genres = ["LETTER", "LETTER", "ADMIN"]

        result = compute_genre_metrics(predictions, references, genres)

        # Should have both genres
        self.assertIn("LETTER", result)
        self.assertIn("ADMIN", result)

        # Each genre should have bleu, chrf, and count
        for genre in ["LETTER", "ADMIN"]:
            self.assertIn("bleu", result[genre])
            self.assertIn("chrf", result[genre])
            self.assertIn("count", result[genre])

            # Values should be in valid ranges
            self.assertGreaterEqual(result[genre]["bleu"], 0.0)
            self.assertLessEqual(result[genre]["bleu"], 100.0)
            self.assertGreaterEqual(result[genre]["chrf"], 0.0)
            self.assertLessEqual(result[genre]["chrf"], 100.0)
            self.assertGreater(result[genre]["count"], 0)

    def test_genre_metrics_correct_counts(self):
        """Genre counts should match input distribution."""
        predictions = ["a", "b", "c", "d", "e"]
        references = ["a", "b", "c", "d", "e"]
        genres = ["LETTER", "LETTER", "ADMIN", "ADMIN", "ADMIN"]

        result = compute_genre_metrics(predictions, references, genres)

        self.assertEqual(result["LETTER"]["count"], 2)
        self.assertEqual(result["ADMIN"]["count"], 3)

    def test_genre_metrics_empty_input(self):
        """Empty input should return empty dict."""
        result = compute_genre_metrics([], [], [])
        self.assertEqual(result, {})

    def test_genre_metrics_raises_on_length_mismatch(self):
        """Mismatched lengths should raise ValueError."""
        predictions = ["a", "b"]
        references = ["a", "b"]
        genres = ["LETTER"]  # Wrong length

        with self.assertRaises(ValueError) as cm:
            compute_genre_metrics(predictions, references, genres)

        self.assertIn("length", str(cm.exception).lower())


class TestLoadPredictions(unittest.TestCase):
    """Tests for load_predictions function."""

    def test_load_predictions_with_id_and_prediction_columns(self):
        """Test 4: load_predictions reads CSV with id, prediction columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "predictions.csv"

            # Create test CSV
            df = pd.DataFrame(
                {
                    "id": ["sample_1", "sample_2", "sample_3"],
                    "prediction": ["pred one", "pred two", "pred three"],
                }
            )
            df.to_csv(csv_path, index=False)

            # Load and verify
            result = load_predictions(csv_path)

            self.assertEqual(len(result), 3)
            self.assertIn("id", result.columns)
            self.assertIn("prediction", result.columns)
            self.assertEqual(result.iloc[0]["id"], "sample_1")
            self.assertEqual(result.iloc[0]["prediction"], "pred one")

    def test_load_predictions_with_hypothesis_column(self):
        """load_predictions accepts 'hypothesis' column as alternative."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "predictions.csv"

            df = pd.DataFrame(
                {
                    "id": ["s1", "s2"],
                    "hypothesis": ["hyp one", "hyp two"],
                }
            )
            df.to_csv(csv_path, index=False)

            result = load_predictions(csv_path)

            # Should normalize to 'prediction' column
            self.assertIn("prediction", result.columns)
            self.assertEqual(result.iloc[0]["prediction"], "hyp one")

    def test_load_predictions_raises_on_missing_file(self):
        """FileNotFoundError when CSV doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_predictions(Path("/nonexistent/path.csv"))

    def test_load_predictions_raises_on_missing_id_column(self):
        """ValueError when 'id' column is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "predictions.csv"

            df = pd.DataFrame({"prediction": ["a", "b"]})
            df.to_csv(csv_path, index=False)

            with self.assertRaises(ValueError) as cm:
                load_predictions(csv_path)

            self.assertIn("id", str(cm.exception).lower())

    def test_load_predictions_raises_on_missing_prediction_column(self):
        """ValueError when prediction column is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "predictions.csv"

            df = pd.DataFrame({"id": ["a", "b"], "other_col": ["x", "y"]})
            df.to_csv(csv_path, index=False)

            with self.assertRaises(ValueError) as cm:
                load_predictions(csv_path)

            self.assertIn("prediction", str(cm.exception).lower())


class TestLoadValidationData(unittest.TestCase):
    """Tests for load_validation_data function."""

    def test_load_validation_data_auto_detects_columns(self):
        """Auto-detects source and target columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "val.csv"

            df = pd.DataFrame(
                {
                    "id": ["s1", "s2"],
                    "transliteration": ["src1", "src2"],
                    "translation": ["tgt1", "tgt2"],
                }
            )
            df.to_csv(csv_path, index=False)

            result = load_validation_data(csv_path)

            self.assertIn("source", result.columns)
            self.assertIn("target", result.columns)
            self.assertEqual(result.iloc[0]["source"], "src1")
            self.assertEqual(result.iloc[0]["target"], "tgt1")

    def test_load_validation_data_with_normalized_columns(self):
        """Prefers normalized column names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "val.csv"

            df = pd.DataFrame(
                {
                    "transliteration": ["raw_src"],
                    "transliteration_normalized": ["norm_src"],
                    "translation": ["raw_tgt"],
                    "translation_normalized": ["norm_tgt"],
                }
            )
            df.to_csv(csv_path, index=False)

            result = load_validation_data(csv_path)

            # Should prefer normalized columns
            self.assertEqual(result.iloc[0]["source"], "norm_src")
            self.assertEqual(result.iloc[0]["target"], "norm_tgt")

    def test_load_validation_data_includes_genre_if_present(self):
        """Includes genre column if present in CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "val.csv"

            df = pd.DataFrame(
                {
                    "source": ["s1", "s2"],
                    "target": ["t1", "t2"],
                    "genre": ["LETTER", "ADMIN"],
                }
            )
            df.to_csv(csv_path, index=False)

            result = load_validation_data(csv_path)

            self.assertIn("genre", result.columns)
            self.assertEqual(result.iloc[0]["genre"], "LETTER")


if __name__ == "__main__":
    unittest.main()
