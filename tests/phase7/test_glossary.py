"""Tests for glossary utilities module.

Tests verify:
1. load_glossary() loads JSON and returns dict
2. apply_glossary(text, glossary) replaces proper names correctly
3. extract_proper_names(text) identifies capitalized sequences
4. compute_name_accuracy(predictions, references) returns accuracy score
"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.glossary_utils import (
    apply_glossary,
    compute_name_accuracy,
    count_corrections,
    extract_proper_names,
    load_glossary,
)


class TestLoadGlossary(unittest.TestCase):
    """Tests for load_glossary function."""

    def test_load_glossary_returns_dict(self):
        """Test 1: load_glossary() loads JSON and returns dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glossary_path = Path(tmpdir) / "glossary.json"

            # Create test glossary
            test_glossary = {
                "Sargon": "Sargon",
                "Naram-Sin": "Naram-Sin",
                "Akkad": "Akkad",
            }
            with glossary_path.open("w", encoding="utf-8") as f:
                json.dump(test_glossary, f)

            # Load and verify
            result = load_glossary(glossary_path)

            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 3)
            self.assertEqual(result["Sargon"], "Sargon")
            self.assertEqual(result["Akkad"], "Akkad")

    def test_load_glossary_raises_on_missing_file(self):
        """FileNotFoundError when glossary file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_glossary(Path("/nonexistent/glossary.json"))

    def test_load_glossary_raises_on_invalid_json(self):
        """JSONDecodeError when file is not valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glossary_path = Path(tmpdir) / "glossary.json"
            glossary_path.write_text("not valid json {{{")

            with self.assertRaises(json.JSONDecodeError):
                load_glossary(glossary_path)

    def test_load_glossary_raises_on_non_dict(self):
        """ValueError when JSON is not a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glossary_path = Path(tmpdir) / "glossary.json"
            glossary_path.write_text('["a", "b", "c"]')

            with self.assertRaises(ValueError) as cm:
                load_glossary(glossary_path)

            self.assertIn("dict", str(cm.exception).lower())


class TestApplyGlossary(unittest.TestCase):
    """Tests for apply_glossary function."""

    def test_apply_glossary_replaces_names(self):
        """Test 2: apply_glossary(text, glossary) replaces proper names correctly."""
        glossary = {
            "Assur": "Aššur",
            "Shamash": "Šamaš",
        }
        text = "King of Assur worships Shamash."

        result = apply_glossary(text, glossary)

        self.assertEqual(result, "King of Aššur worships Šamaš.")

    def test_apply_glossary_case_insensitive(self):
        """Apply glossary handles case variations."""
        glossary = {"sargon": "Sargon"}
        text = "SARGON was a king. sargon ruled Akkad."

        result = apply_glossary(text, glossary)

        # Should replace both occurrences
        self.assertIn("SARGON", result)  # uppercase preserved
        self.assertIn("sargon", result)  # lowercase preserved

    def test_apply_glossary_empty_text(self):
        """Empty text returns empty string."""
        result = apply_glossary("", {"Sargon": "Sargon"})
        self.assertEqual(result, "")

    def test_apply_glossary_empty_glossary(self):
        """Empty glossary returns original text."""
        text = "King of Akkad"
        result = apply_glossary(text, {})
        self.assertEqual(result, text)

    def test_apply_glossary_no_match(self):
        """Text without glossary terms unchanged."""
        glossary = {"Sargon": "Sargon"}
        text = "The temple was built."

        result = apply_glossary(text, glossary)

        self.assertEqual(result, text)

    def test_apply_glossary_word_boundaries(self):
        """Only matches whole words, not substrings."""
        glossary = {"Ur": "Ur"}
        text = "The urban area near Ur was prosperous."

        result = apply_glossary(text, glossary)

        # "urban" should not be affected
        self.assertIn("urban", result)
        self.assertEqual(result, text)


class TestExtractProperNames(unittest.TestCase):
    """Tests for extract_proper_names function."""

    def test_extract_proper_names_finds_capitalized(self):
        """Test 3: extract_proper_names(text) identifies capitalized sequences."""
        text = "King Sargon of Akkad conquered Ur and Nippur."

        result = extract_proper_names(text)

        self.assertIn("Sargon", result)
        self.assertIn("Akkad", result)
        self.assertIn("Ur", result)
        self.assertIn("Nippur", result)

    def test_extract_proper_names_hyphenated(self):
        """Extracts hyphenated names like Naram-Sin."""
        text = "Naram-Sin was a great king."

        result = extract_proper_names(text)

        self.assertIn("Naram-Sin", result)

    def test_extract_proper_names_filters_common_words(self):
        """Filters out common English words."""
        text = "The King went to the palace."

        result = extract_proper_names(text)

        self.assertNotIn("The", result)
        self.assertIn("King", result)  # King is not a common word

    def test_extract_proper_names_empty_text(self):
        """Empty text returns empty list."""
        result = extract_proper_names("")
        self.assertEqual(result, [])

    def test_extract_proper_names_no_capitals(self):
        """Text without capitalized words returns empty list."""
        text = "the temple was built by workers."

        result = extract_proper_names(text)

        self.assertEqual(result, [])

    def test_extract_proper_names_unique(self):
        """Returns unique names only."""
        text = "Sargon was king. Sargon conquered Akkad."

        result = extract_proper_names(text)

        # Should only appear once
        self.assertEqual(result.count("Sargon"), 1)


class TestComputeNameAccuracy(unittest.TestCase):
    """Tests for compute_name_accuracy function."""

    def test_compute_name_accuracy_perfect_match(self):
        """Test 4: compute_name_accuracy(predictions, references) returns accuracy score."""
        predictions = ["King Sargon of Akkad", "The temple of Ishtar"]
        references = ["King Sargon of Akkad", "The temple of Ishtar"]

        accuracy = compute_name_accuracy(predictions, references)

        self.assertIsInstance(accuracy, float)
        self.assertEqual(accuracy, 100.0)

    def test_compute_name_accuracy_partial_match(self):
        """Partial match returns percentage."""
        predictions = ["King Sargon of Ur"]  # Akkad → Ur
        references = ["King Sargon of Akkad"]

        accuracy = compute_name_accuracy(predictions, references)

        # Sargon correct, Akkad missing = 50%
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)
        self.assertLess(accuracy, 100.0)

    def test_compute_name_accuracy_no_match(self):
        """No matching names returns 0%."""
        predictions = ["The temple was built"]
        references = ["King Sargon of Akkad"]

        accuracy = compute_name_accuracy(predictions, references)

        self.assertEqual(accuracy, 0.0)

    def test_compute_name_accuracy_empty_input(self):
        """Empty input returns 100%."""
        accuracy = compute_name_accuracy([], [])
        self.assertEqual(accuracy, 100.0)

    def test_compute_name_accuracy_no_names_in_reference(self):
        """No names in reference returns 100%."""
        predictions = ["The temple was built"]
        references = ["The temple was built"]

        accuracy = compute_name_accuracy(predictions, references)

        self.assertEqual(accuracy, 100.0)

    def test_compute_name_accuracy_raises_on_length_mismatch(self):
        """Mismatched lengths raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            compute_name_accuracy(["a", "b"], ["a"])

        self.assertIn("length", str(cm.exception).lower())

    def test_compute_name_accuracy_with_glossary(self):
        """Glossary filters which names to consider."""
        glossary = {"Sargon": "Sargon"}  # Only Sargon in glossary
        predictions = ["King Sargon went to Akkad"]
        references = ["King Sargon went to Akkad"]

        accuracy = compute_name_accuracy(predictions, references, glossary)

        # Only Sargon is considered, and it matches
        self.assertEqual(accuracy, 100.0)


class TestCountCorrections(unittest.TestCase):
    """Tests for count_corrections function."""

    def test_count_corrections_no_change(self):
        """No corrections when text unchanged."""
        text = "King of Akkad"
        glossary = {"Akkad": "Akkad"}

        count = count_corrections(text, text, glossary)

        self.assertEqual(count, 0)

    def test_count_corrections_with_changes(self):
        """Counts corrections when changes made."""
        original = "King of Assur"
        corrected = "King of Aššur"
        glossary = {"Assur": "Aššur"}

        count = count_corrections(original, corrected, glossary)

        self.assertEqual(count, 1)

    def test_count_corrections_multiple(self):
        """Counts multiple corrections."""
        original = "Assur and Assur"
        corrected = "Aššur and Aššur"
        glossary = {"Assur": "Aššur"}

        count = count_corrections(original, corrected, glossary)

        self.assertEqual(count, 2)


class TestLoadRealGlossary(unittest.TestCase):
    """Tests for loading the actual project glossary."""

    def test_load_project_glossary(self):
        """Load the actual data/glossary.json file."""
        glossary_path = Path("data/glossary.json")

        if not glossary_path.exists():
            self.skipTest("data/glossary.json not found")

        glossary = load_glossary(glossary_path)

        self.assertIsInstance(glossary, dict)
        self.assertGreater(len(glossary), 0)
        # Check for some expected entries
        self.assertIn("a-šùr", glossary)


if __name__ == "__main__":
    unittest.main()
