import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
import os
import shutil


class TestSubmissionValidatorContract(unittest.TestCase):
    def setUp(self):
        # Create temp directory for test CSVs
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp files
        shutil.rmtree(self.temp_dir)

    def _create_csv(self, filename, headers, rows):
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return filepath

    def _run_validator(self, filepath):
        cmd = ["python", "scripts/validate_submission.py", "--csv", filepath]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def test_valid_submission_passes(self):
        filepath = self._create_csv(
            "valid.csv",
            ["id", "akkadian_source", "english_translation"],
            [[1, "text1", "translation1"], [2, "text2", "translation2"]],
        )
        result = self._run_validator(filepath)
        self.assertEqual(result.returncode, 0)
        self.assertTrue("PASS" in result.stdout or "valid" in result.stdout.lower())

    def test_missing_column_fails(self):
        filepath = self._create_csv(
            "missing_col.csv", ["id", "akkadian_source"], [[1, "text1"], [2, "text2"]]
        )
        result = self._run_validator(filepath)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            "missing" in result.stderr.lower() or "required" in result.stderr.lower()
        )

    def test_empty_translation_fails(self):
        filepath = self._create_csv(
            "empty_trans.csv",
            ["id", "akkadian_source", "english_translation"],
            [[1, "text1", ""], [2, "text2", "translation2"]],
        )
        result = self._run_validator(filepath)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            "empty" in result.stderr.lower() or "missing" in result.stderr.lower()
        )

    def test_duplicate_id_fails(self):
        filepath = self._create_csv(
            "dup_id.csv",
            ["id", "akkadian_source", "english_translation"],
            [[1, "text1", "trans1"], [1, "text2", "trans2"]],
        )
        result = self._run_validator(filepath)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue("duplicate" in result.stderr.lower())

    def test_non_sequential_ids_warn_but_pass(self):
        filepath = self._create_csv(
            "non_seq.csv",
            ["id", "akkadian_source", "english_translation"],
            [[1, "text1", "trans1"], [3, "text2", "trans2"]],
        )
        result = self._run_validator(filepath)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(
            "warning" in result.stdout.lower()
            or "not sequential" in result.stdout.lower()
        )


if __name__ == "__main__":
    unittest.main()
