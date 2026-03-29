"""INF-01 contract tests for batched inference CLI.

Tests verify:
1. Dry-run mode writes inference_results.csv with correct schema
2. Missing checkpoint exits non-zero with error message containing "checkpoint"
3. Missing test CSV exits non-zero with error message containing "test-csv"
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestInferenceContract(unittest.TestCase):
    """Contract tests for run_inference.py CLI."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.script_path = self.project_root / "scripts" / "run_inference.py"

    def test_dry_run_writes_csv_with_correct_schema(self):
        """Test 1: Dry-run writes inference_results.csv with header: id,akkadian_source,english_translation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.script_path),
                    "--dry-run",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
            )

            # Assert exit code 0
            self.assertEqual(
                result.returncode,
                0,
                f"Dry-run should exit 0.\nstdout: {result.stdout}\nstderr: {result.stderr}",
            )

            # Assert CSV file exists
            csv_path = Path(tmpdir) / "inference_results.csv"
            self.assertTrue(
                csv_path.exists(), f"Expected inference_results.csv in {tmpdir}"
            )

            # Assert correct schema
            with csv_path.open("r", encoding="utf-8") as f:
                header = f.readline().strip()

            self.assertEqual(
                header,
                "id,akkadian_source,english_translation",
                f"CSV header must be 'id,akkadian_source,english_translation'. Got: {header}",
            )

    def test_missing_checkpoint_exits_nonzero_with_error(self):
        """Test 2: Missing checkpoint path exits non-zero with error containing 'checkpoint'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-existent checkpoint path
            fake_checkpoint = Path(tmpdir) / "nonexistent_checkpoint"

            result = subprocess.run(
                [
                    sys.executable,
                    str(self.script_path),
                    "--checkpoint",
                    str(fake_checkpoint),
                    "--test-csv",
                    "data/stage3_val.csv",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
            )

            # Assert non-zero exit code
            self.assertNotEqual(
                result.returncode, 0, "Missing checkpoint should exit non-zero"
            )

            # Assert error message contains 'checkpoint'
            combined_output = result.stdout.lower() + result.stderr.lower()
            self.assertIn(
                "checkpoint",
                combined_output,
                f"Error should mention 'checkpoint'. Got:\nstdout: {result.stdout}\nstderr: {result.stderr}",
            )

    def test_missing_test_csv_exits_nonzero_with_error(self):
        """Test 3: Missing test CSV exits non-zero with error containing 'test-csv'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake checkpoint directory with config.json
            fake_checkpoint = Path(tmpdir) / "fake_checkpoint"
            fake_checkpoint.mkdir()
            # Create a minimal config.json to make it look like a valid checkpoint
            (fake_checkpoint / "config.json").write_text("{}")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self.script_path),
                    "--checkpoint",
                    str(fake_checkpoint),
                    "--test-csv",
                    "/nonexistent.csv",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
            )

            # Assert non-zero exit code
            self.assertNotEqual(
                result.returncode, 0, "Missing test-csv should exit non-zero"
            )

            # Assert error message contains 'test-csv'
            combined_output = result.stdout.lower() + result.stderr.lower()
            self.assertIn(
                "test-csv",
                combined_output,
                f"Error should mention 'test-csv'. Got:\nstdout: {result.stdout}\nstderr: {result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
