"""TRN1-01 contract tests for Stage 1 training CLI.

Tests validate:
1. Dry-run mode writes run_manifest.json with required fields
2. Invalid config override exits non-zero with error message
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SEED = 42


class TestStage1TrainingCLI(unittest.TestCase):
    """Contract tests for scripts/train_stage1.py CLI."""

    def setUp(self) -> None:
        """Create temporary directories and fixture CSVs."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True)

        # Create minimal fixture CSVs
        self.train_csv = Path(self.temp_dir) / "train.csv"
        self.val_csv = Path(self.temp_dir) / "val.csv"

        train_df = pd.DataFrame(
            {
                "transliteration_normalized": ["a-na šar-ri", "be-li-ia"],
                "translation_normalized": ["to the king", "my lord"],
            }
        )
        val_df = pd.DataFrame(
            {
                "transliteration_normalized": ["i-na e-ka"],
                "translation_normalized": ["in the temple"],
            }
        )

        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)

    def tearDown(self) -> None:
        """Clean up temporary directories."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dry_run_writes_manifest_with_required_fields(self) -> None:
        """Test 1: --dry-run writes run_manifest.json with stage=1, model_id, seed."""
        result = subprocess.run(
            [
                "python",
                "scripts/train_stage1.py",
                "--dry-run",
                "--train-csv",
                str(self.train_csv),
                "--val-csv",
                str(self.val_csv),
                "--output-dir",
                str(self.output_dir),
                "--model-id",
                "facebook/mbart-large-50-many-to-many-mmt",
                "--seed",
                str(SEED),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        combined_output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
        self.assertEqual(0, result.returncode, msg=combined_output)

        # Verify run_manifest.json exists
        manifest_path = self.output_dir / "run_manifest.json"
        self.assertTrue(
            manifest_path.exists(), f"run_manifest.json not found in {self.output_dir}"
        )

        # Verify required fields
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(1, manifest.get("stage"), "stage should be 1")
        self.assertEqual(
            "facebook/mbart-large-50-many-to-many-mmt",
            manifest.get("model_id"),
            "model_id should match",
        )
        self.assertEqual(SEED, manifest.get("seed"), f"seed should be {SEED}")
        self.assertIn("output_dir", manifest, "output_dir should be in manifest")
        self.assertIn("train_csv", manifest, "train_csv should be in manifest")
        self.assertIn(
            "started_at_utc", manifest, "started_at_utc should be in manifest"
        )

    def test_invalid_train_csv_exits_nonzero_with_error(self) -> None:
        """Test 2: Invalid config (missing train_csv file) exits non-zero with error."""
        result = subprocess.run(
            [
                "python",
                "scripts/train_stage1.py",
                "--dry-run",
                "--train-csv",
                "/nonexistent/path/to/train.csv",
                "--output-dir",
                str(self.output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(
            0, result.returncode, "Should exit non-zero for missing train_csv"
        )

        combined_output = f"{result.stdout}\n{result.stderr}".lower()
        self.assertIn("train_csv", combined_output, "Error should mention train_csv")

    def test_dry_run_does_not_import_torch(self) -> None:
        """Test 3: Dry-run mode should work without importing torch."""
        # We test this indirectly: dry-run should succeed even in environments
        # where torch is slow to import. The real test is that Test 1 passes
        # in reasonable time. This test documents the expectation.
        result = subprocess.run(
            [
                "python",
                "scripts/train_stage1.py",
                "--dry-run",
                "--train-csv",
                str(self.train_csv),
                "--output-dir",
                str(self.output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,  # Should complete quickly without ML imports
        )

        # Dry-run should exit 0
        combined_output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
        self.assertEqual(0, result.returncode, msg=combined_output)

    def test_default_seed_is_42(self) -> None:
        """Test 4: Default seed is 42 per AGENTS.md."""
        result = subprocess.run(
            [
                "python",
                "scripts/train_stage1.py",
                "--dry-run",
                "--train-csv",
                str(self.train_csv),
                "--output-dir",
                str(self.output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, f"Failed: {result.stderr}")

        manifest_path = self.output_dir / "run_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(42, manifest.get("seed"), "Default seed should be 42")


if __name__ == "__main__":
    unittest.main()
