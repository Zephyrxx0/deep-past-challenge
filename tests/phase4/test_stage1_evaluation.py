"""TRN1-02 contract tests for Stage 1 evaluation CLI.

Tests verify:
1. Dry-run mode writes stage1_metrics.json with correct schema
2. Missing checkpoint exits non-zero with helpful error message
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestStage1EvaluationContract(unittest.TestCase):
    """Contract tests for evaluate_stage1.py CLI."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.script_path = self.project_root / "scripts" / "evaluate_stage1.py"

    def test_dry_run_writes_metrics_json_with_correct_schema(self):
        """Test 1: Dry-run writes stage1_metrics.json with keys [stage, bleu, chrf, samples]."""
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

            # Assert metrics file exists
            metrics_path = Path(tmpdir) / "stage1_metrics.json"
            self.assertTrue(
                metrics_path.exists(), f"Expected stage1_metrics.json in {tmpdir}"
            )

            # Assert correct schema
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)

            required_keys = {"stage", "bleu", "chrf", "samples"}
            self.assertTrue(
                required_keys.issubset(metrics.keys()),
                f"Metrics must contain keys {required_keys}. Got: {metrics.keys()}",
            )

            # Assert stage is 1
            self.assertEqual(metrics["stage"], 1, "Stage should be 1")

            # In dry-run mode, bleu/chrf/samples should be zero
            self.assertEqual(metrics["bleu"], 0.0, "Dry-run BLEU should be 0.0")
            self.assertEqual(metrics["chrf"], 0.0, "Dry-run chrF should be 0.0")
            self.assertEqual(metrics["samples"], 0, "Dry-run samples should be 0")

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


if __name__ == "__main__":
    unittest.main()
