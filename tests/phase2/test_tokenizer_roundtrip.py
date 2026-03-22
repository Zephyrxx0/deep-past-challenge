import subprocess
import tempfile
import unittest
from pathlib import Path


ROUNDTRIP_SCRIPT = Path("scripts/tokenizer_roundtrip_check.py")
TRAIN_SCRIPT = Path("scripts/train_tokenizer.py")
TRAIN_FIXTURE = Path("tests/fixtures/tokenizer_train_small.txt")
SAMPLES_FIXTURE = Path("tests/fixtures/roundtrip_samples.txt")


class TestTokenizerRoundtripContract(unittest.TestCase):
    def test_roundtrip_cli_success_path_reports_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            train_result = subprocess.run(
                [
                    "python",
                    str(TRAIN_SCRIPT),
                    "--input-file",
                    str(TRAIN_FIXTURE),
                    "--output-dir",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            train_output = f"{train_result.stdout}\n{train_result.stderr}"
            self.assertEqual(0, train_result.returncode, msg=train_output)

            manifest_path = output_dir / "tokenizer_manifest.json"
            result = subprocess.run(
                [
                    "python",
                    str(ROUNDTRIP_SCRIPT),
                    "--samples-file",
                    str(SAMPLES_FIXTURE),
                    "--manifest",
                    str(manifest_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            combined_output = f"{result.stdout}\n{result.stderr}"
            self.assertEqual(0, result.returncode, msg=combined_output)
            self.assertIn("samples_checked:", combined_output)
            self.assertIn("roundtrip_passed:", combined_output)

    def test_roundtrip_cli_fails_for_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_manifest = Path(temp_dir) / "missing_manifest.json"
            result = subprocess.run(
                [
                    "python",
                    str(ROUNDTRIP_SCRIPT),
                    "--samples-file",
                    str(SAMPLES_FIXTURE),
                    "--manifest",
                    str(missing_manifest),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        combined_output = f"{result.stdout}\n{result.stderr}".lower()
        self.assertNotEqual(0, result.returncode)
        self.assertIn("error:", combined_output)


if __name__ == "__main__":
    unittest.main()
