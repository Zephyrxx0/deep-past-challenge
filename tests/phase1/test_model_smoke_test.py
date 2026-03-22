import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_PATH = Path("scripts/model_smoke_test.py")


class TestModelSmokeTest(unittest.TestCase):
    def test_smoke_script_prints_markers_and_exits_zero_with_tiny_override(
        self,
    ) -> None:
        result = subprocess.run(
            [
                "python",
                str(SCRIPT_PATH),
                "--model-id",
                "hf-internal-testing/tiny-random-t5",
                "--max-new-tokens",
                "12",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        combined_output = f"{result.stdout}\n{result.stderr}"
        self.assertEqual(0, result.returncode, msg=combined_output)
        self.assertIn("model_loaded:", combined_output)
        self.assertIn("tokenizer_loaded:", combined_output)
        self.assertIn("translation:", combined_output)

    def test_smoke_script_fails_when_active_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing_active_model.yaml"
            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT_PATH),
                    "--active-config",
                    str(missing_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        combined_output = f"{result.stdout}\n{result.stderr}".lower()
        self.assertNotEqual(0, result.returncode)
        self.assertIn("error:", combined_output)

    def test_smoke_script_accepts_input_text_and_echoes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            active_config_path = Path(temp_dir) / "active_model.yaml"
            payload = {
                "model_id": "hf-internal-testing/tiny-random-t5",
                "hf_checkpoint": "hf-internal-testing/tiny-random-t5",
                "selected_at_utc": "2026-03-22T00:00:00Z",
            }
            with active_config_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(payload, file, sort_keys=False)

            custom_input = "a-na dumu lugal"
            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT_PATH),
                    "--active-config",
                    str(active_config_path),
                    "--input-text",
                    custom_input,
                    "--max-new-tokens",
                    "12",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        combined_output = f"{result.stdout}\n{result.stderr}"
        self.assertEqual(0, result.returncode, msg=combined_output)
        self.assertIn(f"input_text: {custom_input}", combined_output)


if __name__ == "__main__":
    unittest.main()
