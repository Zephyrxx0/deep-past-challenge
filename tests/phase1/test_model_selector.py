import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_PATH = Path("scripts/select_model.py")
CATALOG_PATH = Path("config/models/catalog.yaml")


class TestModelSelector(unittest.TestCase):
    def test_selector_rejects_unsupported_model(self) -> None:
        result = subprocess.run(
            [
                "python",
                str(SCRIPT_PATH),
                "--model-id",
                "unsupported/model-id",
                "--catalog",
                str(CATALOG_PATH),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        combined_output = f"{result.stdout}\n{result.stderr}".lower()
        self.assertNotEqual(0, result.returncode)
        self.assertIn("unsupported model", combined_output)

    def test_selector_writes_active_model_for_supported_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "active_model.yaml"
            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT_PATH),
                    "--model-id",
                    "facebook/mbart-large-50-many-to-many-mmt",
                    "--catalog",
                    str(CATALOG_PATH),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                0, result.returncode, msg=f"{result.stdout}\n{result.stderr}"
            )
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}

            self.assertEqual(
                "facebook/mbart-large-50-many-to-many-mmt", payload.get("model_id")
            )


if __name__ == "__main__":
    unittest.main()
