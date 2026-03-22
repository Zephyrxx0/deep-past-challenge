import json
import subprocess
import unittest
from pathlib import Path

import yaml


SCRIPT_PATH = Path("scripts/print_stage_config.py")
CONFIG_DIR = Path("config/training")
REQUIRED_KEYS = {
    "model_id",
    "train_csv",
    "val_csv",
    "epochs",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "max_source_length",
    "max_target_length",
    "seed",
    "output_dir",
}


class TestStageConfigLoading(unittest.TestCase):
    def test_stage_configs_exist_and_include_required_keys(self) -> None:
        for stage in ["stage1", "stage2", "stage3"]:
            config_path = CONFIG_DIR / f"{stage}.yaml"
            self.assertTrue(config_path.exists(), f"{config_path} must exist")

            with config_path.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}

            missing = REQUIRED_KEYS.difference(payload.keys())
            self.assertFalse(
                missing, f"{stage} missing required keys: {sorted(missing)}"
            )

    def test_stage_learning_rates_match_exact_schedule(self) -> None:
        expected = {
            "stage1": 1e-4,
            "stage2": 5e-5,
            "stage3": 1e-5,
        }
        for stage, learning_rate in expected.items():
            config_path = CONFIG_DIR / f"{stage}.yaml"
            self.assertTrue(config_path.exists(), f"{config_path} must exist")

            with config_path.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}

            self.assertAlmostEqual(learning_rate, float(payload.get("learning_rate")))

    def test_loader_cli_prints_json_with_stage_and_payload(self) -> None:
        result = subprocess.run(
            ["python", str(SCRIPT_PATH), "--stage", "stage2"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=f"{result.stdout}\n{result.stderr}")
        payload = json.loads(result.stdout)
        self.assertEqual("stage2", payload.get("stage"))
        self.assertIn("config", payload)
        self.assertAlmostEqual(5e-5, float(payload["config"].get("learning_rate")))


if __name__ == "__main__":
    unittest.main()
