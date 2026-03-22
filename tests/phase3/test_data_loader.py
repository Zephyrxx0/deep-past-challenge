import json
import subprocess
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_loader import load_stage_data


SEED = 42


class TestDataLoader(unittest.TestCase):
    def test_stage1_deterministic_loading(self) -> None:
        df1 = load_stage_data(stage=1, split="train", seed=SEED)
        df2 = load_stage_data(stage=1, split="train", seed=SEED)

        self.assertTrue(
            df1.index.equals(df2.index), "Row indices should be identical across loads"
        )
        self.assertEqual(
            list(df1["transliteration_normalized"].head(10)),
            list(df2["transliteration_normalized"].head(10)),
            "First 10 transliterations should match exactly",
        )

    def test_stage2_deterministic_loading(self) -> None:
        df1 = load_stage_data(stage=2, split="train", seed=SEED)
        df2 = load_stage_data(stage=2, split="train", seed=SEED)

        self.assertTrue(
            df1.index.equals(df2.index), "Row indices should be identical across loads"
        )
        self.assertEqual(
            list(df1["transliteration_normalized"].head(10)),
            list(df2["transliteration_normalized"].head(10)),
            "First 10 transliterations should match exactly",
        )

    def test_stage3_train_deterministic_loading(self) -> None:
        df1 = load_stage_data(stage=3, split="train", seed=SEED)
        df2 = load_stage_data(stage=3, split="train", seed=SEED)

        self.assertTrue(
            df1.index.equals(df2.index), "Row indices should be identical across loads"
        )
        self.assertEqual(
            list(df1["transliteration_normalized"].head(10)),
            list(df2["transliteration_normalized"].head(10)),
            "First 10 transliterations should match exactly",
        )

    def test_stage3_val_deterministic_loading(self) -> None:
        df1 = load_stage_data(stage=3, split="val", seed=SEED)
        df2 = load_stage_data(stage=3, split="val", seed=SEED)

        self.assertTrue(
            df1.index.equals(df2.index), "Row indices should be identical across loads"
        )
        self.assertEqual(
            list(df1["transliteration_normalized"].head(10)),
            list(df2["transliteration_normalized"].head(10)),
            "First 10 transliterations should match exactly",
        )

    def test_required_columns(self) -> None:
        for stage in [1, 2, 3]:
            with self.subTest(stage=stage):
                df = load_stage_data(stage=stage, split="train", seed=SEED)
                self.assertIn(
                    "transliteration_normalized",
                    df.columns,
                    f"Stage {stage} missing transliteration_normalized",
                )
                self.assertIn(
                    "translation_normalized",
                    df.columns,
                    f"Stage {stage} missing translation_normalized",
                )

        df_val = load_stage_data(stage=3, split="val", seed=SEED)
        self.assertIn(
            "transliteration_normalized",
            df_val.columns,
            "Stage 3 val missing transliteration_normalized",
        )
        self.assertIn(
            "translation_normalized",
            df_val.columns,
            "Stage 3 val missing translation_normalized",
        )

    def test_cli_command(self) -> None:
        result = subprocess.run(
            ["python", "scripts/data_loader.py", "--stage", "1", "--split", "train"],
            capture_output=True,
            text=True,
            check=False,
        )

        combined_output = f"{result.stdout}\n{result.stderr}"
        self.assertEqual(0, result.returncode, msg=combined_output)

        output_data = json.loads(result.stdout)
        self.assertIn(
            "row_count", output_data, "JSON output should contain row_count key"
        )


if __name__ == "__main__":
    unittest.main()
