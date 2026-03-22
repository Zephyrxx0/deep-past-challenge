import subprocess
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from scripts.create_dataloader import create_dataloader


class TestBatchCollation(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = 1
        self.batch_size = 2
        self.model_id = "facebook/mbart-large-50-many-to-many-mmt"

    @unittest.skipUnless(
        Path("data/stage1_train.csv").exists() and torch is not None,
        "Dataset and torch required",
    )
    def test_dataloader_instance(self) -> None:
        loader = create_dataloader(
            stage=self.stage,
            model_id=self.model_id,
            batch_size=self.batch_size,
        )
        self.assertIsInstance(
            loader,
            torch.utils.data.DataLoader,
            "create_dataloader should return DataLoader instance",
        )

    @unittest.skipUnless(
        Path("data/stage1_train.csv").exists() and torch is not None,
        "Dataset and torch required",
    )
    def test_batch_keys(self) -> None:
        loader = create_dataloader(
            stage=self.stage,
            model_id=self.model_id,
            batch_size=self.batch_size,
        )
        batch = next(iter(loader))
        self.assertIn("input_ids", batch, "Batch missing input_ids")
        self.assertIn("attention_mask", batch, "Batch missing attention_mask")
        self.assertIn("labels", batch, "Batch missing labels")

    @unittest.skipUnless(
        Path("data/stage1_train.csv").exists() and torch is not None,
        "Dataset and torch required",
    )
    def test_input_ids_shape(self) -> None:
        loader = create_dataloader(
            stage=self.stage,
            model_id=self.model_id,
            batch_size=self.batch_size,
        )
        batch = next(iter(loader))
        self.assertEqual(
            batch["input_ids"].shape[0],
            self.batch_size,
            "input_ids batch dimension mismatch",
        )
        self.assertEqual(
            len(batch["input_ids"].shape), 2, "input_ids should be 2D tensor"
        )

    @unittest.skipUnless(
        Path("data/stage1_train.csv").exists() and torch is not None,
        "Dataset and torch required",
    )
    def test_attention_mask_shape(self) -> None:
        loader = create_dataloader(
            stage=self.stage,
            model_id=self.model_id,
            batch_size=self.batch_size,
        )
        batch = next(iter(loader))
        self.assertEqual(
            batch["attention_mask"].shape,
            batch["input_ids"].shape,
            "attention_mask shape should match input_ids",
        )

    @unittest.skipUnless(
        Path("data/stage1_train.csv").exists() and torch is not None,
        "Dataset and torch required",
    )
    def test_attention_mask_values(self) -> None:
        loader = create_dataloader(
            stage=self.stage,
            model_id=self.model_id,
            batch_size=self.batch_size,
        )
        batch = next(iter(loader))
        mask = batch["attention_mask"]
        self.assertTrue(
            torch.all((mask == 0) | (mask == 1)), "attention_mask values must be 0 or 1"
        )

    @unittest.skipUnless(
        Path("data/stage1_train.csv").exists() and torch is not None,
        "Dataset and torch required",
    )
    def test_labels_shape(self) -> None:
        loader = create_dataloader(
            stage=self.stage,
            model_id=self.model_id,
            batch_size=self.batch_size,
        )
        batch = next(iter(loader))
        self.assertEqual(
            batch["labels"].shape[0], self.batch_size, "labels batch dimension mismatch"
        )
        self.assertEqual(len(batch["labels"].shape), 2, "labels should be 2D tensor")

    @unittest.skipUnless(
        Path("data/stage1_train.csv").exists() and torch is not None,
        "Dataset and torch required",
    )
    def test_labels_padding_masked(self) -> None:
        loader = create_dataloader(
            stage=self.stage,
            model_id=self.model_id,
            batch_size=self.batch_size,
        )
        batch = next(iter(loader))
        self.assertTrue(
            torch.any(batch["labels"] == -100),
            "Labels should have -100 for padding positions",
        )

    def test_cli_command(self) -> None:
        result = subprocess.run(
            [
                "python",
                "scripts/create_dataloader.py",
                "--stage",
                "1",
                "--batch-size",
                "2",
                "--model-id",
                self.model_id,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        combined_output = f"{result.stdout}\n{result.stderr}"
        self.assertEqual(0, result.returncode, msg=combined_output)


if __name__ == "__main__":
    unittest.main()
