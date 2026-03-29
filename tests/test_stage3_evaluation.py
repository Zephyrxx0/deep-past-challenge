"""Tests for Stage 3 evaluation and checkpoint selection.

Tests TRN3-02: User can select and export best Stage 3 checkpoint based on validation BLEU.
"""

import argparse
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import sys

# Ensure scripts can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_stage3
from scripts import checkpoint_selection


class TestFindBestCheckpoint:
    """Tests for find_best_checkpoint function."""

    @patch("scripts.checkpoint_selection.evaluate_checkpoint")
    @patch("pathlib.Path.glob")
    def test_find_best_checkpoint_basic(self, mock_glob, mock_evaluate):
        """Test find_best_checkpoint returns path to highest BLEU checkpoint."""
        # Setup mock checkpoints
        checkpoints_dir = Path("models/stage3_test")
        epoch1 = checkpoints_dir / "epoch_1"
        epoch2 = checkpoints_dir / "epoch_2"
        epoch3 = checkpoints_dir / "epoch_3"

        mock_glob.return_value = [epoch1, epoch2, epoch3]

        # Mock evaluation results (epoch_2 has highest BLEU)
        def mock_eval(
            checkpoint_path,
            val_csv,
            batch_size,
            max_source_length,
            max_target_length,
            device,
        ):
            if checkpoint_path == epoch1:
                return {"bleu": 25.5, "chrf": 50.2}
            elif checkpoint_path == epoch2:
                return {"bleu": 28.3, "chrf": 52.1}
            else:  # epoch3
                return {"bleu": 26.8, "chrf": 51.5}

        mock_evaluate.side_effect = mock_eval

        # Call find_best_checkpoint
        best_path = checkpoint_selection.find_best_checkpoint(
            checkpoints_dir=checkpoints_dir,
            val_csv=Path("data/stage3_val.csv"),
            metric="bleu",
        )

        assert best_path == epoch2
        assert mock_evaluate.call_count == 3

    @patch("scripts.checkpoint_selection.evaluate_checkpoint")
    @patch("pathlib.Path.glob")
    def test_find_best_checkpoint_with_chrf(self, mock_glob, mock_evaluate):
        """Test find_best_checkpoint with chrF metric."""
        checkpoints_dir = Path("models/stage3_test")
        epoch1 = checkpoints_dir / "epoch_1"
        epoch2 = checkpoints_dir / "epoch_2"

        mock_glob.return_value = [epoch1, epoch2]

        # Mock evaluation results (epoch_1 has higher chrF)
        def mock_eval(
            checkpoint_path,
            val_csv,
            batch_size,
            max_source_length,
            max_target_length,
            device,
        ):
            if checkpoint_path == epoch1:
                return {"bleu": 25.5, "chrf": 55.0}
            else:
                return {"bleu": 26.0, "chrf": 52.0}

        mock_evaluate.side_effect = mock_eval

        best_path = checkpoint_selection.find_best_checkpoint(
            checkpoints_dir=checkpoints_dir,
            val_csv=Path("data/stage3_val.csv"),
            metric="chrf",
        )

        assert best_path == epoch1


class TestCompareCheckpoints:
    """Tests for compare_checkpoints function."""

    @patch("scripts.checkpoint_selection.evaluate_checkpoint")
    def test_compare_checkpoints_returns_sorted_list(self, mock_evaluate):
        """Test compare_checkpoints returns sorted list of checkpoint results."""
        checkpoint_paths = [
            Path("models/stage3_test/epoch_1"),
            Path("models/stage3_test/epoch_2"),
            Path("models/stage3_test/epoch_3"),
        ]

        # Mock evaluation results
        def mock_eval(
            checkpoint_path,
            val_csv,
            batch_size,
            max_source_length,
            max_target_length,
            device,
        ):
            if "epoch_1" in str(checkpoint_path):
                return {"bleu": 25.5, "chrf": 50.2}
            elif "epoch_2" in str(checkpoint_path):
                return {"bleu": 28.3, "chrf": 52.1}
            else:
                return {"bleu": 26.8, "chrf": 51.5}

        mock_evaluate.side_effect = mock_eval

        results = checkpoint_selection.compare_checkpoints(
            checkpoint_paths=checkpoint_paths,
            val_csv=Path("data/stage3_val.csv"),
        )

        # Should be sorted by BLEU (descending)
        assert len(results) == 3
        assert results[0]["bleu"] == 28.3
        assert results[1]["bleu"] == 26.8
        assert results[2]["bleu"] == 25.5
        assert "epoch_2" in str(results[0]["path"])


class TestExportBestCheckpoint:
    """Tests for export_best_checkpoint function."""

    def test_export_best_checkpoint_creates_manifest(self, tmp_path):
        """Test export_best_checkpoint creates manifest and evaluation results."""
        # Create a mock checkpoint directory with files
        best_path = tmp_path / "epoch_2"
        best_path.mkdir()

        # Create dummy checkpoint files
        (best_path / "model.safetensors").write_text("dummy model")
        (best_path / "tokenizer.json").write_text("{}")
        (best_path / "config.json").write_text("{}")

        # Export directory
        export_dir = tmp_path / "stage3_final"

        # Call export_best_checkpoint
        checkpoint_selection.export_best_checkpoint(
            best_path=best_path,
            export_dir=export_dir,
            metrics={"bleu": 28.3, "chrf": 52.1, "epoch": 2},
        )

        # Verify export directory was created
        assert export_dir.exists()

        # Verify files were copied
        assert (export_dir / "model.safetensors").exists()
        assert (export_dir / "tokenizer.json").exists()
        assert (export_dir / "config.json").exists()

        # Verify manifest was created
        manifest_path = export_dir / "best_checkpoint_manifest.json"
        assert manifest_path.exists()

        # Verify manifest contents
        import json

        with manifest_path.open() as f:
            manifest = json.load(f)

        assert manifest["bleu"] == 28.3
        assert manifest["chrf"] == 52.1
        assert manifest["epoch"] == 2

        # Verify evaluation results were created
        results_path = export_dir / "evaluation_results.json"
        assert results_path.exists()


class TestEvaluateCheckpoint:
    """Tests for evaluate_checkpoint function."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("pandas.read_csv")
    @patch("sacrebleu.corpus_bleu")
    @patch("sacrebleu.corpus_chrf")
    @patch("torch.cuda.is_available")
    def test_evaluate_checkpoint_computes_metrics(
        self,
        mock_cuda,
        mock_chrf,
        mock_bleu,
        mock_read_csv,
        mock_model,
        mock_tokenizer,
    ):
        """Test evaluate_checkpoint computes BLEU and chrF."""
        mock_cuda.return_value = False

        # Mock dataframe
        mock_df = MagicMock()
        mock_df.columns = ["transliteration_normalized", "translation_normalized"]
        mock_df.__getitem__.side_effect = lambda x: MagicMock(
            tolist=lambda: ["test1", "test2"]
        )
        mock_read_csv.return_value = mock_df

        # Mock BLEU/chrF results
        mock_bleu.return_value = MagicMock(score=25.5)
        mock_chrf.return_value = MagicMock(score=50.2)

        # Mock tokenizer
        mock_tok_instance = MagicMock()
        mock_tok_instance.return_value = {
            "input_ids": MagicMock(to=lambda x: MagicMock())
        }
        mock_tok_instance.batch_decode.return_value = ["hyp1", "hyp2"]
        mock_tokenizer.return_value = mock_tok_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = [[1, 2, 3]]
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        # Call evaluate_checkpoint
        result = checkpoint_selection.evaluate_checkpoint(
            checkpoint_path=Path("models/stage3_test/epoch_1"),
            val_csv=Path("data/stage3_val.csv"),
            batch_size=4,
            max_source_length=256,
            max_target_length=256,
            device="cpu",
        )

        assert result["bleu"] == 25.5
        assert result["chrf"] == 50.2


class TestEvaluateStage3CLI:
    """Tests for evaluate_stage3.py CLI."""

    def test_cli_args_parsing(self):
        """Test that CLI accepts required arguments."""
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                checkpoints_dir=Path("models/stage3_test"),
                val_data=Path("data/stage3_val.csv"),
                export_dir=Path("models/stage3_final"),
                dry_run=True,
                metric="bleu",
                batch_size=4,
                max_source_length=256,
                max_target_length=256,
                device="cpu",
            )

            args = evaluate_stage3.parse_args()

            assert args.checkpoints_dir == Path("models/stage3_test")
            assert args.metric == "bleu"

    @patch("scripts.evaluate_stage3.validate_checkpoint_completeness")
    @patch("scripts.evaluate_stage3.find_best_checkpoint")
    @patch("scripts.evaluate_stage3.compare_checkpoints")
    @patch("scripts.evaluate_stage3.export_best_checkpoint")
    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_cli_dry_run(
        self,
        mock_glob,
        mock_exists,
        mock_is_dir,
        mock_export,
        mock_compare,
        mock_find_best,
        mock_validate,
    ):
        """Test CLI dry-run mode evaluates but doesn't export."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        mock_validate.return_value = True

        epoch1 = Path("models/stage3_test/epoch_1")
        epoch2 = Path("models/stage3_test/epoch_2")

        mock_glob.return_value = [epoch1, epoch2]
        mock_find_best.return_value = epoch2
        mock_compare.return_value = [
            {
                "path": epoch2,
                "bleu": 28.3,
                "chrf": 52.1,
                "epoch": 2,
            },
            {
                "path": epoch1,
                "bleu": 25.5,
                "chrf": 50.2,
                "epoch": 1,
            },
        ]

        with patch("scripts.evaluate_stage3.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                checkpoints_dir=Path("models/stage3_test"),
                val_data=Path("data/stage3_val.csv"),
                export_dir=Path("models/stage3_final"),
                dry_run=True,
                metric="bleu",
                batch_size=4,
                max_source_length=256,
                max_target_length=256,
                device="cpu",
            )

            ret_code = evaluate_stage3.main()

            assert ret_code == 0
            mock_compare.assert_called_once()
            # In dry-run, export should NOT be called
            mock_export.assert_not_called()
