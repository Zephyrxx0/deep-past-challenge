import argparse
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Ensure scripts can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_stage3
from scripts import training_utils


class TestStage3CLI:
    def test_cli_args_parsing(self):
        """Test that CLI accepts required arguments and defaults."""
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            # Mock return values
            mock_parse.return_value = argparse.Namespace(
                checkpoint=Path("models/stage2_final"),
                early_stopping_patience=3,
                dry_run=True,
                config=Path("config/training/stage3.yaml"),
                output_dir=Path("models/stage3_test"),
                seed=42,
            )

            args = train_stage3.parse_args()

            assert args.checkpoint == Path("models/stage2_final")
            assert args.early_stopping_patience == 3
            assert args.dry_run is True

    @patch("scripts.train_stage3.validate_checkpoint")
    @patch("scripts.train_stage3.load_config")
    @patch("scripts.train_stage3.merge_config")
    @patch("scripts.train_stage3.validate_config")
    @patch("scripts.train_stage3.write_run_manifest")
    @patch("scripts.train_stage3.validate_genre_tags")
    @patch("pathlib.Path.exists")
    def test_dry_run_validation(
        self,
        mock_exists,
        mock_genre,
        mock_write,
        mock_val_conf,
        mock_merge,
        mock_load,
        mock_val_ckpt,
    ):
        """Test that dry-run performs validation but skips training."""
        # Setup mocks
        mock_load.return_value = {"output_dir": "models/stage3_test"}
        mock_merge.return_value = {
            "output_dir": "models/stage3_test",
            "train_csv": "data/train.csv",
        }
        mock_exists.return_value = True

        # Run main with dry-run args
        with patch("scripts.train_stage3.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                checkpoint=Path("models/stage2_final"),
                early_stopping_patience=3,
                dry_run=True,
                config=Path("config/training/stage3.yaml"),
                output_dir=Path("models/stage3_test"),
                seed=42,
                resume_from=None,
                train_csv=None,
                val_csv=None,
            )

            ret_code = train_stage3.main()

            assert ret_code == 0
            mock_val_ckpt.assert_called_once()
            mock_val_conf.assert_called_once()
            mock_genre.assert_called_once()  # Should validate genre tags even in dry run
            mock_write.assert_called_once()


class TestEarlyStopping:
    def test_early_stopping_initialization(self):
        """Test EarlyStopping initializes with correct parameters."""
        es = training_utils.EarlyStopping(patience=5, min_delta=0.01)
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.best_score is None
        assert es.counter == 0

    def test_should_stop_early_improvement(self):
        """Test that improvement resets counter and updates best score."""
        es = training_utils.EarlyStopping(patience=3)

        # First epoch: 10.0 (Best)
        stop = es.should_stop_early(10.0)
        assert not stop
        assert es.best_score == 10.0
        assert es.counter == 0
        assert es.is_best_checkpoint(10.0)

        # Second epoch: 12.0 (Improvement)
        stop = es.should_stop_early(12.0)
        assert not stop
        assert es.best_score == 12.0
        assert es.counter == 0
        assert es.is_best_checkpoint(12.0)

    def test_should_stop_early_plateau(self):
        """Test that lack of improvement increments counter."""
        es = training_utils.EarlyStopping(patience=2, min_delta=0.1)

        # Initial best
        es.should_stop_early(10.0)

        # Tiny improvement < min_delta
        stop = es.should_stop_early(10.05)
        assert not stop
        assert es.counter == 1
        assert not es.is_best_checkpoint(10.05)

        # Another tiny improvement
        stop = es.should_stop_early(10.08)
        assert stop  # Patience reached (2)
        assert es.counter == 2


class TestGenreValidation:
    def test_validate_genre_tags_success(self):
        """Test validation passes with correct tags."""
        # Mock dataframe and tokenizer
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__len__.return_value = 2
        # Create a mock iterate that returns rows with valid tags
        mock_df.itertuples.return_value = [
            MagicMock(transliteration="[LETTER] some text"),
            MagicMock(transliteration="[ADMIN] other text"),
        ]

        mock_tokenizer = MagicMock()
        # Mock encode/decode to return same string (identity)
        mock_tokenizer.decode.return_value = "[LETTER] some text"
        mock_tokenizer.encode.return_value = [1, 2, 3]

        assert training_utils.validate_genre_tags(mock_df, mock_tokenizer) is True

    def test_validate_genre_tags_failure(self):
        """Test validation fails with missing tags."""
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__len__.return_value = 1
        mock_df.itertuples.return_value = [MagicMock(transliteration="no tag here")]

        mock_tokenizer = MagicMock()

        assert training_utils.validate_genre_tags(mock_df, mock_tokenizer) is False
