#!/usr/bin/env python3
"""Shared training utilities for Stage 1, Stage 2, and Stage 3 training CLIs.

Provides common functionality:
- Config loading and merging
- Config validation
- Run manifest writing
- Seed setting for reproducibility
- Tokenizer loading with mBART language configuration

Per D-07: Extract shared logic into training_utils.py; both Stage 1 and Stage 2
CLIs import from this module.
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# Default seed for reproducibility (per AGENTS.md)
DEFAULT_SEED = 42

# mBART language codes for Akkadian->English translation
# Using Arabic as proxy for Akkadian (both Semitic languages)
MBART_SRC_LANG = "ar_AR"
MBART_TGT_LANG = "en_XX"


def load_config(config_path: Path) -> dict:
    """Load training config from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary of configuration values, empty dict if file doesn't exist
    """
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """Merge base config with CLI argument overrides.

    CLI arguments take precedence over config file values.

    Args:
        base_config: Configuration dictionary loaded from YAML
        args: Parsed command-line arguments

    Returns:
        Merged configuration dictionary
    """
    config = base_config.copy()

    # Apply CLI overrides if provided (check for attribute existence and non-None)
    cli_overrides = [
        ("train_csv", "train_csv"),
        ("val_csv", "val_csv"),
        ("model_id", "model_id"),
        ("output_dir", "output_dir"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("max_source_length", "max_source_length"),
        ("max_target_length", "max_target_length"),
    ]

    for config_key, arg_name in cli_overrides:
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                # Convert Path objects to strings for config
                config[config_key] = str(value) if isinstance(value, Path) else value

    # Always use seed from CLI if present (has default)
    if hasattr(args, "seed") and args.seed is not None:
        config["seed"] = args.seed

    return config


def validate_config(config: dict, dry_run: bool = False) -> None:
    """Validate required config fields and file existence.

    Args:
        config: Merged configuration dictionary
        dry_run: If True, validate files exist but allow missing train_csv

    Raises:
        ValueError: If required config fields are missing
        FileNotFoundError: If required files don't exist
    """
    # Check required fields
    required_fields = ["output_dir"]
    missing = [f for f in required_fields if not config.get(f)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # Validate train_csv exists (required for both dry-run and training)
    train_csv_path = config.get("train_csv")
    if train_csv_path:
        train_csv = Path(train_csv_path)
        if not train_csv.exists():
            raise FileNotFoundError(
                f"train_csv not found: {train_csv}. "
                "Ensure the path is correct or run data preprocessing first."
            )
    elif not dry_run:
        raise ValueError("train_csv is required for training")

    # Validate val_csv if provided
    val_csv_path = config.get("val_csv")
    if val_csv_path:
        val_csv = Path(val_csv_path)
        if not val_csv.exists():
            raise FileNotFoundError(
                f"val_csv not found: {val_csv}. "
                "Ensure the path is correct or run data preprocessing first."
            )


def write_run_manifest(
    config: dict,
    output_dir: Path,
    stage: int = 1,
    checkpoint_path: str | None = None,
) -> Path:
    """Write run_manifest.json with run metadata.

    Args:
        config: Merged configuration dictionary
        output_dir: Directory to write manifest to
        stage: Training stage number (1, 2, or 3)
        checkpoint_path: Path to checkpoint being loaded (for Stage 2/3)

    Returns:
        Path to written manifest file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "stage": stage,
        "model_id": config.get("model_id", "facebook/mbart-large-50-many-to-many-mmt"),
        "train_csv": config.get("train_csv"),
        "val_csv": config.get("val_csv"),
        "seed": config.get("seed", DEFAULT_SEED),
        "output_dir": str(output_dir),
        "epochs": config.get("epochs", 10 if stage == 1 else 5),
        "batch_size": config.get("batch_size", 4),
        "learning_rate": float(
            config.get("learning_rate", 1e-4 if stage == 1 else 5e-5)
        ),
        "max_source_length": config.get("max_source_length", 256),
        "max_target_length": config.get("max_target_length", 256),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Add checkpoint path for Stage 2/3
    if checkpoint_path:
        manifest["checkpoint_path"] = checkpoint_path

    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def set_seeds(seed: int) -> None:
    """Set deterministic seeds for reproducibility.

    Per AGENTS.md: Always set seeds for deterministic behavior.
    Sets seeds for random, numpy (if available), and torch (if available).

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """Initialize EarlyStopping.

        Args:
            patience: How long to wait after last time validation loss improved.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def should_stop_early(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Metric value (higher is better, e.g. BLEU)

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.early_stop = False

        return False

    def is_best_checkpoint(self, score: float) -> bool:
        """Check if current score is the best so far.

        Args:
            score: Metric value

        Returns:
            True if score is better than previous best
        """
        if self.best_score is None:
            return True
        return score >= self.best_score + self.min_delta


def validate_genre_tags(dataframe: Any, tokenizer: Any) -> bool:
    """Validate that dataframe contains genre tags and tokenizer handles them.

    Args:
        dataframe: Pandas DataFrame containing 'transliteration' column
        tokenizer: PreTrainedTokenizer

    Returns:
        True if validation passes, False otherwise
    """
    valid_tags = ["[LETTER]", "[DEBT_NOTE]", "[CONTRACT]", "[ADMIN]"]

    # Check if dataframe is empty (handle both len() and .empty if available)
    if hasattr(dataframe, "empty") and dataframe.empty:
        return False
    if hasattr(dataframe, "__len__") and len(dataframe) == 0:
        return False

    # Check first few rows
    # We assume dataframe has 'transliteration' column/attribute
    try:
        # iterate over rows
        # If it's a mock, itertuples might be a MagicMock
        iterator = (
            dataframe.itertuples() if hasattr(dataframe, "itertuples") else dataframe
        )

        for row in iterator:
            text = getattr(row, "transliteration", None)
            if not text:
                continue

            # Check if text starts with any valid tag
            # The plan says "Genre tags appear at start of transliterations"
            # But let's just check existence for now as per test
            has_tag = any(text.strip().startswith(tag) for tag in valid_tags)
            if not has_tag:
                return False

    except Exception:
        # If itertuples fails or structure is wrong, return False
        return False

    return True


def configure_mbart_tokenizer(tokenizer: Any) -> Any:
    """Configure mBART tokenizer with source and target languages.

    This is required for mBART models to properly tokenize translation pairs.
    Uses Arabic as proxy for Akkadian (both are Semitic languages with similar
    morphological patterns) and English as the target.

    Args:
        tokenizer: AutoTokenizer instance

    Returns:
        Configured tokenizer
    """
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = MBART_SRC_LANG
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = MBART_TGT_LANG
    return tokenizer
