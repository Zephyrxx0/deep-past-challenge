import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================


def is_kaggle_environment() -> bool:
    """Detect if running in Kaggle environment."""
    return os.path.exists("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def is_colab_environment() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab

        return True
    except ImportError:
        return False


RUNNING_ON_KAGGLE = is_kaggle_environment()
RUNNING_ON_COLAB = is_colab_environment()


def get_data_dir() -> Path:
    """Get the data directory based on environment."""
    if RUNNING_ON_KAGGLE:
        # Kaggle: Try multiple possible input paths
        kaggle_input = Path("/kaggle/input")
        possible_paths = [
            kaggle_input / "deep-past-akkadian" / "data",
            kaggle_input / "deep-past" / "data",
            kaggle_input / "deep-past-challenge" / "data",
            kaggle_input / "akkadian-translation",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Fallback: check if data files exist directly in any dataset
        for dataset_dir in kaggle_input.iterdir():
            if dataset_dir.is_dir():
                if (dataset_dir / "stage1_train.csv").exists():
                    return dataset_dir
                if (dataset_dir / "data" / "stage1_train.csv").exists():
                    return dataset_dir / "data"

        # Default fallback
        return kaggle_input / "deep-past-akkadian" / "data"

    elif RUNNING_ON_COLAB:
        # Colab: Assume repo cloned to /content
        return Path("/content/deep-past-challenge/data")

    else:
        # Local development
        return Path("data")


DATA_DIR = get_data_dir()


def get_stage_configs() -> dict:
    """Get stage configurations with correct paths for current environment."""
    return {
        1: {"train": DATA_DIR / "stage1_train.csv"},
        2: {"train": DATA_DIR / "stage2_train.csv"},
        3: {"train": DATA_DIR / "stage3_train.csv", "val": DATA_DIR / "stage3_val.csv"},
    }


# Legacy compatibility - these will be dynamically resolved
STAGE_CONFIGS = {
    1: {"train": "data/stage1_train.csv"},
    2: {"train": "data/stage2_train.csv"},
    3: {"train": "data/stage3_train.csv", "val": "data/stage3_val.csv"},
}


def load_stage_data(
    stage: int, split: str = "train", seed: int = SEED, data_dir: Path = None
) -> pd.DataFrame:
    """Load stage dataset with deterministic ordering.

    Args:
        stage: Training stage (1, 2, or 3)
        split: Data split ('train' or 'val')
        seed: Random seed for reproducibility
        data_dir: Optional override for data directory (useful for Kaggle)

    Returns:
        DataFrame with transliteration_normalized and translation_normalized columns

    Raises:
        ValueError: If stage or split is invalid
        FileNotFoundError: If dataset file is missing
    """
    if stage not in [1, 2, 3]:
        raise ValueError(f"Invalid stage {stage}. Must be 1, 2, or 3.")

    # Get stage configs with correct paths
    stage_configs = (
        get_stage_configs()
        if data_dir is None
        else {
            1: {"train": data_dir / "stage1_train.csv"},
            2: {"train": data_dir / "stage2_train.csv"},
            3: {
                "train": data_dir / "stage3_train.csv",
                "val": data_dir / "stage3_val.csv",
            },
        }
    )

    if split not in stage_configs[stage]:
        raise ValueError(
            f'Invalid split "{split}" for stage {stage}. Available: {list(stage_configs[stage].keys())}'
        )

    csv_path = stage_configs[stage][split]
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run data preprocessing first."
        )

    # Set random seeds for deterministic behavior
    np.random.seed(seed)
    pd.options.mode.chained_assignment = None

    # Load CSV
    df = pd.read_csv(csv_path)

    # Handle different column structures across stages
    # Stage 1, 3: Have 'transliteration' and 'translation' columns
    # Stage 2: Uses Sentences_Oare structure (text_uuid, translation, first_word_spelling)

    if "transliteration" in df.columns and "translation" in df.columns:
        # Standard structure (stages 1 and 3)
        df = df.rename(
            columns={
                "transliteration": "transliteration_normalized",
                "translation": "translation_normalized",
            }
        )
    elif "first_word_spelling" in df.columns and "translation" in df.columns:
        # Stage 2 structure (Sentences_Oare): use first_word_spelling as transliteration
        # This is a simplified approach - full transliteration would require re-running preprocessing
        df = df.rename(
            columns={
                "first_word_spelling": "transliteration_normalized",
                "translation": "translation_normalized",
            }
        )
    else:
        raise ValueError(
            f"Dataset missing required columns. Expected either (transliteration, translation) or (first_word_spelling, translation)"
        )

    # Drop rows with null values in required columns (prevents tokenizer errors)
    initial_count = len(df)
    df = df.dropna(subset=["transliteration_normalized", "translation_normalized"])
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"[INFO] Dropped {dropped_count} rows with null values")

    # Reset index to ensure deterministic ordering
    df = df.reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Load stage datasets with deterministic seed"
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Training stage (1, 2, or 3)",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Data split (train or val)"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    try:
        df = load_stage_data(args.stage, args.split, args.seed)

        output = {
            "row_count": len(df),
            "columns": list(df.columns),
            "stage": args.stage,
            "split": args.split,
        }

        print(json.dumps(output))
        sys.exit(0)

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
