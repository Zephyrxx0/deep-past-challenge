import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
STAGE_CONFIGS = {
    1: {"train": "data/stage1_train.csv"},
    2: {"train": "data/stage2_train.csv"},
    3: {"train": "data/stage3_train.csv", "val": "data/stage3_val.csv"},
}


def load_stage_data(stage: int, split: str = "train", seed: int = SEED) -> pd.DataFrame:
    """Load stage dataset with deterministic ordering.

    Args:
        stage: Training stage (1, 2, or 3)
        split: Data split ('train' or 'val')
        seed: Random seed for reproducibility

    Returns:
        DataFrame with transliteration_normalized and translation_normalized columns

    Raises:
        ValueError: If stage or split is invalid
        FileNotFoundError: If dataset file is missing
    """
    if stage not in [1, 2, 3]:
        raise ValueError(f"Invalid stage {stage}. Must be 1, 2, or 3.")

    if split not in STAGE_CONFIGS[stage]:
        raise ValueError(
            f'Invalid split "{split}" for stage {stage}. Available: {list(STAGE_CONFIGS[stage].keys())}'
        )

    csv_path = STAGE_CONFIGS[stage][split]
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
