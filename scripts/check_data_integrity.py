import argparse
import sys
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
REQUIRED_FILES = {
    "stage1_train": DATA_DIR / "stage1_train.csv",
    "stage2_train": DATA_DIR / "stage2_train.csv",
    "stage3_train": DATA_DIR / "stage3_train.csv",
    "stage3_val": DATA_DIR / "stage3_val.csv",
}
REQUIRED_COLUMNS = ["transliteration", "translation"]
STAGE2_COLUMNS = [
    "first_word_spelling",
    "translation",
]  # Stage 2 uses Sentences_Oare structure
COMPETITION_TEST = Path("dataset/competition/test.csv")  # Original competition test set


def check_provenance(data_dir: Path = DATA_DIR) -> bool:
    """Validate all required stage CSVs exist with required columns.

    Args:
        data_dir: Directory containing stage CSV files

    Returns:
        True if all files exist with correct columns

    Raises:
        FileNotFoundError: If any required CSV is missing
        ValueError: If required columns are missing from a CSV
    """
    for stage_name, filepath in REQUIRED_FILES.items():
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}. Run data preprocessing first."
            )

        # Read only header + few rows for efficiency
        df = pd.read_csv(filepath, nrows=5)

        # Stage 2 has different structure (Sentences_Oare)
        if stage_name == "stage2_train":
            expected_cols = STAGE2_COLUMNS
        else:
            expected_cols = REQUIRED_COLUMNS

        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Dataset {filepath} missing required columns: {missing_cols}"
            )

    return True


def check_split_integrity(data_dir: Path = DATA_DIR) -> bool:
    """Validate no overlap between validation and training splits.

    Checks:
    - Rule 1: stage3_val must not overlap with stage1_train or stage2_train
    - Rule 2: stage3_train must not overlap with competition test set (if exists)

    Args:
        data_dir: Directory containing stage CSV files

    Returns:
        True if no overlaps detected

    Raises:
        ValueError: If overlap detected between splits
    """
    # Load datasets
    stage1_df = pd.read_csv(REQUIRED_FILES["stage1_train"])
    stage2_df = pd.read_csv(REQUIRED_FILES["stage2_train"])
    stage3_train_df = pd.read_csv(REQUIRED_FILES["stage3_train"])
    stage3_val_df = pd.read_csv(REQUIRED_FILES["stage3_val"])

    # Rule 1: Check stage3_val doesn't overlap with stage1/stage2 training data
    # Use transliteration as key (stage 1/3 have 'transliteration', stage 2 has 'first_word_spelling')

    stage1_translit = set(stage1_df["transliteration"].dropna())
    stage2_translit = set(
        stage2_df["first_word_spelling"].dropna()
    )  # Stage 2 uses different column
    stage3_val_translit = set(stage3_val_df["transliteration"].dropna())

    overlap_stage1 = stage3_val_translit.intersection(stage1_translit)
    overlap_stage2 = stage3_val_translit.intersection(stage2_translit)

    if overlap_stage1:
        raise ValueError(
            f"Split integrity violation: {len(overlap_stage1)} validation examples found in stage1_train. "
            f"Example: {list(overlap_stage1)[:3]}"
        )

    if overlap_stage2:
        raise ValueError(
            f"Split integrity violation: {len(overlap_stage2)} validation examples found in stage2_train. "
            f"Example: {list(overlap_stage2)[:3]}"
        )

    # Rule 2: Check stage3_train doesn't overlap with competition test set (if it exists)
    if COMPETITION_TEST.exists():
        test_df = pd.read_csv(COMPETITION_TEST)
        if "transliteration" in test_df.columns:
            test_translit = set(test_df["transliteration"].dropna())
            stage3_train_translit = set(stage3_train_df["transliteration"].dropna())
            overlap_test = stage3_train_translit.intersection(test_translit)

            if overlap_test:
                raise ValueError(
                    f"Test set contamination: {len(overlap_test)} test examples found in stage3_train. "
                    f"Example: {list(overlap_test)[:3]}"
                )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate data integrity and split boundaries"
    )
    parser.add_argument(
        "--check",
        choices=["all", "provenance", "splits"],
        default="all",
        help="Type of check to run (default: all)",
    )

    args = parser.parse_args()

    try:
        if args.check in ["all", "provenance"]:
            check_provenance()
            print(
                "[OK] Provenance check passed: all required datasets exist with correct columns"
            )

        if args.check in ["all", "splits"]:
            check_split_integrity()
            print(
                "[OK] Split integrity check passed: no train/validation overlap detected"
            )

        print(f"\n[OK] Data integrity validation complete: {args.check}")
        sys.exit(0)

    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Data integrity check failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
