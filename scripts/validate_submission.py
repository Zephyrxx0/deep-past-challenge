#!/usr/bin/env python3
"""Submission file validator for competition format compliance.

Implements INF-02: Validate submission schema completeness.

Usage:
    python scripts/validate_submission.py --csv outputs/inference/inference_results.csv
    python scripts/validate_submission.py --csv submission.csv --expected-count 1234 --strict
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate competition submission CSV")
    parser.add_argument("--csv", type=Path, required=True, help="Submission CSV file")
    parser.add_argument("--expected-count", type=int, help="Expected row count")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )
    return parser.parse_args()


def validate_submission(
    csv_path: Path, expected_count: int = None, strict: bool = False
) -> int:
    """Validate submission CSV. Returns 0 on success, 1 on failure."""
    required_columns = ["id", "akkadian_source", "english_translation"]

    # File exists check
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    # Schema check
    df = pd.read_csv(
        csv_path,
        dtype={"id": int, "akkadian_source": str, "english_translation": str},
        keep_default_na=False,
    )
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: missing required column(s): {missing_cols}", file=sys.stderr)
        return 1

    # Completeness check
    # Check if string columns are empty strings
    empty_cells = (df == "") | df.isnull()
    if empty_cells.any().any():
        empty_rows = df[empty_cells.any(axis=1)]
        for idx in empty_rows.index:
            print(
                f"Error: row {idx + 1} has empty translation or missing value",
                file=sys.stderr,
            )
        return 1

    # Duplicate ID check
    duplicates = df[df.duplicated(subset=["id"], keep=False)]
    if not duplicates.empty:
        dup_ids = duplicates["id"].unique().tolist()
        print(
            f"Error: found {len(duplicates)} duplicate ID(s): {dup_ids}",
            file=sys.stderr,
        )
        return 1

    # Sequential ID check (warning)
    actual_ids = sorted(df["id"].unique())
    expected_ids = list(range(1, len(df) + 1))
    if actual_ids != expected_ids:
        missing_ids = set(expected_ids) - set(actual_ids)
        print(
            f"Warning: IDs are not sequential. Missing: {sorted(missing_ids)}",
            file=sys.stdout,
        )
        if strict:
            return 1

    # Row count check
    if expected_count and len(df) != expected_count:
        print(
            f"Error: expected {expected_count} rows, found {len(df)}", file=sys.stderr
        )
        return 1

    # Success
    print(f"Validation PASSED")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  ID range: {df['id'].min()}-{df['id'].max()}")
    print(f"  No missing values")
    print(f"  No duplicates")
    return 0


def main():
    args = parse_args()
    exit_code = validate_submission(args.csv, args.expected_count, args.strict)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
