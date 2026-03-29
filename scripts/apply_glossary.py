#!/usr/bin/env python3
"""Glossary application CLI for post-processing translations.

Applies glossary corrections to translation outputs and reports
proper-name accuracy before and after correction.

Implements GLS-01 and GLS-02 requirements.

Usage:
    # Show help:
    python scripts/apply_glossary.py --help

    # Apply glossary to predictions:
    python scripts/apply_glossary.py --input predictions.csv --output corrected.csv

    # With accuracy comparison:
    python scripts/apply_glossary.py --input predictions.csv --output corrected.csv \
        --references data/stage3_val.csv --report accuracy_report.json

    # Dry-run to preview corrections:
    python scripts/apply_glossary.py --input predictions.csv --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.glossary_utils import (
    apply_glossary,
    compute_name_accuracy,
    count_corrections,
    load_glossary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply glossary corrections to translations and report accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input predictions CSV (id, translation columns).",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output CSV with glossary-corrected translations.",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("data/glossary.json"),
        help="Path to glossary JSON file.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        help="Path to reference translations CSV for accuracy comparison.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to output accuracy report JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show sample corrections without writing output.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample corrections to show in dry-run mode.",
    )

    return parser.parse_args()


def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Load predictions CSV with id and translation columns.

    Args:
        csv_path: Path to predictions CSV

    Returns:
        DataFrame with id and translation columns

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check for id column
    if "id" not in df.columns:
        # Try to use index as id
        df["id"] = df.index

    # Check for translation column (support multiple names)
    trans_col = None
    for col in ["translation", "prediction", "hypothesis"]:
        if col in df.columns:
            trans_col = col
            break

    if trans_col is None:
        raise ValueError(
            f"CSV must have 'translation', 'prediction', or 'hypothesis' column. "
            f"Found: {list(df.columns)}"
        )

    # Normalize to 'translation'
    if trans_col != "translation":
        df = df.rename(columns={trans_col: "translation"})

    return df


def load_references(csv_path: Path) -> List[str]:
    """Load reference translations from CSV.

    Args:
        csv_path: Path to references CSV

    Returns:
        List of reference translation strings
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"References file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Find translation column
    trans_col = None
    for col in ["translation", "target", "translation_normalized"]:
        if col in df.columns:
            trans_col = col
            break

    if trans_col is None:
        raise ValueError(
            f"References CSV must have 'translation' or 'target' column. "
            f"Found: {list(df.columns)}"
        )

    return df[trans_col].fillna("").tolist()


def apply_corrections(
    predictions: List[str],
    glossary: Dict[str, str],
) -> Tuple[List[str], int]:
    """Apply glossary corrections to all predictions.

    Args:
        predictions: List of prediction strings
        glossary: Glossary dictionary

    Returns:
        Tuple of (corrected predictions, total corrections count)
    """
    corrected = []
    total_corrections = 0

    for pred in predictions:
        corrected_text = apply_glossary(str(pred) if pd.notna(pred) else "", glossary)
        corrected.append(corrected_text)
        total_corrections += count_corrections(
            str(pred) if pd.notna(pred) else "", corrected_text, glossary
        )

    return corrected, total_corrections


def show_sample_corrections(
    original: List[str],
    corrected: List[str],
    glossary: Dict[str, str],
    num_samples: int = 5,
) -> None:
    """Display sample corrections for dry-run mode.

    Args:
        original: Original texts
        corrected: Corrected texts
        glossary: Glossary used
        num_samples: Number of samples to show
    """
    print("\nSample Corrections")
    print("=" * 60)

    samples_shown = 0
    for i, (orig, corr) in enumerate(zip(original, corrected)):
        if orig != corr:
            print(f"\n[Sample {samples_shown + 1}]")
            print(f"  Original:  {orig[:100]}{'...' if len(str(orig)) > 100 else ''}")
            print(f"  Corrected: {corr[:100]}{'...' if len(str(corr)) > 100 else ''}")
            samples_shown += 1

            if samples_shown >= num_samples:
                break

    if samples_shown == 0:
        print("\nNo corrections were made.")
    else:
        print(f"\n[Showing {samples_shown} of {num_samples} requested samples]")

    print("=" * 60)


def run_glossary_application(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the glossary application pipeline.

    Args:
        args: Parsed command-line arguments

    Returns:
        Results dictionary
    """
    # Load glossary
    print(f"[INFO] Loading glossary: {args.glossary}")
    glossary = load_glossary(args.glossary)
    print(f"[INFO] Loaded {len(glossary)} glossary entries")

    # Load predictions
    print(f"[INFO] Loading predictions: {args.input}")
    pred_df = load_predictions(args.input)
    predictions = pred_df["translation"].fillna("").tolist()
    print(f"[INFO] Loaded {len(predictions)} predictions")

    # Apply corrections
    print("[INFO] Applying glossary corrections...")
    corrected, total_corrections = apply_corrections(predictions, glossary)
    print(f"[INFO] Made {total_corrections} corrections")

    results: Dict[str, Any] = {
        "input": str(args.input),
        "glossary": str(args.glossary),
        "corrections_made": total_corrections,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # If references provided, compute accuracy
    if args.references:
        print(f"[INFO] Loading references: {args.references}")
        references = load_references(args.references)

        if len(references) != len(predictions):
            print(
                f"[WARNING] Reference count ({len(references)}) != "
                f"prediction count ({len(predictions)})"
            )
            # Truncate to shorter length
            min_len = min(len(references), len(predictions))
            references = references[:min_len]
            predictions = predictions[:min_len]
            corrected = corrected[:min_len]

        print("[INFO] Computing proper-name accuracy...")
        accuracy_before = compute_name_accuracy(predictions, references, glossary)
        accuracy_after = compute_name_accuracy(corrected, references, glossary)
        improvement = accuracy_after - accuracy_before

        results["name_accuracy_before"] = round(accuracy_before, 2)
        results["name_accuracy_after"] = round(accuracy_after, 2)
        results["improvement"] = round(improvement, 2)

        print(f"[RESULT] Name accuracy before: {accuracy_before:.2f}%")
        print(f"[RESULT] Name accuracy after:  {accuracy_after:.2f}%")
        print(f"[RESULT] Improvement:          {improvement:+.2f}%")

    # Handle dry-run or output
    if args.dry_run:
        show_sample_corrections(predictions, corrected, glossary, args.samples)
        print("\n[DRY-RUN] No files written.")
    else:
        if args.output:
            # Save corrected predictions
            output_df = pred_df.copy()
            output_df["translation"] = corrected
            output_df.to_csv(args.output, index=False)
            print(f"[OK] Corrected predictions saved to {args.output}")

        if args.report:
            # Save accuracy report
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"[OK] Accuracy report saved to {args.report}")

    return results


def main() -> int:
    """Main entry point for glossary application CLI."""
    args = parse_args()

    try:
        run_glossary_application(args)
        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
