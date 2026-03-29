#!/usr/bin/env python3
"""Stage 3 Evaluation CLI with best checkpoint selection.

Implements TRN3-02: User can select and export best Stage 3 checkpoint based on validation BLEU.

This script:
1. Evaluates all epoch checkpoints from Stage 3 training
2. Ranks them by validation BLEU (or chrF)
3. Exports the best checkpoint to final model directory

Usage:
    # Dry-run (evaluate and rank without exporting):
    python scripts/evaluate_stage3.py --checkpoints-dir models/stage3_test --dry-run

    # Full evaluation with export:
    python scripts/evaluate_stage3.py --checkpoints-dir models/stage3_test --export-dir models/stage3_final

    # Use chrF for selection instead of BLEU:
    python scripts/evaluate_stage3.py --checkpoints-dir models/stage3_test --metric chrf
"""

import argparse
import sys
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.checkpoint_selection import (
    find_best_checkpoint,
    compare_checkpoints,
    export_best_checkpoint,
    validate_checkpoint_completeness,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 3 evaluation CLI with best checkpoint selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        required=True,
        help="Directory containing epoch_* checkpoints from Stage 3 training.",
    )

    # Optional arguments
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/stage3_val.csv"),
        help="Validation CSV for evaluation.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("models/stage3_final"),
        help="Directory to export best checkpoint to.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and rank checkpoints without exporting.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="bleu",
        choices=["bleu", "chrf"],
        help="Metric to use for checkpoint selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=256,
        help="Maximum source sequence length.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=256,
        help="Maximum target sequence length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on.",
    )

    return parser.parse_args()


def validate_inputs(checkpoints_dir: Path, val_data: Path) -> None:
    """Validate input paths exist.

    Args:
        checkpoints_dir: Directory containing checkpoints
        val_data: Validation CSV file

    Raises:
        FileNotFoundError: If required paths don't exist
        ValueError: If checkpoints directory is invalid
    """
    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_dir}. "
            "Ensure you have run Stage 3 training first."
        )

    if not checkpoints_dir.is_dir():
        raise ValueError(f"Checkpoints path must be a directory: {checkpoints_dir}.")

    # Check for epoch subdirectories
    epoch_dirs = list(checkpoints_dir.glob("epoch_*"))
    if not epoch_dirs:
        raise ValueError(
            f"No epoch_* subdirectories found in {checkpoints_dir}. "
            "Ensure this is a valid Stage 3 training output directory."
        )

    if not val_data.exists():
        raise FileNotFoundError(
            f"Validation data not found: {val_data}. "
            "Ensure the path is correct or run data preprocessing first."
        )


def print_ranking_table(results: list) -> None:
    """Print formatted ranking table of checkpoint results.

    Args:
        results: List of checkpoint evaluation results
    """
    print()
    print("Checkpoint Ranking")
    print("=" * 70)
    print(f"{'Rank':<6} {'Checkpoint':<25} {'BLEU':<10} {'chrF':<10} {'Epoch':<8}")
    print("-" * 70)

    for i, result in enumerate(results, start=1):
        checkpoint_name = result["path"].name
        bleu = result["bleu"]
        chrf = result["chrf"]
        epoch = result["epoch"]

        print(f"{i:<6} {checkpoint_name:<25} {bleu:<10.2f} {chrf:<10.2f} {epoch:<8}")

    print("=" * 70)
    print()


def main() -> int:
    """Main entry point for Stage 3 evaluation CLI."""
    args = parse_args()

    try:
        # Validate inputs
        print(f"[INFO] Validating inputs...")
        validate_inputs(args.checkpoints_dir, args.val_data)
        print("[OK] Input validation passed")

        # Find all epoch checkpoints
        epoch_dirs = sorted(args.checkpoints_dir.glob("epoch_*"))
        print(f"[INFO] Found {len(epoch_dirs)} epoch checkpoints")

        # Validate checkpoint completeness
        print("[INFO] Validating checkpoint completeness...")
        for epoch_dir in epoch_dirs:
            if not validate_checkpoint_completeness(epoch_dir):
                print(
                    f"[WARNING] Checkpoint {epoch_dir.name} is incomplete, skipping..."
                )
                epoch_dirs.remove(epoch_dir)

        if not epoch_dirs:
            raise ValueError(
                "No valid checkpoints found. All checkpoints are incomplete."
            )

        print(f"[OK] {len(epoch_dirs)} valid checkpoints")

        # Evaluate and compare all checkpoints
        print(f"[INFO] Evaluating checkpoints on {args.val_data}...")
        print(f"[INFO] Using device: {args.device}")
        print(f"[INFO] This may take a few minutes...")

        results = compare_checkpoints(
            checkpoint_paths=epoch_dirs,
            val_csv=args.val_data,
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            device=args.device,
        )

        # Print ranking table
        print_ranking_table(results)

        # Determine best checkpoint based on metric
        if args.metric == "bleu":
            best_result = results[0]  # Already sorted by BLEU
        else:  # chrf
            best_result = max(results, key=lambda x: x["chrf"])

        best_path = best_result["path"]

        print(f"[RESULT] Best checkpoint: {best_path.name}")
        print(f"[RESULT] BLEU: {best_result['bleu']:.2f}")
        print(f"[RESULT] chrF: {best_result['chrf']:.2f}")
        print(f"[RESULT] Epoch: {best_result['epoch']}")
        print()

        if args.dry_run:
            print("[DRY-RUN] Evaluation complete. No export performed.")
            print(f"[INFO] To export best checkpoint, run without --dry-run flag")
            return 0

        # Export best checkpoint
        print(f"[INFO] Exporting best checkpoint to {args.export_dir}...")
        export_best_checkpoint(
            best_path=best_path,
            export_dir=args.export_dir,
            metrics=best_result,
        )

        print(f"[OK] Best checkpoint exported to {args.export_dir}")
        print(
            f"[INFO] Manifest written: {args.export_dir}/best_checkpoint_manifest.json"
        )
        print(f"[INFO] Evaluation results: {args.export_dir}/evaluation_results.json")

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
