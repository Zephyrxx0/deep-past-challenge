#!/usr/bin/env python3
"""Unified evaluation CLI with BLEU, chrF, and genre-specific metrics.

Implements EVAL-01 and EVAL-02: Reproducible evaluation with genre breakdown.

Usage:
    # Show help:
    python scripts/evaluate.py --help

    # Dry-run (validate inputs without running evaluation):
    python scripts/evaluate.py --checkpoint models/stage3_final --dry-run

    # Full evaluation:
    python scripts/evaluate.py --checkpoint models/stage3_final --output results.json

    # With genre breakdown:
    python scripts/evaluate.py --checkpoint models/stage3_final --genre-breakdown --output results.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation_utils import (
    compute_bleu,
    compute_chrf,
    compute_genre_metrics,
    extract_genre_tag,
    load_validation_data,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation CLI with BLEU, chrF, and genre-specific metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint directory.",
    )

    # Optional arguments
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/stage3_val.csv"),
        help="Path to validation CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output JSON for results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to run evaluation on (auto-detect by default).",
    )
    parser.add_argument(
        "--genre-breakdown",
        action="store_true",
        help="Enable genre-specific metrics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without running evaluation.",
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

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input paths and configuration.

    Args:
        args: Parsed command-line arguments

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If configuration is invalid
    """
    # Validate checkpoint path (skip in dry-run for flexibility)
    if not args.dry_run:
        if not args.checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}. "
                "Ensure the checkpoint path is correct."
            )

        # Check for required model files
        config_path = args.checkpoint / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Checkpoint config not found: {config_path}. "
                "Ensure this is a valid model checkpoint directory."
            )

    # Validate val-data path (skip in dry-run)
    if not args.dry_run and not args.val_data.exists():
        raise FileNotFoundError(
            f"Validation data not found: {args.val_data}. "
            "Ensure the path is correct or run data preprocessing first."
        )


def get_device(device_arg: str) -> str:
    """Determine the device to use for evaluation.

    Args:
        device_arg: User-specified device or 'auto'

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device_arg == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def generate_predictions(
    checkpoint_path: Path,
    sources: List[str],
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    device: str,
) -> List[str]:
    """Generate predictions using the model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        sources: List of source texts
        batch_size: Batch size for inference
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        device: Device to run inference on

    Returns:
        List of predicted translations
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from scripts.training_utils import configure_mbart_tokenizer

    # Load model and tokenizer
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer = configure_mbart_tokenizer(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    # Generate predictions in batches
    predictions = []
    total_batches = (len(sources) + batch_size - 1) // batch_size

    for i in range(0, len(sources), batch_size):
        batch_sources = sources[i : i + batch_size]
        batch_idx = i // batch_size + 1

        # Tokenize inputs
        inputs = tokenizer(
            batch_sources,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        # Decode
        batch_predictions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        predictions.extend(batch_predictions)

        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(f"[INFO] Processed batch {batch_idx}/{total_batches}")

    return predictions


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run full evaluation pipeline.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary with evaluation results
    """
    # Determine device
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Load validation data
    print(f"[INFO] Loading validation data: {args.val_data}")
    val_df = load_validation_data(args.val_data)
    sources = val_df["source"].tolist()
    references = val_df["target"].tolist()
    print(f"[INFO] Loaded {len(sources)} validation samples")

    # Generate predictions
    predictions = generate_predictions(
        checkpoint_path=args.checkpoint,
        sources=sources,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        device=device,
    )

    # Compute overall metrics
    print("[INFO] Computing BLEU and chrF metrics...")
    bleu = compute_bleu(predictions, references)
    chrf = compute_chrf(predictions, references)

    results = {
        "checkpoint": str(args.checkpoint),
        "val_data": str(args.val_data),
        "overall": {
            "bleu": round(bleu, 2),
            "chrf": round(chrf, 2),
            "samples": len(sources),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"[RESULT] Overall BLEU: {bleu:.2f}")
    print(f"[RESULT] Overall chrF: {chrf:.2f}")

    # Compute genre-specific metrics if requested
    if args.genre_breakdown:
        print("[INFO] Computing genre-specific metrics...")

        # Extract genre tags from sources
        if "genre" in val_df.columns:
            genres = val_df["genre"].tolist()
        else:
            # Try to extract from source text
            genres = [extract_genre_tag(src) for src in sources]

        genre_metrics = compute_genre_metrics(predictions, references, genres)
        results["by_genre"] = {
            genre: {
                "bleu": round(metrics["bleu"], 2),
                "chrf": round(metrics["chrf"], 2),
                "count": metrics["count"],
            }
            for genre, metrics in genre_metrics.items()
        }

        # Print genre breakdown
        print()
        print("Genre Breakdown")
        print("=" * 60)
        print(f"{'Genre':<15} {'BLEU':<10} {'chrF':<10} {'Count':<10}")
        print("-" * 60)
        for genre, metrics in sorted(results["by_genre"].items()):
            print(
                f"{genre:<15} {metrics['bleu']:<10.2f} "
                f"{metrics['chrf']:<10.2f} {metrics['count']:<10}"
            )
        print("=" * 60)

    return results


def main() -> int:
    """Main entry point for unified evaluation CLI."""
    args = parse_args()

    try:
        # Validate inputs
        print("[INFO] Validating inputs...")
        validate_inputs(args)
        print("[OK] Input validation passed")

        if args.dry_run:
            print("[DRY-RUN] Validation complete. No evaluation performed.")
            return 0

        # Run evaluation
        results = run_evaluation(args)

        # Save results to JSON if output path specified
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"[OK] Results saved to {args.output}")

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
