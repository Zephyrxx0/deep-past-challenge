#!/usr/bin/env python3
"""Batched inference CLI for competition submission.

Implements INF-01: Run batched inference on held-out test data and produce one translation per input.

Usage:
    # Dry-run (validation only, writes empty CSV, no torch required):
    python scripts/run_inference.py --dry-run --output-dir outputs/inference

    # Full inference:
    python scripts/run_inference.py \
        --checkpoint models/stage3_final \
        --test-csv data/stage3_val.csv \
        --output-dir outputs/inference
"""

import argparse
import sys
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Batched inference CLI for Akkadian-to-English translation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write empty CSV without loading model or data (fast validation).",
    )

    # Required paths
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to trained model checkpoint directory (required for real inference).",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        help="Path to test data CSV file (required for real inference).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Directory to write inference_results.csv.",
    )

    # Inference parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=512,
        help="Max tokenization length for source.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=512,
        help="Max generation length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def validate_inputs(checkpoint: Path, test_csv: Path) -> None:
    """Validate checkpoint and test CSV exist.

    Args:
        checkpoint: Path to model checkpoint directory
        test_csv: Path to test data CSV file

    Raises:
        FileNotFoundError: If required files don't exist
    """
    # Check checkpoint exists (look for config.json or model.safetensors)
    if not checkpoint.exists():
        print(f"Error: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Verify checkpoint contains model files
    has_config = (checkpoint / "config.json").exists()
    has_model = (checkpoint / "model.safetensors").exists() or (
        checkpoint / "pytorch_model.bin"
    ).exists()
    if not (has_config or has_model):
        print(
            f"Error: checkpoint directory exists but missing model files: {checkpoint}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check test CSV exists
    if not test_csv.exists():
        print(f"Error: test-csv not found: {test_csv}", file=sys.stderr)
        sys.exit(1)


def run_inference(
    checkpoint: Path,
    test_csv: Path,
    output_dir: Path,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    seed: int,
) -> None:
    """Run batched inference on test data.

    Args:
        checkpoint: Path to model checkpoint directory
        test_csv: Path to test data CSV file
        output_dir: Directory to write results
        batch_size: Inference batch size
        max_source_length: Max tokenization length for source
        max_target_length: Max generation length
        seed: Random seed for reproducibility
    """
    # Lazy imports for inference dependencies (not needed in dry-run)
    import pandas as pd
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from scripts.training_utils import set_seeds

    # Set seeds for reproducibility
    set_seeds(seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model and tokenizer
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint}")
    print(f"Loaded tokenizer from {checkpoint}")

    # Load test data
    print(f"[INFO] Loading test data: {test_csv}")
    df = pd.read_csv(test_csv)

    # Determine source column name
    if "transliteration" in df.columns:
        src_col = "transliteration"
    elif "transliteration_normalized" in df.columns:
        src_col = "transliteration_normalized"
    else:
        print(
            f"Error: test CSV must contain 'transliteration' column. Found: {df.columns.tolist()}",
            file=sys.stderr,
        )
        sys.exit(1)

    sources = df[src_col].tolist()
    print(f"Processing {len(sources)} test examples in batches of {batch_size}...")

    # Generate translations in batches
    translations = []
    total_batches = (len(sources) + batch_size - 1) // batch_size

    for i in range(0, len(sources), batch_size):
        batch_sources = sources[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Tokenize inputs
        inputs = tokenizer(
            batch_sources,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate translations
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        # Decode outputs
        batch_translations = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        translations.extend(batch_translations)

        # Progress indicator
        progress = (batch_num / total_batches) * 100
        print(
            f"Batch {batch_num}/{total_batches} [{'=' * int(progress / 10):<10}] {progress:.0f}%"
        )

    # Write results to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "inference_results.csv"

    with csv_path.open("w", encoding="utf-8") as f:
        # Write header
        f.write("id,akkadian_source,english_translation\n")

        # Write data rows
        for idx, (source, translation) in enumerate(
            zip(sources, translations), start=1
        ):
            # Escape quotes and commas in CSV fields
            source_escaped = source.replace('"', '""')
            translation_escaped = translation.replace('"', '""')
            f.write(f'{idx},"{source_escaped}","{translation_escaped}"\n')

    print(f"Wrote inference results to {csv_path}")
    print(f"Total examples: {len(sources)}")


def main() -> int:
    """Main entry point for batched inference CLI."""
    args = parse_args()

    try:
        # Get output directory
        output_dir = args.output_dir

        if args.dry_run:
            # Dry-run: write empty CSV without loading model/data
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / "inference_results.csv"
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("id,akkadian_source,english_translation\n")
                f.write("1,<dry-run>,<dry-run>\n")
            print(f"Dry-run: wrote {csv_path}")
            return 0

        # Validate required arguments for real inference
        if not args.checkpoint:
            print(
                "Error: --checkpoint is required for inference (or use --dry-run)",
                file=sys.stderr,
            )
            return 1

        if not args.test_csv:
            print(
                "Error: --test-csv is required for inference (or use --dry-run)",
                file=sys.stderr,
            )
            return 1

        # Validate inputs
        validate_inputs(args.checkpoint, args.test_csv)

        # Run inference
        run_inference(
            checkpoint=args.checkpoint,
            test_csv=args.test_csv,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            seed=args.seed,
        )

        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
