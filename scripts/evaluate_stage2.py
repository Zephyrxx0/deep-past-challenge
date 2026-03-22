#!/usr/bin/env python3
"""Stage 2 Evaluation CLI with comparison reporting.

Implements TRN2-02: Stage 2 evaluation that compares Stage 2 vs Stage 1
performance and detects forgetting.

Per D-08: Separate CLI for Stage 2 evaluation
Per D-09: Writes stage2_comparison.json artifact

Usage:
    # Dry-run (validation only, writes zero metrics, no torch required):
    python scripts/evaluate_stage2.py --dry-run --checkpoint models/stage2_test --stage1-checkpoint models/stage1_test --output-dir outputs/stage2_eval

    # Full evaluation:
    python scripts/evaluate_stage2.py --checkpoint models/stage2_final/epoch_5 --stage1-checkpoint models/stage1_final/epoch_10 --output-dir outputs/stage2
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import shared training utilities (per D-07)
from scripts.training_utils import (
    load_config,
    merge_config,
    validate_config,
    set_seeds,
)

# Import forgetting detection utilities
from scripts.forgetting_detection import (
    compute_forgetting_baseline,
    detect_forgetting,
    load_baseline,
    format_forgetting_report,
)


# Default config path
DEFAULT_CONFIG_PATH = Path("config/training/stage2.yaml")
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with config overrides."""
    parser = argparse.ArgumentParser(
        description="Stage 2 evaluation CLI with comparison reporting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write zero metrics without loading model or data (fast validation).",
    )

    # Required checkpoint arguments
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Stage 2 checkpoint directory (required).",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        required=True,
        help="Path to Stage 1 checkpoint directory for comparison (required).",
    )

    # Config and paths
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to stage2 training config YAML.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        help="Override validation CSV path (default: data/stage3_val.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write stage2_comparison.json.",
    )

    # Evaluation parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        help="Override max source sequence length.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        help="Override max target sequence length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def validate_checkpoints(
    stage2_checkpoint: Path,
    stage1_checkpoint: Path,
    dry_run: bool = False,
) -> None:
    """Validate both checkpoints exist and have required files.

    Args:
        stage2_checkpoint: Path to Stage 2 checkpoint directory
        stage1_checkpoint: Path to Stage 1 checkpoint directory
        dry_run: If True, only check paths exist (skip model file check)

    Raises:
        FileNotFoundError: If checkpoint directory or required files don't exist
        ValueError: If checkpoint is invalid
    """
    # Validate Stage 2 checkpoint
    if not dry_run:
        if not stage2_checkpoint.exists():
            raise FileNotFoundError(
                f"Stage 2 checkpoint directory not found: {stage2_checkpoint}. "
                "Ensure the path points to a valid Stage 2 checkpoint directory."
            )
        if not stage2_checkpoint.is_dir():
            raise ValueError(
                f"Stage 2 checkpoint path must be a directory: {stage2_checkpoint}."
            )
        # Check for model file
        model_safetensors = stage2_checkpoint / "model.safetensors"
        pytorch_model = stage2_checkpoint / "pytorch_model.bin"
        if not model_safetensors.exists() and not pytorch_model.exists():
            raise FileNotFoundError(
                f"No model file found in Stage 2 checkpoint: {stage2_checkpoint}. "
                "Expected model.safetensors or pytorch_model.bin."
            )

    # Validate Stage 1 checkpoint
    if not dry_run:
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint directory not found: {stage1_checkpoint}. "
                "Ensure the path points to a valid Stage 1 checkpoint directory."
            )
        if not stage1_checkpoint.is_dir():
            raise ValueError(
                f"Stage 1 checkpoint path must be a directory: {stage1_checkpoint}."
            )
        # Check for model file
        model_safetensors = stage1_checkpoint / "model.safetensors"
        pytorch_model = stage1_checkpoint / "pytorch_model.bin"
        if not model_safetensors.exists() and not pytorch_model.exists():
            raise FileNotFoundError(
                f"No model file found in Stage 1 checkpoint: {stage1_checkpoint}. "
                "Expected model.safetensors or pytorch_model.bin."
            )


def run_evaluation(
    config: dict,
    stage2_checkpoint: Path,
    stage1_checkpoint: Path,
    output_dir: Path,
) -> dict:
    """Run Stage 2 evaluation with comparison to Stage 1.

    Args:
        config: Merged evaluation configuration
        stage2_checkpoint: Path to Stage 2 checkpoint
        stage1_checkpoint: Path to Stage 1 checkpoint
        output_dir: Directory to write results

    Returns:
        Comparison metrics dictionary
    """
    # Lazy imports for evaluation dependencies
    import pandas as pd
    import sacrebleu
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Extract config values
    val_csv_path = Path(config.get("val_csv", "data/stage3_val.csv"))
    batch_size = config.get("batch_size", 4)
    max_source_length = config.get("max_source_length", 256)
    max_target_length = config.get("max_target_length", 256)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Validate val_csv
    if not val_csv_path.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {val_csv_path}. "
            "Ensure the path is correct or run data preprocessing first."
        )

    # Load validation data
    print(f"[INFO] Loading validation data: {val_csv_path}")
    df = pd.read_csv(val_csv_path)

    # Determine column names
    src_col = (
        "transliteration_normalized"
        if "transliteration_normalized" in df.columns
        else "transliteration"
    )
    tgt_col = (
        "translation_normalized"
        if "translation_normalized" in df.columns
        else "translation"
    )

    sources = df[src_col].tolist()
    references = df[tgt_col].tolist()
    print(f"[INFO] Loaded {len(sources)} validation samples")

    # Helper function to evaluate a checkpoint
    def evaluate_checkpoint(checkpoint_path: Path, label: str) -> tuple[float, float]:
        print(f"[INFO] Loading {label} checkpoint: {checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        model.to(device)
        model.eval()

        hypotheses = []
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i : i + batch_size]

            inputs = tokenizer(
                batch_sources,
                max_length=max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_target_length,
                    num_beams=4,
                    early_stopping=True,
                )

            batch_hypotheses = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            hypotheses.extend(batch_hypotheses)

        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

        print(f"[RESULT] {label}: BLEU={bleu:.2f}, chrF={chrf:.2f}")
        return bleu, chrf

    # Evaluate both checkpoints
    stage1_bleu, stage1_chrf = evaluate_checkpoint(stage1_checkpoint, "Stage 1")
    stage2_bleu, stage2_chrf = evaluate_checkpoint(stage2_checkpoint, "Stage 2")

    # Compute deltas
    delta_bleu = stage2_bleu - stage1_bleu
    delta_chrf = stage2_chrf - stage1_chrf

    # Compute forgetting metrics
    print("[INFO] Computing forgetting metrics...")
    baseline = {
        "bleu": stage1_bleu,
        "chrf": stage1_chrf,
        "stage1_checkpoint": str(stage1_checkpoint),
    }
    forgetting = detect_forgetting(
        stage2_checkpoint=stage2_checkpoint,
        baseline=baseline,
        val_csv=val_csv_path,
        batch_size=batch_size,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    # Build comparison result
    comparison = {
        "stage1_bleu": round(stage1_bleu, 2),
        "stage1_chrf": round(stage1_chrf, 2),
        "stage2_bleu": round(stage2_bleu, 2),
        "stage2_chrf": round(stage2_chrf, 2),
        "delta_bleu": round(delta_bleu, 2),
        "delta_chrf": round(delta_chrf, 2),
        "forgetting_bleu": forgetting["forgetting_bleu"],
        "forgetting_delta": forgetting["forgetting_delta"],
        "checkpoint_path": str(stage2_checkpoint),
        "stage1_checkpoint_path": str(stage1_checkpoint),
        "val_csv": str(val_csv_path),
        "samples": len(sources),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    return comparison


def write_comparison(output_dir: Path, comparison: dict) -> Path:
    """Write stage2_comparison.json artifact.

    Per D-09: Write comparison artifact containing:
    {stage1_bleu, stage1_chrf, stage2_bleu, stage2_chrf, delta_bleu, delta_chrf,
     forgetting_bleu, forgetting_delta}

    Args:
        output_dir: Directory to write comparison file
        comparison: Comparison metrics dictionary

    Returns:
        Path to written comparison file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "stage2_comparison.json"

    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return comparison_path


def print_summary_report(comparison: dict) -> None:
    """Print human-readable summary report.

    Args:
        comparison: Comparison metrics dictionary
    """
    print()
    print("Stage 2 Evaluation Complete")
    print("=" * 40)
    print(
        f"Stage 1 Baseline: BLEU={comparison['stage1_bleu']:.2f}, chrF={comparison['stage1_chrf']:.2f}"
    )
    print(
        f"Stage 2 Result:   BLEU={comparison['stage2_bleu']:.2f}, chrF={comparison['stage2_chrf']:.2f}"
    )

    # Format deltas with sign
    delta_bleu = comparison["delta_bleu"]
    delta_chrf = comparison["delta_chrf"]
    delta_bleu_str = f"+{delta_bleu:.2f}" if delta_bleu >= 0 else f"{delta_bleu:.2f}"
    delta_chrf_str = f"+{delta_chrf:.2f}" if delta_chrf >= 0 else f"{delta_chrf:.2f}"
    print(f"Improvement:      BLEU={delta_bleu_str}, chrF={delta_chrf_str}")
    print()

    # Forgetting check
    forgetting_delta = comparison["forgetting_delta"]
    if forgetting_delta < -2.0:
        status = f"WARNING - BLEU dropped {abs(forgetting_delta):.2f} points"
    else:
        delta_str = (
            f"+{forgetting_delta:.2f}"
            if forgetting_delta >= 0
            else f"{forgetting_delta:.2f}"
        )
        status = f"OK - {delta_str} BLEU within threshold"
    print(
        f"Forgetting Check: {comparison['forgetting_delta']:+.2f} BLEU on Stage 1 val ({status})"
    )
    print()


def main() -> int:
    """Main entry point for Stage 2 evaluation CLI."""
    args = parse_args()

    try:
        # Get paths
        stage2_checkpoint = args.checkpoint
        stage1_checkpoint = args.stage1_checkpoint
        output_dir = args.output_dir

        # Validate checkpoints
        print(f"[INFO] Validating Stage 2 checkpoint: {stage2_checkpoint}")
        print(f"[INFO] Validating Stage 1 checkpoint: {stage1_checkpoint}")
        validate_checkpoints(stage2_checkpoint, stage1_checkpoint, dry_run=args.dry_run)
        print("[OK] Checkpoint validation passed")

        # Load and merge config
        base_config = load_config(args.config)
        config = merge_config(base_config, args)

        if args.dry_run:
            # Dry-run: write zero metrics without loading model/data
            comparison = {
                "stage1_bleu": 0.0,
                "stage1_chrf": 0.0,
                "stage2_bleu": 0.0,
                "stage2_chrf": 0.0,
                "delta_bleu": 0.0,
                "delta_chrf": 0.0,
                "forgetting_bleu": 0.0,
                "forgetting_delta": 0.0,
                "checkpoint_path": str(stage2_checkpoint),
                "stage1_checkpoint_path": str(stage1_checkpoint),
                "val_csv": config.get("val_csv", "data/stage3_val.csv"),
                "samples": 0,
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }
            comparison_path = write_comparison(output_dir, comparison)
            print(f"[INFO] Comparison written: {comparison_path}")
            print("[DRY-RUN] Validation complete. No evaluation performed.")
            return 0

        # Set seeds for reproducibility
        set_seeds(config.get("seed", SEED))

        # Run evaluation
        comparison = run_evaluation(
            config, stage2_checkpoint, stage1_checkpoint, output_dir
        )

        # Write comparison artifact
        comparison_path = write_comparison(output_dir, comparison)
        print(f"[INFO] Comparison written: {comparison_path}")

        # Print summary
        print_summary_report(comparison)

        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
