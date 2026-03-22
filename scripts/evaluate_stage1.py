#!/usr/bin/env python3
"""Stage 1 Evaluation CLI with BLEU/chrF metrics output.

Implements TRN1-02: Evaluation produces BLEU/chrF metrics artifacts for Stage 1.

Usage:
    # Dry-run (validation only, writes zero metrics, no torch required):
    python scripts/evaluate_stage1.py --dry-run --output-dir outputs/stage1_eval

    # Full evaluation:
    python scripts/evaluate_stage1.py --checkpoint models/stage1_final --output-dir outputs/stage1_eval
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Default config path
DEFAULT_CONFIG_PATH = Path("config/training/stage1.yaml")
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with config overrides."""
    parser = argparse.ArgumentParser(
        description="Stage 1 evaluation CLI with BLEU/chrF metrics output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write zero metrics without loading model or data (fast validation).",
    )

    # Config and paths
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to stage1 training config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint directory (required for real evaluation).",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        help="Override validation CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write stage1_metrics.json.",
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


def load_config(config_path: Path) -> dict:
    """Load training config from YAML file."""
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_config(base_config: dict, args: argparse.Namespace) -> dict:
    """Merge base config with CLI overrides."""
    config = base_config.copy()

    # Apply CLI overrides if provided
    if args.checkpoint is not None:
        config["checkpoint"] = str(args.checkpoint)
    if args.val_csv is not None:
        config["val_csv"] = str(args.val_csv)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.max_source_length is not None:
        config["max_source_length"] = args.max_source_length
    if args.max_target_length is not None:
        config["max_target_length"] = args.max_target_length

    # Always use seed from CLI (has default)
    config["seed"] = args.seed

    return config


def validate_config(config: dict, dry_run: bool = False) -> None:
    """Validate required config fields and file existence.

    Args:
        config: Merged configuration dictionary
        dry_run: If True, skip checkpoint validation

    Raises:
        ValueError: If required config is missing
        FileNotFoundError: If required files don't exist
    """
    # Check required fields
    required_fields = ["output_dir"]
    missing = [f for f in required_fields if not config.get(f)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # In non-dry-run mode, checkpoint is required
    if not dry_run:
        checkpoint_path = config.get("checkpoint")
        if not checkpoint_path:
            raise ValueError(
                "error: checkpoint path is required for evaluation. "
                "Use --checkpoint <path> or --dry-run for validation mode."
            )
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"error: checkpoint not found at {checkpoint}. "
                "Ensure the checkpoint path is correct."
            )

    # Validate val_csv if provided and not in dry-run mode
    val_csv_path = config.get("val_csv")
    if val_csv_path and not dry_run:
        val_csv = Path(val_csv_path)
        if not val_csv.exists():
            raise FileNotFoundError(
                f"val_csv not found: {val_csv}. "
                "Ensure the path is correct or run data preprocessing first."
            )


def write_metrics(
    output_dir: Path,
    stage: int,
    bleu: float,
    chrf: float,
    samples: int,
    checkpoint_path: str | None = None,
) -> Path:
    """Write stage1_metrics.json with evaluation results.

    Args:
        output_dir: Directory to write metrics to
        stage: Training stage (1, 2, or 3)
        bleu: BLEU score
        chrf: chrF score
        samples: Number of samples evaluated
        checkpoint_path: Path to checkpoint used

    Returns:
        Path to written metrics file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "stage": stage,
        "bleu": bleu,
        "chrf": chrf,
        "samples": samples,
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = output_dir / "stage1_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics_path


def run_evaluation(config: dict) -> dict:
    """Run Stage 1 evaluation with BLEU/chrF computation.

    Args:
        config: Merged evaluation configuration

    Returns:
        Dictionary with bleu, chrf, samples keys
    """
    # Lazy imports for evaluation dependencies (not needed in dry-run)
    import torch
    import pandas as pd
    import sacrebleu
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Extract config values
    checkpoint_path = Path(config["checkpoint"])
    val_csv_path = Path(config.get("val_csv", "data/stage3_val.csv"))
    batch_size = config.get("batch_size", 4)
    max_source_length = config.get("max_source_length", 256)
    max_target_length = config.get("max_target_length", 256)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model and tokenizer
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    # Load validation data
    print(f"[INFO] Loading validation data: {val_csv_path}")
    df = pd.read_csv(val_csv_path)

    # Determine column names (support both normalized and raw)
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

    # Generate translations
    hypotheses = []
    for i in range(0, len(sources), batch_size):
        batch_sources = sources[i : i + batch_size]

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
        batch_hypotheses = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        hypotheses.extend(batch_hypotheses)

        if (i // batch_size + 1) % 10 == 0:
            print(
                f"[INFO] Processed {min(i + batch_size, len(sources))}/{len(sources)} samples"
            )

    # Compute metrics
    print("[INFO] Computing BLEU and chrF metrics...")
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

    print(f"[RESULT] BLEU: {bleu:.2f}")
    print(f"[RESULT] chrF: {chrf:.2f}")

    return {
        "bleu": bleu,
        "chrf": chrf,
        "samples": len(sources),
    }


def main() -> int:
    """Main entry point for Stage 1 evaluation CLI."""
    args = parse_args()

    try:
        # Load and merge config
        base_config = load_config(args.config)
        config = merge_config(base_config, args)

        # Validate config
        validate_config(config, dry_run=args.dry_run)

        # Get output directory
        output_dir = Path(config["output_dir"])

        if args.dry_run:
            # Dry-run: write zero metrics without loading model/data
            metrics_path = write_metrics(
                output_dir=output_dir,
                stage=1,
                bleu=0.0,
                chrf=0.0,
                samples=0,
                checkpoint_path=None,
            )
            print(f"[INFO] Metrics written: {metrics_path}")
            print("[DRY-RUN] Validation complete. No evaluation performed.")
            return 0

        # Run evaluation
        results = run_evaluation(config)

        # Write metrics
        metrics_path = write_metrics(
            output_dir=output_dir,
            stage=1,
            bleu=results["bleu"],
            chrf=results["chrf"],
            samples=results["samples"],
            checkpoint_path=config.get("checkpoint"),
        )
        print(f"[INFO] Metrics written: {metrics_path}")
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
