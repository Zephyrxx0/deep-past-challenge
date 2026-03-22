#!/usr/bin/env python3
"""Stage 2 Domain Adaptation Training CLI.

Implements TRN2-01: Stage 2 training that loads from Stage 1 checkpoint
and trains with domain-specific OARE data using lower learning rate.

Per D-06: Separate CLI for Stage 2 training
Per D-07: Uses shared utilities from training_utils.py

Usage:
    # Dry-run (validation only, no torch required):
    python scripts/train_stage2.py --dry-run --checkpoint models/stage1_final/epoch_10 --output-dir outputs/stage2_test

    # Full training:
    python scripts/train_stage2.py --checkpoint models/stage1_final/epoch_10 --output-dir models/stage2_final
"""

import argparse
import json
import sys
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
    write_run_manifest,
    set_seeds,
)


# Default config path (per D-06)
DEFAULT_CONFIG_PATH = Path("config/training/stage2.yaml")
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with config overrides."""
    parser = argparse.ArgumentParser(
        description="Stage 2 domain adaptation training CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and checkpoint without training.",
    )

    # Required checkpoint argument (per D-01)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Stage 1 checkpoint directory (required).",
    )

    # Config and overrides
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to stage2 training config YAML.",
    )
    parser.add_argument("--train-csv", type=Path, help="Override train CSV path.")
    parser.add_argument("--val-csv", type=Path, help="Override validation CSV path.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--max-source-length", type=int, help="Override max source sequence length."
    )
    parser.add_argument(
        "--max-target-length", type=int, help="Override max target sequence length."
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume Stage 2 training from checkpoint directory.",
    )

    return parser.parse_args()


def validate_checkpoint(checkpoint_path: Path, dry_run: bool = False) -> None:
    """Validate Stage 1 checkpoint exists and has required files.

    Per D-02: Check file existence (model.safetensors or pytorch_model.bin,
    tokenizer.json) + one forward pass test before training starts.

    Args:
        checkpoint_path: Path to Stage 1 checkpoint directory
        dry_run: If True, skip forward pass test

    Raises:
        FileNotFoundError: If checkpoint directory or required files don't exist
        ValueError: If checkpoint is invalid
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}. "
            "Ensure the path points to a valid Stage 1 checkpoint directory."
        )

    if not checkpoint_path.is_dir():
        raise ValueError(
            f"Checkpoint path must be a directory: {checkpoint_path}. "
            "Expected a directory containing model files (e.g., models/stage1_final/epoch_10)."
        )

    # Check for model file (safetensors or pytorch_model.bin)
    model_safetensors = checkpoint_path / "model.safetensors"
    pytorch_model = checkpoint_path / "pytorch_model.bin"
    if not model_safetensors.exists() and not pytorch_model.exists():
        raise FileNotFoundError(
            f"No model file found in {checkpoint_path}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    # Check for tokenizer files
    tokenizer_json = checkpoint_path / "tokenizer.json"
    tokenizer_config = checkpoint_path / "tokenizer_config.json"
    if not tokenizer_json.exists() and not tokenizer_config.exists():
        raise FileNotFoundError(
            f"No tokenizer configuration found in {checkpoint_path}. "
            "Expected tokenizer.json or tokenizer_config.json."
        )

    # Forward pass test (only in training mode)
    if not dry_run:
        _run_forward_pass_test(checkpoint_path)


def _run_forward_pass_test(checkpoint_path: Path) -> None:
    """Run one forward pass to validate checkpoint loads correctly.

    Args:
        checkpoint_path: Path to checkpoint directory

    Raises:
        RuntimeError: If forward pass fails
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"[INFO] Running forward pass test on checkpoint: {checkpoint_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

        # Run a simple forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Simple test input
        test_input = "a-na šar-ri"
        inputs = tokenizer(test_input, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Test that model can generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=1,
            )

        print("[OK] Forward pass test passed")

    except Exception as e:
        raise RuntimeError(
            f"Forward pass test failed for checkpoint {checkpoint_path}: {e}"
        )


def run_training(
    config: dict, checkpoint_path: Path, resume_from: Path | None = None
) -> None:
    """Run the Stage 2 training loop with checkpointing.

    Args:
        config: Merged training configuration
        checkpoint_path: Path to Stage 1 checkpoint to load from
        resume_from: Optional Stage 2 checkpoint directory to resume from
    """
    # Lazy imports for training dependencies (not needed in dry-run)
    import torch
    from torch.optim import AdamW
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from scripts.check_data_integrity import check_provenance, check_split_integrity
    from scripts.create_dataloader import create_dataloader

    # Run data integrity checks
    print("[INFO] Running data integrity checks...")
    check_provenance()
    check_split_integrity()
    print("[OK] Data integrity checks passed")

    # Extract config values (Stage 2 defaults: LR 5e-5, 5 epochs)
    model_id = config.get("model_id", "facebook/mbart-large-50-many-to-many-mmt")
    output_dir = Path(config["output_dir"])
    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 4)
    learning_rate = float(config.get("learning_rate", 5e-5))  # Stage 2 default
    warmup_steps = config.get("warmup_steps", 300)
    gradient_accumulation_steps = config.get(
        "gradient_accumulation_steps", 8
    )  # Stage 2 default
    max_source_length = config.get("max_source_length", 256)
    max_target_length = config.get("max_target_length", 256)
    seed = config.get("seed", SEED)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model and tokenizer
    if resume_from and resume_from.exists():
        print(f"[INFO] Resuming Stage 2 training from: {resume_from}")
        tokenizer = AutoTokenizer.from_pretrained(resume_from)
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_from)
    else:
        print(f"[INFO] Loading Stage 1 checkpoint: {checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    model.to(device)

    # Create training dataloader (Stage 2 data)
    print("[INFO] Creating Stage 2 training dataloader...")
    train_loader = create_dataloader(
        stage=2,
        model_id=model_id,
        batch_size=batch_size,
        split="train",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        seed=seed,
    )
    print(f"[INFO] Training dataloader created: {len(train_loader)} batches")

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Resume optimizer/scheduler state if available
    start_epoch = 0
    global_step = 0
    if resume_from:
        training_state_path = resume_from / "training_state.pt"
        if training_state_path.exists():
            print(f"[INFO] Loading training state from: {training_state_path}")
            training_state = torch.load(training_state_path, map_location=device)
            start_epoch = training_state.get("epoch", 0) + 1
            global_step = training_state.get("global_step", 0)
            optimizer.load_state_dict(training_state.get("optimizer_state_dict", {}))
            scheduler.load_state_dict(training_state.get("scheduler_state_dict", {}))
            print(
                f"[INFO] Resuming from epoch {start_epoch}, global_step {global_step}"
            )

    # Metrics log file
    metrics_path = output_dir / "train_metrics.jsonl"

    # Training loop
    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += outputs.loss.item()
            epoch_steps += 1

            # Gradient accumulation step
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Average epoch loss
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        print(
            f"[EPOCH {epoch + 1}/{epochs}] train_loss: {avg_loss:.4f}, steps: {global_step}"
        )

        # Save epoch checkpoint
        epoch_dir = output_dir / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

        # Save training state for resumption
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(training_state, epoch_dir / "training_state.pt")

        # Append metrics to JSONL
        metrics_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "steps": global_step,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_entry) + "\n")

        print(f"[INFO] Checkpoint saved: {epoch_dir}")

    print(
        f"[DONE] Stage 2 training complete. Final checkpoint: {output_dir / f'epoch_{epochs}'}"
    )


def main() -> int:
    """Main entry point for Stage 2 training CLI."""
    args = parse_args()

    try:
        # Validate checkpoint path (per D-01: fail if missing)
        checkpoint_path = args.checkpoint
        print(f"[INFO] Validating checkpoint: {checkpoint_path}")
        validate_checkpoint(checkpoint_path, dry_run=args.dry_run)
        print("[OK] Checkpoint validation passed")

        # Load and merge config
        base_config = load_config(args.config)
        config = merge_config(base_config, args)

        # Validate config (validates files in both modes)
        validate_config(config, dry_run=args.dry_run)

        # Get output directory
        output_dir = Path(config["output_dir"])

        # Write run manifest (stage=2, include checkpoint path)
        manifest_path = write_run_manifest(
            config,
            output_dir,
            stage=2,
            checkpoint_path=str(checkpoint_path),
        )
        print(f"[INFO] Run manifest written: {manifest_path}")

        if args.dry_run:
            print("[DRY-RUN] Validation complete. No training performed.")
            return 0

        # Set seeds for reproducibility
        set_seeds(config.get("seed", SEED))

        # Run training
        run_training(config, checkpoint_path, resume_from=args.resume_from)
        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
