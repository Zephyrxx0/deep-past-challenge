#!/usr/bin/env python3
"""Stage 1 Training CLI with checkpointing and run manifests.

Implements TRN1-01: Runnable Stage 1 training that produces resumable
checkpoints and structured run artifacts.

Usage:
    # Dry-run (validation only, no torch required):
    python scripts/train_stage1.py --dry-run --output-dir outputs/stage1_test

    # Full training:
    python scripts/train_stage1.py --epochs 10 --output-dir models/stage1_final
"""

import argparse
import json
import os
import random
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
        description="Stage 1 training CLI with checkpointing and run manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and write run_manifest.json without training.",
    )

    # Config and overrides
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to stage1 training config YAML.",
    )
    parser.add_argument("--train-csv", type=Path, help="Override train CSV path.")
    parser.add_argument("--val-csv", type=Path, help="Override validation CSV path.")
    parser.add_argument("--model-id", type=str, help="Override model checkpoint ID.")
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
        help="Resume training from checkpoint directory.",
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
    if args.train_csv is not None:
        config["train_csv"] = str(args.train_csv)
    if args.val_csv is not None:
        config["val_csv"] = str(args.val_csv)
    if args.model_id is not None:
        config["model_id"] = args.model_id
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
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
        dry_run: If True, validate files exist

    Raises:
        ValueError: If required config is missing
        FileNotFoundError: If required files don't exist (in dry_run or train mode)
    """
    # Check required fields
    required_fields = ["output_dir"]
    missing = [f for f in required_fields if not config.get(f)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # Validate train_csv exists (required for both dry-run and training)
    train_csv_path = config.get("train_csv")
    if train_csv_path:
        train_csv = Path(train_csv_path)
        if not train_csv.exists():
            raise FileNotFoundError(
                f"train_csv not found: {train_csv}. "
                "Ensure the path is correct or run data preprocessing first."
            )
    elif not dry_run:
        raise ValueError("train_csv is required for training")

    # Validate val_csv if provided
    val_csv_path = config.get("val_csv")
    if val_csv_path:
        val_csv = Path(val_csv_path)
        if not val_csv.exists():
            raise FileNotFoundError(
                f"val_csv not found: {val_csv}. "
                "Ensure the path is correct or run data preprocessing first."
            )


def write_run_manifest(config: dict, output_dir: Path) -> Path:
    """Write run_manifest.json with run metadata.

    Args:
        config: Merged configuration
        output_dir: Directory to write manifest to

    Returns:
        Path to written manifest
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": 1,
        "model_id": config.get("model_id", "facebook/mbart-large-50-many-to-many-mmt"),
        "train_csv": config.get("train_csv"),
        "val_csv": config.get("val_csv"),
        "seed": config.get("seed", SEED),
        "output_dir": str(output_dir),
        "epochs": config.get("epochs", 10),
        "batch_size": config.get("batch_size", 4),
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "max_source_length": config.get("max_source_length", 256),
        "max_target_length": config.get("max_target_length", 256),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def set_seeds(seed: int) -> None:
    """Set deterministic seeds for reproducibility.

    Per AGENTS.md: Always set seeds for deterministic behavior.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def run_training(config: dict, resume_from: Path | None = None) -> None:
    """Run the Stage 1 training loop with checkpointing.

    Args:
        config: Merged training configuration
        resume_from: Optional checkpoint directory to resume from
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

    # Extract config values
    model_id = config.get("model_id", "facebook/mbart-large-50-many-to-many-mmt")
    output_dir = Path(config["output_dir"])
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 4)
    learning_rate = float(config.get("learning_rate", 1e-4))
    warmup_steps = config.get("warmup_steps", 500)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    max_source_length = config.get("max_source_length", 256)
    max_target_length = config.get("max_target_length", 256)
    seed = config.get("seed", SEED)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model and tokenizer
    if resume_from and resume_from.exists():
        print(f"[INFO] Resuming from checkpoint: {resume_from}")
        tokenizer = AutoTokenizer.from_pretrained(resume_from)
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_from)
    else:
        print(f"[INFO] Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    model.to(device)

    # Create training dataloader
    print("[INFO] Creating training dataloader...")
    train_loader = create_dataloader(
        stage=1,
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
        f"[DONE] Training complete. Final checkpoint: {output_dir / f'epoch_{epochs}'}"
    )


def main() -> int:
    """Main entry point for Stage 1 training CLI."""
    args = parse_args()

    try:
        # Load and merge config
        base_config = load_config(args.config)
        config = merge_config(base_config, args)

        # Validate config (validates files in both modes)
        validate_config(config, dry_run=args.dry_run)

        # Get output directory
        output_dir = Path(config["output_dir"])

        # Write run manifest
        manifest_path = write_run_manifest(config, output_dir)
        print(f"[INFO] Run manifest written: {manifest_path}")

        if args.dry_run:
            print("[DRY-RUN] Validation complete. No training performed.")
            return 0

        # Set seeds for reproducibility
        set_seeds(config.get("seed", SEED))

        # Run training
        run_training(config, resume_from=args.resume_from)
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
