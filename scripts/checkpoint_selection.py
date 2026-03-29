#!/usr/bin/env python3
"""Checkpoint selection utilities for Stage 3 evaluation.

Provides functions to:
- Evaluate individual checkpoints on validation data
- Compare multiple checkpoints and rank them
- Export best checkpoint to final model directory

Used by evaluate_stage3.py to automate best checkpoint selection.
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_checkpoint(
    checkpoint_path: Path,
    val_csv: Path,
    batch_size: int = 16,
    max_source_length: int = 256,
    max_target_length: int = 256,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate a single checkpoint on validation data.

    Args:
        checkpoint_path: Path to checkpoint directory
        val_csv: Path to validation CSV file
        batch_size: Batch size for inference
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        device: Device to run evaluation on (cuda/cpu)

    Returns:
        Dictionary with bleu and chrf scores
    """
    # Lazy imports for evaluation dependencies
    import pandas as pd
    import sacrebleu
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Determine actual device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_obj = torch.device(device)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    model.to(device_obj)
    model.eval()

    # Load validation data
    df = pd.read_csv(val_csv)

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
        inputs = {k: v.to(device_obj) for k, v in inputs.items()}

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

    # Compute metrics
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

    return {
        "bleu": bleu,
        "chrf": chrf,
    }


def compare_checkpoints(
    checkpoint_paths: List[Path],
    val_csv: Path,
    batch_size: int = 16,
    max_source_length: int = 256,
    max_target_length: int = 256,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Evaluate multiple checkpoints and return sorted comparison.

    Args:
        checkpoint_paths: List of checkpoint directory paths
        val_csv: Path to validation CSV file
        batch_size: Batch size for inference
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        device: Device to run evaluation on

    Returns:
        List of dicts with {path, bleu, chrf, epoch}, sorted by BLEU descending
    """
    results = []

    for checkpoint_path in checkpoint_paths:
        # Extract epoch number from path (e.g., epoch_1 -> 1)
        epoch_str = checkpoint_path.name
        epoch = int(epoch_str.split("_")[-1]) if "epoch_" in epoch_str else 0

        # Evaluate checkpoint
        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            val_csv=val_csv,
            batch_size=batch_size,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            device=device,
        )

        results.append(
            {
                "path": checkpoint_path,
                "bleu": metrics["bleu"],
                "chrf": metrics["chrf"],
                "epoch": epoch,
            }
        )

    # Sort by BLEU descending
    results.sort(key=lambda x: x["bleu"], reverse=True)

    return results


def find_best_checkpoint(
    checkpoints_dir: Path,
    val_csv: Path,
    metric: str = "bleu",
    batch_size: int = 16,
    max_source_length: int = 256,
    max_target_length: int = 256,
    device: str = "cuda",
) -> Path:
    """Find the best checkpoint based on validation metric.

    Scans checkpoints_dir/epoch_*/ directories and evaluates each.

    Args:
        checkpoints_dir: Directory containing epoch_* checkpoint subdirectories
        val_csv: Path to validation CSV file
        metric: Metric to use for selection (bleu or chrf)
        batch_size: Batch size for inference
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        device: Device to run evaluation on

    Returns:
        Path to best checkpoint directory
    """
    # Find all epoch checkpoints
    epoch_dirs = sorted(checkpoints_dir.glob("epoch_*"))

    if not epoch_dirs:
        raise ValueError(f"No epoch_* directories found in {checkpoints_dir}")

    # Compare all checkpoints
    results = compare_checkpoints(
        checkpoint_paths=epoch_dirs,
        val_csv=val_csv,
        batch_size=batch_size,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        device=device,
    )

    # Select best based on metric
    if metric == "bleu":
        # Already sorted by BLEU
        best = results[0]
    elif metric == "chrf":
        # Re-sort by chrF
        best = max(results, key=lambda x: x["chrf"])
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'bleu' or 'chrf'.")

    return best["path"]


def export_best_checkpoint(
    best_path: Path,
    export_dir: Path,
    metrics: Dict[str, Any],
) -> None:
    """Export best checkpoint to final model directory.

    Copies checkpoint files and creates manifest with metadata.

    Args:
        best_path: Path to best checkpoint directory
        export_dir: Directory to export checkpoint to
        metrics: Dictionary with evaluation metrics and metadata
    """
    # Create export directory
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from checkpoint to export directory
    for file_path in best_path.glob("*"):
        if file_path.is_file():
            dest_path = export_dir / file_path.name
            shutil.copy2(file_path, dest_path)

    # Create best checkpoint manifest
    manifest = {
        "best_checkpoint_path": str(best_path),
        "bleu": metrics.get("bleu"),
        "chrf": metrics.get("chrf"),
        "epoch": metrics.get("epoch"),
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = export_dir / "best_checkpoint_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Create evaluation results file
    eval_results = {
        "bleu": metrics.get("bleu"),
        "chrf": metrics.get("chrf"),
        "checkpoint_path": str(best_path),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    results_path = export_dir / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)


def validate_checkpoint_completeness(checkpoint_path: Path) -> bool:
    """Validate that checkpoint has all required files.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        True if checkpoint is complete, False otherwise
    """
    required_files = ["model.safetensors", "tokenizer.json", "config.json"]
    alternative_model_files = ["pytorch_model.bin"]

    # Check for model file (safetensors or pytorch_model.bin)
    has_model = (checkpoint_path / "model.safetensors").exists() or (
        checkpoint_path / "pytorch_model.bin"
    ).exists()

    # Check for tokenizer and config
    has_tokenizer = (checkpoint_path / "tokenizer.json").exists()
    has_config = (checkpoint_path / "config.json").exists()

    return has_model and has_tokenizer and has_config
