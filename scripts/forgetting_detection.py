#!/usr/bin/env python3
"""Forgetting detection utilities for Stage 2 domain adaptation.

Implements D-03, D-04, D-05:
- D-03: Detect forgetting by running Stage 2 model on Stage 1 val samples
- D-04: Compute Stage 1 baseline at Stage 2 training start
- D-05: Warn (not fail) if BLEU drops >2 points from Stage 1 baseline

Provides:
- compute_forgetting_baseline: Compute baseline metrics from Stage 1 checkpoint
- detect_forgetting: Compare Stage 2 metrics against Stage 1 baseline
- load_baseline: Load cached baseline from disk
- format_forgetting_report: Human-readable forgetting report
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Forgetting threshold: warn if BLEU drops more than 2 points (D-05)
FORGETTING_THRESHOLD = 2.0


def compute_forgetting_baseline(
    stage1_checkpoint: Path,
    val_csv: Path,
    output_dir: Path,
    batch_size: int = 4,
    max_source_length: int = 256,
    max_target_length: int = 256,
) -> dict[str, Any]:
    """Compute baseline metrics from Stage 1 checkpoint on validation data.

    Per D-04: Compute Stage 1 baseline at Stage 2 training start; cache in
    `stage2_forgetting_baseline.json` in output directory.

    Args:
        stage1_checkpoint: Path to Stage 1 checkpoint directory
        val_csv: Path to validation CSV (Stage 1 val or Stage 3 val subset)
        output_dir: Directory to save baseline JSON
        batch_size: Batch size for inference
        max_source_length: Max source sequence length
        max_target_length: Max target sequence length

    Returns:
        Dictionary with baseline metrics: {stage1_checkpoint, bleu, chrf, samples, computed_at}

    Raises:
        FileNotFoundError: If checkpoint or val_csv doesn't exist
    """
    # Lazy imports for evaluation dependencies
    import pandas as pd
    import sacrebleu
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Validate inputs
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {stage1_checkpoint}. "
            "Ensure the path points to a valid Stage 1 checkpoint directory."
        )

    if not val_csv.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {val_csv}. "
            "Run data preprocessing first or provide a valid path."
        )

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Computing forgetting baseline on device: {device}")

    # Load model and tokenizer
    print(f"[INFO] Loading Stage 1 checkpoint: {stage1_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(stage1_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(stage1_checkpoint)
    model.to(device)
    model.eval()

    # Load validation data
    print(f"[INFO] Loading validation data: {val_csv}")
    df = pd.read_csv(val_csv)

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
    print(f"[INFO] Loaded {len(sources)} validation samples for baseline")

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

    # Compute metrics
    print("[INFO] Computing BLEU and chrF metrics for baseline...")
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

    # Build baseline result
    baseline = {
        "stage1_checkpoint": str(stage1_checkpoint),
        "bleu": round(bleu, 2),
        "chrf": round(chrf, 2),
        "samples": len(sources),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save baseline to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "stage2_forgetting_baseline.json"
    with baseline_path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print(f"[INFO] Baseline saved: {baseline_path}")
    print(f"[BASELINE] BLEU: {baseline['bleu']:.2f}, chrF: {baseline['chrf']:.2f}")

    return baseline


def detect_forgetting(
    stage2_checkpoint: Path,
    baseline: dict[str, Any],
    val_csv: Path,
    batch_size: int = 4,
    max_source_length: int = 256,
    max_target_length: int = 256,
) -> dict[str, Any]:
    """Detect catastrophic forgetting by comparing Stage 2 to Stage 1 baseline.

    Per D-03: Detect forgetting by running Stage 2 final model on Stage 1 val
    samples, comparing BLEU/chrF to Stage 1 baseline.

    Per D-05: Warn (not fail) if BLEU drops >2 points from Stage 1 baseline;
    training continues regardless.

    Args:
        stage2_checkpoint: Path to Stage 2 checkpoint directory
        baseline: Baseline metrics dict from compute_forgetting_baseline()
        val_csv: Path to validation CSV (same as used for baseline)
        batch_size: Batch size for inference
        max_source_length: Max source sequence length
        max_target_length: Max target sequence length

    Returns:
        Dictionary with forgetting metrics:
        {forgetting_bleu, forgetting_delta, stage2_bleu, stage1_bleu, warning}

    Raises:
        FileNotFoundError: If checkpoint or val_csv doesn't exist
    """
    # Lazy imports for evaluation dependencies
    import pandas as pd
    import sacrebleu
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Validate inputs
    if not stage2_checkpoint.exists():
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {stage2_checkpoint}. "
            "Ensure the path points to a valid Stage 2 checkpoint directory."
        )

    if not val_csv.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {val_csv}. "
            "Use the same validation CSV as compute_forgetting_baseline()."
        )

    # Extract baseline metrics
    stage1_bleu = baseline.get("bleu", 0.0)
    stage1_chrf = baseline.get("chrf", 0.0)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Detecting forgetting on device: {device}")

    # Load Stage 2 model and tokenizer
    print(f"[INFO] Loading Stage 2 checkpoint: {stage2_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(stage2_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(stage2_checkpoint)
    model.to(device)
    model.eval()

    # Load validation data
    print(f"[INFO] Loading validation data: {val_csv}")
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
    print(f"[INFO] Loaded {len(sources)} validation samples for forgetting detection")

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

    # Compute metrics for Stage 2
    print("[INFO] Computing BLEU and chrF metrics for Stage 2...")
    stage2_bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    stage2_chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

    # Compute deltas (negative = forgetting, positive = retained/improved)
    delta_bleu = stage2_bleu - stage1_bleu
    delta_chrf = stage2_chrf - stage1_chrf

    # D-05: Check forgetting threshold
    has_warning = delta_bleu < -FORGETTING_THRESHOLD
    warning_msg = None
    if has_warning:
        warning_msg = (
            f"WARNING: Forgetting detected! BLEU dropped {abs(delta_bleu):.2f} points "
            f"(threshold: {FORGETTING_THRESHOLD}). Training continues, but review may be needed."
        )
        print(f"[WARNING] {warning_msg}")
    else:
        print(f"[OK] No significant forgetting: BLEU delta = {delta_bleu:+.2f}")

    # Build forgetting result
    forgetting = {
        "forgetting_bleu": round(stage2_bleu, 2),
        "forgetting_delta": round(delta_bleu, 2),
        "forgetting_chrf": round(stage2_chrf, 2),
        "forgetting_chrf_delta": round(delta_chrf, 2),
        "stage2_bleu": round(stage2_bleu, 2),
        "stage2_chrf": round(stage2_chrf, 2),
        "stage1_bleu": round(stage1_bleu, 2),
        "stage1_chrf": round(stage1_chrf, 2),
        "threshold": FORGETTING_THRESHOLD,
        "warning": warning_msg,
    }

    return forgetting


def load_baseline(output_dir: Path) -> dict[str, Any]:
    """Load cached forgetting baseline from disk.

    Args:
        output_dir: Directory containing stage2_forgetting_baseline.json

    Returns:
        Baseline metrics dictionary

    Raises:
        FileNotFoundError: If baseline file doesn't exist
    """
    baseline_path = output_dir / "stage2_forgetting_baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Forgetting baseline not found: {baseline_path}. "
            "Run compute_forgetting_baseline() first to create the baseline."
        )

    with baseline_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_forgetting_report(forgetting: dict[str, Any]) -> str:
    """Format forgetting metrics as human-readable report.

    Args:
        forgetting: Forgetting metrics dictionary from detect_forgetting()

    Returns:
        Formatted report string
    """
    lines = [
        "Forgetting Detection Report",
        "=" * 40,
        f"Stage 1 Baseline: BLEU={forgetting['stage1_bleu']:.2f}, chrF={forgetting['stage1_chrf']:.2f}",
        f"Stage 2 on S1 Val: BLEU={forgetting['stage2_bleu']:.2f}, chrF={forgetting['stage2_chrf']:.2f}",
        f"Delta:             BLEU={forgetting['forgetting_delta']:+.2f}, chrF={forgetting.get('forgetting_chrf_delta', 0):+.2f}",
        "",
    ]

    # Status line
    threshold = forgetting.get("threshold", FORGETTING_THRESHOLD)
    if forgetting.get("warning"):
        lines.append(f"Status: FORGETTING DETECTED (BLEU drop > {threshold})")
        lines.append(f"Warning: {forgetting['warning']}")
    else:
        lines.append(f"Status: OK (BLEU within threshold of {threshold})")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test when run directly
    print("[INFO] forgetting_detection.py module loaded successfully")
    print(
        f"[INFO] Exports: compute_forgetting_baseline, detect_forgetting, load_baseline, format_forgetting_report"
    )
