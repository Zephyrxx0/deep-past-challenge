#!/usr/bin/env python3
"""Shared evaluation utilities for BLEU, chrF, and genre-specific metrics.

Provides reproducible evaluation functions used by evaluate.py CLI:
- compute_bleu: Corpus-level BLEU using sacrebleu
- compute_chrf: Corpus-level chrF using sacrebleu
- compute_genre_metrics: Per-genre breakdown of metrics
- load_predictions: Load prediction CSV with validation
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute corpus-level BLEU score.

    Uses sacrebleu for reproducible BLEU computation.

    Args:
        predictions: List of hypothesis translations
        references: List of reference translations

    Returns:
        BLEU score in range [0, 100]

    Raises:
        ValueError: If predictions and references have different lengths
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )

    if not predictions:
        return 0.0

    import sacrebleu

    # sacrebleu expects references as list of lists (multiple references per hypothesis)
    result = sacrebleu.corpus_bleu(predictions, [references])
    return result.score


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """Compute corpus-level chrF score.

    Uses sacrebleu for reproducible chrF computation.

    Args:
        predictions: List of hypothesis translations
        references: List of reference translations

    Returns:
        chrF score in range [0, 100]

    Raises:
        ValueError: If predictions and references have different lengths
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )

    if not predictions:
        return 0.0

    import sacrebleu

    result = sacrebleu.corpus_chrf(predictions, [references])
    return result.score


def extract_genre_tag(text: str) -> str:
    """Extract genre tag from text if present.

    Genre tags are in format [TAG] at the start of text, e.g., [LETTER], [ADMIN].

    Args:
        text: Input text potentially containing genre tag

    Returns:
        Genre tag without brackets (e.g., "LETTER") or "UNKNOWN" if no tag found
    """
    if not text:
        return "UNKNOWN"

    match = re.match(r"^\[([A-Z_]+)\]", text.strip())
    if match:
        return match.group(1)
    return "UNKNOWN"


def compute_genre_metrics(
    predictions: List[str],
    references: List[str],
    genres: List[str],
) -> Dict[str, Dict]:
    """Compute per-genre BLEU and chrF metrics.

    Groups predictions/references by genre tag and computes metrics for each group.

    Args:
        predictions: List of hypothesis translations
        references: List of reference translations
        genres: List of genre tags (e.g., ["LETTER", "ADMIN", ...])

    Returns:
        Dictionary mapping genre to metrics:
        {
            "LETTER": {"bleu": 28.1, "chrf": 51.3, "count": 120},
            "ADMIN": {"bleu": 22.3, "chrf": 45.1, "count": 180},
            ...
        }

    Raises:
        ValueError: If inputs have mismatched lengths
    """
    if not (len(predictions) == len(references) == len(genres)):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"references={len(references)}, genres={len(genres)}"
        )

    if not predictions:
        return {}

    # Group by genre
    genre_data: Dict[str, Tuple[List[str], List[str]]] = {}
    for pred, ref, genre in zip(predictions, references, genres):
        if genre not in genre_data:
            genre_data[genre] = ([], [])
        genre_data[genre][0].append(pred)
        genre_data[genre][1].append(ref)

    # Compute metrics for each genre
    results = {}
    for genre, (preds, refs) in genre_data.items():
        results[genre] = {
            "bleu": compute_bleu(preds, refs),
            "chrf": compute_chrf(preds, refs),
            "count": len(preds),
        }

    return results


def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Load predictions CSV with validation.

    Expected columns: id, prediction (or hypothesis)

    Args:
        csv_path: Path to predictions CSV file

    Returns:
        DataFrame with id and prediction columns

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check for id column
    if "id" not in df.columns:
        raise ValueError(
            f"Predictions CSV must have 'id' column. Found: {list(df.columns)}"
        )

    # Check for prediction column (support both 'prediction' and 'hypothesis')
    pred_col = None
    for col in ["prediction", "hypothesis", "translation"]:
        if col in df.columns:
            pred_col = col
            break

    if pred_col is None:
        raise ValueError(
            f"Predictions CSV must have 'prediction', 'hypothesis', or 'translation' column. "
            f"Found: {list(df.columns)}"
        )

    # Normalize column name to 'prediction'
    if pred_col != "prediction":
        df = df.rename(columns={pred_col: "prediction"})

    return df[["id", "prediction"]]


def load_validation_data(
    csv_path: Path,
    source_col: str = None,
    target_col: str = None,
) -> pd.DataFrame:
    """Load validation CSV with source and target columns.

    Auto-detects column names if not specified.

    Args:
        csv_path: Path to validation CSV file
        source_col: Source column name (auto-detect if None)
        target_col: Target column name (auto-detect if None)

    Returns:
        DataFrame with source, target, and optionally genre columns

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Auto-detect source column
    if source_col is None:
        for col in ["transliteration_normalized", "transliteration", "source"]:
            if col in df.columns:
                source_col = col
                break

    if source_col is None or source_col not in df.columns:
        raise ValueError(
            f"Could not find source column. Expected one of: "
            f"transliteration_normalized, transliteration, source. Found: {list(df.columns)}"
        )

    # Auto-detect target column
    if target_col is None:
        for col in ["translation_normalized", "translation", "target"]:
            if col in df.columns:
                target_col = col
                break

    if target_col is None or target_col not in df.columns:
        raise ValueError(
            f"Could not find target column. Expected one of: "
            f"translation_normalized, translation, target. Found: {list(df.columns)}"
        )

    result = df[[source_col, target_col]].copy()
    result.columns = ["source", "target"]

    # Add id column if present
    if "id" in df.columns:
        result["id"] = df["id"]

    # Add genre column if present
    if "genre" in df.columns:
        result["genre"] = df["genre"]

    return result
