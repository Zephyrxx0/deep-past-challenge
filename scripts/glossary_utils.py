#!/usr/bin/env python3
"""Glossary utilities for proper-name post-processing.

Provides functions for loading glossaries and applying proper-name corrections
to translation outputs.

Functions:
    - load_glossary: Load glossary JSON file
    - apply_glossary: Apply glossary corrections to text
    - extract_proper_names: Extract capitalized word sequences from text
    - compute_name_accuracy: Compute accuracy of proper names vs references
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set


def load_glossary(path: Path) -> Dict[str, str]:
    """Load glossary from JSON file.

    Args:
        path: Path to glossary JSON file

    Returns:
        Dictionary mapping source names to target names

    Raises:
        FileNotFoundError: If glossary file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Glossary file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        glossary = json.load(f)

    if not isinstance(glossary, dict):
        raise ValueError("Glossary must be a JSON object (dict)")

    return glossary


def apply_glossary(text: str, glossary: Dict[str, str]) -> str:
    """Apply glossary corrections to text.

    Replaces terms in the text with their glossary equivalents.
    Handles case variations by matching case-insensitively and
    preserving the original case pattern where possible.

    Args:
        text: Input text to correct
        glossary: Dictionary mapping source terms to target terms

    Returns:
        Text with glossary corrections applied
    """
    if not text or not glossary:
        return text

    result = text

    # Sort glossary entries by length (longest first) to avoid partial replacements
    sorted_entries = sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True)

    for source, target in sorted_entries:
        # Create case-insensitive pattern with word boundaries
        pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)

        def replace_match(match: re.Match) -> str:
            """Replace match preserving case pattern."""
            matched_text = match.group(0)

            # If the matched text is all uppercase, return target in uppercase
            if matched_text.isupper():
                return target.upper()
            # If the matched text starts with uppercase, capitalize target
            elif matched_text[0].isupper():
                return (
                    target[0].upper() + target[1:]
                    if len(target) > 1
                    else target.upper()
                )
            # Otherwise return target as-is (or lowercase if source was lowercase)
            else:
                return target.lower() if matched_text.islower() else target

        result = pattern.sub(replace_match, result)

    return result


def extract_proper_names(text: str) -> List[str]:
    """Extract proper names (capitalized word sequences) from text.

    Identifies capitalized words that likely represent proper names,
    including multi-word names connected by hyphens.

    Args:
        text: Input text to extract names from

    Returns:
        List of proper names found in text
    """
    if not text:
        return []

    # Pattern for capitalized words or hyphenated capitalized sequences
    # Matches: "Sargon", "Naram-Sin", "Tiglath-Pileser"
    pattern = re.compile(r"\b[A-Z][a-z]*(?:-[A-Z][a-z]*)*\b")

    matches = pattern.findall(text)

    # Filter out common English words that happen to be capitalized at sentence start
    common_words = {
        "The",
        "A",
        "An",
        "In",
        "On",
        "At",
        "To",
        "For",
        "Of",
        "And",
        "Or",
        "But",
        "If",
        "So",
        "As",
        "By",
        "From",
        "With",
        "Into",
        "When",
        "Where",
        "How",
        "What",
        "Who",
        "Which",
        "That",
        "This",
        "These",
        "Those",
        "He",
        "She",
        "It",
        "They",
        "We",
        "I",
        "You",
        "His",
        "Her",
        "Its",
        "Their",
        "Our",
        "My",
        "Your",
        "Here",
        "There",
        "Now",
        "Then",
    }

    # Return unique names, preserving order
    seen: Set[str] = set()
    result = []
    for name in matches:
        if name not in common_words and name not in seen:
            seen.add(name)
            result.append(name)

    return result


def compute_name_accuracy(
    predictions: List[str],
    references: List[str],
    glossary: Dict[str, str] = None,
) -> float:
    """Compute accuracy of proper names in predictions vs references.

    Measures what percentage of proper names in the reference translations
    also appear in the corresponding predictions.

    Args:
        predictions: List of predicted translations
        references: List of reference translations
        glossary: Optional glossary to define the set of valid proper names.
                  If provided, only names in the glossary are considered.

    Returns:
        Accuracy as percentage (0-100)

    Raises:
        ValueError: If predictions and references have different lengths
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )

    if not predictions:
        return 100.0  # Perfect score for empty input

    total_names = 0
    correct_names = 0

    # Build set of valid glossary terms for filtering
    glossary_terms = set(glossary.keys()) if glossary else None

    for pred, ref in zip(predictions, references):
        ref_names = extract_proper_names(ref)

        # If glossary provided, filter to only glossary terms
        if glossary_terms:
            ref_names = [n for n in ref_names if n in glossary_terms]

        if not ref_names:
            continue

        # Extract names from prediction (case-insensitive comparison)
        pred_names_lower = {n.lower() for n in extract_proper_names(pred)}

        for name in ref_names:
            total_names += 1
            if name.lower() in pred_names_lower:
                correct_names += 1

    if total_names == 0:
        return 100.0  # Perfect score if no names to compare

    return (correct_names / total_names) * 100


def count_corrections(
    original: str,
    corrected: str,
    glossary: Dict[str, str],
) -> int:
    """Count number of glossary corrections applied.

    Args:
        original: Original text before corrections
        corrected: Text after glossary corrections
        glossary: Glossary used for corrections

    Returns:
        Number of corrections made
    """
    if original == corrected:
        return 0

    count = 0
    for source, target in glossary.items():
        if source != target:
            # Count occurrences of source in original that became target in corrected
            source_pattern = re.compile(
                r"\b" + re.escape(source) + r"\b", re.IGNORECASE
            )
            source_matches = len(source_pattern.findall(original))
            target_pattern = re.compile(
                r"\b" + re.escape(target) + r"\b", re.IGNORECASE
            )
            target_matches = len(target_pattern.findall(corrected))

            # If target appears more in corrected than original, corrections were made
            original_target_matches = len(target_pattern.findall(original))
            if target_matches > original_target_matches:
                count += target_matches - original_target_matches

    return count
