---
phase: 07-evaluation-and-glossary-quality
plan: 01
status: completed
completed_at: 2026-03-30
---

# Summary: Unified Evaluation CLI

## Objective
Implemented unified evaluation CLI with BLEU, chrF, and genre-specific metrics to satisfy EVAL-01 and EVAL-02 requirements.

## Tasks Completed

### Task 1: Create evaluation utilities and tests
- **File**: `scripts/evaluation_utils.py`
- **Functions implemented**:
  - `compute_bleu(predictions, references)` - Corpus-level BLEU using sacrebleu
  - `compute_chrf(predictions, references)` - Corpus-level chrF using sacrebleu
  - `compute_genre_metrics(predictions, references, genres)` - Per-genre breakdown
  - `extract_genre_tag(text)` - Extract genre tags like [LETTER], [ADMIN]
  - `load_predictions(csv_path)` - Load prediction CSV with validation
  - `load_validation_data(csv_path)` - Load validation CSV with auto-detection

- **Test file**: `tests/phase7/test_evaluation.py`
- **Test coverage**: 26 tests covering all utility functions

### Task 2: Implement unified evaluation CLI
- **File**: `scripts/evaluate.py`
- **CLI Arguments**:
  - `--checkpoint`: Path to model checkpoint directory (required)
  - `--val-data`: Path to validation CSV (default: data/stage3_val.csv)
  - `--output`: Path to output JSON for results
  - `--batch-size`: Batch size for inference (default: 16)
  - `--device`: cuda/cpu/auto (default: auto)
  - `--genre-breakdown`: Enable genre-specific metrics
  - `--dry-run`: Validate inputs without running evaluation

- **Output Format**:
```json
{
  "checkpoint": "models/stage3_final",
  "val_data": "data/stage3_val.csv",
  "overall": {"bleu": 25.4, "chrf": 48.2, "samples": 500},
  "by_genre": {
    "LETTER": {"bleu": 28.1, "chrf": 51.3, "count": 120},
    "ADMIN": {"bleu": 22.3, "chrf": 45.1, "count": 180}
  },
  "timestamp": "2026-03-30T12:00:00+00:00"
}
```

## Verification Results

### Tests
```
python -m pytest tests/phase7/test_evaluation.py -xvs
============================= 26 passed ==============================
```

### CLI Help
```
python scripts/evaluate.py --help
# Shows all arguments including --checkpoint, --genre-breakdown, --dry-run
```

### Dry-run
```
python scripts/evaluate.py --checkpoint models/stage3_final --dry-run
[INFO] Validating inputs...
[OK] Input validation passed
[DRY-RUN] Validation complete. No evaluation performed.
```

## Artifacts Created

| Artifact | Description | Lines |
|----------|-------------|-------|
| `scripts/evaluation_utils.py` | Shared evaluation utilities | ~230 |
| `scripts/evaluate.py` | Unified evaluation CLI | ~300 |
| `tests/phase7/__init__.py` | Phase 7 test package | 1 |
| `tests/phase7/test_evaluation.py` | Evaluation tests | ~280 |

## Requirements Satisfied
- **EVAL-01**: User can compute BLEU and chrF for any checkpoint with CLI
- **EVAL-02**: Genre-specific performance breakdown available with --genre-breakdown flag

## Key Design Decisions
1. Used sacrebleu library for reproducible metric computation
2. Reused patterns from `checkpoint_selection.py` for model loading
3. Auto-detects column names in validation data (normalized vs raw)
4. Supports both explicit genre column and genre extraction from text
5. Results saved as JSON for downstream analysis
