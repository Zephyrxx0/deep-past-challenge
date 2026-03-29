---
phase: 07-evaluation-and-glossary-quality
plan: 02
status: completed
completed_at: 2026-03-30
---

# Summary: Glossary-Aware Post-Processing

## Objective
Implemented glossary-aware post-processing and proper-name accuracy reporting to satisfy GLS-01 and GLS-02 requirements.

## Tasks Completed

### Task 1: Create glossary utilities and tests
- **File**: `scripts/glossary_utils.py`
- **Functions implemented**:
  - `load_glossary(path)` - Load JSON glossary and return dict
  - `apply_glossary(text, glossary)` - Replace proper names with standardized forms (case-insensitive, word boundaries)
  - `extract_proper_names(text)` - Extract capitalized word sequences (filters common words)
  - `compute_name_accuracy(predictions, references, glossary)` - Compute proper-name accuracy percentage
  - `count_corrections(original, corrected)` - Count number of corrections made

- **Test file**: `tests/phase7/test_glossary.py`
- **Test coverage**: 27 tests covering all utility functions

### Task 2: Implement glossary application CLI
- **File**: `scripts/apply_glossary.py`
- **CLI Arguments**:
  - `--input`: Path to predictions CSV (id, translation columns) - required
  - `--output`: Path to output CSV with glossary-corrected translations
  - `--glossary`: Path to glossary JSON (default: data/glossary.json)
  - `--references`: Path to reference translations for accuracy comparison
  - `--report`: Path to output accuracy report JSON
  - `--dry-run`: Show sample corrections without writing output
  - `--samples`: Number of sample corrections to show in dry-run mode (default: 5)

- **Output Report Format**:
```json
{
  "input": "outputs/submission.csv",
  "glossary": "data/glossary.json",
  "corrections_made": 45,
  "name_accuracy_before": 72.3,
  "name_accuracy_after": 89.1,
  "improvement": 16.8
}
```

## Verification Results

### Tests
```
python -m pytest tests/phase7/test_glossary.py -xvs
============================= 27 passed ==============================
```

### CLI Help
```
python scripts/apply_glossary.py --help
# Shows all arguments including --input, --glossary, --references, --dry-run
```

### Glossary Verification
```
# data/glossary.json exists with thousands of Akkadian proper name mappings
# Example entries: "a-šùr": "Aššur", "sar-ru-gi": "Sargon"
```

## Artifacts Created

| Artifact | Description | Lines |
|----------|-------------|-------|
| `scripts/glossary_utils.py` | Glossary loading and application utilities | ~120 |
| `scripts/apply_glossary.py` | CLI for glossary post-processing | ~180 |
| `tests/phase7/test_glossary.py` | Glossary tests | 306 |

## Pre-existing Artifacts Used

| Artifact | Description |
|----------|-------------|
| `data/glossary.json` | Comprehensive Akkadian proper name glossary (thousands of entries) |

## Requirements Satisfied
- **GLS-01**: User can apply glossary-aware post-processing to translations
- **GLS-02**: User can measure proper-name accuracy before and after glossary

## Key Design Decisions
1. Case-insensitive matching with word boundary awareness for robust name replacement
2. Leveraged existing comprehensive glossary at `data/glossary.json` rather than creating minimal sample
3. Common English words filtered from proper name extraction to improve accuracy
4. Supports hyphenated names (e.g., "Naram-Sin") in extraction
5. Reports before/after accuracy to quantify glossary benefit
6. Dry-run mode shows sample corrections for verification before committing changes
