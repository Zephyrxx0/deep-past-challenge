---
phase: 04-stage-1-training
plan: 02
subsystem: evaluation
tags:
  - evaluation
  - metrics
  - bleu
  - chrf
  - tdd
dependency_graph:
  requires:
    - scripts/train_stage1.py
    - config/training/stage1.yaml
    - data/stage3_val.csv
  provides:
    - scripts/evaluate_stage1.py
    - outputs/stage1_metrics.json
  affects:
    - models/stage1_final
tech_stack:
  added:
    - sacrebleu (BLEU/chrF computation)
  patterns:
    - TDD contract testing
    - CLI argparse with config merge
    - Lazy imports for torch dependencies
key_files:
  created:
    - scripts/evaluate_stage1.py
    - tests/phase4/test_stage1_evaluation.py
  modified: []
decisions:
  - Dry-run mode writes zero metrics without torch imports for fast CI validation
  - Metrics JSON includes checkpoint path and timestamp for traceability
  - Uses sacrebleu for standardized BLEU/chrF computation
metrics:
  duration: 2m
  completed: "2026-03-22T19:44:30Z"
---

# Phase 04 Plan 02: Stage 1 Evaluation CLI Summary

Stage 1 evaluation CLI producing BLEU/chrF metrics artifacts via sacrebleu with dry-run validation mode.

## Objective Achieved

✅ TRN1-02 satisfied: evaluation produces BLEU/chrF metrics artifacts for Stage 1.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add TRN1-02 evaluation contract tests | 849fef4 | tests/phase4/test_stage1_evaluation.py |
| 2 | Implement Stage 1 evaluation CLI | 0a72353 | scripts/evaluate_stage1.py |

## Implementation Details

### Task 1: TRN1-02 Contract Tests (TDD RED → GREEN)

Created contract tests in `tests/phase4/test_stage1_evaluation.py`:
- **Test 1**: Verify `--dry-run` writes `stage1_metrics.json` with keys `[stage, bleu, chrf, samples]`
- **Test 2**: Verify missing checkpoint exits non-zero with error message containing "checkpoint"

Tests use subprocess to invoke the CLI, ensuring end-to-end validation without torch dependencies.

### Task 2: Evaluation CLI Implementation

Implemented `scripts/evaluate_stage1.py` with:
- **CLI arguments**: `--checkpoint`, `--val-csv`, `--output-dir`, `--batch-size`, `--max-source-length`, `--max-target-length`, `--seed`, `--dry-run`
- **Config loading**: Reads `config/training/stage1.yaml` and merges with CLI overrides
- **Dry-run mode**: Writes zero metrics without importing torch for fast validation
- **Full evaluation mode**: Loads model/tokenizer from checkpoint, generates translations, computes BLEU/chrF via sacrebleu
- **Metrics artifact**: Writes `stage1_metrics.json` with stage, bleu, chrf, samples, checkpoint, timestamp

## Verification Results

```
$ python -m unittest tests.phase4.test_stage1_evaluation -v
test_dry_run_writes_metrics_json_with_correct_schema ... ok
test_missing_checkpoint_exits_nonzero_with_error ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.185s

OK
```

## Deviations from Plan

None - plan executed exactly as written.

## Key Artifacts

### scripts/evaluate_stage1.py
- Evaluation CLI matching train_stage1.py pattern
- Supports both dry-run validation and full evaluation
- Outputs structured metrics JSON

### tests/phase4/test_stage1_evaluation.py
- Contract tests for TRN1-02 requirements
- Subprocess-based testing for CLI validation
- No torch dependencies required

## Usage Examples

```bash
# Dry-run (fast validation, no torch required)
python scripts/evaluate_stage1.py --dry-run --output-dir outputs/stage1_eval

# Full evaluation
python scripts/evaluate_stage1.py --checkpoint models/stage1_final --output-dir outputs/stage1_eval

# With custom validation data
python scripts/evaluate_stage1.py --checkpoint models/stage1_final --val-csv data/stage3_val.csv --output-dir outputs/stage1_eval
```

## Metrics Artifact Schema

```json
{
  "stage": 1,
  "bleu": 0.0,
  "chrf": 0.0,
  "samples": 0,
  "checkpoint": null,
  "timestamp": "2026-03-22T19:44:20.678537+00:00"
}
```

## Self-Check: PASSED

- [x] scripts/evaluate_stage1.py exists
- [x] tests/phase4/test_stage1_evaluation.py exists
- [x] Commit 849fef4 found
- [x] Commit 0a72353 found
- [x] All tests pass
