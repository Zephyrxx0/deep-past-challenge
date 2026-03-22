---
phase: 04-stage-1-training
plan: 01
subsystem: training
tags:
  - cli
  - checkpointing
  - stage1
  - tdd
dependency_graph:
  requires:
    - config/training/stage1.yaml
    - scripts/create_dataloader.py
    - scripts/check_data_integrity.py
  provides:
    - Stage 1 training CLI with dry-run mode
    - Resumable checkpoints (per-epoch model + training_state.pt)
    - Run manifest (run_manifest.json)
    - Training metrics log (train_metrics.jsonl)
  affects:
    - models/stage1_final/
    - outputs/
tech_stack:
  added:
    - argparse CLI framework for training scripts
  patterns:
    - TDD contract tests via subprocess for CLI validation
    - Lazy torch imports for dry-run performance
    - YAML config with CLI override merge pattern
key_files:
  created:
    - scripts/train_stage1.py
    - tests/phase4/__init__.py
    - tests/phase4/test_stage1_training.py
  modified: []
decisions:
  - Dry-run mode validates config and writes manifest without torch imports for fast feedback
  - Per-epoch checkpoints include model, tokenizer, and training_state.pt for full resumability
  - Metrics logged to JSONL format for easy incremental parsing
metrics:
  duration: 3m
  completed: "2026-03-22T19:40:29Z"
---

# Phase 04 Plan 01: Stage 1 Training CLI Summary

Stage 1 training CLI with argparse, YAML config loading, dry-run validation, per-epoch checkpointing, and resume support using AdamW + linear warmup scheduler.

## Tasks Completed

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | Add TRN1-01 training CLI dry-run contract tests | `109521d` | ✅ |
| 2 | Implement Stage 1 training CLI with checkpointing | `a13ee5b` | ✅ |

## Implementation Details

### Stage 1 Training CLI (`scripts/train_stage1.py`)

The CLI provides:
- **Config loading**: YAML base config from `config/training/stage1.yaml` with CLI overrides
- **Dry-run mode**: Validates config, checks file existence, writes `run_manifest.json` without importing torch
- **Full training**: AdamW optimizer with linear warmup scheduler, gradient accumulation, per-epoch checkpoints
- **Resumability**: `--resume-from` loads model/tokenizer and restores optimizer/scheduler state from `training_state.pt`
- **Metrics**: Appends epoch metrics to `train_metrics.jsonl` after each epoch

### Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dry-run` | false | Validate config only, no training |
| `--config` | `config/training/stage1.yaml` | Base config path |
| `--train-csv` | from config | Override training data path |
| `--output-dir` | from config | Override checkpoint output dir |
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 4 | Training batch size |
| `--learning-rate` | 1e-4 | Learning rate (Stage 1 default) |
| `--seed` | 42 | Reproducibility seed per AGENTS.md |
| `--resume-from` | none | Checkpoint directory to resume from |

### Checkpoint Structure

```
output_dir/
├── run_manifest.json       # Run metadata (stage, model_id, seed, timestamps)
├── train_metrics.jsonl     # Per-epoch metrics (epoch, train_loss, steps)
├── epoch_1/
│   ├── config.json         # Model config
│   ├── pytorch_model.bin   # Model weights
│   ├── tokenizer.json      # Tokenizer
│   └── training_state.pt   # Optimizer + scheduler state for resume
├── epoch_2/
│   └── ...
```

### Contract Tests

4 tests covering TRN1-01 requirements:
1. **test_dry_run_writes_manifest_with_required_fields**: Verifies stage=1, model_id, seed in manifest
2. **test_invalid_train_csv_exits_nonzero_with_error**: Validates error handling
3. **test_dry_run_does_not_import_torch**: Confirms fast validation path
4. **test_default_seed_is_42**: Enforces AGENTS.md SEED convention

## Verification Results

```
$ python -m unittest tests.phase4.test_stage1_training -v
test_default_seed_is_42 ... ok
test_dry_run_does_not_import_torch ... ok
test_dry_run_writes_manifest_with_required_fields ... ok
test_invalid_train_csv_exits_nonzero_with_error ... ok
----------------------------------------------------------------------
Ran 4 tests in 0.388s
OK
```

## Deviations from Plan

None - plan executed exactly as written.

## Requirements Satisfied

- **TRN1-01**: Stage 1 training is runnable and produces resumable checkpoints
  - Dry-run validation provides fast feedback
  - Run manifest and per-epoch metrics are persisted
  - Resume via `--resume-from` checkpoint directory

## Known Stubs

None - all functionality is fully implemented.

## Self-Check: PASSED

- [x] `scripts/train_stage1.py` exists
- [x] `tests/phase4/__init__.py` exists
- [x] `tests/phase4/test_stage1_training.py` exists
- [x] Commit `109521d` exists (test RED)
- [x] Commit `a13ee5b` exists (implementation GREEN)
