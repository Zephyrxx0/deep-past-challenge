# Phase 5: Stage 2 Domain Adaptation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-23
**Phase:** 05-stage-2-domain-adaptation
**Areas discussed:** Checkpoint sourcing, Forgetting detection, Training CLI structure, Evaluation comparison

---

## Checkpoint Sourcing

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit path required | Require --checkpoint pointing to a Stage 1 epoch dir; fail if missing or invalid | ✓ |
| Auto-detect with default | Default to models/stage1_final/epoch_{last} but allow override; warn if auto-detected | |
| Config-driven path | Require config key stage1_checkpoint in stage2.yaml; CLI override optional | |

**User's choice:** Explicit path required
**Notes:** Clear, explicit checkpoint sourcing prevents accidental use of wrong checkpoint.

### Checkpoint Validation

| Option | Description | Selected |
|--------|-------------|----------|
| File check + forward pass | Check model.safetensors/tokenizer.json exist + run one forward pass | ✓ |
| File existence only | Only check required files exist (fast, but silent corruption possible) | |
| Full load + generation | Full model load + one sample generation (thorough but slower startup) | |

**User's choice:** File check + forward pass
**Notes:** Balances thoroughness with startup speed.

---

## Forgetting Detection

| Option | Description | Selected |
|--------|-------------|----------|
| Val-set BLEU comparison | Run Stage 2 model on Stage 1 val samples, report BLEU/chrF drop vs Stage 1 baseline | ✓ |
| Per-epoch sampling | Sample N random Stage 1 val examples, track per-epoch BLEU during Stage 2 training | |
| Embedding drift analysis | Compute embedding drift between Stage 1 and Stage 2 models on held-out samples | |

**User's choice:** Val-set BLEU comparison
**Notes:** Standard approach using existing evaluation infrastructure.

### Baseline Capture

| Option | Description | Selected |
|--------|-------------|----------|
| Compute at Stage 2 start | Run Stage 1 eval at Stage 2 start, cache metrics in stage2_forgetting_baseline.json | ✓ |
| Require pre-computed file | Require baseline metrics file from prior Stage 1 eval run via --baseline flag | |
| Auto-read Stage 1 output | Read Stage 1 metrics from models/stage1_final/stage1_metrics.json if exists | |

**User's choice:** Compute at Stage 2 start
**Notes:** Self-contained approach that doesn't depend on prior evaluation runs.

### Threshold Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Warn on 2+ point drop | Log warning if BLEU drops >2 points, but continue training | ✓ |
| Fail on threshold | Hard fail if BLEU drops below configurable threshold | |
| Report only, no action | Just report the delta, no warnings or thresholds | |

**User's choice:** Warn on 2+ point drop
**Notes:** Alerts user without blocking potentially useful training runs.

---

## Training CLI Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Separate CLI, shared module | New train_stage2.py for clarity; share common logic via module imports | ✓ |
| Unified CLI with stage flag | Single train.py with --stage 1|2|3 flag, internal dispatch | |
| Separate CLIs, no sharing | Copy train_stage1.py to train_stage2.py with modifications | |

**User's choice:** Separate CLI, shared module
**Notes:** Clear separation while avoiding code duplication.

### Shared Module Location

| Option | Description | Selected |
|--------|-------------|----------|
| training_utils.py | New scripts/training_utils.py with shared functions (config loading, checkpointing, metrics) | ✓ |
| Inline duplication OK | Keep inline in each CLI, accept some duplication for simplicity | |
| Trainer class | Use a class-based trainer that both CLIs instantiate | |

**User's choice:** training_utils.py
**Notes:** Clean separation of concerns with function-based utilities.

---

## Evaluation Comparison

| Option | Description | Selected |
|--------|-------------|----------|
| Single comparison JSON | stage2_comparison.json with {stage1_bleu, stage2_bleu, delta, forgetting_bleu, forgetting_delta} | ✓ |
| Separate files + compare CLI | Separate metrics files + CLI that reads both and prints comparison table | |
| Markdown report | Markdown report with tables and pass/fail summary | |

**User's choice:** Single comparison JSON
**Notes:** Single artifact makes downstream consumption simple.

### Evaluation CLI

| Option | Description | Selected |
|--------|-------------|----------|
| Separate evaluate_stage2.py | New evaluate_stage2.py that runs both comparisons and writes stage2_comparison.json | ✓ |
| Extend evaluate_stage1.py | Extend evaluate_stage1.py with --stage and --compare flags | |
| Unified evaluate.py | Generic evaluate.py that handles all stages based on args | |

**User's choice:** Separate evaluate_stage2.py
**Notes:** Consistent with training CLI structure decision.

---

## Agent's Discretion

- Exact structure of `training_utils.py` internals (function signatures, helper breakdown)
- Specific error messages and logging verbosity
- Test organization within Phase 5 test module

## Deferred Ideas

None — discussion stayed within phase scope.
