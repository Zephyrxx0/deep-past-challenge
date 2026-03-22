# Roadmap: Deep Past Akkadian-English Translation

**Created:** 2026-03-22
**Granularity:** Fine
**Scope:** End-to-end implementation from model setup to submission artifacts

## Overview

This roadmap converts v1 requirements into executable phases with explicit success criteria and requirement coverage.

| # | Phase | Goal | Requirements |
|---|-------|------|--------------|
| 1 | Model Foundation Setup | Establish model selection, config, and smoke-tested loading path | ARCH-01, ARCH-02, ARCH-03 |
| 2 | Tokenizer and Text Interface | Ensure tokenization supports Akkadian-specific notation and roundtrip integrity | TOK-01, TOK-02 |
| 3 | Data Pipeline and Integrity | Build deterministic data loaders and enforce anti-leakage checks | DATA-01, DATA-02, DATA-03 |
| 4 | 2/2 | Complete   | 2026-03-22 |
| 5 | Stage 2 Domain Adaptation | Adapt model to OARE domain while monitoring forgetting | TRN2-01, TRN2-02 |
| 6 | Stage 3 Competition Fine-Tuning | Fine-tune for competition domain with genre conditioning and checkpoint selection | TRN3-01, TRN3-02 |
| 7 | Evaluation and Glossary Quality | Produce full metric suite including genre and proper-name quality | EVAL-01, EVAL-02, GLS-01, GLS-02 |
| 8 | Inference and Submission | Generate and validate competition submission outputs | INF-01, INF-02 |
| 9 | Reproducibility and Run Ops | Standardize reruns and artifact governance | OPS-01, OPS-02 |

## Phase Details

## Phase 1: Model Foundation Setup

**Goal:** Establish a stable model-selection and run configuration baseline.

**Requirements:** ARCH-01, ARCH-02, ARCH-03

**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md — Add config-driven model selector and approved model catalog (ARCH-01)
- [x] 01-02-PLAN.md — Add model/tokenizer smoke-test command with generation check (ARCH-02)
- [x] 01-03-PLAN.md — Add versioned stage training configs and loader CLI (ARCH-03)

**Success criteria:**
1. A committed config path supports switching among approved candidate checkpoints.
2. Smoke test script loads model/tokenizer and produces one translation without errors.
3. Stage-specific training configs are versioned and callable by scripts.

**UI hint:** no

## Phase 2: Tokenizer and Text Interface

**Goal:** Guarantee text interface correctness for Akkadian-specific inputs and special tokens.

**Requirements:** TOK-01, TOK-02

**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md — Build config-driven tokenizer training/resolver CLIs with special-token preservation tests (TOK-01)
- [ ] 02-02-PLAN.md — Add roundtrip verification CLI and representative transliteration roundtrip tests (TOK-02)

**Success criteria:**
1. Tokenizer recognizes `<gap>` and genre tags consistently.
2. Roundtrip tests pass on representative transliterations.
3. Tokenizer artifacts are versioned and reusable by training/inference scripts.

**UI hint:** no

## Phase 3: Data Pipeline and Integrity

**Goal:** Build deterministic loaders/collators with robust split integrity safeguards.

**Requirements:** DATA-01, DATA-02, DATA-03

**Plans:** 3 plans

Plans:
- [x] 03-01-PLAN.md — Build deterministic data loaders with fixed-seed reproducibility (DATA-01)
- [x] 03-02-PLAN.md — Create DataLoader factory with seq2seq batch collation and masking (DATA-02)
- [x] 03-03-PLAN.md — Implement split/provenance integrity validation checks (DATA-03)

**Success criteria:**
1. Data loaders work for all stage CSVs with fixed-seed reproducibility.
2. Batch tensors include correct attention and label masking for seq2seq training.
3. Integrity checks fail fast on split/provenance violations.

**UI hint:** no

## Phase 4: Stage 1 Training

**Goal:** Train a general Akkadian-English foundation model and persist checkpoints/metrics.

**Requirements:** TRN1-01, TRN1-02

**Plans:** 2/2 plans complete

Plans:
- [x] 04-01-PLAN.md — Stage 1 training CLI with checkpoints and run manifests (TRN1-01)
- [x] 04-02-PLAN.md — Stage 1 evaluation CLI with BLEU/chrF metrics artifacts (TRN1-02)

**Success criteria:**
1. Stage 1 training run completes and saves resumable checkpoints.
2. Validation metrics are computed and written to structured artifacts.
3. Training logs enable convergence diagnosis and rerun comparability.

**UI hint:** no

## Phase 5: Stage 2 Domain Adaptation

**Goal:** Improve OARE-domain quality while detecting and limiting catastrophic forgetting.

**Requirements:** TRN2-01, TRN2-02

**Plans:** 1/3 plans complete

Plans:
- [x] 05-01-PLAN.md — Stage 2 training CLI with shared utilities (TRN2-01)
- [ ] 05-02-PLAN.md — Forgetting detection baseline and comparison
- [ ] 05-03-PLAN.md — Stage 2 evaluation CLI with comparison reporting (TRN2-02)

**Success criteria:**
1. Stage 2 starts from Stage 1 checkpoint and trains with lower LR profile.
2. Stage 2 validation quality improves over Stage 1 on domain-focused evaluation.
3. Forgetting report quantifies retained Stage 1 performance.

**UI hint:** no

## Phase 6: Stage 3 Competition Fine-Tuning

**Goal:** Optimize competition-domain performance with genre conditioning and best-checkpoint selection.

**Requirements:** TRN3-01, TRN3-02

**Success criteria:**
1. Stage 3 training consumes genre-conditioned inputs and runs with early stopping.
2. Best checkpoint is selected by validation BLEU and exported.
3. Final stage artifacts support downstream evaluation and inference.

**UI hint:** no

## Phase 7: Evaluation and Glossary Quality

**Goal:** Validate translation quality using global, genre-specific, and proper-name-aware metrics.

**Requirements:** EVAL-01, EVAL-02, GLS-01, GLS-02

**Success criteria:**
1. BLEU and chrF are generated for all staged checkpoints.
2. Genre-level performance breakdown is reported for Stage 3 validation.
3. Glossary integration improves proper-name fidelity and is quantitatively reported.

**UI hint:** no

## Phase 8: Inference and Submission

**Goal:** Produce competition-ready outputs from the selected final checkpoint.

**Requirements:** INF-01, INF-02

**Success criteria:**
1. Inference generates one output per test input with no missing rows.
2. Submission file matches expected schema and passes validation checks.
3. Output artifacts are stored in standard locations for handoff.

**UI hint:** no

## Phase 9: Reproducibility and Run Ops

**Goal:** Ensure reruns are deterministic and artifacts are auditable.

**Requirements:** OPS-01, OPS-02

**Success criteria:**
1. Full run can be replayed from documented commands/configs.
2. Every run stores metrics, checkpoints, and config snapshots.
3. Artifact structure enables quick diagnosis and comparison across runs.

**UI hint:** no

## Coverage Validation

- v1 requirements total: 22
- Mapped to phases: 22
- Unmapped requirements: 0
- Coverage status: PASS

---
*Last updated: 2026-03-22 after Phase 05 Plan 01 completion*
