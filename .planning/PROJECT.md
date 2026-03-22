# Deep Past Akkadian-English Translation

## What This Is

This project builds a neural machine translation pipeline that translates Old Akkadian cuneiform transliterations into English. The preprocessing foundation is already complete, and the current scope focuses on implementing and validating a 3-stage curriculum training pipeline for competition-quality inference. The primary users are researchers and competition participants working with low-resource historical language translation.

## Core Value

Deliver reliable Akkadian-to-English translation quality through a reproducible 3-stage training pipeline that avoids leakage and preserves domain adaptation gains.

## Requirements

### Validated

- ✓ Multi-source text normalization pipeline implemented (Unicode harmonization, subscript normalization, lacunae handling)
- ✓ Cross-dataset deduplication and leakage prevention implemented
- ✓ Stage datasets prepared (`stage1_train.csv`, `stage2_train.csv`, `stage3_train.csv`, `stage3_val.csv`)
- ✓ Proper-name glossary extracted (`data/glossary.json`)
- ✓ Config-driven base model selection with approved checkpoints (ARCH-01, validated in Phase 1)
- ✓ Model/tokenizer smoke command with one-step generation path (ARCH-02, validated in Phase 1)
- ✓ Versioned stage-specific training configs and loader CLI (ARCH-03, validated in Phase 1)
- ✓ Tokenizer training/resolver interface preserves `<gap>` and genre tags via manifest contract (TOK-01, validated in Phase 2)
- ✓ Deterministic tokenizer roundtrip verification command for representative transliterations (TOK-02, validated in Phase 2)
- ✓ Deterministic stage data loading, seq2seq batching, and split integrity checks (DATA-01/02/03, validated in Phase 3)

### Active

- [ ] Implement training infrastructure for staged fine-tuning
- [ ] Train and evaluate Stage 1, Stage 2, and Stage 3 models with reproducible metrics
- [ ] Integrate glossary-aware decoding/post-processing for proper name accuracy
- [ ] Generate competition-ready test predictions and submission artifacts

### Out of Scope

- Production deployment APIs and online serving infrastructure — not needed for current research/competition goal
- New data collection/annotation campaigns — current focus is exploiting the prepared corpus effectively
- Generalized multilingual productization beyond Akkadian-English — would dilute immediate competition objective

## Context

- Existing assets:
  - `data_preprocessing.ipynb`
  - `data/stage1_train.csv`, `data/stage2_train.csv`, `data/stage3_train.csv`, `data/stage3_val.csv`
  - `data/glossary.json`
  - `explanation.md`, `data.md`, `TRAINING_PLAN.md`
- Candidate model families researched: mBART-50, mT5-base, NLLB-200-distilled
- Research direction selected: architecture-focused research for low-resource historical NMT
- Training strategy: curriculum fine-tuning with learning-rate decay (1e-4 -> 5e-5 -> 1e-5)
- Current state: Phase 2 complete — tokenizer training/resolution and roundtrip integrity gates are now implemented and test-covered.
 - Current state: Phase 3 complete — deterministic data loaders, seq2seq collation, and integrity checks are implemented and test-covered.

## Constraints

- **Tech stack**: PyTorch + HuggingFace Transformers + SentencePiece + sacrebleu — aligns with current project dependencies
- **Data integrity**: Competition test leakage must remain zero — evaluation validity depends on strict separation
- **Reproducibility**: Deterministic seeds and logged configs required — comparisons across stages must be trustworthy
- **Compute**: Single-GPU-friendly execution path required — architecture and batch settings must fit realistic hardware
- **Scope**: Current milestone is training/evaluation/submission workflow, not deployment — keeps effort focused on measurable outcomes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep preprocessing as validated baseline and focus roadmap on implementation/training | Data prep is complete and documented; highest leverage is model execution quality | ✓ Good |
| Use detailed phase slicing | User requested granular implementation phases for better control and tracking | ✓ Good |
| Run architecture research before roadmap decisions | Model choice strongly affects memory, quality, and training protocol | ✓ Good |
| Prefer mBART-50 as primary candidate with mT5-base fallback | Strong low-resource MT performance plus practical fallback when memory is constrained | ✓ Adopted in Phase 1 configs |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -> still the right priority?
3. Audit Out of Scope -> reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-22 after Phase 3 completion*
