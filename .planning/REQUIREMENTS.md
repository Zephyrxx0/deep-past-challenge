# Requirements: Deep Past Akkadian-English Translation

**Defined:** 2026-03-22
**Core Value:** Deliver reliable Akkadian-to-English translation quality through a reproducible 3-stage training pipeline that avoids leakage and preserves domain adaptation gains.

## v1 Requirements

### Architecture and Configuration

- [x] **ARCH-01**: User can run a single command to select a supported base model (mBART-50, mT5-base, or NLLB-200-distilled) via config.
- [x] **ARCH-02**: User can run a model smoke test that loads the model/tokenizer and generates one sample translation.
- [x] **ARCH-03**: User can run training scripts with stage-specific hyperparameters loaded from versioned config files.

### Tokenization and Data Pipeline

- [ ] **TOK-01**: User can train or initialize a tokenizer that preserves project special tokens (`<gap>`, genre tags) and normalized transliteration forms.
- [ ] **TOK-02**: User can verify encode/decode roundtrip behavior on representative transliteration examples.
- [ ] **DATA-01**: User can load each stage dataset (`stage1_train`, `stage2_train`, `stage3_train`, `stage3_val`) with deterministic sampling.
- [ ] **DATA-02**: User can create batched seq2seq inputs with correct padding, masks, and label handling.
- [ ] **DATA-03**: User can enforce split/provenance checks that prevent train/validation/test leakage during staged training.

### Stage Training

- [ ] **TRN1-01**: User can train Stage 1 from prepared general corpus and save resumable checkpoints.
- [ ] **TRN1-02**: User can evaluate Stage 1 on validation metrics and persist results.
- [ ] **TRN2-01**: User can train Stage 2 from Stage 1 checkpoint with lower learning rate and domain-focused data.
- [ ] **TRN2-02**: User can report Stage 2 improvement against Stage 1 baseline and detect forgetting on Stage 1 validation samples.
- [ ] **TRN3-01**: User can train Stage 3 from Stage 2 checkpoint with genre-conditioned inputs and early stopping.
- [ ] **TRN3-02**: User can select and export best Stage 3 checkpoint based on validation BLEU.

### Evaluation and Glossary Integration

- [ ] **EVAL-01**: User can compute BLEU and chrF for each stage with reproducible command-line evaluation.
- [ ] **EVAL-02**: User can compute genre-specific performance for Stage 3 validation examples.
- [ ] **GLS-01**: User can apply glossary-aware post-processing during inference.
- [ ] **GLS-02**: User can report proper-name accuracy before and after glossary integration.

### Inference, Submission, and Reproducibility

- [ ] **INF-01**: User can run batched inference on held-out competition test data and produce one translation per input.
- [ ] **INF-02**: User can generate a competition-formatted submission file and validate schema completeness.
- [ ] **OPS-01**: User can reproduce a full training/evaluation run with fixed seeds and logged configs.
- [ ] **OPS-02**: User can review run artifacts (configs, metrics, checkpoints, logs) for each stage in standard output directories.

## v2 Requirements

### Advanced Optimization

- **ADV-01**: User can enable optional catastrophic-forgetting controls (EWC or distillation) as experimental toggles.
- **ADV-02**: User can run automated hyperparameter search for stage-specific settings.

### Productization

- **PROD-01**: User can serve trained translation model through a production API.
- **PROD-02**: User can monitor live inference quality and drift in production.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Production API deployment | Current milestone is research/competition execution, not serving infrastructure |
| New data annotation pipeline | Existing preprocessed corpora are sufficient for current objective |
| Multi-language expansion beyond Akkadian-English | Would dilute focus from immediate competition-quality target |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ARCH-01 | Phase 1 | Complete |
| ARCH-02 | Phase 1 | Complete |
| ARCH-03 | Phase 1 | Complete |
| TOK-01 | Phase 2 | Pending |
| TOK-02 | Phase 2 | Pending |
| DATA-01 | Phase 3 | Pending |
| DATA-02 | Phase 3 | Pending |
| DATA-03 | Phase 3 | Pending |
| TRN1-01 | Phase 4 | Pending |
| TRN1-02 | Phase 4 | Pending |
| TRN2-01 | Phase 5 | Pending |
| TRN2-02 | Phase 5 | Pending |
| TRN3-01 | Phase 6 | Pending |
| TRN3-02 | Phase 6 | Pending |
| EVAL-01 | Phase 7 | Pending |
| EVAL-02 | Phase 7 | Pending |
| GLS-01 | Phase 7 | Pending |
| GLS-02 | Phase 7 | Pending |
| INF-01 | Phase 8 | Pending |
| INF-02 | Phase 8 | Pending |
| OPS-01 | Phase 9 | Pending |
| OPS-02 | Phase 9 | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-03-22*
*Last updated: 2026-03-22 after initial definition*
