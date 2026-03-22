# Phase 5: Stage 2 Domain Adaptation - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Adapt the Stage 1 general model to OARE domain data while detecting and limiting catastrophic forgetting. This phase delivers:
1. Stage 2 training CLI that loads from Stage 1 checkpoint and trains with domain data
2. Forgetting detection that compares retained Stage 1 performance
3. Evaluation CLI that reports Stage 2 improvement vs Stage 1 baseline

</domain>

<decisions>
## Implementation Decisions

### Checkpoint Sourcing
- **D-01:** Require explicit `--checkpoint` flag pointing to a Stage 1 epoch directory; fail if missing or invalid
- **D-02:** Validate checkpoint with file existence check (model.safetensors, tokenizer.json) + one forward pass before training starts

### Forgetting Detection
- **D-03:** Detect forgetting by running Stage 2 final model on Stage 1 val samples, comparing BLEU/chrF to Stage 1 baseline
- **D-04:** Compute Stage 1 baseline at Stage 2 training start; cache in `stage2_forgetting_baseline.json` in output directory
- **D-05:** Warn (not fail) if BLEU drops >2 points from Stage 1 baseline; training continues regardless

### Training CLI Structure
- **D-06:** Create separate `scripts/train_stage2.py` CLI for clarity
- **D-07:** Extract shared logic (config loading, checkpointing, metrics logging) into `scripts/training_utils.py`; both Stage 1 and Stage 2 CLIs import from this module

### Evaluation Comparison
- **D-08:** Create separate `scripts/evaluate_stage2.py` CLI
- **D-09:** Write `stage2_comparison.json` artifact containing: `{stage1_bleu, stage1_chrf, stage2_bleu, stage2_chrf, delta_bleu, delta_chrf, forgetting_bleu, forgetting_delta}`

### Agent's Discretion
- Exact structure of `training_utils.py` internals (function signatures, helper breakdown)
- Specific error messages and logging verbosity
- Test organization within Phase 5 test module

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Training Infrastructure
- `scripts/train_stage1.py` — Reference implementation for Stage 1 training CLI (checkpointing, resume, metrics logging patterns)
- `scripts/evaluate_stage1.py` — Reference implementation for evaluation CLI (BLEU/chrF computation, metrics artifacts)

### Configuration
- `config/training/stage2.yaml` — Stage 2 hyperparameters (LR 5e-5, 5 epochs, gradient_accumulation=8)
- `config/training/stage1.yaml` — Stage 1 config for comparison

### Data Pipeline
- `scripts/create_dataloader.py` — DataLoader factory with seq2seq collation
- `scripts/data_loader.py` — Stage data loading with `load_stage_data(stage, split)`

### Project Standards
- `AGENTS.md` — Code style, import order, naming conventions, error handling patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `train_stage1.py` training loop structure — can be refactored into `training_utils.py`
- `evaluate_stage1.py` BLEU/chrF computation — can be shared via utils
- `create_dataloader()` — already supports stage parameter, no changes needed
- `check_provenance()` and `check_split_integrity()` — data integrity checks to reuse

### Established Patterns
- Per-epoch checkpointing: `output_dir/epoch_N/` with model, tokenizer, `training_state.pt`
- Metrics logging: JSONL format to `train_metrics.jsonl`
- Dry-run mode: validate config/paths without torch imports
- CLI arg pattern: config defaults + CLI overrides

### Integration Points
- Stage 2 training connects to Stage 1 via `--checkpoint` flag
- Stage 2 evaluation produces `stage2_comparison.json` in output directory
- Forgetting baseline stored alongside other Stage 2 artifacts

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches following Phase 4 patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-stage-2-domain-adaptation*
*Context gathered: 2026-03-23*
