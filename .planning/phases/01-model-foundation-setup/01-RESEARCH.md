---
phase: 01
name: Model Foundation Setup
status: complete
created: 2026-03-22
sources:
  - .planning/research/SUMMARY.md
  - .planning/research/ARCHITECTURE.md
  - .planning/research/STACK.md
  - .planning/research/PITFALLS.md
  - .planning/research/model_architectures.md
---

# Phase 1 Research — Model Foundation Setup

## Scope

Research what is needed to plan and implement ARCH-01, ARCH-02, and ARCH-03 for a stable model foundation baseline.

## Requirement Mapping

- **ARCH-01**: Provide a single-command model selector with exactly three supported checkpoints:
  - `facebook/mbart-large-50-many-to-many-mmt`
  - `google/mt5-base`
  - `facebook/nllb-200-distilled-600M`
- **ARCH-02**: Provide a smoke-test command that loads model + tokenizer and generates one translation sample.
- **ARCH-03**: Version stage-specific training hyperparameters for Stage 1/2/3 and make them callable from scripts.

## Technical Decisions for Planning

1. **Primary default model**: mBART-50 (`facebook/mbart-large-50-many-to-many-mmt`)
2. **Fallback model**: mT5-base when memory constraints exist
3. **Alternative baseline**: NLLB-200-distilled-600M
4. **Stage LR schedule to encode in config**:
   - Stage 1: `1e-4`
   - Stage 2: `5e-5`
   - Stage 3: `1e-5`
5. **Reproducibility requirement**: fixed seed and versioned config files

## Implementation Guardrails

- Keep configuration declarative (versioned YAML/JSON), not hardcoded constants spread across scripts.
- Ensure smoke test supports lightweight override model IDs for CI sanity checks.
- Keep selection command deterministic and explicit (no heuristic auto-detection).
- Preserve tokenizer special-token compatibility for upcoming Phase 2 (`<gap>`, genre tags).

## Risks to Mitigate in Phase 1

- Heavy checkpoint downloads can make smoke tests flaky/slow.
- Model/tokenizer loading behavior differs across mBART/mT5/NLLB families.
- Configuration drift between stage scripts if values are duplicated.

## Validation Architecture

- Add focused phase tests under `tests/phase1/`.
- Validate selector behavior independently of heavy model loading.
- Validate stage config schema/required keys.
- Smoke script must support both:
  - **real load mode** (for requirement completion)
  - **tiny override mode** (for quick automated verification)

## Output Artifacts Required by This Phase

- Model catalog + active-model config files
- Single-command model selector script
- Stage 1/2/3 versioned training config files
- Smoke test script
- Tests covering config + selector + smoke command wiring
