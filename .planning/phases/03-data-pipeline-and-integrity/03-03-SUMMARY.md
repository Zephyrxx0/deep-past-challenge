---
phase: 03-data-pipeline-and-integrity
plan: 03
subsystem: data
tags: [pandas, data-validation, integrity-checks, split-contamination, tdd]

# Dependency graph
requires:
  - phase: preprocessing (pre-GSD)
    provides: Stage CSV files with proper split boundaries
provides:
  - Split integrity validation preventing train/val overlap
  - Provenance checks for required dataset files and columns
  - CLI command for granular integrity checks (all, provenance, splits)
  - Fail-fast validation with actionable error messages
affects: [04-stage1-training, 05-stage2-training, 06-stage3-training, verification]

# Tech tracking
tech-stack:
  added: [pandas set operations for overlap detection]
  patterns: [TDD with RED-GREEN phases, fail-fast validation, CLI with check type selection]

key-files:
  created:
    - tests/phase3/test_data_integrity.py
    - scripts/check_data_integrity.py
  modified: []

key-decisions:
  - "Use transliteration as overlap key for stage1/3, first_word_spelling for stage2 (different column structures)"
  - "Check stage3_val against both stage1_train and stage2_train to prevent indirect leakage"
  - "Check stage3_train against competition test set if it exists (prevents test contamination)"
  - "Use ASCII output markers ([OK], [ERROR]) instead of Unicode for Windows compatibility"

patterns-established:
  - "Fail-fast validation: raise specific exceptions (FileNotFoundError, ValueError) with actionable messages"
  - "Overlap detection pattern: set intersection with example output in error messages"
  - "Provenance validation pattern: check file existence before reading, validate columns after reading"

requirements-completed: [DATA-03]

# Metrics
duration: 2min
completed: 2026-03-22
---

# Phase 03 Plan 03: Split Integrity Validation Summary

**Fail-fast split/provenance validation preventing train/val overlap and test contamination, with granular CLI checks and actionable error messages**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-22T18:46:10Z
- **Completed:** 2026-03-22T18:48:00Z
- **Tasks:** 2 (TDD with RED/GREEN phases)
- **Files modified:** 2

## Accomplishments
- DATA-03 requirement satisfied: integrity checks detect split contamination and provenance violations
- All 7 contract tests pass: clean integrity, overlap detection, provenance, CLI commands
- Prevents validation leakage (stage3_val vs stage1/2 training data)
- Prevents test contamination (stage3_train vs competition test set)

## Task Commits

Each task was committed atomically following TDD RED-GREEN cycle:

1. **Task 1 (RED)**: Add DATA-03 integrity check contract tests - `3ca76da` (test)
2. **Task 1 (GREEN)**: Implement split/provenance integrity validation - `169937a` (feat)

## Files Created/Modified
- `tests/phase3/test_data_integrity.py` - 7 contract tests validating DATA-03 (clean splits, overlap detection, provenance, CLI)
- `scripts/check_data_integrity.py` - check_provenance() and check_split_integrity() with CLI, handles different stage structures

## Decisions Made
- **Overlap key selection**: Use transliteration for stage1/3, first_word_spelling for stage2 to match different column structures (stage2 uses Sentences_Oare format).
- **Comprehensive validation**: Check stage3_val against BOTH stage1_train and stage2_train to prevent indirect leakage through combined training stages.
- **Test contamination protection**: Check stage3_train against competition/test.csv if it exists (guards against accidental test leakage).
- **Windows compatibility**: Use ASCII markers ([OK], [ERROR]) instead of Unicode (✓, ✗) to avoid Windows console encoding issues.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Windows Unicode console encoding issue**
- **Found during:** Task 1 GREEN phase
- **Issue:** CLI output used Unicode checkmarks (✓, ✗) which caused UnicodeEncodeError on Windows console (cp1252 codec)
- **Fix:** Replaced Unicode symbols with ASCII-safe markers ([OK], [ERROR])
- **Files modified:** scripts/check_data_integrity.py
- **Verification:** CLI tests pass on Windows
- **Committed in:** 169937a (Task 1 GREEN commit - fixed before final commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix for Windows encoding)
**Impact on plan:** Auto-fix necessary for cross-platform compatibility. No scope creep.

## Issues Encountered
None - plan executed smoothly after Windows encoding fix.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Integrity validation ready for use as pre-training gate (run before Stage 1/2/3 training scripts)
- DATA-03 contract tests provide regression protection against future data modifications
- Granular CLI checks (--check provenance/splits) enable targeted debugging

---
*Phase: 03-data-pipeline-and-integrity*
*Completed: 2026-03-22*
