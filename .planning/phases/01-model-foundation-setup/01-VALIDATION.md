---
phase: 1
slug: model-foundation-setup
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-22
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python `unittest` (stdlib) |
| **Config file** | none — stdlib discovery |
| **Quick run command** | `python -m unittest tests.phase1.test_model_selector tests.phase1.test_model_catalog_contract tests.phase1.test_model_smoke_test tests.phase1.test_stage_config_loading -v` |
| **Full suite command** | `python -m unittest discover -s tests -p "test_*.py" -v` |
| **Estimated runtime** | ~40 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m unittest tests.phase1.test_model_selector tests.phase1.test_model_catalog_contract tests.phase1.test_model_smoke_test tests.phase1.test_stage_config_loading -v`
- **After every plan wave:** Run `python -m unittest discover -s tests -p "test_*.py" -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | ARCH-01 | contract | `python -m unittest tests.phase1.test_model_catalog_contract -v` | ✅ / ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | ARCH-01 | unit | `python -m unittest tests.phase1.test_model_selector -v` | ✅ / ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 2 | ARCH-02 | unit | `python -m unittest tests.phase1.test_model_smoke_test -v` | ✅ / ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 2 | ARCH-03 | unit | `python -m unittest tests.phase1.test_stage_config_loading -v` | ✅ / ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/phase1/__init__.py` — package scaffolding for discovery
- [ ] `tests/phase1/test_model_catalog_contract.py` — contract stubs for ARCH-01
- [ ] `tests/phase1/test_model_selector.py` — selector behavior stubs for ARCH-01
- [ ] `tests/phase1/test_model_smoke_test.py` — smoke flow stubs for ARCH-02
- [ ] `tests/phase1/test_stage_config_loading.py` — stage config stubs for ARCH-03

---

## Manual-Only Verifications

All Phase 1 behaviors have automated verification. No manual-only checks required.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
