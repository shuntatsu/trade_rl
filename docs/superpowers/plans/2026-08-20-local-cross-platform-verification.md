# Local Cross-Platform Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the maintained developer verification contracts behave consistently on Windows and Linux without weakening Docker-image or behavior-cloning capability checks.

**Architecture:** Keep the training-image job as the real Docker integration oracle. Make fast provenance argument validation executable without a Docker daemon, persist audit report bytes through an explicit byte writer, and add the Windows-sensitive training-audit contracts to the compatibility matrix before changing behavior.

**Tech Stack:** Python 3.12, pytest, GitHub Actions, Docker, Stable-Baselines3, PyTorch.

**Spec:** User-reported full-suite failures on Windows, 2026-08-20.

## Global Constraints

- Do not change behavior-cloning gate thresholds or production/research training profiles.
- Preserve `full_training_capability_audit_v1` and its digest calculation.
- Preserve report bytes as sorted, indented UTF-8 JSON with one LF terminator.
- Preserve Docker provenance validation semantics and `/provenance.valid` payload.
- Preserve the real training-image build/probe CI job.
- Do not merge to `main` in this plan.

---

### Task 1: Reproduce Windows-sensitive contracts in CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: existing compatibility matrix.
- Produces: explicit Windows/Ubuntu execution of the three user-reported contracts.

- [ ] Add the report-byte, sequence-BC, and provenance-validation tests to the compatibility command.
- [ ] Push the test-only commit and confirm Windows RED before implementation changes.

### Task 2: Preserve audit-report bytes across platforms

**Files:**
- Modify: `trade_rl/operations/_training_capability_audit_impl.py`
- Test: `tests/operations/test_training_capability_audit.py`

**Interfaces:**
- Consumes: report dict and existing byte contract.
- Produces: identical LF-terminated UTF-8 bytes on Windows/Linux.

- [ ] Keep the existing byte assertion as the test oracle.
- [ ] Replace platform-translating text persistence with explicit bytes.
- [ ] Verify targeted Windows and Linux tests.

### Task 3: Remove Docker-daemon dependency from provenance argument validation

**Files:**
- Create: `scripts/validate_training_image_provenance.py`
- Modify: `docker/Dockerfile.training`
- Modify: `tests/examples/test_docker_training_assets.py`

**Interfaces:**
- Consumes: five provenance values.
- Produces: exit status plus exact colon-separated marker bytes.

- [ ] Add direct subprocess tests for invalid and valid arguments that do not invoke Docker.
- [ ] Implement stdlib-only validator and invoke it from the provenance stage.
- [ ] Keep the real training-image CI build as integration oracle.

### Task 4: Stabilize the real sequence capability probe on Windows without weakening gates

**Files:**
- Modify only after Windows RED diagnostics identify the unstable boundary.
- Test: `tests/operations/test_training_capability_audit.py`

**Interfaces:**
- Consumes: deterministic audit fixture and existing BC gate contract.
- Produces: real hierarchical sequence BC pass on both compatibility OSes.

- [ ] Use Windows RED diagnostics to locate the platform-sensitive behavior.
- [ ] Apply the smallest audit-only correction; do not change shared gate thresholds.
- [ ] Verify target reconstruction, non-collapse, cash-after-cost, and regret admission on both OSes.

### Task 5: Final verification

- [ ] Ruff and format check.
- [ ] Mypy.
- [ ] Import Linter.
- [ ] Full Ubuntu tests and coverage.
- [ ] Windows/Ubuntu compatibility including the three targeted contracts.
- [ ] Training image build/probe.
- [ ] Review final diff for unrelated changes and document residual risks.
