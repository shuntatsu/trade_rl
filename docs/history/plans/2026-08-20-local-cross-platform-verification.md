# Local Cross-Platform Verification Implementation Plan

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

### Task 1: Reproduce Windows-sensitive contracts in CI

- Modify `.github/workflows/ci.yml` so Ubuntu and Windows compatibility run the three user-reported contracts.
- Confirm RED before production changes.

### Task 2: Preserve audit-report bytes across platforms

- Keep the existing byte assertion as the oracle.
- Replace platform-translating text persistence with explicit bytes.
- Verify on Windows and Linux.

### Task 3: Remove Docker-daemon dependency from provenance argument validation

- Add a stdlib-only provenance validator.
- Test it directly without Docker.
- Invoke the same validator from `Dockerfile.training`.
- Keep the real training-image build/probe CI job as the Docker integration oracle.

### Task 4: Stabilize the real sequence capability probe on Windows without weakening gates

- Use Windows RED diagnostics to locate the unstable boundary.
- Apply only the smallest audit-only correction supported by evidence.
- Preserve target reconstruction, non-collapse, cash-after-cost, and regret admission gates.

### Task 5: Final verification

- Ruff and format check.
- Mypy.
- Import Linter.
- Full Ubuntu tests and coverage.
- Windows/Ubuntu compatibility including the three targeted contracts.
- Training image build/probe.
- Final diff and residual-risk review.
