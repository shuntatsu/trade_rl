# Sequence Projection CI Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Continuously prevent restoration of the backend-sensitive projection test and repeat the stable contracts on both supported CI operating systems only when relevant files change.

**Architecture:** A Python AST test owns source-structure enforcement. A narrowly path-filtered workflow owns repeated Ubuntu and Windows execution. Production code remains unchanged.

**Tech Stack:** Python 3.12, AST, pytest, PyTorch, GitHub Actions.

## Constraints

- Base on `main` commit `464c14669bd2355b6922e6813870030bcf6cc745`.
- Do not modify `trade_rl/`.
- Keep external Actions pinned by full SHA.
- Use read-only workflow permissions.
- Run 10 repetitions per operating system.
- Do not trigger for unrelated files.
- Keep the PR Draft and unmerged.

### Task 1: Add the AST contract

**Files:**
- Create: `tests/architecture/test_sequence_projection_stability.py`

- [ ] Parse `tests/rl/test_sequence_policy_core.py` with `ast.parse`.
- [ ] Require the float64 and float32 stable test names.
- [ ] Forbid the old flaky test name.
- [ ] Run:

```bash
pytest tests/architecture/test_sequence_projection_stability.py -q
```

Expected: one passing test.

### Task 2: Add the targeted matrix

**Files:**
- Create: `.github/workflows/sequence-projection-stability.yml`

- [ ] Trigger only for four relevant paths on push or PR to `main`.
- [ ] Configure Ubuntu and Windows with `fail-fast: false`.
- [ ] Install `dev` and `train-sb3` dependencies.
- [ ] Repeat the AST test and two focused numerical tests 10 times.
- [ ] Upload per-platform diagnostics even on failure.
- [ ] Run workflow-security validation through normal CI.

Expected: 10/10 successful repetitions on both operating systems.

### Task 3: Verify and document

**Files:**
- Create: `docs/verification/2026-07-23-sequence-projection-ci-guard.md`

- [ ] Record the current-main core remediation evidence.
- [ ] Record final exact head and targeted workflow artifacts.
- [ ] Record normal CI and PostgreSQL results.
- [ ] Compare against base and confirm no production file changed.
- [ ] Update the Draft PR body without merging.
