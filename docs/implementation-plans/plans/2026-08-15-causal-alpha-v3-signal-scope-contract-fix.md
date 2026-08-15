# Causal Alpha V3 Signal Scope Contract Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for behavioral changes and superpowers:verification-before-completion before completion claims.

**Goal:** Restore the intended `minimum_scope_count` contract after chronological episode clustering so the maintained V3 example is structurally passable without weakening signal-quality thresholds.

**Architecture:** Keep raw `(fit, symbol, episode)` scope records as the population used for scope-count and coverage gates. Keep chronological episode clusters as the only inputs to moving-block bootstrap statistics, so correlated same-episode symbol copies do not inflate confidence. Preserve the existing config schema and authored threshold values.

**Tech Stack:** Python 3.12, pytest, NumPy, existing V3 signal-gate contracts and GitHub Actions.

## Global Constraints

- Do not relax rank-IC, top/bottom-spread, direction-accuracy, or coverage thresholds.
- Do not change V3 candidate/model/controller behavior.
- Do not change holdout, economic selection, admission, reward, risk, or execution contracts.
- Preserve `minimum_scope_count` as the count of persisted raw signal scopes, consistent with `expected_scope_count` and `scope_coverage`.
- Bootstrap uncertainty only over chronological episode clusters.
- Add a regression test that fails on the current clustered implementation and passes only when raw-scope counting is restored.

---

### Task 1: Reproduce the scope-count regression

**Files:**
- Modify: `tests/workflows/test_universal_causal_alpha_v3_architecture_hardening.py`

**Test Oracle:** Two chronological episodes with four symbol scopes each provide eight raw scopes but only two independent episode clusters. With `minimum_scope_count=4` and uniformly strong positive cluster statistics, the gate must pass: raw scope count satisfies the count gate while bootstrap remains clustered.

- [ ] Replace the existing single-episode clustering test with a two-episode/four-symbol regression fixture.
- [ ] Run exact-head CI and confirm the test fails with `scope_count` on the current implementation.

### Task 2: Restore raw-scope count semantics

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_v2.py`

**Implementation:** Keep `_episode_clusters(...)` and all bootstrap inputs unchanged. Change only the count gate from `len(rank_values)` to `len(values)` and add a concise comment documenting that count/coverage operate on raw scopes while CI operates on independent chronological clusters.

- [ ] Apply the minimal implementation.
- [ ] Run targeted/full CI and confirm the regression test passes with no new failures.

### Task 3: Falsification and completion review

- [ ] Verify same-episode symbol copies still do not enter bootstrap as independent observations.
- [ ] Verify missing raw scopes still fail `minimum_scope_coverage`.
- [ ] Verify the maintained example `minimum_scope_count=24` is reachable with 9 symbols × 8 signal episodes (=72 raw scopes per fit).
- [ ] Review final diff for unrelated changes.
- [ ] Confirm exact-HEAD required checks before marking the PR ready.
