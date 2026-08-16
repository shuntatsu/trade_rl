# Causal Alpha V3 Signal Scope Contract Fix Implementation Plan

> **Superseded:** This minimal V1-scope plan is retained only as historical TDD context. It was superseded by `2026-08-16-causal-alpha-v3-signal-contract-v2.md` and must not be used as the current implementation contract. In particular, active V2 code uses `minimum_independent_episode_count` and `minimum_raw_scope_coverage`; the ambiguous `minimum_scope_count` / `minimum_scope_coverage` names below describe the pre-V2 state only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for behavioral changes and superpowers:verification-before-completion before completion claims.

**Goal:** Make the maintained V3 authored configuration structurally compatible with the chronological independent-scope Signal Gate without weakening signal-quality statistics or treating correlated symbol copies as independent evidence.

**Architecture:** Preserve chronological episode clustering in `evaluate_causal_alpha_v3_signal_gate_clustered`: same-episode symbol metrics contribute one independent scope to the count and one cluster value to each bootstrap statistic. Fix the authored example threshold to the available number of signal episodes and fail fast when any research config requires more independent scopes than `signal_contract_count` can provide.

**Tech Stack:** Python 3.12, pytest, existing strict V3 config contracts and GitHub Actions.

## Global Constraints

- Do not change clustered Signal Gate numerical semantics.
- Do not relax rank-IC, top/bottom-spread, direction-accuracy, or coverage thresholds.
- Do not change V3 candidate/model/controller behavior.
- Do not change holdout, economic selection, admission, reward, risk, or execution contracts.
- Preserve the architecture-hardening invariant that same-episode symbol duplication does not increase the independent scope count.
- Reject structurally impossible authored configs before any signal computation is started.

---

### Task 1: Reproduce the invalid authored-config contract

**Files:**
- Modify: `tests/workflows/test_universal_causal_alpha_v3_runner_config.py`

**Test Oracle:** A config with `signal_contract_count=2` and `minimum_scope_count=3` is impossible because the clustered gate can produce at most two independent chronological episode scopes. Config construction must fail before the runner starts.

- [ ] Add the failing strict-config regression test.
- [ ] Run exact-head CI and confirm RED because the current config accepts the impossible combination.

### Task 2: Fail fast on impossible independent-scope requirements

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_config.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_runner_config.py`

**Implementation:** In `CausalAlphaV3ResearchConfig.__post_init__`, require `signal_gate.minimum_scope_count <= nested_selection.signal_contract_count` with an explicit error identifying both fields.

- [ ] Add the cross-field validation.
- [ ] Re-run targeted/full CI and confirm the regression test passes.

### Task 3: Repair the maintained example

**Files:**
- Modify: `examples/binance/universal-causal-alpha-v3-research.json`
- Modify: `tests/test_causal_alpha_v3_runner_example_contract.py`

**Implementation:** Change `minimum_scope_count` from 24 to 8, matching the eight predeclared signal contracts. This changes only the impossible independent-scope threshold; coverage remains 1.0 and all lower-CI thresholds remain 0.0.

- [ ] Assert the maintained example requires all eight independent signal episodes.
- [ ] Verify the example loads under the new fail-fast contract.

### Task 4: Falsification and completion review

- [ ] Verify same-episode symbol copies still do not increase `minimum_scope_count` evidence.
- [ ] Verify the original clustered evaluator source is unchanged.
- [ ] Verify `minimum_scope_count=9` with eight signal contracts fails config loading.
- [ ] Verify no economic-selection, holdout, reward, risk, execution, model, or controller files changed.
- [ ] Review final diff for unrelated changes.
- [ ] Confirm exact-final-HEAD tests, static checks, compatibility, training image, coverage, and required CI checks before marking the PR ready.
