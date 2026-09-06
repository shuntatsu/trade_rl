# Universal Trade RL U2 Selection Final Gate Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the missing normative U2 Development Selection decision layer so primary-seed core gates, D1/D2/D1+D2 seed robustness, checkpoint anti-cherry-picking, and Admission eligibility are evaluated as one canonical fail-closed artifact.

**Architecture:** Keep the existing same-path replay, Selection leaf metric derivation, symbol reduction, and segmented moving-block bootstrap as lower-level evidence producers. Correct the metric summary boundary to be one `cell × training_seed`, propagate hard-risk/rejection evidence from replay, add deterministic D robustness evaluators over fixed seeds `(0,1,2)`, then build one content-addressed `UniversalTradeRLU2DevelopmentSelectionEvidence` that ANDs every preregistered gate and can select only seed-0's exact final checkpoint. No U1 economic/runtime behavior changes.

**Tech Stack:** Python 3.12, dataclasses, numpy, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u2-base-ppo-selection-design.md`

## Global Constraints

- Production remains `NO-GO`; this work does not authorize real U2 training.
- Admission numeric data remains sealed; Selection only computes whether Admission would be eligible for a later explicit authorization step.
- U1 reward/action/execution/risk/accounting semantics must not change.
- U2 seeds remain exactly `(0,1,2)` and the primary candidate remains seed `0`.
- Selection candidate remains exact-final checkpoint only; no best-seed/checkpoint substitution.
- Primary mandatory cells are exactly `B`, `C1`, `C2`, `D1`, `D2`; cell `A` is diagnostic and `E` is sealed.
- Core gate thresholds are frozen: balanced gross wealth `>1`, balanced net wealth `>1`, median symbol net wealth `>=1`, minimum symbol net wealth `>=1`, positive scope fraction `>=0.50`, CVaR10 `>=-0.01`, turnover p95/day `<=1`, meaningful-execution symbol fraction `==1`, hard-risk violations `==0`, unexplained execution rejections `==0`, and when gross log growth is positive net/gross log-growth retention `>=0.50`.
- D seed robustness is required separately for `D1`, `D2`, and `D1+D2`: median seed symbol-balanced net wealth `>1`, worst seed symbol-balanced net wealth `>=1`, all-seed hard-risk violations `==0`, all-seed turnover p95/day `<=1`, and paired excess moving-block bootstrap lower 95% CI `>0`.
- Bootstrap remains deterministic: 2,000 resamples, seed 0, linear quantiles, circular moving blocks, and no block crossing D1/D2 boundaries.
- Until a normative explained-rejection allowlist/schema exists, any replay execution rejection is conservatively counted as unexplained (fail closed rather than inventing an allowlist).

---

### Task 1: Make Selection summaries one `cell × seed` and retain gate safety evidence

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_selection.py`
- Modify: `tests/workflows/test_universal_trade_rl_u2_selection_metrics.py`
- Modify: `tests/integrations/test_universal_trade_rl_u2_selection_metrics_integration.py`

**Interfaces:**
- Consumes: `UniversalTradeRLU2ReplayEvidence`.
- Produces: `UniversalTradeRLU2SelectionLeafMetrics` with hard-risk/rejection counts and `UniversalTradeRLU2SelectionMetricSummary` bound to exactly one training seed.

- [ ] **Step 1: Write RED tests for seed mixing and replay safety evidence**

Add tests equivalent to:

```python
def test_u2_selection_summary_rejects_mixed_training_seeds() -> None:
    leaves = (_leaf(training_seed=0, ...), _leaf(training_seed=1, ...))
    with pytest.raises(ValueError, match="seed|single|cell"):
        summarize_universal_trade_rl_u2_selection_metrics(leaves=leaves)


def test_u2_selection_leaf_retains_hard_risk_and_rejection_counts(...):
    leaf = build_universal_trade_rl_u2_selection_leaf_metrics(...)
    assert leaf.hard_risk_violation_count == evidence.hard_risk_violation_count
    assert leaf.unexplained_execution_rejection_count == evidence.execution_rejection_count
```

- [ ] **Step 2: Run focused tests and verify RED**

Run the U2 focused workflow/test set. Expected: failure because mixed seeds are currently accepted and the new safety-evidence fields do not exist.

- [ ] **Step 3: Implement minimal seed/safety contract**

Add required immutable count fields, validate them as non-negative integers, include them in artifact payload/digest, propagate from replay evidence, and require one exact training seed in `summarize_universal_trade_rl_u2_selection_metrics`.

- [ ] **Step 4: Verify GREEN**

Run Selection unit + replay integration tests, Ruff, format, and Mypy.

---

### Task 2: Implement primary core gate with exact threshold boundaries

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_selection.py`
- Create: `tests/workflows/test_universal_trade_rl_u2_selection_gate.py`

**Interfaces:**
- Consumes: one `UniversalTradeRLU2SelectionMetricSummary` for primary seed 0.
- Produces: canonical per-cell gate evidence with deterministic rejection reasons and `passed`.

- [ ] **Step 1: Write one RED case per rejection reason**

Cover exact boundary behavior for all preregistered conditions. In particular, equality must pass for `median/minimum wealth ==1`, `positive fraction ==0.5`, `CVaR10 ==-0.01`, `turnover ==1`, and `retention ==0.5`; balanced gross/net wealth require strict `>1`.

- [ ] **Step 2: Verify RED**

Expected: missing gate API/type.

- [ ] **Step 3: Implement minimal `evaluate_universal_trade_rl_u2_primary_cell_gate(...)`**

Require `training_seed == 0`, mandatory cell membership, and return immutable threshold/reason evidence. Never mutate thresholds from observed Development values.

- [ ] **Step 4: Verify GREEN**

Run the gate test file plus existing Selection metric/integrity tests.

---

### Task 3: Implement D1, D2, and D1+D2 fixed-seed robustness

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_selection.py`
- Modify: `tests/workflows/test_universal_trade_rl_u2_selection.py`
- Modify: `tests/workflows/test_universal_trade_rl_u2_selection_gate.py`

**Interfaces:**
- Consumes: D1/D2 leaves for exact seeds `(0,1,2)` and paired-excess reduced segments.
- Produces: three canonical seed-robustness results: `D1`, `D2`, `D1+D2`.

- [ ] **Step 1: RED exact seed/scope closure**

Reject missing/extra seed, mismatched symbol/tile closure across seeds, missing D1 or D2, and unexpected cells.

- [ ] **Step 2: RED robustness thresholds**

For each of D1, D2, D1+D2 verify rejection when median seed balanced net wealth is `<=1`, worst seed wealth is `<1`, any hard-risk violation exists, or any seed's p95 turnover is `>1/day`.

- [ ] **Step 3: RED separate bootstrap gates**

Prove D1 can fail while D2 passes, D2 can fail while D1 passes, and aggregate can fail independently. Preserve no-cross-boundary behavior for D1+D2 aggregate.

- [ ] **Step 4: Implement minimal robustness evaluator**

Compute per-seed symbol-balanced log growth from the selected D cell leaves, derive wealth, max turnover p95 and summed hard-risk count, and pair each robustness scope with its deterministic bootstrap result.

- [ ] **Step 5: Verify GREEN**

Run all Selection/bootstrap tests and static checks.

---

### Task 4: Build canonical final Development Selection evidence

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_selection.py`
- Modify: `tests/workflows/test_universal_trade_rl_u2_selection_gate.py`
- Modify: `.github/workflows/u2-contracts.yml`

**Interfaces:**
- Consumes: exact final checkpoint closure, five primary seed-0 cell summaries/core-gate results, three D robustness results, and the pre-Development/Development identity closure needed to prove Admission stayed sealed.
- Produces: `UniversalTradeRLU2DevelopmentSelectionEvidence`.

- [ ] **Step 1: RED all-cell AND rule**

Build a fully passing fixture, then make exactly one of `B`, `C1`, `C2`, `D1`, `D2`, D1 robustness, D2 robustness, or D1+D2 robustness fail. Assert overall failure, `admission_eligible is False`, and `selected_checkpoint_digest is None`.

- [ ] **Step 2: RED primary checkpoint closure**

Even when seed 1/2 economics are stronger, passing Selection must choose only seed 0's exact-final checkpoint from `UniversalTradeRLU2FinalCheckpointClosure`. Reject incomplete/noncanonical checkpoint closure or substituted checkpoint digest.

- [ ] **Step 3: RED sealed Admission and immutable Production state**

Require authoritative Development lock/evidence proving `admission_numeric_open_count == 0`. Assert `promotion_eligible is False` unconditionally.

- [ ] **Step 4: Implement final artifact**

The constructor/builder must recompute overall pass from child evidence and bind every child digest. `admission_eligible` is true iff every normative condition passes and Admission remained unopened; no caller-supplied boolean may override the result.

- [ ] **Step 5: Put the new gate tests in focused CI**

Add the test file to Ruff/format/static/focused pytest lists in `.github/workflows/u2-contracts.yml` so a broken final gate cannot hide behind the repository-wide suite.

---

### Task 5: Falsification and exact-head verification

**Files:**
- Review only after GREEN; no threshold changes are permitted to make tests pass.

- [ ] **Step 1: Tamper artifacts and recompute outer digests**

Attempt child-summary replacement, seed substitution, omitted cell, duplicated leaf identity, bootstrap-result substitution, selected-checkpoint substitution, hard-risk/rejection suppression, and Admission-open drift. Each must fail closed or produce a failing Selection result as specified.

- [ ] **Step 2: Run required layers**

Run focused U2 contracts, replay integration, static checks, repository full pytest with branch coverage and enforced `--cov-fail-under=80`, critical coverage ratchet, package/uv identity, training capability/image, Ubuntu/Windows compatibility, PostgreSQL Catalog, and Nautilus Capability on one exact HEAD.

- [ ] **Step 3: Independent requirement reconstruction**

Re-read Sections 13–15 of the normative Selection spec and Acceptance Criteria 8–14 without using implementation conclusions as premises. Compare every condition against final code, tests, and CI evidence.

- [ ] **Step 4: Restore stacked PR base**

After exact-head main-base repository verification, retarget PR #434 back to `codex/universal-trade-rl-u2-base-ppo-selection-design` without changing the verified head tree, update the PR body with exact run IDs/results, and leave merge/real training/Admission/Production unauthorized.
