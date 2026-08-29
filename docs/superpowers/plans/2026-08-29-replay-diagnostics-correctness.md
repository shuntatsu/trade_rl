# Replay Diagnostics Correctness Implementation Plan

> **Execution status:** implementation and source-bearing verification are complete. Exact PR-head CI and repository cleanup evidence are tracked in PR #424 rather than encoded as transient status in this plan.

**Goal:** Persist simulator-observed execution-boundary evidence and trustworthy V10 diagnostics without changing trading economics, generic replay artifact semantics, or fixed research gates.

**Architecture:** Add an immutable execution trace to `ActionPathEvaluation`. Keep V10-oriented change classes in that trace/V10 leaf diagnostics only. Persist V10 replay leaf v3 evidence and validate compact diagnostics by recomputing them from the persisted trace. Preserve canonical V7/V8 PnL attribution because a single decision-boundary weight is not exact whole-interval exposure. Separate generic proposal-reference compatibility from forensic book-state tracing so flat observations remain supported without fabricating realized state.

**Tech Stack:** Python 3.12, NumPy, pytest, GitHub Actions, Ruff, Mypy, import-linter.

**Spec:** `docs/implementation-plans/specs/2026-08-29-replay-diagnostics-correctness.md`

## Global Constraints

- No V8/V9/V10 strategy constants, Signal/Selection/Admission numerical gates, reward, execution, cost, or target behavior changes.
- Do not change generic `ActionPathCollapseEvidence` or historical V5/V6/V7/V8 replay artifact schemas.
- Do not interpret post-step weights as exact whole-interval PnL exposure.
- Hard-risk gate evidence is about the authoritative final risk projection, not ordinary market drift after projection.
- Missing optional observation `current_weights` must not break historical flat-observation generic collapse behavior; forensic trace must use authoritative book state instead.
- Present-but-malformed observation weights, malformed risk evidence, or malformed authoritative book state fail closed.
- RED -> GREEN -> Refactor applies to each contract change.
- Final PR remains Draft/unmerged unless explicitly authorized.

---

### Task 1: Execution-boundary trace, change classification, and reference separation

**Files:**
- `trade_rl/learning/rollout_evaluation.py`
- `tests/learning/test_rollout_evaluation.py`
- `tests/learning/test_rollout_risk_timing.py`
- `tests/learning/test_rollout_active_mask_classification.py`
- `tests/learning/test_rollout_flat_observation_trace.py`

**Implemented trace:**

```python
ActionPathExecutionTrace(
    pre_action_weights=...,
    risk_constrained_weights=...,
    post_step_weights=...,
    applied_risk_scales=...,
    strategy_intent_changes=...,
    realized_state_follows=...,
    rebalance_reassertions=...,
    hard_risk_violations=...,
)
```

- [x] RED: prove missing execution trace.
- [x] GREEN: record aligned immutable boundary states without changing step economics.
- [x] RED: distinguish strategy intent, realized-state follow, and rebalance reassertion.
- [x] GREEN: classify those events at decision level over active dimensions.
- [x] Falsification correction: remove V10 change counters from generic `ActionPathCollapseEvidence` to prevent V5/V6/BC schema drift.
- [x] Falsification correction: rename `passive_weight_drift` to causal-neutral `realized_state_follow`.
- [x] RED/GREEN: reject non-boolean event arrays rather than coercing strings such as `"false"`.
- [x] RED: a target emitted while a dimension was inactive was incorrectly reused as prior active intent when the dimension became active.
- [x] GREEN: track `previous_active`; only continuously active dimensions compare with prior requested intent, while newly active proposed targets are fresh strategy intent.
- [x] Falsification RED: requiring observation `current_weights` broke the maintained flat-observation behavior-cloning audit.
- [x] GREEN: split references. Generic proposal collapse uses valid observation weights when present, otherwise first-decision zero / later previous action. Forensic trace uses valid observation weights when present, otherwise `environment.hybrid.weights`.
- [x] Regression: post-step forensic trace also falls back to authoritative book weights when flat observations omit `current_weights`.

**Oracle:** evaluated actions and gross/net return, reward, cost, turnover, and execution events remain unchanged; generic flat-observation collapse semantics remain compatible; forensic trace records actual book state rather than generic fallback values.

---

### Task 2: Hard-risk evidence at the correct boundary

**Files:**
- `trade_rl/learning/rollout_evaluation.py`
- `tests/learning/test_rollout_evaluation.py`
- `tests/learning/test_rollout_risk_timing.py`

The authoritative oracle is the final risk target returned as `hybrid_risk.weights` plus the `hybrid_risk.risk_scale` actually applied before execution.

- [x] RED: fixed `hard_risk_violation=False` hides violations.
- [x] GREEN: calculate violation from maintained PreTrade hard limits.
- [x] Falsification correction: stop recomputing scale from end-of-step drawdown; use the applied risk scale from the step result.
- [x] Falsification correction: stop gating on post-step actual weight because ordinary market movement can drift weight after a valid projection.
- [x] RED/GREEN: a valid projected `0.05` with later post-step `0.06` at a `0.05` cap is not a hard-risk projection violation.
- [x] RED/GREEN: an invalid projected `0.06` at a `0.05` cap is a violation even if post-step movement later returns actual weight to `0.04`.

**Oracle:** hard-risk status matches the maintained risk-projection contract while post-step realized weight remains available only as diagnostic evidence.

---

### Task 3: Persist V10-owned trace and compact diagnostics

**Files:**
- `trade_rl/workflows/universal_causal_alpha_v10_diagnostics.py`
- `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`
- `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`
- `tests/workflows/test_universal_causal_alpha_v10_resume_identity.py`

- [x] RED: V10 leaf lacked execution trace/diagnostics.
- [x] GREEN: bump only V10 replay leaf `v2 -> v3` and persist full trace plus compact diagnostics.
- [x] Bind trace digest to all boundary weight arrays, applied risk scales, and boolean event vectors.
- [x] Strictly validate boolean trace fields.
- [x] Validate diagnostics content digest and trace identity.
- [x] RED: changing a derived diagnostics counter and recomputing the diagnostics self-digest was still accepted.
- [x] GREEN: derive compact diagnostics through one canonical trace function and require resume payload equality with recomputed values.
- [x] Falsification RED: a self-consistent one-step trace could be resumed against a two-decision replay metric because trace length was not bound to replay identity.
- [x] GREEN: include canonical diagnostics `decision_count` and require equality with `metric.v6_metric.decision_count` on resume.

**Compact diagnostics include:** decision count; strategy-intent-change count; realized-state-follow count; rebalance-reassertion count; hard-risk-violation boolean; minimum applied risk scale; mean absolute pre-action/risk-constrained/post-step weights; maximum absolute post-step weight; trace digest.

**Resume rule:** old V10 v2 leaves are not silently reused as v3 evidence. Because artifact leaves are immutable and schema-strict, a fresh v3 replay uses a new output/artifact root rather than overwriting a v2 leaf in place. A v3 trace must also cover exactly the replay metric's decision count.

---

### Task 4: Regression, capability, static, and repository verification

**Required targeted regression surface:**

```bash
uv run pytest -q \
  tests/learning/test_rollout_flat_observation_trace.py \
  tests/learning/test_rollout_evaluation.py \
  tests/learning/test_rollout_risk_timing.py \
  tests/learning/test_rollout_active_mask_classification.py \
  tests/learning/test_causal_alpha_v10_closed_loop.py \
  tests/learning/test_causal_alpha_v10_closed_loop_falsification.py \
  tests/learning/test_causal_alpha_v10_closed_loop_additional_falsification.py \
  tests/learning/test_causal_alpha_v10_target_trace.py \
  tests/workflows/test_universal_causal_alpha_v5_replay.py \
  tests/workflows/test_universal_causal_alpha_v6_replay.py \
  tests/workflows/test_universal_causal_alpha_v7_attribution.py \
  tests/workflows/test_universal_causal_alpha_v8_attribution.py \
  tests/workflows/test_universal_causal_alpha_v8_replay.py \
  tests/workflows/test_universal_causal_alpha_v10_stage_entry.py \
  tests/workflows/test_universal_causal_alpha_v10_resume_identity.py \
  tests/workflows/test_universal_causal_alpha_v10_gates.py \
  tests/simulation/test_causal_alpha_v10_execution_contract.py
```

Static/capability checks:

```bash
uv run ruff check <changed Python files and affected tests>
uv run ruff format --check <changed Python files and affected tests>
uv run mypy \
  trade_rl/learning/rollout_evaluation.py \
  trade_rl/workflows/universal_causal_alpha_v10_diagnostics.py \
  trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py
uv run lint-imports
uv run python scripts/run_training_capability_audit.py --output <fresh-output>
uv build
```

Verified source-bearing head `f03c6b9e60d5452af341c3eca32302bb8c7c98a1`:

- [x] 69 targeted/compatibility tests passed.
- [x] Ruff affected surface passed.
- [x] Ruff format affected surface passed.
- [x] Affected Mypy passed.
- [x] Import architecture passed: 13 kept, 0 broken.
- [x] Full training capability audit passed, including behavior cloning.
- [x] Package build passed.
- [x] PR diff contained no `trade_rl/learning/evaluation.py` change and no temporary `.github` verification helper.
- [x] Full-suite comparator used the same environment for PR and base main. PR: 4457 passed, 26 skipped, 3 failed. Main: 4444 passed, 26 skipped, 3 failed. The same three pre-existing architecture/documentation failures occurred on both sides; no PR-only full-suite failure was observed.

Exact CI status for subsequent documentation-only cleanup commits is recorded in PR #424. Do not reuse an older successful run as evidence for a newer HEAD.

---

### Task 5: Final falsification and handoff

Review from the original requirements rather than implementation assumptions:

- [x] Requested, generic proposal-reference, forensic pre-action, risk-constrained, and post-step states have distinct semantics.
- [x] Ordinary price movement does not falsely trip the hard-risk Selection gate.
- [x] An invalid final risk projection cannot be hidden by later movement.
- [x] Non-boolean trace payloads are rejected rather than coerced.
- [x] Compact diagnostics cannot be changed and self-rehashed without matching the persisted trace.
- [x] V10-specific counters do not leak into generic V5/V6/BC collapse evidence.
- [x] Inactive output is not treated as prior active intent after reactivation.
- [x] A self-consistent trace with a different decision count cannot be resumed for this replay.
- [x] A v2 leaf cannot be mistaken for v3 evidence.
- [x] Strategy/gate constants and economic output are outside this change and no such production file appears in the final diff.
- [x] Attribution limitations are explicit rather than disguised as realized-PnL attribution.
- [x] Flat-observation compatibility was independently falsified by the Full training capability failure before the reference split and is covered by a dedicated regression after the fix.

Final handoff evidence must record the current PR HEAD, final diff, targeted/full tests, static checks, build, normal CI, PR state, remaining limitations, and the fact that a fresh DB-backed V10 Selection run is still required before the new diagnostics can explain real economic outcomes.
