# Replay Diagnostics Correctness Implementation Plan

> **Execution status:** implementation complete on the working branch; final cleanup, full verification, and PR verification remain.

**Goal:** Persist simulator-observed execution-boundary evidence and trustworthy V10 diagnostics without changing trading economics, generic replay artifact semantics, or fixed research gates.

**Architecture:** Add an immutable execution trace to `ActionPathEvaluation`. Keep V10-oriented change classes in that trace/V10 leaf diagnostics only. Persist V10 replay leaf v3 evidence and validate compact diagnostics by recomputing them from the persisted trace. Preserve canonical V7/V8 PnL attribution because a single decision-boundary weight is not exact whole-interval exposure.

**Tech Stack:** Python 3.12, NumPy, pytest, GitHub Actions, Ruff, Mypy, import-linter.

**Spec:** `docs/implementation-plans/specs/2026-08-29-replay-diagnostics-correctness.md`

## Global Constraints

- No V8/V9/V10 strategy constants, Signal/Selection/Admission numerical gates, reward, execution, cost, or target behavior changes.
- Do not change generic `ActionPathCollapseEvidence` or historical V5/V6/V7/V8 replay artifact schemas.
- Do not interpret post-step weights as exact whole-interval PnL exposure.
- Hard-risk gate evidence is about the authoritative final risk projection, not ordinary market drift after projection.
- RED -> GREEN -> Refactor applies to each contract change.
- Final PR remains Draft/unmerged unless explicitly authorized.

---

### Task 1: Execution-boundary trace and change classification

**Files:**
- `trade_rl/learning/rollout_evaluation.py`
- `tests/learning/test_rollout_evaluation.py`
- `tests/learning/test_rollout_risk_timing.py`
- `tests/learning/test_rollout_active_mask_classification.py`

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

**Oracle:** evaluated actions and gross/net return, reward, cost, turnover, and execution events remain unchanged; only additional observational trace is produced.

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
- [x] Falsification RED: a self-consistent one-step trace could be resumed against a two-decision replay metric because trace length was not bound to replay identity (`1 failed / 3 passed`).
- [x] GREEN: include canonical diagnostics `decision_count` and require equality with `metric.v6_metric.decision_count` on resume.

**Compact diagnostics include:**
- decision count;
- strategy-intent-change count;
- realized-state-follow count;
- rebalance-reassertion count;
- hard-risk-violation boolean;
- minimum applied risk scale;
- mean absolute pre-action/risk-constrained/post-step weights;
- maximum absolute post-step weight;
- trace digest.

**Resume rule:** old V10 v2 leaves are not silently reused as v3 evidence. Because artifact leaves are immutable and schema-strict, a fresh v3 replay uses a new output/artifact root rather than overwriting a v2 leaf in place. A v3 trace must also cover exactly the replay metric's decision count.

---

### Task 4: Regression and static verification

**Required targeted regression surface:**

```bash
uv run pytest -q \
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

Static checks:

```bash
uv run ruff check <changed Python files>
uv run ruff format --check <changed Python files>
uv run mypy \
  trade_rl/learning/rollout_evaluation.py \
  trade_rl/workflows/universal_causal_alpha_v10_diagnostics.py \
  trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py
uv run lint-imports
```

- [x] Related V5-V10 regression coverage established.
- [x] Ruff issues found and corrected.
- [x] Mypy narrowing issues found and corrected.
- [x] Import-layer contract verified after the implementation.
- [x] Complete targeted/static matrix passed after the active-mask correction: 67 targeted tests plus Ruff, format, affected Mypy, import-linter, and scope invariants.
- [x] Targeted resume/stage tests plus Ruff/format/Mypy passed after the decision-count identity correction.
- [x] Delete temporary `.github/patch_*` helpers used before the final active-mask patch; delete the final active-mask helper before final diff verification.
- [ ] Delete temporary verification workflow after final full comparator.
- [ ] Verify final diff contains no `trade_rl/learning/evaluation.py` change and no temporary helper.
- [ ] Re-run the complete targeted/static matrix on the exact final PR HEAD including the new decision-count test.
- [ ] Run full-suite/build comparator against current main and classify any baseline failures symmetrically on the final PR HEAD.
- [ ] Run normal GitHub CI on final Draft PR HEAD.

---

### Task 5: Final falsification and handoff

Review from the original requirements rather than implementation assumptions:

- [ ] Can requested, risk-constrained, and post-step weights still be confused?
- [ ] Can ordinary price movement falsely trip the hard-risk Selection gate?
- [ ] Can an invalid final risk projection be hidden by later movement?
- [ ] Can non-boolean trace payloads be coerced and accepted?
- [ ] Can compact diagnostics be changed and self-rehashed without changing the trace?
- [ ] Can V10-specific counters leak into generic V5/V6/BC artifacts?
- [ ] Can an inactive output be mistaken for prior active intent after reactivation?
- [ ] Can a self-consistent trace with a different decision count be resumed for this replay?
- [ ] Can a v2 leaf be mistaken for v3 evidence?
- [ ] Did any strategy/gate constant or economic output change?
- [ ] Are attribution limitations explicit rather than disguised as realized-PnL attribution?

Final evidence must record final HEAD, final diff, targeted/full tests, static checks, build, normal CI, PR status, remaining limitations, and the fact that a fresh DB-backed V10 Selection run is still required before the new diagnostics can explain real economic outcomes.
