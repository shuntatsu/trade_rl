# Universal Trade RL U2 Replay Evidence Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining Task 7C-1 gap between the frozen deterministic Development replay spec and the current implementation without changing U1 economics, defining gross metrics, or opening real Development/Admission data.

**Architecture:** Keep `UniversalTradeRLU2DevelopmentReplaySession` as the sole replay owner and the frozen U1 `UniversalTradeEnvironment` as the sole Risk/Execution/Accounting authority. Harden the immutable replay evidence so it explicitly binds canonical scope boundaries and retains step-aligned action/target/execution/risk diagnostics sufficient for later Selection metrics. Correct the synthetic early-termination oracle to use a positive-wealth U1 economic termination instead of an invalid zero-wealth trajectory.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pytest, Ruff, MyPy, import-linter, GitHub Actions.

**Spec:**
- `docs/implementation-plans/specs/2026-09-05-universal-trade-rl-u2-deterministic-development-replay-design.md`
- `docs/implementation-plans/specs/2026-09-05-universal-trade-rl-u2-development-replay-seed-amendment.md`
- `docs/implementation-plans/specs/2026-08-31-universal-trade-rl-u1-observation-reward-design.md`
- `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u2-base-ppo-selection-design.md`

## Quality Contract

### Objective

1. Make Task 7C-1 raw replay evidence explicitly satisfy the frozen written spec.
2. Preserve exact U1 runtime/economic semantics and the exclusive evaluation-stop boundary.
3. Keep economic early termination observable without constructing a trajectory that violates U1 reward invariants.

### Non-goals

- Do not change U1 Observation/Action/Reward/Risk/Execution/Accounting semantics.
- Do not define gross return, gross wealth, or cost add-back semantics.
- Do not run real Development numeric evaluation.
- Do not open Admission.
- Do not implement Selection/bootstrap/gating in this task.
- Do not change PPO training or checkpoint selection.

### Acceptance Criteria

1. Replay evidence has an explicit schema version and explicitly records `outcome_start_bar_index`, `outcome_stop_bar_index_exclusive`, `evaluation_start_bar_index`, and exclusive `evaluation_stop_bar_index`.
2. Evidence validates `evaluation_start = outcome_start - 1`, `evaluation_stop = outcome_stop`, `runtime_start = evaluation_start`, and `runtime_end = outcome_stop - 1`.
3. Every observed replay decision retains step-aligned normalized action, submitted target, executed target, risk-projected target, realized exposure, requested/filled turnover, filled notional/fill count, rejection reasons, hard-risk evidence, and realized transition/sign-flip evidence.
4. Evidence records and validates `target_change_count`; later submitted/executed-change, hard-risk, rejection, sign-flip, and meaningful-execution metrics can be reconstructed without rerunning the environment.
5. All evidence fields that can influence later Selection metrics participate in the evidence digest.
6. Normal 720 h replay remains exactly 2880 decisions and ends at runtime index `O_stop - 1` by truncation without terminal liquidation.
7. Synthetic economic early termination returns explicit non-normal evidence while final wealth remains positive and finite, consistent with U1 net-log-growth reward.
8. Candidate/Cash/+1/-1 continue to use independent mutable U1 environments with identical canonical scope/dataset/seed/checkpoint identity.
9. No gross metric is introduced.

### Invariants

- `evaluation_stop_bar_index` is metadata-exclusive and is never compared directly to final `current_index`.
- `runtime_end_bar_index == outcome_stop_bar_index_exclusive - 1`.
- U1 valid reward trajectories require positive finite before/after wealth; no epsilon clipping is added.
- Admission remains sealed; real Development artifacts are not loaded by tests.
- Candidate inference remains `deterministic=True`.
- Baselines remain exact normalized actions `0.0`, `+1.0`, `-1.0`.
- Mutable environment/base-environment identity is never shared across replay variants.

### Failure Modes

- Boundary off-by-one hidden by a digest-only scope reference.
- Step evidence shorter/longer than `observed_decision_count`.
- Submitted/executed/risk/realized stages accidentally conflated.
- Rejected executions counted without reason evidence or reason/event counts disagree.
- Hard-risk metadata missing or a projected target violates the frozen hard limits.
- Target-change or sign-flip counts disagree with the retained trace.
- Economic termination fixture produces zero/non-finite wealth and raises from U1 reward instead of returning a valid terminal transition.
- Normal time-limit replay is mislabeled as economic termination or performs terminal liquidation.
- Evidence tampering leaves digest unchanged.

### Risk

High for research validity: missing or ambiguous raw evidence can make later U2 Selection metrics unreconstructable or silently misclassify execution/risk behavior. Runtime/economic changes are intentionally excluded to keep blast radius low.

### Test Oracle

- Canonical scope object for metadata boundaries.
- Actual `UniversalTradeEnvironment.reset/step` state for runtime indices and Gymnasium flags.
- `info["submitted_target"]`, `info["executed_target"]`, `info["hybrid_risk"]`, `info["hybrid_execution"]`, and `info["effective_filled_weights"]` for step lifecycle evidence.
- Final `BookState` for wealth/cost/funding/borrow/fill/trade reconciliation.
- Recomputed counts/digest from retained evidence for tamper detection.

### Required Test Layers

- Unit/contract: dataclass validation, digest binding, boundary relationships, trace/count reconciliation.
- Integration: actual synthetic U1 environment, 2880-step normal replay, candidate and baselines, positive-wealth economic early termination.
- Falsification/regression: malformed action, shared environment, U1 contract drift, evidence tampering, missing/misaligned execution/risk evidence.
- Static: Ruff, Ruff format, MyPy, import architecture and repository-required static checks.
- Repository: full test suite and triggered compatibility/build/audit workflows.

### Quality Gate

Task 7C-1 is not complete until the exact final HEAD satisfies the Acceptance Criteria, targeted/integration/falsification tests, repository static checks, full required CI, self-review, and independent/falsification review. Any failed/skipped required check or unverified spec item keeps the task incomplete.

---

### Task 1: Correct the economic-termination oracle

**Files:**
- Modify: `tests/integrations/test_universal_trade_rl_u2_replay.py`
- Modify: `tests/integrations/test_universal_trade_rl_u2_replay_runtime.py`

**Interfaces:** Existing U1 environment and Task 7C-1 replay API only.

- [ ] **Step 1: Preserve the current RED as evidence of the invalid zero-wealth fixture**

The current `fee_rate=1.0` path reaches a fill, drives wealth to zero, and fails in `universal_net_log_growth_reward()` because U1 requires positive finite wealth. Record this as a fixture defect, not a production replay defect.

- [ ] **Step 2: Replace the fixture with a positive-wealth drawdown-stop path**

Use zero synthetic price drift for deterministic pre-trade admission and a high but sub-100% fee (fixed `0.25`) so the first actual long fill plus emergency deleveraging crosses the maintained drawdown stop while leaving positive finite wealth.

- [ ] **Step 3: Keep execution assertions strong**

Continue asserting that the +1 execution intent has positive requested/fill notional, `rejected_count == 0`, and `fill_count > 0`. Assert the returned terminal reason is economic and the evidence is non-normal with fewer than 2880 decisions.

- [ ] **Step 4: Run the exact U2 replay integration pair**

Run:

```text
uv run pytest -q tests/integrations/test_universal_trade_rl_u2_replay.py tests/integrations/test_universal_trade_rl_u2_replay_runtime.py
```

Expected: all replay integration tests pass before evidence hardening begins.

---

### Task 2: Freeze missing raw-evidence requirements as RED tests

**Files:**
- Create: `tests/integrations/test_universal_trade_rl_u2_replay_evidence_contract.py`
- Modify: `.github/workflows/u2-contracts.yml`

**Interfaces:** `UniversalTradeRLU2ReplayEvidence` and `UniversalTradeRLU2DevelopmentReplaySession.replay()`.

- [ ] **Step 1: Add explicit scope-boundary assertions**

For a real synthetic U1 Cash replay, require schema version plus all four canonical boundary indices and exact runtime relations.

- [ ] **Step 2: Add step-evidence coverage assertions**

Require `len(step_evidence) == observed_decision_count` and verify each step exposes submitted/executed/risk/realized stages, turnover/fill/rejection evidence, hard-risk evidence, and transition class.

- [ ] **Step 3: Add reconstructability assertions**

Recompute target changes, sign flips, hard-risk violations, rejection totals, fill totals, and meaningful-execution inputs from retained raw evidence. Require aggregate fields, where present, to reconcile exactly.

- [ ] **Step 4: Add the new file to the maintained U2 focused workflow and run RED**

Expected: tests fail because the current evidence class does not expose the normative boundary and lifecycle fields. Existing replay tests must remain green.

---

### Task 3: Implement minimal immutable evidence hardening

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_replay.py`

**Interfaces:**
- Extend `UniversalTradeRLU2ReplayEvidence` without changing replay request/session semantics.
- Add one immutable per-decision replay-step evidence structure local to the U2 replay workflow.

- [ ] **Step 1: Add schema and explicit canonical boundaries**

Bind the scope fields directly into the evidence payload/digest and validate their exclusive/inclusive relations.

- [ ] **Step 2: Capture lifecycle stages from maintained U1 `info`**

For every decision capture normalized action, submitted target, execution-intent target, post-risk target, realized exposure, requested/filled turnover, filled notional/fill count, rejection reasons, hard-risk evidence, and transition class.

- [ ] **Step 3: Fail closed on malformed maintained evidence**

Reject non-finite/wrong-shaped stage vectors, missing hard-risk limit metadata, mismatch between `rejected_count` and rejected order events, invalid rejection reasons, and inconsistent trace lengths.

- [ ] **Step 4: Reconcile derived counters**

Use a fixed `1e-6` diagnostic tolerance. Recompute target changes/sign flips/hard-risk violations from step evidence and require stored aggregates to match.

- [ ] **Step 5: Bind all later-selection inputs into the evidence digest**

`to_payload(include_digest=False)` must contain the schema, all scope boundaries, all per-step raw lifecycle fields, and aggregate counts. Reusing an old digest after any mutation must fail.

- [ ] **Step 6: Run focused RED→GREEN tests**

Run the new evidence-contract test plus existing workflow and integration replay tests until all pass without changing their acceptance conditions.

---

### Task 4: Falsification and exact-head verification

**Files:**
- Modify production/tests only for independently reproduced defects.

- [ ] **Step 1: Falsify trace alignment and tamper resistance**

Mutate one boundary, one step target, one rejection reason, one hard-risk flag, and one counter while keeping the old digest; each must fail reconstruction or change digest.

- [ ] **Step 2: Falsify off-by-one semantics**

Prove that setting runtime end to exclusive `evaluation_stop_bar_index` or final current index to that exclusive boundary is rejected.

- [ ] **Step 3: Run U2 focused contract workflow locally/CI**

Require workflow/unit/integration/SB3/router regressions on the exact head.

- [ ] **Step 4: Run repository static and full gates**

Require Ruff, Ruff format, MyPy, import architecture, full tests, and all triggered exact-head CI/compatibility/audit workflows.

- [ ] **Step 5: Self-review and independent falsification review**

Review the original spec, final diff, assertions, runtime behavior, and CI from scratch. Search specifically for evidence fields that later metrics could require but are still absent or ambiguous.

- [ ] **Step 6: Report guarantees and limitations**

Keep explicit: synthetic-only verification, no real Development evaluation, no real PPO checkpoint evaluation, gross metrics undefined, Selection not implemented, Admission sealed, Production `NO-GO`.
