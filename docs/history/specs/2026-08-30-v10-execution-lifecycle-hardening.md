# V10 Execution Lifecycle Hardening Design

## Objective

Correct the Causal Alpha V10 execution evidence and risk-reduction path without changing the frozen research hypothesis. The change must:

1. derive `hard_risk_violation` from the authoritative risk projection instead of a fixed boolean;
2. preserve the policy -> submitted target -> delayed execution intent -> PreTrade -> feasible target -> realized-weight lifecycle in replay evidence;
3. let explicitly marked reduce-only targets bypass ordinary hysteresis/no-trade suppression while never increasing absolute exposure, gross exposure, or crossing through zero into the opposite sign;
4. use that reduce-only path only for V10 hierarchical risk reduction / explicit exit intent so V8/V9 controls remain unchanged.

## Non-goals

- No change to V10 4h fast horizon, 72h slow horizon, lookbacks, confirmation counts, edge margin, or target magnitude.
- No change to Signal, Selection, Admission, BC/RL numerical gates.
- No change to reward (`100 * net_log_return`), fee model, symbol universe, or symbol-ID prohibition.
- No one-minute data.
- No V9 prediction/calibration/horizon change in this change.
- No claim of profitability until a fresh DB-backed Selection is run.

## Root causes

### Hard-risk evidence is not authoritative

`evaluate_action_path()` currently constructs `ActionPathCollapseEvidence` with `hard_risk_violation=False`. Therefore the Selection field exists but is not derived from the simulator risk projection.

### The lifecycle trace skips the execution-delay boundary

`ResidualMarketEnv.step()` first converts the policy action into `submitted_target`, then applies the configured signal delay and produces `executed_target`. `EnvironmentRiskProjector` receives that delayed target. The existing step trace stores the policy request and the later risk/realized states, but not the submitted and delayed execution-intent targets. This makes a one-decision delayed entry/exit impossible to attribute exactly from the persisted trace.

### Risk-reducing micro adjustments share ordinary rebalance suppression

`PreTradeRisk` applies entry/hold/reversal hysteresis and the no-trade band to every ordinary target. V10 also treats `risk_weight_caps` as a safety sizing boundary. When a required same-direction reduction is smaller than the no-trade band or falls below entry hysteresis, the current V10 hierarchy may request a complete flatten rather than the capped reduction. This can convert small risk corrections into round-trip exits.

## Design

### 1. Explicit reduce-only mask in PreTrade

Add an optional boolean `reduce_only_mask` to `PreTradeRisk.constrain()`.

For each `True` dimension the target is valid only when:

- current exposure is non-zero;
- target has the same sign as current or is zero;
- `abs(target) <= abs(current)` within the maintained tolerance.

Invalid reduce-only adds or flips fail closed.

A valid reduce-only dimension bypasses entry/hold/reversal hysteresis and the ordinary no-trade band. It does not bypass:

- max-turnover soft control;
- max-absolute-weight and max-gross hard controls;
- drawdown scaling;
- emergency flatten;
- downstream portfolio risk;
- execution feasibility or fill constraints.

This is intentionally not inferred from every reduction. Callers must opt in explicitly.

### 2. Carry reduce-only intent through execution delay

Extend `EnvironmentRiskRequest` with an optional reduce-only mask and pass it to `PreTradeRisk`.

`ResidualMarketEnv` gains a narrow API for the next policy submission's reduce-only mask. The environment stores a pending mask alongside the existing delayed target so the mask and target are delayed together. At risk projection time the mask associated with `decision.executed_hybrid_target` is used. The normal `step()` path has an all-false mask and therefore preserves existing behavior.

For V10 hierarchical replay only, a wrapper reads the policy's already-produced hierarchy metadata before environment execution and marks explicit `risk_cap_projection`, `risk_cap_flatten`, and strategic `exit` targets as reduce-only. V8/V9 replay does not use this wrapper.

### 3. V10 risk-cap behavior

For a same-direction observed position above `risk_weight_cap`, V10 requests the capped partial reduction even when the delta is below the ordinary no-trade band. It does not fall back to flat solely because the partial reduction would have been suppressed by ordinary rebalance controls.

`FLATTEN_ON_RISK_BREACH` remains an explicitly separate preregistered boundary mode and continues to request flat.

The V10 compiler/input identity is bumped with a fixed reduce-only execution-contract schema marker so stale replay leaves cannot be reused.

### 4. Authoritative hard-risk evidence

Extend `RiskConstrainedTarget` with the hard `max_abs_weight` used by PreTrade. `EnvironmentRiskProjector` carries it to the final risk result together with `max_gross` and the applied `risk_scale`.

For each evaluated step, compare the final feasible risk target (`hybrid_risk.weights`) against:

- `max_abs_weight * risk_scale`;
- `max_gross * risk_scale`;
- zero exposure when `risk_scale == 0`.

Use the maintained fail-closed tolerance. Any violation sets the per-path canonical `hard_risk_violation=True`. Post-step market-weight drift is not used as the hard-risk oracle.

### 5. Lifecycle evidence

Bump the generic action-path step-trace schema with backward decoding support for v1. New traces add at least:

- policy requested targets;
- submitted targets;
- delayed execution-intent targets;
- PreTrade targets;
- final feasible/risk-projected targets;
- realized weights;
- applied risk scale;
- per-step hard-risk violation;
- PreTrade/risk reason tuples;
- transition class;
- flatten initiator when a non-flat position becomes flat.

Transition classification is based on pre-step realized weight and post-step realized weight, not on the requested target. Flatten initiators prioritize liquidation/emergency/hard-risk evidence, then explicit policy exit/risk flatten, then other risk reasons, otherwise `unexplained`. A V10 forensic trace is not accepted as complete if a realized non-flat -> flat transition is left unexplained.

### 6. Attribution boundary

This change does not redefine canonical V6/V7 economic attribution or Selection thresholds. A 15-minute step may include pre-fill exposure, execution, and post-fill exposure, so post-step weight alone is not exact whole-interval PnL attribution. The new transition evidence is for lifecycle diagnosis and future attribution work.

## Acceptance Criteria

1. `hard_risk_violation` is false for an in-bounds final risk projection and true for a synthetic projection that violates max-abs/gross/applied-risk-scale limits.
2. Post-step price drift cannot create a hard-risk violation when the final risk projection was valid.
3. A micro same-direction reduce-only target (for example `0.1004 -> 0.1000` with no-trade band `0.05`) is preserved through PreTrade rather than held at `0.1004` or flattened.
4. A reduce-only add (`0.1000 -> 0.1004`) fails closed.
5. A reduce-only sign flip fails closed.
6. The same micro change without reduce-only intent remains suppressed by the existing no-trade contract.
7. Emergency flatten remains executable and unchanged.
8. V10 risk-cap projection requests the same-direction cap instead of flat solely because delta is below no-trade/hysteresis thresholds.
9. V10 `FLATTEN_ON_RISK_BREACH` still requests flat.
10. Reduce-only intent follows signal delay: the mask used by PreTrade corresponds to the delayed `executed_target`, not the newest submitted target.
11. New step traces distinguish policy submitted target and delayed execution intent and preserve risk reasons.
12. Synthetic non-flat -> flat transitions have a non-`unexplained` flatten initiator; a deliberately unexplained transition is rejected by V10 forensic validation.
13. V8/V9 control target/action behavior is unchanged under identical inputs.
14. V10 constants and all numerical promotion gates are unchanged.

## Invariants

- Reduce-only cannot add exposure, increase absolute exposure, or directly flip sign.
- PreTrade hard limits remain final and fail closed.
- V10 cannot directly long-to-short or short-to-long flip.
- Signal-delay causality remains unchanged; only matching intent metadata is delayed with the target.
- Reward, fees, Selection gates, holdout separation, and symbol-general modeling remain unchanged.

## Failure Modes

- Reduce-only mask is delayed out of sync with the target.
- A malformed/non-boolean mask is coerced instead of rejected.
- Reduce-only is accidentally enabled for V8/V9 controls.
- Normal alpha reductions bypass no-trade without explicit intent.
- Hard-risk evidence checks post-step market drift rather than the authoritative projection.
- A risk violation is hidden by downstream portfolio reduction or later execution.
- Lifecycle classification labels an exit bar as a flat holding state without identifying its transition.
- New trace schema silently accepts stale V10 leaves as current evidence.

## Test Oracle

Observe and reconcile:

- exact PreTrade output weights and reasons;
- submitted target vs delayed execution intent;
- PreTrade and final risk targets;
- risk scale/max-abs/max-gross;
- pre/post realized weights;
- execution/liquidation/risk reasons;
- canonical collapse hard-risk boolean;
- V10 hierarchy reason and final requested target;
- unchanged V8/V9 control action paths.

## Required Test Layers

- Unit: `PreTradeRisk` reduce-only contract and malformed masks.
- Unit/state machine: V10 partial risk reduction and explicit flat-on-breach.
- Integration: environment risk projector and delayed target/mask alignment.
- Evaluation contract: hard-risk derivation and lifecycle trace.
- Workflow: V10 replay persistence/resume identity and V8/V9 control regressions.
- Static: Ruff, format, Mypy, import-linter, architecture checks.
- Regression: related V5-V10 suites, then full suite/build.
- CI: checks on the exact final PR HEAD.

## Quality Gate

Do not call the work complete unless the RED tests were observed failing for the intended missing behavior, targeted and related suites pass, static/import/build checks pass, the final diff is self-reviewed, falsification cases pass, full-suite results are compared with exact base `main` if repository baseline failures remain, and CI is checked on the same final HEAD. A fresh DB-backed 216-leaf Selection is required for any economic-success claim but is not required to prove the code-correctness fix itself.
