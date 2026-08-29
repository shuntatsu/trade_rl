# Replay Diagnostics Correctness Design

## Status

Implemented on a diagnostics-only follow-up branch; final repository/CI verification remains required before completion.

## Objective

Make replay diagnostics describe simulator-observed execution boundaries and authoritative risk projection facts instead of assuming that the requested target equals the realized position, without changing any strategy action, economic gate, reward, cost, execution, risk, fit, horizon, confirmation, or target-selection behavior.

## Non-goals

This change does not:

- change V8/V9/V10 target generation or numerical constants;
- change Signal, Selection, or Admission thresholds;
- change reward (`100 * net_log_return`), costs, execution resolution, or PreTrade behavior;
- reinterpret V10 r5 as passed;
- change the generic `ActionPathCollapseEvidence` schema or historical V5/V6/V7/V8 replay artifact schemas;
- claim that one post-step weight exactly explains a whole decision interval's PnL;
- replace canonical V7/V8 PnL attribution with a requested/realized approximation;
- introduce one-minute data, BC, or RL.

## Root-cause evidence

1. `evaluate_action_path` previously constructed `ActionPathCollapseEvidence` with `hard_risk_violation=False` unconditionally, while Selection/Admission require zero hard-risk violations.
2. Closed-loop V10 can receive a simulator-observed current weight that differs from its preceding requested target because of execution, risk projection, or market movement. Existing target/submission counters cannot distinguish a genuine new strategy intent from a policy following realized state or reasserting its prior requested target.
3. Existing V7/V8 attribution uses simulator-authoritative step economics but classifies exposure using requested target paths. Stateful execution can include existing-position gap PnL, fills, and post-fill intrabar PnL inside one decision interval, so no single post-step weight can be treated as exact whole-interval exposure.
4. V10 replay leaves did not persist enough execution-boundary evidence to audit requested/risk-constrained/realized divergence or to prove that compact diagnostics were derived from the persisted trace.

## Architecture

`ActionPathEvaluation` gains an immutable `ActionPathExecutionTrace` containing decision-boundary evidence:

- `pre_action_weights`: simulator current weights before the policy/action;
- `risk_constrained_weights`: authoritative final `hybrid_risk.weights` passed to execution;
- `post_step_weights`: simulator current weights returned after the step;
- `applied_risk_scales`: the `hybrid_risk.risk_scale` actually applied for each decision;
- `strategy_intent_changes`: requested target changes that are neither simply following realized state nor reasserting the prior request;
- `realized_state_follows`: the realized state diverged from the prior request and the new request follows that realized state;
- `rebalance_reassertions`: the realized state diverged from the prior request and the policy repeats the prior request;
- `hard_risk_violations`: authoritative final risk projections that violate the maintained PreTrade hard-limit contract.

The trace is observational. It does not alter action generation or execution. The three V10-oriented change classifications remain in the trace/V10 diagnostics only; they are deliberately not added to generic `ActionPathCollapseEvidence`, preventing V5/V6/BC artifact schema drift.

V10 replay leaf schema is bumped from `causal_alpha_v10_replay_leaf_v2` to `causal_alpha_v10_replay_leaf_v3`. Each leaf persists the full execution trace plus compact V10 diagnostics. Resume validates the trace digest, strict boolean types, compact-diagnostics digest, exact semantic reconciliation of every derived compact diagnostic with the persisted trace, and equality between trace `decision_count` and replay metric `decision_count`. Because the artifact store is immutable and schema-strict, an existing v2 leaf in the same output root is rejected rather than overwritten; producing v3 evidence requires a fresh output/artifact root.

Canonical V7/V8 PnL attribution remains unchanged until exact bar-level exposure attribution is available.

## Change-class semantics

For tolerance `tol`, let `current` be the pre-action simulator weight, `action` the current requested action, `previous_action` the preceding requested action, and `active`/`previous_active` the current/preceding active masks. Classification is decision-level over active dimensions.

- First decision: if an active `action != current`, mark a strategy intent change.
- A dimension that is active now but was inactive on the preceding decision has no valid preceding active intent. If its `action != current`, it is a fresh strategy intent change.
- For dimensions active on both decisions:
  - realized-state follow: `current != previous_action` and `action == current`;
  - rebalance reassertion: `current != previous_action`, `action == previous_action`, and `action != current`;
  - strategy intent change: `action != previous_action` and `action != current`.

This prevents an output value emitted while a dimension was inactive from being misused later as evidence of a previously active strategic target.

`realized_state_follow` is intentionally causal-neutral terminology. It does not claim whether the divergence came from price movement, partial fill, minimum-notional rejection, PreTrade suppression, or another maintained execution effect.

## Hard-risk projection oracle

Hard-risk evidence is evaluated at the risk-projection boundary, not from end-of-step market-drifted exposure.

For each decision:

1. read authoritative final `hybrid_risk.weights` from the environment step result;
2. read the `hybrid_risk.risk_scale` actually applied for that decision;
3. compare that projected target with `environment.pre_trade_risk.config`.

A hard-risk projection violation is true when any of these holds beyond `fail_closed_tolerance`:

- `max(abs(risk_constrained_weights)) > max_abs_weight * applied_risk_scale`;
- `sum(abs(risk_constrained_weights)) > max_gross * applied_risk_scale`;
- `applied_risk_scale == 0` and any non-zero projected exposure remains.

Post-step actual weights are still persisted for diagnostics, but ordinary price movement after a valid projection must not create a false Selection hard-risk failure. Conversely, an invalid final projection must be detected even if subsequent market movement happens to bring the post-step weight back inside the cap.

If the maintained replay environment does not expose authoritative risk configuration, risk-constrained weights, or the applied risk scale, evaluation fails closed instead of fabricating verified safety.

## V10 diagnostic identity

Compact diagnostics are a deterministic function of `ActionPathExecutionTrace` and include:

- decision count;
- strategy-intent-change count;
- realized-state-follow count;
- rebalance-reassertion count;
- hard-risk-violation boolean;
- minimum applied risk scale;
- mean absolute pre-action, risk-constrained, and post-step weight;
- maximum absolute post-step weight;
- trace digest.

Resume validation does not trust a self-consistent diagnostics JSON digest alone. It reconstructs the trace, recomputes the canonical diagnostics, and requires exact equality. It also requires the reconstructed trace decision count to equal the replay metric decision count. Therefore changing a derived counter/metric and recomputing only the diagnostics digest, or substituting a self-consistent trace from a different-length replay, is rejected.

## Invariants

- Evaluated actions are identical before and after this change for the same policy/environment inputs.
- Gross returns, net returns, rewards, costs, turnover, trade count, and execution events are unchanged.
- V8/V9/V10 strategy constants and candidate mappings are unchanged.
- Existing generic `submitted_change_count`, `executed_change_count`, and economic gates retain their meanings.
- Generic `ActionPathCollapseEvidence` does not gain V10-specific change counters.
- Historical V5-V8 artifacts are not silently reinterpreted under a changed schema.
- Existing V10 v2 replay leaves are not silently accepted as v3 evidence; a fresh artifact root/replay is required to populate the new trace.
- A V10 trace cannot be resumed against a replay with a different decision count.

## Failure modes

- Missing or malformed current-weight observation: fail closed.
- Missing/malformed `hybrid_risk.weights`: fail closed.
- Missing/invalid applied `risk_scale`: fail closed.
- Missing authoritative `PreTradeRisk` config: fail closed.
- Shape drift between action/current/risk/post-step weights: fail closed.
- Non-boolean event arrays, including string values such as `"false"`: fail closed.
- Realized-state follow miscounted as strategy change: regression test.
- Reasserting an unchanged strategic target miscounted as new intent: regression test.
- Newly active target misclassified using an output from a preceding inactive decision: regression test.
- Normal post-step price drift misclassified as a hard-risk projection violation: regression test.
- Invalid final risk projection hidden by later post-step movement: regression test.
- Diagnostics payload changed with a recomputed self-digest but unchanged trace: resume rejection test.
- Self-consistent execution trace from a different decision count accepted for a replay: resume rejection test.
- Diagnostics changing PnL/reward/cost: regression test.

## Test Oracle

Correctness is observed through:

- exact recorded pre-action/risk-constrained/post-step weight arrays and applied risk scales;
- hand-computable strategy-intent/realized-state-follow/reassertion event vectors, including inactive-to-active transitions;
- hard-risk true/false from explicit `PreTradeRiskConfig`, authoritative projected weights, and applied risk scale;
- normal post-step drift not affecting hard-risk projection status;
- equality of performance/economics with the pre-change action path;
- V10 leaf trace and compact diagnostics identity validation, including semantic tamper rejection and replay/trace decision-count equality;
- unchanged V5/V6/V7/V8 replay/attribution behavior and V10 target/gate behavior.

## Required Test Layers

1. Unit: execution-trace validation, strict boolean validation, change classification including active-mask transitions, hard-risk projection oracle.
2. Integration: `evaluate_action_path` with an explicit maintained-style environment exposing risk and boundary weights.
3. Workflow: V10 leaf write/load/resume identity, semantic-tamper rejection, and decision-count mismatch rejection.
4. Regression: V5-V10 replay/attribution/closed-loop/gate tests.
5. Static: Ruff, Ruff format, affected Mypy, import-linter.
6. Repository comparison: full suite against current main with independently reproduced baseline failures handled symmetrically.
7. Normal GitHub CI on the final PR HEAD.

## Quality Gate

Do not mark complete unless:

- RED tests fail for intended missing/wrong behavior before the corresponding production change;
- all targeted and required regression tests pass after implementation;
- explicit invariance tests show no economic output change;
- V10 diagnostics are persisted, strictly typed, digest-bound, semantically resume-safe, and decision-count bound to the replay metric;
- affected static/architecture checks pass;
- final diff contains no strategy/gate constant changes and no temporary verification helpers;
- generic evidence schema remains unchanged;
- full-suite/build/normal-CI regressions are compared against main;
- independent/falsification review finds no remaining material contract mismatch;
- remaining limitations are documented, especially that exact PnL-by-realized-exposure attribution still requires finer-grained execution evidence and a fresh DB-backed V10 run is required to populate the new diagnostics.
