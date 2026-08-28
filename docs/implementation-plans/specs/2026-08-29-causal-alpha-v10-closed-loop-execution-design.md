# Causal Alpha V10 Closed-Loop Execution Design

## Objective

Make the V10 `hierarchical_wave` candidate decide from the simulator-authoritative realized position on every replay step, so downstream PreTrade/risk/execution changes cannot leave the V10 state machine reasoning from a stale requested target.

## Non-goals

- Do not change V8/V9 control target behavior.
- Do not change Selection or Admission numerical gates.
- Do not change the 4-hour fast horizon, 72-hour slow horizon, entry confirmation count, exit confirmation count, or slow-neutral expiry count.
- Do not add one-minute data, holdout tuning, BC, or RL changes.
- Do not redesign `ResidualMarketEnv` or the generic `evaluate_action_path` API.
- Do not claim economic improvement until a new DB-backed immutable V10 Selection is run.
- Do not fix unrelated repository-wide CI baseline debt in this change.

## Root cause and residual failure modes

The first V10 execution-contract fix stopped fresh targets below the static PreTrade entry floor and stopped soft-liquidity target churn. It still compiled the whole hierarchical target path before replay. The compiler's internal `current` therefore represented the last requested target, while `PreTradeRisk`, drawdown scaling, execution limits, liquidation, or partial execution could make the realized portfolio different.

A second conflict exists for hard V10 `risk_weight_caps`. A same-direction partial reduction can be rewritten to the current position by PreTrade when the target is below `entry_threshold`, or suppressed by `no_trade_band`. A hard-risk reduction must not become a repeated ineffective request.

The old V10 cadence also used episode-local row offsets (`offset % 16`) rather than the absolute decision index. This can change whether an identical market decision is a 4-hour decision when an episode is sliced at a different phase.

Finally, `execution_contract_hold` mixed two different diagnostics: a flat entry below the execution floor and a held position above a soft liquidity capacity.

## Architecture

### 1. Stateful V10 hierarchy policy

Add an immutable V10 hierarchy input/spec object containing the precomputed causal arrays and identities used by the hierarchy: decision indices, fast/slow head predictions, one-way costs, liquidity/risk caps, state-regime arrays, actionable mask, attribution boundaries, fit/forecast identity, frozen V10 config, initial weight, and execution contract.

A stateful V10 hierarchy policy implements the existing evaluator model interface:

```python
predict(observation, deterministic=True) -> tuple[np.ndarray, object]
```

The policy is called once per replay step by `evaluate_action_path(model=...)`. It reads `observation["current_weights"]` and uses that realized one-symbol weight as the current position. The policy never assumes that the prior requested action was achieved.

The default action for an already-held position is the observed current weight. Therefore, if drawdown/risk/execution changes a prior requested `0.10` to realized `0.05`, the next ordinary hold requests `0.05`, not `0.10`. This prevents the closed-loop from continually fighting a simulator-authoritative reduction.

A realized transition from non-zero to flat resets held-state confirmation counters. An impossible unrequested sign flip fails closed instead of silently reinterpreting state.

### 2. Absolute 4-hour cadence

A fast decision is owned by the absolute decision index:

```python
decision_index % config.fast_horizon_decisions == 0
```

not by the episode-local row offset. Slicing the same absolute decision sequence at a different episode start must not change the cadence classification of an overlapping decision.

### 3. Full PreTrade identity, minimal policy semantics

Resolve the actual environment `PreTradeRiskConfig` into a frozen V10 execution-contract value. Its identity includes every maintained config value that can change replay behavior:

- `max_gross`
- `max_abs_weight`
- `max_turnover`
- `entry_threshold`
- `exit_threshold`
- `no_trade_band`
- `drawdown_start`
- `drawdown_stop`
- `emergency_turnover_override`
- `fail_closed_tolerance`

The hierarchy policy directly needs `entry_threshold`, `exit_threshold`, and `no_trade_band`; the remaining fields are still bound into the execution-contract digest so a changed runtime cannot reuse a stale replay leaf.

The contract is re-read immediately before replay. Any mismatch fails closed.

### 4. Hard risk-cap behavior

`risk_weight_caps` remain strategy-owned hard caps. If realized absolute exposure exceeds the current cap:

1. Compute the same-direction capped target.
2. If that partial target is executable under the known rebalance hysteresis/band contract, request the partial reduction.
3. If PreTrade would hold/suppress that partial reduction, request flat instead and record `risk_cap_flatten`.
4. Once a risk-cap flatten is requested, keep requesting flat until the realized position is flat; do not cancel the safety action merely because one partial fill moved exposure below the cap.

This policy logic does not override `PreTradeRisk`; PreTrade remains the final safety authority.

### 5. V10-specific diagnostics

Do not encode V10-only diagnostics as new generic V6 reasons. Keep the V6-compatible target reasons on the existing generic vocabulary and attach a V10 hierarchy trace to the V10 target artifact.

The trace distinguishes at minimum:

- `entry_floor_hold`: coherent/economic flat entry is below the executable floor.
- `liquidity_capacity_hold`: held realized exposure is above the soft liquidity capacity and is deliberately not resized.
- `risk_cap_projection`: an executable hard-cap partial reduction was requested.
- `risk_cap_flatten`: a hard-cap partial reduction was not executable under hysteresis/band semantics, so flat was requested.
- `realized_state_reset`: simulator-authoritative flatten reset hierarchy state.

The trace covers every decision and is included in the V10 target identity.

### 6. Restart-safe identity

A hierarchical target path is no longer fully known before replay. Introduce a pre-replay hierarchy policy-input digest that binds all causal inputs and the full execution contract.

Replay leaves store this input digest. Resume checks the current input digest before reusing a leaf. The post-replay V10 target artifact additionally binds the realized closed-loop requested action path and hierarchy trace.

Bump the V10 replay-leaf schema so old open-loop leaves cannot be interpreted as closed-loop leaves.

V8/V9 control leaves continue to use their existing deterministic target-path identity; they are not converted to a stateful policy.

## Acceptance criteria

1. If V10 requests `0.10` and the simulator exposes `current_weights == 0.05` on the next step, the next V10 hold decision is based on `0.05` and does not restore `0.10` without a new strategic transition.
2. A simulator-authoritative external flatten resets the hierarchy's held-state counters; an unrequested realized sign flip fails closed.
3. If `current=0.10`, `risk_cap=0.04`, `entry_threshold=0.10`, `exit_threshold=0.03`, and `no_trade_band=0.05`, V10 requests flat rather than repeatedly requesting the non-executable `0.04` partial reduction.
4. If a hard-cap partial reduction clears entry hysteresis and the no-trade band, V10 requests the capped reduction instead of unnecessarily flattening.
5. Two episode slices containing the same absolute decision index classify that index identically for 4-hour cadence.
6. Flat entry below the effective entry floor is diagnosed as `entry_floor_hold`.
7. Held exposure above a lower soft liquidity cap is diagnosed separately as `liquidity_capacity_hold`.
8. Changing `exit_threshold`, drawdown thresholds, turnover limit, or another bound PreTrade field changes the hierarchy input identity and prevents stale leaf reuse.
9. V8/V9 control target generation and Selection/Admission numerical gates have no behavior change.
10. Existing no-direct-flip, fast/slow confirmation, slow-neutral expiry, causal fitting, reward, and after-cost entry contracts remain intact.

## Invariants

- Reward remains `100 * net_log_return`.
- Each symbol remains an independent long/short position; no symbol-ID feature or symbol-specific learned parameter is introduced.
- PreTrade and simulator execution remain authoritative for realized exposure.
- The hierarchy cannot directly request a long-to-short or short-to-long flip; ordinary reversal still exits before a later opposite entry.
- V8/V9 controls remain immutable comparators.
- All causal predictions, costs, capacities, and regime inputs remain decision-time available; no future execution-row market data is introduced.

## Failure modes and test oracle

| Failure mode | Test oracle |
| --- | --- |
| Requested/realized state divergence | Feed sequential observations with realized weight different from prior request and assert the next requested hold equals realized weight. |
| Hard-cap reduction blocked by hysteresis | Exercise `0.10 -> cap 0.04` under maintained thresholds and assert flat request plus `risk_cap_flatten`. |
| Executable hard-cap reduction flattened unnecessarily | Use a cap above entry threshold and a delta at/above the no-trade band; assert partial reduction. |
| Episode-phase cadence drift | Compare overlapping absolute decision indices from differently shifted arrays. |
| Diagnostic ambiguity | Assert entry-floor and held-liquidity cases produce distinct V10 trace reasons. |
| Stale resume after execution-config change | Change each newly bound PreTrade field and assert policy-input/leaf identity mismatch. |
| State reset hidden by simulator action | Externally flatten after holding and assert state reset; inject an unrequested sign flip and assert fail-closed. |
| Control regression | Compare V8/V9 target-path digests/arrays under unchanged inputs. |

## Required test layers

- Unit: hierarchy state machine, absolute cadence, V10 trace, input identity.
- Integration: `evaluate_action_path(model=...)` with a simulator/fake environment exposing changing `current_weights`.
- Workflow: environment contract resolution, drift rejection, leaf resume identity, V8/V9 unchanged wiring.
- Static: Ruff, Ruff format for newly touched files, Mypy, import-linter, vulture.
- Regression: V6/V8/V9/V10 related suites and repository full suite under the same baseline separation used by the existing PR.
- CI: exact final PR HEAD workflow status, including compatibility/training checks where reached.

## Quality gate

Do not mark the change complete unless all new RED tests were observed failing for the intended missing behavior, the targeted suites pass on final source, static/import checks pass for affected code, the full-suite comparison shows no new failure relative to `main`, the final PR diff is self-reviewed, and remaining CI baseline blockers and the unrun DB-backed 216-leaf Selection are reported explicitly.
