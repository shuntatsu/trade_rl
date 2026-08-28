# Causal Alpha V10 Execution Contract Fix Design

## Objective

Make the V10 `hierarchical_wave` target compiler emit targets that are executable under the same `PreTradeRisk` rebalance contract used by the DB-backed simulator, while preserving the frozen Signal -> Selection -> Admission -> BC/RL order and all existing economic gates.

## Non-goals

- Do not change Selection or Admission numerical thresholds.
- Do not change the 72-hour slow horizon, 4-hour fast horizon, or confirmation counts.
- Do not introduce one-minute data.
- Do not modify V8/V9 control behavior.
- Do not open holdout, BC, or RL before the existing gates pass.
- Do not add tunable V10 execution thresholds.

## Root cause

The DB-backed V10 replay submits target weights to `ResidualMarketEnv`, whose `PreTradeRisk` applies entry hysteresis before the no-trade band. A flat position is suppressed when `abs(target) < entry_threshold`; after hysteresis, ordinary target changes are suppressed when `abs(target - current) < no_trade_band`.

The maintained Binance configuration has `entry_threshold = 0.10` and `no_trade_band = 0.05`. Therefore a flat-entry target in `[0.05, 0.10)` is not executable even though it clears the no-trade band: it is suppressed by `entry_hysteresis`. Equality is executable because both checks use strict `<` comparisons.

The V10 hierarchy clips a fresh entry to `min(0.10, liquidity_cap, risk_cap)` without reading either execution threshold. Any cap below the effective flat-entry floor can therefore create a persistent non-zero target that the simulator repeatedly suppresses.

The previous compiler also followed liquidity-cap changes while a position was held. In the maintained PreTrade contract, a same-direction target below `entry_threshold` is held at the current position rather than partially reduced. Emitting such intermediate targets is therefore not a meaningful strategic action.

Finally, `one_way_cost_rates` are recorded by V10 but were not used to decide whether a coherent fast/slow entry has positive after-cost edge.

## Fixed execution semantics

1. The replay environment is authoritative for `entry_threshold` and `no_trade_band`; neither is authored in `CausalAlphaV10Config`.
2. The flat-entry execution floor is `max(entry_threshold, no_trade_band)`.
3. The compiler matches `PreTradeRisk` strict-boundary semantics: equality with the active floor is executable.
4. A coherent entry below the execution floor remains flat and resets entry confirmation because the observation is not an executable entry observation.
5. Liquidity cap is treated as soft entry/capacity information. Once a V10 position is held, liquidity-cap reductions do not emit intermediate smaller targets; the previous target is held. Strategic exit logic remains the owner of ordinary exits.
6. Existing `risk_weight_caps` projection remains immediate and is not weakened. This change does not add a generic bypass around risk or execution controls.
7. The execution contract is re-read from the actual replay environment before evaluation; a mismatch with the compiler contract fails closed.

## Cost-aware entry

V10 reuses the already-authored V6 economic hurdle instead of introducing new parameters:

`abs(fast_mean) > fast_uncertainty + 1.5 * one_way_cost_rate + edge_margin`

A coherent fast/slow observation that does not clear this hurdle does not advance entry confirmation and remains flat. The V10 objective evidence uses the same V6 objective so recorded economics and entry eligibility share one contract.

## Identity and provenance

The V6-compatible target path's `config_digest` is changed from the bare V10 config digest to a V10 compiler-contract digest containing:

- V10 config digest
- execution entry threshold
- execution no-trade band
- inherited V6 economic-config digest
- a dedicated schema version

Therefore a replay resume with a different execution contract produces a different target-path digest and fails existing immutable-leaf identity checks rather than reusing stale hierarchical results. V8/V9 control target generation remains unchanged.

## Files

- `trade_rl/learning/causal_alpha_v6.py`: allow one explicit V10 diagnostic reason for execution-contract holds.
- `trade_rl/learning/causal_alpha_v10_hierarchy.py`: enforce the flat-entry floor, soft-liquidity holding semantics, after-cost entry, and compiler identity.
- `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`: resolve the actual environment rebalance contract, pass it into the compiler, and fail closed on replay drift.
- `tests/learning/test_causal_alpha_v10_hierarchy.py`: entry-floor, liquidity, cost, state-machine, and identity regressions.
- `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`: environment contract resolution, close, and drift behavior.
- `tests/simulation/test_causal_alpha_v10_execution_contract.py`: compiler-to-`PreTradeRisk` integration at the actual hysteresis/band boundaries.
- `tests/risk/test_pretrade.py`: characterization test pinning strict `< no_trade_band` suppression and equality executability.

## Acceptance criteria

- With `entry_threshold = 0.10` and `no_trade_band = 0.05`, a coherent flat entry capped to `0.099` remains flat.
- A coherent flat entry capped to exactly `0.10` may enter when all other conditions pass.
- With `entry_threshold = 0`, a target exactly equal to a `0.05` no-trade band remains executable.
- A coherent entry that fails the existing V6 after-cost hurdle remains flat.
- Liquidity-cap jitter between 4-hour evaluations does not change a held target.
- A lower soft liquidity cap at a 4-hour evaluation does not emit a smaller target that PreTrade would hold/rewrite.
- The execution contract is resolved from the replay environment and the resolver always closes its temporary environment.
- A replay environment whose contract differs from the compiler contract is rejected before evaluation.
- Changing either execution threshold changes the V10 target-path digest/config identity.
- Existing V10 state-machine tests remain green.
- Existing Selection/Admission gate code is unchanged.

## Invariants

- Reward remains `100 * net_log_return`.
- Per-symbol long/short positions remain independent.
- No symbol ID or symbol-specific learned parameter is introduced.
- No direct position flip is introduced.
- V8/V9 controls and their artifacts remain immutable controls.
- PreTrade hard limits continue to be authoritative and fail-closed.

## Failure modes and test oracle

- Hysteresis/band target loop: compare compiled targets with `PreTradeRisk` at `0.099`, `0.10`, and the zero-entry-threshold `0.05` boundary.
- Soft-liquidity churn: inject liquidity-cap changes and assert the held strategic target does not shrink.
- Cost-blind entry: inject a coherent forecast whose gross edge is positive but below the V6 after-cost hurdle and assert no entry.
- Execution identity drift: compile the same forecasts with two execution contracts and assert different target identities.
- Runtime contract drift: instantiate a replay environment with a different rebalance contract and require fail-closed rejection.
- Resource leak: resolve the contract from a fake environment and assert `close()` is called.

## Required verification

Targeted unit tests, related V10 workflow tests, full pytest/coverage, Ruff, Ruff format, Mypy, import-linter, architecture tests, compatibility jobs, training image build, full training capability gate, final diff review, and GitHub Actions checks on the same final PR HEAD. DB-backed V10 economic Selection is not automatically rerun by CI and remains a separate research run after the code-quality gate passes.
