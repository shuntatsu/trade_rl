# Causal Alpha V10 Execution Contract Fix Design

## Objective

Make the V10 `hierarchical_wave` target compiler emit targets that are executable under the same `PreTradeRisk.no_trade_band` used by the DB-backed simulator, while preserving the frozen Signal -> Selection -> Admission -> BC/RL order and all existing economic gates.

## Non-goals

- Do not change Selection or Admission numerical thresholds.
- Do not change the 72-hour slow horizon, 4-hour fast horizon, or confirmation counts.
- Do not introduce one-minute data.
- Do not modify V8/V9 control behavior.
- Do not open holdout, BC, or RL before the existing gates pass.
- Do not add a tunable V10 no-trade-band hyperparameter.

## Root cause

The DB-backed V10 replay submits target weights to `ResidualMarketEnv`, whose `PreTradeRisk` suppresses ordinary target changes when `abs(target - current) < no_trade_band`. The maintained V10 run uses `no_trade_band = 0.05`.

The V10 hierarchy currently clips a new entry to `min(0.10, liquidity_cap, risk_cap)` without checking that execution contract. A liquidity cap below 0.05 can therefore create a persistent non-zero target that the simulator repeatedly suppresses.

The V10 compiler also applies liquidity-cap shrinkage on every 15-minute row before the 4-hour decision cadence, even though the V10 design says the emitted strategic target is held between 4-hour evaluations. In the current V10 path, `risk_weight_caps` are inherited from the V6/V7 compatibility path and default to 0.25, so the observed mass suppression is driven primarily by liquidity caps, not a dynamic hard-risk cap.

Finally, `one_way_cost_rates` are recorded by V10 but not used to decide whether a coherent fast/slow entry has positive after-cost edge.

## Fixed execution semantics

1. The canonical execution floor is read from the actual replay environment's `pre_trade_risk.config.no_trade_band`.
2. The target compiler receives that value explicitly. It is not authored in `CausalAlphaV10Config`.
3. The compiler matches `PreTradeRisk` exactly: a flat-entry target is execution-eligible when its absolute change is **greater than or equal to** the band; changes strictly below the band are suppressed.
4. A coherent entry below the execution floor remains flat and resets entry confirmation because the observation is not an executable entry observation.
5. Liquidity-cap reductions are strategic/soft sizing changes and may alter the published target only on the existing 4-hour cadence. Sub-band liquidity reductions are held rather than repeatedly emitted.
6. Existing risk-cap projection remains fail-closed and is not weakened. This change does not add a generic bypass around risk or execution controls.

## Cost-aware entry

V10 reuses the already-authored V6 economic hurdle instead of introducing new parameters:

`abs(fast_mean) > fast_uncertainty + 1.5 * one_way_cost_rate + edge_margin`

A coherent fast/slow observation that does not clear this hurdle does not advance entry confirmation and remains flat. The V10 objective evidence includes the same cost multiplier so the recorded score matches the entry rule.

## Identity and provenance

The V6-compatible target path's `config_digest` is changed from the bare V10 config digest to a V10 compiler-contract digest containing:

- V10 config digest
- execution no-trade band
- inherited V6 execution-cost multiplier
- a dedicated schema version

Therefore a replay resume with a different execution band produces a different target-path digest and fails existing immutable-leaf identity checks rather than reusing stale results.

## Files

- `trade_rl/learning/causal_alpha_v6.py`: allow one explicit V10 diagnostic reason for execution-band holds.
- `trade_rl/learning/causal_alpha_v10_hierarchy.py`: enforce execution floor, cadence-aware liquidity sizing, after-cost entry, and compiler identity.
- `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`: resolve the actual environment no-trade band and pass it into the compiler.
- `tests/learning/test_causal_alpha_v10_hierarchy.py`: boundary, cadence, cost, and identity regressions.
- `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`: environment contract resolution and close behavior.
- `tests/risk/test_pretrade.py`: characterization test pinning strict `< band` suppression and equality executability.

## Acceptance criteria

- A coherent flat entry capped to 0.049 remains 0.0 when the execution band is 0.05.
- A coherent flat entry capped to exactly 0.05 may enter when all other conditions pass.
- A coherent entry that fails the existing V6 after-cost hurdle remains flat.
- Liquidity-cap jitter between 4-hour evaluations does not change a held target.
- A material liquidity reduction at a 4-hour evaluation is still applied.
- The execution band is resolved from the same environment used for replay and the environment is always closed.
- Changing the execution band changes the V10 target-path digest/config identity.
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

- Sub-band target loop: detect with exact target arrays and reason evidence.
- Boundary mismatch at 0.05: compare V10 target behavior with `PreTradeRisk` characterization.
- Cadence drift: inject 15-minute liquidity-cap changes and assert target constancy between 4-hour decisions.
- Cost-blind entry: inject a coherent forecast whose gross edge is positive but below the V6 after-cost hurdle and assert no entry.
- Execution identity drift: compile the same forecasts with two execution bands and assert different target path identities.
- Resource leak: resolve the band from a fake environment and assert `close()` is called.

## Required verification

Targeted unit tests, related V10 workflow tests, full pytest/coverage, Ruff, Ruff format, Mypy, import-linter, architecture tests, compatibility jobs, training image build, full training capability gate, final diff review, and GitHub Actions checks on the same final PR HEAD. DB-backed V10 economic Selection is not automatically rerun by CI and remains a separate research run after the code-quality gate passes.
