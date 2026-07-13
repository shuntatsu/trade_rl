# Trade RL Architecture

## Status

`trade_rl` is a research-grade baseline-anchored residual-RL core. It is not authorized for production trading. Architecture quality and empirical profitability are separate gates.

## Responsibility map

```text
trade_rl/
  domain/        immutable dataset, policy, selection and release identities
  artifacts/     canonical serialization, hashing and atomic publication
  data/          market calendar, feature and execution-data contracts
  strategies/    deterministic causal baselines
  risk/          pure hard/soft pre-trade constraints
  simulation/    execution, costs, carry, margin and accounting
  evaluation/    metrics, paired tests, folds, gates and AUM capacity
  rl/            actions, observations, normalization, rewards, environment, policies, training
  workflows/     typed orchestration
  serving/       identity-bound bundles, registry and fail-closed runtime
  cli/           typed configuration entry points
```

The dependency direction remains `cli -> workflows -> serving/rl/evaluation/artifacts`, `serving -> rl actions/artifacts/domain`, `rl -> risk/simulation/strategies/data/evaluation/artifacts/domain`, and `domain -> standard library`.

## Action contract

`baseline_residual_v2` contains `fast_tilt`, `slow_tilt`, `risk_tilt`, optional `alpha_scale`, and zero or more named factor controls. The environment derives the exact dimension from `ActionSpec`; alpha-disabled environments do not expose an unused alpha coordinate. Zero action is exact baseline identity. Training may clip with diagnostics; evaluation and serving can reject out-of-range actions fail closed.

## Baseline contract

Trend baselines distinguish time-series direction from cross-sectional ranking. `auto` uses time-series trend for a one-symbol universe and cross-sectional trend for multiple symbols. Directional modes preserve signal confidence and cash allocation rather than always normalizing to full gross exposure.

## Market and execution contract

Bar timestamps are close times and decisions first execute at the next open. Continuous datasets require regular cadence; session datasets use wall-clock lookup helpers across overnight, weekend and holiday gaps. Execution uses only information available at the decision and execution timestamps. Hybrid and shadow books share one episode RNG stream while different episodes receive distinct deterministic seeds.

The execution layer models partial fills, fees, spread, impact, random and tail slippage, per-bar constraints, minimum notional, lot/tick rules, borrow, carry, latency, market/limit orders, margin, dividends, splits and delisting recovery. Economic failures become structured terminal transitions; malformed market data remains an exception.

## Risk ordering

Turnover is a soft operational constraint. Concentration, gross leverage and emergency drawdown limits are hard constraints applied afterward and validated again. A hard deleveraging requirement may override turnover, and every projection exposes reasons and L1 distance.

## Observation and normalization

Observation schema v3 carries feature values, feature-level availability/staleness/reasons, active/tradable state, all baseline and factor inputs, current and requested portfolios, fill/cost/capacity state, cash/net/gross/margin state and previous action. Remaining time is included only for an explicitly finite-horizon MDP. Normalization statistics are fitted on an explicit train range, frozen elsewhere, content-addressed, and preserve mask/categorical coordinates exactly.

## Reward contract

Reward schema v3 prioritizes absolute log-wealth growth. Baseline-relative growth is secondary. Drawdown is penalized only on new excess drawdown beyond a dead zone. Baseline underperformance uses a fixed real-time rolling window, tolerance and progressive hinge. Terminal penalties are continuous in equity shortfall rather than fixed jackpots. Every component is returned for audit.

## Training, evaluation and serving

Training verifies decision cadence, action names/spec digest, observation size, environment digest and AUM. Shared per-asset encoding plus masked attention supports inactive assets and avoids arbitrary symbol-order dependence. PPO, SAC, TD3 and TQC are compared only under the same sealed walk-forward protocol.

Serving bundle v3 binds action size/names/spec digest, observation schema/size, environment, normalizer, alpha/factor artifacts and release identity. Runtime actions must be finite, correctly shaped and inside `[-1, 1]`; violations fail closed.
