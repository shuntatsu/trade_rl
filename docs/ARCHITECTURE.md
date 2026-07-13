# Trade RL Architecture

## Status

`trade_rl` is a research-grade baseline-anchored residual reinforcement-learning core. It is not authorized for production trading. The software architecture and the empirical strategy result are evaluated separately; a clean architecture does not make a failed research gate pass.

## Authoritative package

Only `trade_rl` is maintained. The former `mars_lite`, direct-action PPO, legacy scripts, and legacy tests have been removed. Git history is the archive; there is no compatibility layer.

## Responsibility map

```text
trade_rl/
  domain/        immutable identities and state invariants
  artifacts/     canonical serialization, hashing, staging, publication
  data/          causal sources, feature construction and dataset contracts
  strategies/    deterministic baseline strategies
  risk/          pure pre-trade portfolio constraints
  simulation/    execution, costs, funding and accounting
  evaluation/    metrics, paired comparisons, bootstrap, gates, folds
  rl/            residual actions, observations, rewards, environment, training
  workflows/     typed application orchestration
  serving/       immutable bundles, registry activation and runtime snapshots
  cli/           argument parsing and conversion to typed configuration
```

The directory tree reflects real responsibilities. Empty placeholder packages are not added.

## Dependency direction

The permitted direction is:

```text
cli -> workflows -> serving / rl / evaluation / artifacts
serving -> rl actions / observations / artifacts / domain
rl -> risk / simulation / strategies / data / evaluation / artifacts / domain
simulation -> data
strategies -> data
evaluation -> domain
artifacts -> domain
domain -> Python standard library only
```

`trade_rl.domain` does not import NumPy, Gymnasium, Stable-Baselines3, filesystem code, serving code, or CLI code. Training code does not import the serving runtime. Serving code does not import the training backend.

Import Linter enforces these boundaries in CI.

## Causal market data

`trade_rl.data` owns the maintained source-to-dataset path. `CsvMarketDataSource` reads real per-symbol bars, while `MarketDatasetBuilder` resolves a regular union clock and builds causal trailing features. Missing symbol bars are not removed from the market clock. Listing, delisting, source-row presence and source-reported trading status are represented separately.

`MarketDataset` carries:

- ordered symbols and timestamps;
- OHLCV and funding arrays;
- point-in-time `symbol_active` and `tradable` masks;
- feature values, per-feature availability and normalized staleness;
- explicit volume units and base-asset contract multipliers;
- feature-configuration and normalization digests.

The dataset identity is content-addressed over the ordered contracts, feature and normalization configuration, timestamps, all resolved market arrays, active/tradable masks, feature values, availability and staleness. Changing symbol order, a lookback, a normalization parameter, a lifetime, a volume unit, a contract multiplier or any resolved input value changes `dataset_id`.

All maintained feature calculations are prefix invariant: extending a source with future rows cannot alter already-built prefix features. Warm-up and missing data are unavailable observations rather than genuine zero signals.

## Policy model

The maintained action schema is `baseline_residual_v1`:

1. `trend_mix` interpolates from the base trend target toward the fast or slow target.
2. `alpha_budget` controls a separately gated alpha vector.
3. The zero vector is the exact baseline identity action.

There is no maintained direct-action mode.

## Observation contract

`ObservationBuilder` is the only maintained structured policy-input builder. Training and Serving call the same implementation. A decision at row `t` includes only row `t` market state, current book state and deterministic targets computed from history ending at `t`.

Per-symbol observations contain feature values, per-feature availability, per-feature staleness, current active/tradable state, fast/base/slow trend targets, optional alpha, hybrid and shadow weights, and their difference. The observation never includes `tradable[t + 1]` or any other next-row value.

Execution may consult row `t + 1` because it simulates the realized next-open fill. This execution truth remains outside the policy observation. Trend targets exclude instruments that were not active throughout the required lookback.

## Environment timing

One policy action controls one complete decision interval. Market execution advances through every base bar in that interval, then emits one reward based on the hybrid book's excess log return over an independent shadow baseline book. Base-bar returns remain available for annualized evaluation and are never silently mixed with decision-step returns.

Liquidity capacity uses explicit volume semantics. Base-asset volume is multiplied by price, quote-notional volume is used directly, and contract volume is multiplied by both the base-asset contract multiplier and price.

## Evaluation

All total-return, Sharpe, Sortino, drawdown, turnover, cost, funding and paired-excess calculations live in `trade_rl.evaluation`. Every return series declares its temporal identity and annualization factor.

Walk-forward evaluation is split into pure fold construction, fold-local execution, sealed outer-OOS results and chronological stitching. Purge boundaries and non-overlapping outer test windows are explicit invariants.

## Artifacts

Dataset, signal, policy ensemble, evaluation, selection and release identities are separate immutable records. Artifacts use canonical JSON and SHA-256 content digests. Run publication is staged, validated, and atomically pointed to as latest. A failed run cannot overwrite the last successful run.

## Serving

Serving bundles list every file with a size and SHA-256 digest. A registry validates a bundle before installing it and atomically changes the active pointer. The runtime fully validates and loads a replacement policy before swapping its in-memory snapshot; failed hot swaps preserve the previous snapshot.

Structured Serving input is built through the same `ObservationBuilder` used by the environment. A baseline-only bundle explicitly has no policy digest. It is a research or safety fallback identity, not evidence of production eligibility.

## Quality gates

CI runs Ruff, formatting checks, mypy, Import Linter, Vulture advisory reporting, unit and property-based tests, migration tests and branch coverage. Regression tests cover prefix invariance, future-row observation invariance, next-bar tradability isolation, point-in-time trend eligibility, explicit volume conversion and training/Serving observation parity. The architecture contract also verifies that legacy execution trees and direct-action mode are absent.
