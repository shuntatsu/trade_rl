# Residual Reward Hardening Design

## Status

Approved for implementation from the prior reward-design review. This specification targets the current `trade_rl` residual core and composes with the OHLCV, self-financing accounting, and next-open execution contracts already present on `main`.

Production remains **NO-GO**.

## Scope adjustment after the residual-core rebuild

The rebuilt repository already removed the legacy direct-action PPO environment, turnover-penalty reward, ignored-action decision timing, curriculum DSR mode, and all compatibility paths. Those items are complete by deletion and must not be reintroduced.

The remaining reward-design work is limited to four residual-core issues:

1. preserve long-horizon paired wealth optimization through the training discount contract;
2. expose the shadow-relative state needed to predict the paired reward;
3. separate controllable hybrid insolvency from uncontrollable shadow insolvency;
4. use per-period excess log returns consistently for reward, bootstrap inference, checkpoint evidence, and reporting.

## Training discount contract

Residual training uses a default discount factor of `0.99`. `ResidualTrainingConfig` rejects `gamma < 0.95` unless `allow_low_gamma=True` is explicitly supplied for a research ablation.

The CLI makes `--gamma` optional with default `0.99` and exposes `--allow-low-gamma`. The resolved configuration output records both values and uses schema `residual_training_config_v2`.

## Paired observation state

The residual policy observation keeps all existing market, trend, alpha, and hybrid-book fields, and adds:

- per-symbol `hybrid_weight - shadow_weight`;
- global log wealth ratio `log(hybrid_value / shadow_value)`;
- shadow drawdown;
- shadow gross exposure;
- cumulative turnover excess `hybrid_turnover - shadow_turnover`.

The zero action must remain exact identity. At reset, all paired-state additions are zero.

## Insolvency semantics

The environment computes separate flags:

- `hybrid_insolvent`: controlled candidate book is below the configured equity threshold;
- `shadow_insolvent`: deterministic reference book is below the threshold.

The realized interval excess log return is always retained. A configurable `hybrid_insolvency_penalty` in unscaled log-return units is subtracted only when the hybrid book is insolvent. Shadow insolvency does not apply that penalty because the agent cannot control the reference book.

Either insolvency terminates the episode. `info` exposes both flags and `rollout_valid = not shadow_insolvent`. Evaluation and selection code must fail closed when `rollout_valid` is false.

## Paired statistical quantity

For each paired base period:

```text
period_excess_log_return = log1p(candidate_return) - log1p(benchmark_return)
```

Moving-block bootstrap inference, confidence intervals, p-values, and `mean_period_excess` use this series. Arithmetic return differences remain available only as `mean_period_simple_excess` diagnostic output.

Total arithmetic-return difference and total excess log return remain separate fields.

## Required tests

1. default CLI gamma is `0.99`;
2. `gamma < 0.95` fails without the explicit override and succeeds with it;
3. paired observation layout and reset values are exact;
4. zero action still matches the shadow book exactly;
5. hybrid-only insolvency applies the configured penalty;
6. shadow-only insolvency does not apply the hybrid penalty and marks the rollout invalid;
7. paired bootstrap and period mean use log differences, with arithmetic difference retained only as a diagnostic;
8. the architecture continues to contain no direct-action or DSR training path;
9. all quality checks and the complete test suite pass.