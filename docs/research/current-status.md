# Current research status

## Status boundary

- M1 lean core: **complete**
- M2 comparison infrastructure: **complete**
- M2 real-data development comparison: **not run yet**
- M3 frozen final evaluation / stress / deletion: **not started**
- Profitability claim: **none**
- Production/live order routing: **not authorized**

“Complete” above means the named infrastructure/contracts are implemented and covered by repository verification. It does not mean a profitable strategy has been demonstrated.

## Current comparison set

The first development comparison is intentionally fixed to five candidates plus three controls:

| Name | Family | Role |
| --- | --- | --- |
| `trend` | rule | Trend signal with hysteresis |
| `mean_reversion` | rule | Mean-reversion signal with hysteresis |
| `ridge24` | forecast | Universal 24h Ridge forecast + common controller |
| `lightgbm24` | forecast | Universal shallow 24h LightGBM forecast + common controller |
| `ppo` | RL | Universal teacher-free PPO |
| `cash` | control | Always FLAT |
| `constant_long` | control | Always LONG |
| `constant_short` | control | Always SHORT |

Do not expand this initial comparison with large model/hyperparameter families merely because the existing candidates do not win.

## Universal fit and evaluation scope

Ridge, LightGBM, and PPO are trained without symbol identity features. `fit_symbol_names` determines which symbols may contribute supervised rows or PPO training episodes. Evaluation symbols are a separate scope; a symbol may be evaluated without being part of fit scope when the experiment contract allows it.

Ridge and LightGBM equalize each fit symbol’s total sample weight. PPO cycles fit symbols round-robin by episode. After fitting, the same frozen model/policy/strategy is replayed independently for each evaluation symbol.

Per-symbol results remain visible. An aggregate return is not sufficient evidence if one symbol’s loss is hidden by another symbol’s gain.

## Development evidence protocol

A development run requires:

1. a canonical filesystem dataset artifact;
2. one resolved JSON run configuration;
3. a new output directory.

Representative configuration fields are:

```json
{
  "signal_name": "<rule-signal-feature>",
  "feature_names": ["<feature-a>", "<feature-b>"],
  "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
  "fit_cutoff": "<exact-dataset-timestamp>",
  "evaluation_start": "<exact-dataset-timestamp>",
  "evaluation_stop_exclusive": "<exact-dataset-timestamp>",
  "rule_entry_threshold": 0.10,
  "rule_exit_threshold": 0.02,
  "forecast_entry_threshold": 0.01,
  "forecast_exit_threshold": 0.002,
  "ppo_total_timesteps": 100000,
  "ppo_seed": 0,
  "gross_budget": 0.5,
  "initial_capital": 100000.0
}
```

Run:

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.runs.candidate \
  --dataset <dataset-artifact-dir> \
  --config <run-config.json> \
  --output <new-result-dir>
```

The runner refuses to overwrite an existing result directory. The immutable evidence directory contains:

```text
summary.json
returns.npz
```

`summary.json` records dataset identity, the resolved configuration, fit/evaluation scope, and per-symbol × per-strategy metrics/diagnostics. `returns.npz` preserves raw interval-return series for paired or block-bootstrap analysis.

## What M2 still requires

Before calling the development comparison complete:

1. choose the real-data development dataset artifact;
2. freeze feature names, `fit_symbol_names`, fit cutoff, and development evaluation window;
3. run all five candidates and three controls under one shared contract;
4. report all symbol-level evidence, including costs, drawdowns, termination, turnover, and raw returns;
5. freeze a supported winner or conclude **no winner**.

A valid no winner result is preferable to adapting the experiment until something appears profitable.

## Decision rules after the development run

Ask, in order:

1. Does a candidate beat the controls on unused development data?
2. Is the result acceptable per symbol rather than only in aggregate?
3. Is edge concentrated in one symbol or one regime?
4. Do turnover, fees, spread, impact, funding, or borrow erase the gross edge?
5. Is a forecast candidate materially better than the rule candidates?
6. Does PPO add repeatable value over the simpler candidates?
7. Are raw returns robust to time-block dependence and reasonable stress?

Prefer the simpler candidate when evidence is not strong enough to justify added complexity.

## M3 boundary

M3 starts only after M2 freezes a candidate. It consists of frozen unused-future/zero-shot evaluation, pre-registered stress, removal of unsupported strategy families, and further repository simplification. Research evidence and Production authorization remain separate decisions.

Nothing in the current repository state establishes profitability or authorizes live order routing.
