# Research History

This document records the role of historical experiments without making them normative system requirements or profitability claims.

## Scope

The repository has explored:

- synthetic positive and negative controls;
- feature predictive-power and leakage diagnostics;
- forecast horizons and target definitions;
- PPO hyperparameters and behavioral-cloning warm starts;
- multi-timeframe feature encoders;
- post-processing, turnover penalties, no-trade bands, and volatility targeting;
- seed ensembles and disagreement scaling;
- rule-based and RL risk overlays;
- walk-forward, cost sensitivity, bootstrap comparisons, and replay simulation;
- several public exchange data sources with different history limitations.

## Interpretation rules

Historical results are not directly comparable unless data, symbols, date range, target, horizon, decision frequency, fees, execution model, random seeds, code revision, and evaluation protocol are identical.

Synthetic returns and Sharpe ratios are software or hypothesis diagnostics. They are not estimates of achievable live performance.

Single-split, single-seed, or repeatedly inspected holdout results are exploratory. Promotion evidence requires the current gated Control Plane and exact ServingBundle identity.

## Important historical lessons

1. Apparent predictive power can disappear after proper warmup removal, walk-forward evaluation, or correction for repeated horizon/target search.
2. A high in-sample or diagnostic IC does not establish post-cost trading value.
3. Fixed baselines can look strong in one split and fail across folds; they must be evaluated under the same protocol as RL.
4. Turnover penalties, smoothing, and no-trade bands can change both learned action scale and execution behavior; their effects must be isolated rather than attributed from one run.
5. Training/serving parity requires the complete observation, preprocessing, portfolio state, and decision path—not merely the same model file.
6. Public OHLCV-only data may not contain enough stable edge. Continuing parameter search after a defined withdrawal criterion increases selection bias.

## Where detailed numbers went

Earlier Markdown files contained many experiment-specific numbers without one durable machine-readable provenance standard. Those files were removed from the normative documentation set. Git history preserves them for forensic reference.

New research evidence should be stored as immutable artifacts containing:

- command and full configuration;
- dataset and time range identity;
- symbols and sample counts;
- Git SHA and bundle digest where applicable;
- fold and seed details;
- raw metrics and confidence intervals;
- limitations and whether the result is exploratory or promotion evidence.

## Production boundary

No entry in this document can change Production from NO-GO to GO. Only the evidence checklist in [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) and the approved deployment process can authorize that decision.
