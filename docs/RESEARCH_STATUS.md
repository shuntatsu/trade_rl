# Research Status

## Current classification

```text
Architecture: HARDENED
Environment implementation: VERIFIED BY UNIT/PROPERTY TESTS
Historical real-data evidence: STALE AFTER ENVIRONMENT CHANGE
ResidualPolicyCandidate: NOT_SELECTED
BaselineFallback: SELECTED_FOR_ANALYSIS
ProductionRelease: BLOCKED
```

## 2026-07-13 archived real-data result

The archived result remains immutable and is classified as follows:

```text
ResearchRun: COMPLETED
SignalArtifact: REJECTED
ResidualPolicyCandidate: NOT_SELECTED
BaselineFallback: SELECTED_FOR_ANALYSIS
ProductionRelease: BLOCKED
```

Configuration A was the identity baseline and had no selected PPO model path. The signal gate failed because mean OOS IC was below its required threshold. The final production gate also failed, including the positive-return significance check. Positive holdout return and positive 2x-cost return do not override failed mandatory gates.

## Why the result must not be reused as current evidence

The maintained environment now differs materially from the archived experiment:

- quantity/cash self-financing accounting replaces weight-only accounting;
- target weights drift naturally with prices;
- orders execute from the next open rather than the decision close;
- liquidity, partial fills, exchange filters and dynamic costs are modeled;
- mark-price margin and liquidation are modeled;
- operational guardrails and pre-trade constraints are shared with serving;
- missing data, warm-up and tradability are explicit;
- time horizons and discounting can be normalized in physical hours;
- terminal liquidation and account-state handoff are explicit.

Consequently, the old return, Sharpe, turnover and candidate-selection outputs are historical migration evidence only. They cannot establish the performance of the hardened environment. A new nested walk-forward run, including normal and stressed costs, multiple seeds and a sealed outer OOS evaluation, is mandatory before any strategy conclusion is updated.
