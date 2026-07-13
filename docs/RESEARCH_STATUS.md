# Research Status

## 2026-07-13 archived real-data result

The archived result is classified as follows:

```text
ResearchRun: COMPLETED
SignalArtifact: REJECTED
ResidualPolicyCandidate: NOT_SELECTED
BaselineFallback: SELECTED_FOR_ANALYSIS
ProductionRelease: BLOCKED
```

Configuration A was the identity baseline and had no selected PPO model path. The signal gate failed because mean OOS IC was below its required threshold. The final production gate also failed, including the positive-return significance check. Positive holdout return and positive 2x-cost return remain evidence, but they do not override failed mandatory gates.

The migration fixture exists to prevent this evidence from being mislabeled as a selected policy ensemble or a production release.

## Paired reward and inference contract

The maintained residual environment optimizes candidate growth relative to the independent shadow baseline in log-return space. Paired moving-block inference therefore uses the same per-period quantity, `log1p(candidate_return) - log1p(shadow_return)`, for its mean, confidence interval, and p-value. Arithmetic period-return differences are retained only as a diagnostic field and do not drive statistical superiority decisions.
