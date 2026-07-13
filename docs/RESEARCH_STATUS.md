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

## Causal training contract

Random training episodes end as time-limit truncations without forced liquidation. Stable-Baselines3 may therefore bootstrap their terminal observations. Explicit end-of-window liquidation is reserved for sealed evaluation, is reported as a terminal transition, and fails closed if liquidity prevents a complete exit.

Policy observations do not include synthetic episode progress or next-bar tradability. Next-open execution uses the last completed bar's volume as its capacity proxy, while actual next-bar tradability remains part of transition dynamics.

Every policy ensemble records the observation schema, complete PPO configuration digest, requested timesteps, observed actual timesteps, and resolved compute device. Low GPU utilization for the current small single-environment MLP is not treated as a quality failure; throughput and sealed OOS evidence remain the relevant criteria.

## AUM and environment identity contract

Initial capital is an explicit quote-currency research input rather than a scale-free default. The environment refuses construction when AUM is omitted. This prevents a one-dollar simulation from silently disabling participation, impact, and liquidation constraints that matter for the intended deployment capital.

The environment identity hashes the dataset, resolved timing, trend configuration, risk limits, execution costs, reward configuration, alpha mode, action and observation schemas, and initial capital. Policy ensembles record the environment digest and AUM, and fail closed when seeds report inconsistent environment or capital identities.

Capacity conclusions must therefore be evaluated at predeclared AUM scenarios. Performance at one capital scale does not establish performance at a larger scale.

## Nested walk-forward execution contract

The maintained workflow now has a concrete adapter-driven fold runner. Candidate training receives only the fold train and checkpoint-validation ranges. Frozen candidates and the identity baseline are compared only on the configuration-selection range. The selected candidate and baseline are then evaluated on the sealed outer-test range exactly once; baseline fallback avoids a duplicate test evaluation.

All fold results bind dataset identity, selected configuration, policy digest when applicable, selection evidence, and sealed-test evidence. Selected and baseline OOS series are stitched separately and a final content digest identifies the complete walk-forward evaluation.

Gate decisions now preserve the evaluated dataset, selected policy identity when applicable, and final evaluation digest. Release construction fails closed when those identities do not match the selection and dataset artifacts.

## Serving and activation contract

Serving bundle schema v2 binds the action schema, observation schema, flattened observation size, environment digest, initial capital, dataset, signal, selection, policy, release, and every included file into one immutable digest.

Runtime activation rejects action- or observation-schema mismatches before loading a replacement policy. Inference rejects vectors whose length differs from the active observation contract. Both runtime and registry require an approved release identity by default, while unreleased research bundles require the explicit `allow_unreleased=True` override.

This closes the previous gap where a bundle could contain a valid policy output shape but receive a different feature layout, observation width, environment configuration, or capital scale.

The remaining gap is application-specific adapter integration: a real-data loader, fold-local preprocessing, PPO trainer, and evaluator must be connected to these typed requests. The repository still makes no profitability or production-readiness claim.
