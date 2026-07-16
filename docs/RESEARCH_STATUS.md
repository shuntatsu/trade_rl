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

## Absolute-growth reward and paired inference contract

The maintained residual environment now optimizes the hybrid book's net absolute log growth. It uses the independent shadow baseline only for a light rolling non-inferiority hinge and for sealed paired evaluation. The reward does not add raw interval excess return as a second growth objective.

The baseline hinge uses a 30-day rolling window, a seven-day minimum history, and a 1.5% full-window log-growth tolerance. Only increases in the hinge level are penalized. Drawdown shaping is free through 5%, becomes progressively steeper through 20%, and likewise penalizes only newly worsening severity. Zero action still produces identical hybrid and shadow books, but its reward is the baseline strategy's absolute growth rather than zero.

Paired moving-block inference continues to use `log1p(candidate_return) - log1p(shadow_return)` for its mean, confidence interval, and p-value. That paired quantity is a selection and non-inferiority contract, not the primary step reward. Arithmetic period-return differences remain diagnostic only.

## Causal training contract

Maintained finite-horizon training, behavior cloning, checkpoint validation, configuration selection, and sealed outer evaluation all use the same `liquidate_at_close` terminal-accounting contract. Terminal liquidation costs therefore enter both optimization and evaluation, and every stage fails closed if liquidity prevents a complete exit. Legacy bootstrap-compatible truncation remains available only through non-maintained custom configurations and is not eligible for the complete research workflow.

Policy observations do not include synthetic episode progress or next-bar tradability. They do include rolling hybrid and shadow growth, their growth gap, baseline shortfall, scaled tolerance, hinge level, and emergency-deleverage state because those values determine future reward and termination semantics. Next-open execution uses the last completed bar's volume as its capacity proxy, while actual next-bar tradability remains part of transition dynamics.

A 20% hybrid drawdown triggers current-close liquidation of the hybrid policy book and a true `drawdown_stop` terminal transition. The independent shadow book is preserved rather than charged for a policy failure. Actual hybrid liquidation costs are included in final wealth and reward; there is no fixed terminal jackpot or penalty. Explicit sealed end-of-window evaluation remains the separate mode that liquidates both books.

Every policy ensemble records the observation schema, complete PPO configuration digest, requested timesteps, observed actual timesteps, and resolved compute device. Low GPU utilization for the current small single-environment MLP is not treated as a quality failure; throughput and sealed OOS evidence remain the relevant criteria.

## AUM and environment identity contract

Initial capital is an explicit quote-currency research input rather than a scale-free default. The environment refuses construction when AUM is omitted. This prevents a one-dollar simulation from silently disabling participation, impact, and liquidation constraints that matter for the intended deployment capital.

The environment identity hashes the dataset, resolved timing, trend configuration, risk limits, execution costs, complete reward configuration and resolved rolling windows, alpha mode, action and observation schemas, and initial capital. Policy ensembles record the environment digest and AUM, and fail closed when seeds report inconsistent environment or capital identities.

Capacity conclusions must therefore be evaluated at predeclared AUM scenarios. Performance at one capital scale does not establish performance at a larger scale.

## Nested walk-forward execution contract

The maintained workflow uses fold-local signal lineage, stage-scoped dataset capabilities and a one-shot sealed-test access ledger. Alpha and factor artifacts identify fit and prediction ranges, generator configuration/code digests, validity masks and per-row availability times. Training predictions must be causal inside the train capability; checkpoint, selection and test predictions must be generated from the authorized train fit.

Every fold preserves execution evidence including turnover, fees, funding, borrow, dividends, cash interest, fills, participation and economic termination. Candidate eligibility is still computed from the fixed seed distribution, while configuration selection and sealed outer testing evaluate the exact deterministic mean-action ensemble that final serving loads. Independent folds are summarized as a distribution with median, weighted mean, win rate and worst fold. They are not mislabeled as one continuous portfolio return or drawdown. Continuous metrics require contiguous ranges and verified opening/closing state digests. Session-market annualization and carry use actual elapsed time.

Training and walk-forward runs have separate manifest schemas, exact file closure and content-addressed provenance. Git commit, dirty state, lockfile digest, runtime/library versions, platform/hardware and deterministic seed configuration are captured automatically.

## Serving and activation contract

Serving candidate bundle schema v4 binds the exact runtime contract and declared files but deliberately contains no release digest. A separate external `ReleaseAttestation` binds the candidate bundle to verified dataset, selection/evaluation and evidence-bound gate digests, selected policy, source commit, dependency provenance, approver and approval time. This removes the former bundle/release circular hash.

Registry and runtime activation require an HMAC-SHA256-authenticated external attestation issued under an explicitly trusted key ID. Unknown-key, unsigned, or tampered attestations fail closed. Before swapping live state, the runtime verifies exact file closure, rejects symlinks, loads the shared observation normalizer and executes deterministic probe observations through every policy member. Structured predictions additionally require a monotonic identity-bound `ServingStateSnapshot` so stale portfolio or pending-target state cannot be reused. Shape, finite-value, bounds, action-name, observation-schema, normalizer, and state-identity mismatches fail before activation or inference.

## Capability boundary

The repository is research-ready and supports attested local/paper serving. It still does not implement direct exchange websocket ingestion, order submit/cancel/replace, broker reconciliation, production secrets, venue kill switches or operational alerting. Those capabilities remain a separate live-trading integration phase and the project makes no profitability claim.

## Maintained reward identity

The maintained reward contract is **Reward schema v4** with a complete 720-hour baseline window, fixed 1.5% tolerance, worsening-only staged drawdown shaping, and continuous economic terminal penalties.
