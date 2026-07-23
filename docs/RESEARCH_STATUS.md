# Research Status

## Current capability status — 2026-07-23

```text
RepositoryIntegrity: VERIFIED_ON_ARCHITECTURE_REMEDIATION_HEAD
ResearchWorkflows: AVAILABLE
StatefulOHLCExecution: AVAILABLE_WITH_OHLCV_LIMITATIONS
TradeRLStudio: AVAILABLE_FOR_DIAGNOSTIC_REPLAY_WITH_EPISODE_ISOLATION
AttestedPaperServing: AVAILABLE_FOR_ELIGIBLE_SELECTED_FINAL_BUNDLES
DirectExchangeRouting: NOT_IMPLEMENTED
EmpiricalProductionGate: NO-GO
ProfitabilityClaim: NONE
```

The current baseline includes the conservative stateful order simulator and the follow-up architecture remediation that unified compatibility execution with the stateful engine, enforced the telemetry dependency boundary, hardened telemetry parsing and indexed reads, centralized canonical JSON, separated PostgreSQL sealed-test reservation responsibilities, decomposed `ResidualMarketEnv`, isolated Live Training replay to one selected vector environment and its current episode, and added producer-issued telemetry episode identity. Exact commit, CI, PostgreSQL, container, and test evidence belongs in the dated [architecture remediation record](verification/2026-07-22-architecture-audit-remediation.md), [post-remediation audit](verification/2026-07-22-post-merge-architecture-audit.md), and [architecture audit closeout](verification/2026-07-23-architecture-audit-closeout.md).

Software verification establishes code, packaging, artifact, and test integrity for a source head. Research validity additionally requires causal data and sealed evaluation. Profitability requires positive repeatable evidence. Release eligibility requires external authorization and attestation. Direct exchange operation requires a separate broker integration. None of those later judgments follows automatically from passing CI.

## 2026-07-21 P0 validation boundaries

The maintained P0 validation work added three explicit trust boundaries:

1. a PostgreSQL-backed persistent sealed-test ledger that rejects a second opening of the same experiment-plan, dataset, and fold identity across processes;
2. a real non-zero Training–Serving observation parity test covering symbol/feature order, availability, staleness, hybrid/shadow books, pending target and orders, previous action, raw and normalized observations, policy-member actions, and deterministic ensemble action;
3. a historical-metadata promotion gate that requires dataset-bound, point-in-time, Ed25519-authenticated, effective-dated full-interval evidence.

A three-seed CPU smoke selected the baseline because both the candidate seed distribution and deployable ensemble failed their declared thresholds. The sealed outer test was opened once and production status remained `NO-GO`. This is correct fail-closed behavior, not a failed pipeline.

## Conservative stateful execution status

Normal RL transitions, compatibility target execution, baseline reward pre-roll, sensitivity replay, and deterministic replay share the stateful order engine. The maintained configuration uses:

```text
path mode: conservative
processing-bar shared capacity: enabled
partial-fill carry: enabled
trigger-volume fractions: 1.00 / 0.50 / 0.25 / 0.00
stateful environment time in force: GTC
```

Order quantity is fixed from decision-time known state. Explicit market, limit, and stop-market orders retain latency, eligibility, trigger state, remaining quantity, time in force, replacement linkage, status, and deterministic events across decisions. Fills share one symbol-level processing-bar capacity pool. Final selected-policy promotion requires complete conservative execution evidence and a matching execution-policy digest. Neutral and optimistic bar paths are diagnostics only.

The deterministic replay smoke reproduced order-event, equity-curve, and observation-trace digests. Its candidate was not promoted because the declared performance requirements failed; baseline fallback remained selected.

OHLCV still cannot reconstruct true intrabar event order, exchange queue position, hidden liquidity, auctions, adverse selection, or L2 depth. Stateful simulation is a conservative research approximation, not proof of exchange-equivalent fills.

## Trade RL Studio and training telemetry

Trade RL Studio is a local research console for validated datasets, configs, exploratory jobs, runs, evidence, comparisons, and read-only serving state. Live Training shows exploratory rollout telemetry as a market replay.

`training_telemetry_v1` is append-only diagnostic data. It is excluded from checkpoint selection, configuration selection, sealed evaluation, run identity, promotion, release approval, and order execution. BUY/SELL markers represent target-exposure changes, not exchange orders.

The maintained browser path selects one vector environment and derives the chart, cursor, price, PnL, baseline, drawdown, events, playback, and jump controls from that selected vector environment's current episode. Producer-issued nullable `episode_id` values are preferred for episode selection. Historical records with `null` identity retain the terminal and counter-rollback fallback, so existing `training_telemetry_v1` streams remain readable.

Telemetry remains exploratory and is excluded from fitting, checkpoint selection, configuration selection, sealed evaluation, run identity, promotion, release approval, Serving activation, and order routing. The isolation contract prevents false cross-environment or cross-episode continuity; it does not turn telemetry into profitability or exchange-execution evidence.

## PostgreSQL artifact catalog

The optional PostgreSQL catalog stores verified artifact metadata, canonical cache keys, locations, sizes, dependency edges, lifecycle status, and persistent sealed-test reservations. Datasets, arrays, checkpoints, models, and run evidence remain immutable filesystem artifacts.

Catalog registration is idempotent for identical metadata and rejects digest/cache-key conflicts. PostgreSQL is optional for ordinary filesystem operation, but the durable cross-process sealed-test uniqueness guarantee requires the persistent ledger.

## Absolute-growth reward and paired inference contract

The maintained environment optimizes the hybrid book's net absolute log growth. The independent shadow baseline supplies a light rolling non-inferiority hinge and sealed paired evaluation. Raw interval excess return is not added as a second primary growth objective.

The baseline hinge uses a 30-day rolling window, seven-day minimum history, and 1.5% full-window log-growth tolerance. Only increases in hinge severity are penalized. Drawdown shaping is free through 5%, becomes progressively steeper through 20%, and likewise penalizes only newly worsening severity. Zero residual action produces identical hybrid and shadow books, but its reward is the baseline strategy's absolute growth rather than zero.

Paired moving-block inference uses `log1p(candidate_return) - log1p(shadow_return)` for its mean, confidence interval, and p-value. That quantity is a selection/non-inferiority contract, not the primary step reward.

The maintained reward contract is **Reward schema v4**.

## Causal training contract

Maintained finite-horizon training, behavior cloning, checkpoint validation, configuration selection, baseline reward pre-roll, and sealed evaluation use the same stateful execution and liquidation-at-close terminal accounting contracts. Policy observations do not include synthetic episode progress, future tradability, or processing-bar OHLCV before action. They do include the current account, reward/risk state, and persistent pending-order coordinates required for a Markov contract.

A 20% hybrid drawdown triggers current-close liquidation of the hybrid policy book and a true `drawdown_stop` terminal transition. The independent shadow book is not charged for a policy-specific failure. Actual liquidation costs enter final wealth and reward; no fixed terminal jackpot is used.

Every policy ensemble records observation schema, action identity, training-configuration digest, requested/actual timesteps, compute device, dataset/environment identity, AUM, normalizers, execution policy, and policy-member digests. Low GPU utilization for a small model is not itself a quality failure; throughput and sealed OOS evidence are the relevant criteria.

## AUM and environment identity

Initial capital is an explicit quote-currency research input. The environment refuses construction when AUM is omitted. Environment identity hashes dataset, timing, trend, risk, execution policy, reward, rolling windows, alpha/factor mode, action and observation schemas, sequence settings, and initial capital.

Capacity conclusions must be evaluated at predeclared AUM scenarios. Performance at one capital scale does not establish performance at a larger scale.

## Nested walk-forward execution

The maintained workflow uses fold-local signal lineage, stage-scoped dataset capabilities, and a one-shot sealed-test access ledger. Alpha/factor artifacts identify fit/prediction ranges, generator configuration/code digests, validity masks, and row availability times.

Each fold preserves execution evidence including turnover, fees, funding, borrow, dividends, cash interest, fills, participation, pending-order events, and economic termination. Candidate eligibility is computed from a fixed seed distribution. Configuration selection and sealed testing evaluate the exact deterministic mean-action ensemble that serving loads.

Independent folds are summarized as a distribution with median, weighted mean, win rate, and worst fold. They are not mislabeled as one continuous portfolio return or drawdown. Continuous metrics require contiguous ranges and verified opening/closing state digests.

## Serving and activation contract

Serving candidate bundle v5 identity is `serving_bundle_v5`. The bundle binds runtime contracts and declared files but contains no private approval material. A detached `ReleaseAttestation` binds the immutable bundle to verified dataset, selection/evaluation/gate evidence, fresh confirmation, conservative execution evidence, selected policy, source commit, dependency provenance, approver, approval time, and expiry.

Registry and runtime require a purpose-bound trusted public key. Unknown-key, unsigned, expired, or tampered attestations fail closed. Before activation, the runtime verifies file closure, rejects symlinks, loads shared normalizers and adapters, and executes deterministic probe observations through every policy member. Structured predictions require a monotonic identity-bound `ServingStateSnapshot` including persistent execution state.

## 2026-07-13 archived real-data result

The archived result remains classified as:

```text
ResearchRun: COMPLETED
SignalArtifact: REJECTED
ResidualPolicyCandidate: NOT_SELECTED
BaselineFallback: SELECTED_FOR_ANALYSIS
ProductionRelease: BLOCKED
```

Configuration A was the identity baseline and had no selected PPO path. The signal gate failed because mean OOS IC was below threshold. The final production gate also failed, including the positive-return significance check. Positive holdout return and positive 2x-cost return remain evidence, but they do not override mandatory failed gates.

The migration fixture prevents this evidence from being mislabeled as a selected policy ensemble or production release.

## Capability boundary

The repository supports research workflows and attested local/paper serving for eligible selected-final bundles. It does not implement direct exchange websocket ingestion, authenticated account access, order submission/cancellation/replacement, broker reconciliation, production secrets, venue kill switches, or operational alerting.

Production remains `NO-GO` until maintained GPU verification, at least 180 OOS days, a strictly positive paired block-bootstrap lower bound on RL-minus-baseline daily log excess, signed fresh confirmation, complete conservative execution evidence, and paper-trading reconciliation all pass.

See [Architecture](ARCHITECTURE.md), [P0 validation evidence](verification/2026-07-21-p0-validation-baseline.md), [stateful execution verification](verification/2026-07-21-conservative-order-simulator.md), the original [2026-07-22 documentation and architecture audit](verification/2026-07-22-documentation-and-architecture-audit.md), the [architecture remediation record](verification/2026-07-22-architecture-audit-remediation.md), the [post-remediation architecture audit](verification/2026-07-22-post-merge-architecture-audit.md), and the [architecture audit closeout](verification/2026-07-23-architecture-audit-closeout.md).
