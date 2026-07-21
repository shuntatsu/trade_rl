# Research Status

## Current capability status — 2026-07-22

```text
RepositoryIntegrity: VERIFIED_ON_PR_75_HEAD
ResearchWorkflows: AVAILABLE
StatefulOHLCExecution: AVAILABLE_WITH_KNOWN_COMPATIBILITY_GAPS
TradeRLStudio: AVAILABLE_FOR_DIAGNOSTIC_REPLAY
AttestedPaperServing: AVAILABLE_FOR_ELIGIBLE_SELECTED_FINAL_BUNDLES
DirectExchangeRouting: NOT_IMPLEMENTED
EmpiricalProductionGate: NO-GO
ProfitabilityClaim: NONE
```

The latest integrated verification record is PR #75's conservative stateful order simulator. On exact product head `27a564313f64a4ebbd4001fc77518c9985af78b8`, repository CI run `29858620871` and PostgreSQL run `29858620803` succeeded. The full Python result was `1131 passed, 2 skipped`, with 83.43% total coverage. Studio tests/build/layout, Ruff, Mypy, Import Linter, critical coverage, Ubuntu/Windows suites, PostgreSQL migration/integration, structured Serving smoke, and the non-root training image were included in that verification record.

This evidence establishes code, packaging, artifact, and test integrity for that source head. It does not establish profitability, exchange-equivalent fills, paper/live reconciliation, or permission to deploy capital.

## 2026-07-21 P0 validation boundaries

The maintained P0 validation work added three explicit trust boundaries:

1. a PostgreSQL-backed persistent sealed-test ledger that rejects a second opening of the same experiment-plan, dataset, and fold identity across processes;
2. a real, non-zero Training–Serving observation parity test covering symbol/feature order, availability, staleness, hybrid/shadow books, pending target, previous action, raw and normalized observations, policy-member actions, and deterministic ensemble action;
3. a historical-metadata promotion gate that requires dataset-bound, point-in-time, Ed25519-authenticated, effective-dated full-interval evidence.

A three-seed CPU smoke selected the baseline because both the candidate median seed score and deployable ensemble score were below the declared threshold. The sealed outer test was opened once and production status remained `NO-GO`. This is correct fail-closed behavior rather than a failed pipeline.

## Conservative stateful execution status

Normal RL environment transitions now use persistent market, limit, and stop-market instructions. The maintained configuration selects:

```text
path mode: conservative
processing-bar volume capacity: true
partial-fill carry: true
trigger-volume fractions: 1.00 / 0.50 / 0.25 / 0.00
stateful environment time in force: GTC
```

Order quantity is fixed using decision-time information. Pending orders preserve residual quantity, latency, trigger state, replacement linkage, status, and deterministic event evidence across decisions. Fills share one symbol-level processing-bar capacity pool. Final promotion requires complete conservative execution evidence and a matching execution-policy digest.

The deterministic replay smoke reproduced identical order-event, equity-curve, and observation-trace digests. Its candidate was not promoted because the declared performance requirements failed. No profitability claim was made.

Two boundaries remain important:

- OHLCV cannot reconstruct true intrabar order, exchange queue position, hidden liquidity, auctions, or L2 depth.
- `MarketExecutor.execute_interval` remains a separate compatibility target-filling implementation instead of a thin stateful adapter. Normal episode steps are stateful, but baseline reward pre-roll and some compatibility/sensitivity callers still use this path. Their outputs must not be described as complete persistent-order evidence until migrated.

## Trade RL Studio and training telemetry

Trade RL Studio provides a local research console for validated datasets, configs, exploratory jobs, runs, evidence, comparisons, and read-only serving state. Live Training shows seed-scoped exploration telemetry as a market replay.

`training_telemetry_v1` is append-only diagnostic data. It does not participate in checkpoint selection, configuration selection, sealed evaluation, run identity, release approval, or order execution. BUY/SELL markers represent exposure changes, not exchange orders.

Current telemetry limitations are recorded in the architecture audit:

- status and paged reads scan the JSONL file from the beginning, so repeated polling grows with accumulated file size;
- JSON boolean fields are currently coerced with Python truthiness rather than parsed strictly;
- duplicate files resolving to the same seed are selected by discovery order instead of being rejected as ambiguous;
- `trade_rl.telemetry` is not yet placed inside the enforced Import Linter layer stack.

These issues affect diagnostics, scaling, and architectural enforcement. They do not alter the fact that telemetry is excluded from promotion evidence.

## PostgreSQL artifact catalog

The optional PostgreSQL catalog stores verified artifact metadata, canonical cache keys, locations, sizes, dependency edges, lifecycle status, and persistent sealed-test reservations. Datasets, arrays, checkpoints, models, and run evidence remain immutable filesystem artifacts.

Catalog registration is idempotent for identical metadata and rejects digest/cache-key conflicts. PostgreSQL is optional for ordinary filesystem operation, but the durable cross-process sealed-test uniqueness guarantee requires the persistent ledger.

## Absolute-growth reward and paired inference contract

The maintained environment optimizes the hybrid book's net absolute log growth. The independent shadow baseline supplies a light rolling non-inferiority hinge and sealed paired evaluation. Raw interval excess return is not added as a second primary growth objective.

The baseline hinge uses a 30-day rolling window, seven-day minimum history, and 1.5% full-window log-growth tolerance. Only increases in hinge severity are penalized. Drawdown shaping is free through 5%, becomes progressively steeper through 20%, and likewise penalizes only newly worsening severity. Zero residual action produces identical hybrid and shadow books, but its reward is the baseline strategy's absolute growth rather than zero.

Paired moving-block inference uses `log1p(candidate_return) - log1p(shadow_return)` for its mean, confidence interval, and p-value. That quantity is a selection/non-inferiority contract, not the primary step reward.

The maintained reward contract is **Reward schema v4**.

## Causal training contract

Maintained finite-horizon training, behavior cloning, checkpoint validation, configuration selection, and sealed evaluation use liquidation-at-close terminal accounting. Policy observations do not include synthetic episode progress or future tradability. They do include the state required by reward, risk, and persistent execution semantics.

A 20% hybrid drawdown triggers current-close liquidation of the hybrid policy book and a true `drawdown_stop` terminal transition. The independent shadow book is not charged for a policy-specific failure. Actual liquidation costs enter final wealth and reward; no fixed terminal jackpot is used.

Every policy ensemble records observation schema, action identity, PPO configuration digest, requested/actual timesteps, compute device, dataset/environment identity, AUM, normalizers, and policy-member digests. Low GPU utilization for a small model is not itself a quality failure; throughput and sealed OOS evidence are the relevant criteria.

The baseline reward pre-roll currently uses the compatibility execution path, while normal transitions use the stateful engine. Under non-zero latency, persistent partial fills, limit/stop semantics, or cancel-and-replace, this can create an economic-state mismatch at episode reset. Production promotion remains blocked, and this gap is prioritized in the 2026-07-22 architecture audit.

## AUM and environment identity

Initial capital is an explicit quote-currency research input. The environment refuses construction when AUM is omitted. Environment identity hashes dataset, timing, trend, risk, execution policy, reward, rolling windows, alpha/factor mode, action and observation schemas, sequence settings, and initial capital.

Capacity conclusions must be evaluated at predeclared AUM scenarios. Performance at one capital scale does not establish performance at a larger scale.

## Nested walk-forward execution

The maintained workflow uses fold-local signal lineage, stage-scoped dataset capabilities, and a one-shot sealed-test access ledger. Alpha/factor artifacts identify fit/prediction ranges, generator configuration/code digests, validity masks, and row availability times.

Each fold preserves execution evidence including turnover, fees, funding, borrow, dividends, cash interest, fills, participation, pending-order events, and economic termination. Candidate eligibility is computed from a fixed seed distribution. Configuration selection and sealed testing evaluate the exact deterministic mean-action ensemble that serving loads.

Independent folds are summarized as a distribution with median, weighted mean, win rate, and worst fold. They are not mislabeled as one continuous portfolio return or drawdown. Continuous metrics require contiguous ranges and verified opening/closing state digests.

## Serving and activation contract

Serving candidate bundle schema is **v5**. The bundle binds runtime contracts and declared files but deliberately contains no private approval material. A detached `ReleaseAttestation` binds the immutable bundle to verified dataset, selection/evaluation/gate evidence, fresh confirmation, conservative execution evidence, selected policy, source commit, dependency provenance, approver, approval time, and expiry.

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

The repository supports research workflows and attested local/paper serving for eligible selected-final bundles. It does not implement direct exchange websocket ingestion, order submission/cancellation/replacement, broker reconciliation, production secrets, venue kill switches, or operational alerting.

Production remains `NO-GO` until maintained GPU verification, at least 180 OOS days, a strictly positive paired block-bootstrap lower bound on RL-minus-baseline daily log excess, signed fresh confirmation, complete conservative execution evidence, and paper-trading reconciliation all pass.

See [Architecture](ARCHITECTURE.md), [P0 validation evidence](verification/2026-07-21-p0-validation-baseline.md), [stateful execution verification](verification/2026-07-21-conservative-order-simulator.md), and the [2026-07-22 documentation and architecture audit](verification/2026-07-22-documentation-and-architecture-audit.md).
