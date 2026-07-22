# Post-Merge Architecture Audit — 2026-07-22

## 1. Audit target and environment

The implementation baseline audited here is `main` commit `6bec98e43599c98fb4b86a1522ab455f5acd396b`, which merged PR #78 after the conservative stateful order simulator. The documentation work is isolated on `docs/current-architecture-sync-20260722` in draft PR #82.

The first documentation-contract head was `6229330f1c79acdd9e28ff9e230300f5a15acd80`. Its exact source archive was produced by GitHub Actions run `29912139543`, artifact `8526308504`, digest `sha256:7cfd64cb89264106bdd5d243ff22c0563abf578148a3b64e6027a1dbbc38cc3f`.

Local inspection used that exact archive. The local host provides Python 3.13 rather than the repository's required Python 3.12.x and does not provide Gymnasium, Stable-Baselines3, psycopg, Docker, Ruff, Mypy, or Import Linter. Therefore local execution was limited to dependency-light contracts; final authority remains exact-head GitHub Actions and PostgreSQL CI.

Local evidence collected before the final branch verification:

```text
Documentation contracts, before report creation: 7 passed, 1 expected link failure
Causality/data/sequence/simulation/telemetry/Studio API/catalog/offline-key tests: 193 passed
Sealed-ledger/execution-promotion/bundle/attestation/catalog-service tests: 22 passed
Workflow security checker: passed
```

Collection of Gymnasium-dependent E2E and structured-serving tests failed locally only because `gymnasium` is not installed. That environmental limitation is not recorded as a product failure.


### Verification candidate head

A code-identical CI rerun was made at head `a4d0959dac089c6c3afd98065ca97d797bad1caf`; the only additional file was the inert `.github/ci-rerun-marker.txt`. GitHub Actions CI run `29916374600` completed successfully:

```text
Full pytest: 1163 passed, 2 skipped, 11 warnings
Total coverage: 83.16%
Critical branch-coverage ratchet: passed
CLI smoke: passed
Studio tests, TypeScript, build, and fixed viewport: passed
Workflow security, Ruff, format, Mypy, Import Linter, dead code: passed
Recovery and structured Serving smoke: passed
Ubuntu and Windows compatibility: passed
Training image build, identity capture, and non-root probe: passed
```

Evidence artifacts:

```text
pytest diagnostics: 8528112948
pytest digest: sha256:4b7ca85c3df94d347de9366d6dedb24221455cc321466d0ad4e1179e7bb64ae0
architecture diagnostics: 8528071873
architecture digest: sha256:a75c326f9b1c0230d97dd0cc1a1086ae1826dbc4b226df4cd429f6b2554e1477
static diagnostics: 8528071373
static digest: sha256:891392eb6ea79813217bc79a8670b124ccce60c5c6c7e6590ab3218fe86f3536
training image evidence: 8528069570
training image digest: sha256:47f916488bccb9d7d0814a83ba1e36484332ea5a13f16ed2b69d908cf50ce582
image id: sha256:48b331ebb3e01b68e4d3876536e4a406e12d78016a0e31222605d694f2a6050b
source tree digest: 430c6dd7aaebb9aeb9104990899613a120e6432f466d6a743dd6bf5da8fb6d18
lockfile digest: d2fb04f4bca12cb1b0702033aa46db27dc6a821764aad864f373bc490b012c79
```

## 2. Capability boundary

The repository is research-ready and can build externally attested, read-only paper-serving bundles for eligible selected-final runs. It does not implement authenticated account access, direct exchange order routing, broker reconciliation, venue kill switches, production secret distribution, or operational alerting.

The following judgments remain independent:

```text
Software and artifact integrity: verified per exact source head
Research validity: requires causal data and sealed evaluation
Profitability: not established
Release eligibility: requires external authorization, confirmation, and attestation
Direct exchange capability: not implemented
Production status: NO-GO
```

A conservative OHLCV simulator is not an exchange emulator. Queue priority, hidden liquidity, event-level intrabar ordering, auctions, adverse selection, and L2 depth remain outside the evidence boundary.

## 3. Responsibility and dependency reality

`.importlinter` enforces this order:

```text
trade_rl.cli
trade_rl.studio
trade_rl.workflows
trade_rl.integrations
trade_rl.serving
trade_rl.learning
trade_rl.rl
trade_rl.risk
trade_rl.simulation
trade_rl.strategies
trade_rl.data
trade_rl.catalog
trade_rl.evaluation
trade_rl.release
trade_rl.artifacts
trade_rl.telemetry
trade_rl.domain
```

The remediation commit made `trade_rl.telemetry` an explicit standard-library-only layer. Domain, release, serving, learning, workflow-framework, training-core, offline-signer, and catalog contracts are separately enforced. The workflow security checker passed on the audited source.

The effective public classes/functions for simulation, telemetry, and Studio are currently installed by package-initializer mutation:

```text
trade_rl.simulation.execution.MarketExecutor
  -> trade_rl.simulation.execution_adapter.StatefulCompatibilityMarketExecutor
trade_rl.telemetry.training.TrainingTelemetryRecord
  -> trade_rl.telemetry.indexed_training.StrictTrainingTelemetryRecord
trade_rl.telemetry.training.read_training_telemetry
  -> trade_rl.telemetry.indexed_training.read_indexed_training_telemetry
trade_rl.studio.telemetry.StudioTelemetryReader
  -> trade_rl.studio.strict_telemetry.StrictStudioTelemetryReader
```

Ordinary imports receive the hardened implementations, but the maintained behavior is not defined where the public symbol appears in source. Finding `AUD-ARCH-001` records this non-local boundary.

## 4. Market data and causality

The audit traced market-source timestamps, `available_at`, union clocks, tradability, feature availability/staleness, flat and structured observations, sequence normalization, and fold capabilities.

Relevant code and tests include:

```text
trade_rl/data/market.py
trade_rl/data/builder.py
trade_rl/data/features.py
trade_rl/rl/market_inputs.py
trade_rl/rl/observations.py
trade_rl/rl/sequence_observations.py
trade_rl/rl/normalization.py
trade_rl/rl/sequence_normalization.py
tests/data/test_extended_prefix_causality.py
tests/data/test_feature_availability_v2.py
tests/data/test_information_availability.py
tests/rl/test_observation_causality.py
tests/rl/test_extended_observation_causality.py
tests/rl/test_sequence_observations.py
tests/rl/test_sequence_normalization.py
```

The dependency-light causality group passed locally. No path was found that exposes future return, future high/low, next-bar tradability, evaluation-range normalization statistics, or post-period metadata revisions to the action-time observation. Processing-bar OHLCV is consumed by transition execution after action and is not inserted into the preceding policy observation.

Result: `AUD-DATA-001`, `NOT_FOUND`, P0-class leakage check.

## 5. Orders, execution, and accounting

The maintained path represents explicit market, limit, and stop-market orders with deterministic identity, latency, eligibility, trigger state, time in force, remaining quantity, rejection, expiration, cancellation/replacement, shared symbol capacity, and deterministic events. Decision-time target reconciliation fixes quantity from known equity and reference prices rather than resizing it from future execution prices.

Base-quantity, contract-quantity, and quote-notional volume are separate contracts. Quote-notional capacity is already denominated in quote currency and is not multiplied by price a second time. The simulation test suite, including independent accounting-oracle, precision, branch-ratchet, stateful execution, liquidity, volume-unit, gap, liquidation, replay, and promotion tests, passed in the dependency-light local run.

PR #78 made the compatibility facade delegate to `execute_target_statefully()`, so baseline reward pre-roll and compatibility callers now use the persistent order lifecycle when they chain the returned `BookState` through the same executor.

Two structural issues remain:

- the public compatibility class is installed through `trade_rl.simulation.__init__` mutation rather than defined directly at its public module boundary (`AUD-ARCH-001`);
- `execute_stateful_orders()` is a 614-line function coordinating validation, lifecycle transitions, trigger/path decisions, capacity allocation, fills, carry, margin, termination, and evidence accumulation (`AUD-SIM-001`).

No overfill, quote-notional double conversion, accounting conservation break, or future-price quantity resize was found.

Result: `AUD-EXEC-001`, `NOT_FOUND`, P0-class accounting/execution check.

## 6. Training, selection, and sealed evaluation

The audit traced behavior cloning, PPO, intermediate checkpoint selection, fixed-seed aggregation, baseline fallback, configuration selection, sealed fold evaluation, outer access counts, execution sensitivity, selected-final authorization, and publication.

The one-shot ledger binds plan, dataset, and fold identity. The fold runner authorizes sealed access only after candidate selection and verifies the expected evaluation count. Post-selection execution sensitivity replays the selected policy and baseline without feeding results back into recipe selection. Failed candidate or promotion gates retain baseline fallback or block selected-final publication.

Dependency-light sealed-ledger and execution-promotion tests passed locally. Existing exact-head CI and PostgreSQL integration remain required for the concrete Gymnasium/SB3 workflow path.

No source path was found that feeds sealed outer returns, telemetry, sensitivity results, or fresh confirmation back into model fitting or configuration selection.

Result: `AUD-EVAL-001`, `NOT_FOUND`, P0-class sealed-contamination check.

## 7. Training-serving parity

Flat observation identity is `baseline_residual_observation_v5`. Per symbol it includes seven pending-order coordinates: remaining-notional ratio, order type, order status, age, eligible delay, trigger state, and expiry distance. Structured serving additionally binds symbol order, native feature order, sequence windows, normalizers, account state, pending target/orders, decision index, and execution-policy digest.

Candidate identity is `serving_bundle_v5`. Runtime validates file closure, bundle/evidence identity, normalizers, action schema and bounds, observation schema and shape, monotonic state snapshots, execution-policy identity, and deterministic policy probes before activation.

Offline private-key modules are excluded from serving, integrations, Studio, RL, and online workflow paths by Import Linter. The dependency-light bundle and release-attestation tests passed locally; full structured parity remains delegated to exact-head CI because Gymnasium is unavailable locally.

No fail-open schema mismatch, pending-order omission, private-key import, or action clipping-at-runtime path was found.

Result: `AUD-SERVE-001`, `NOT_FOUND`, P0-class parity/key-boundary check.

## 8. Artifacts, PostgreSQL, and release

Canonical datasets, arrays, checkpoints, policy archives, manifests, evidence, and bundles remain immutable filesystem artifacts. PostgreSQL stores searchable metadata, provenance, dependency edges, cache identities, lifecycle status, locations, and durable sealed-test reservations. Catalog writes do not redefine artifact identity.

PR #78 separated evaluation-specific reservation SQL into `PostgresSealedTestReservationStore`; `PostgresArtifactCatalog.reserve_sealed_test_access()` remains a compatibility delegate. Canonical JSON bytes are centralized in `trade_rl.domain.canonical_json` and shared by artifact and catalog identity paths.

The local catalog contract/service and offline-key tests passed. This documentation-only PR does not match the PostgreSQL workflow path filter. The same product baseline was verified by PR #78 PostgreSQL Catalog run `29891993502`: Compose validation, PostgreSQL 16 startup/readiness, migrations, and unit/integration tests all passed.

No model/dataset BLOB authority, circular release hash, runtime private-key use, or catalog-driven identity mutation was found.

Result: `AUD-CAT-001`, `NOT_FOUND`, P0-class artifact/release check.

## 9. Studio and Live Training

Studio reads maintained artifacts and starts exploratory training through maintained workflows. It does not calculate checkpoint ranking, open sealed ranges, sign approval evidence, activate bundles, accept private keys, or route orders.

Telemetry hardening from PR #78 is effective under ordinary imports: strict JSON booleans, a sparse identity-bound sidecar index, append-only refresh from prior EOF, cursor seeking, replacement/truncation rebuild, duplicate-seed rejection, root/symlink checks, and an enforced telemetry layer.

### Confirmed vector-environment and episode mixing

`TrainingTelemetryRecord` carries `environment_id` but no episode identity. `TrainingTelemetrySampler.consume()` iterates all vector environments and appends them to one seed-scoped JSONL with one global sequence. `useTrainingTelemetry()` buffers and sorts only by sequence. `LiveTrainingPage` computes first portfolio value, PnL, equity, baseline, and drawdown arrays over the complete seed buffer. `MarketReplayChart` filters only event type and finite close, then draws the records as one candle sequence.

A dependency-light reproduction emitted two environments in one sampler call:

```text
emitted=2
sequence=[1, 2]
environment_id=[0, 1]
close=[101.0, 601.0]
portfolio_value=[1000.0, 1500.0]
```

The browser would treat those records as adjacent points in one price/equity path. Auto-reset episodes have the same issue because the schema has an `episode_end` event but no `episode_id`, and the browser does not segment on the event.

This is diagnostic misrepresentation, not selection or execution corruption, because telemetry is excluded from those evidence paths.

Result: `AUD-STUDIO-001`, `CONFIRMED`, P1.

## 10. CI, Docker, and privileged execution

PR CI runs on GitHub-hosted Ubuntu/Windows jobs and checks exact pull-request head, pinned dependencies/actions, Studio tests/build/layout, workflow security, Ruff, format, Mypy, Import Linter, dead-code reporting, Serving smoke, full pytest/coverage, critical branch coverage, CLI smoke, compatibility suites, training-image construction, and non-root runtime behavior.

Self-hosted GPU workflows are restricted to `main`, repository-owner controls, and the protected `gpu-full-training` environment. External Actions in audited workflows are pinned to immutable SHAs. The hourly monitor is read-only and does not use the protected environment. Logs and inspect evidence are captured before retained-container deletion.

The local workflow-security checker passed. Docker/Compose are unavailable on the local audit host. GitHub Actions run `29916374600` supplied the training-image identity and non-root evidence; PR #78 PostgreSQL run `29891993502` supplied the unchanged product baseline's PostgreSQL 16 service evidence.

No pull-request-controlled self-hosted execution or unpinned external Action was found in the audited maintained workflows.

Result: `AUD-CI-001`, `NOT_FOUND`, P0-class privileged-execution check.

## 11. Findings

### AUD-DOC-001 — maintained-document drift

```text
Status: CONFIRMED
Priority: P3
Affected: README.md, README.ja.md, START.md, docs/ARCHITECTURE.md,
          docs/RESEARCH_STATUS.md, docs/BINANCE.md,
          docs/operations/docker-gpu-full-training.md, studio/README.md
Observed fact: maintained documents retained observation-v3/bundle-v4 language and
               findings already remediated by PR #78.
Invariant: current-state documentation must describe the current implementation and
           must not present closed findings as active defects.
Impact: operators and reviewers could select obsolete commands, contracts, or risk assumptions.
Reproduction: tests/test_current_documentation_contract.py was RED for schema, layer,
              capability-boundary, and stale-finding checks.
Recommended boundary: executable documentation contracts derived from source constants
                      and .importlinter.
Independent remediation PR: PR #82 (this documentation/audit PR).
```

### AUD-STUDIO-001 — mixed vector-environment and episode replay

```text
Status: CONFIRMED
Priority: P1
Affected: trade_rl/telemetry/training.py,
          trade_rl/rl/training_telemetry.py,
          studio/src/live/useTrainingTelemetry.ts,
          studio/src/pages/LiveTrainingPage.tsx,
          studio/src/live/MarketReplayChart.tsx
Observed fact: one seed stream contains multiple environment_id values and reset episodes;
               the UI builds one continuous price/equity/PnL path from all records.
Invariant: a displayed market/equity trajectory must identify one environment and one episode,
           or explicitly segment discontinuities.
Impact: exploratory diagnostics can display false jumps and misleading continuous PnL.
Reproduction: the two-environment sampler reproduction in section 9.
Recommended boundary: add explicit episode identity, expose environment/episode selection in
                      the API/UI, and segment or reset derived metrics at boundaries.
Independent remediation PR: fix/live-training-stream-isolation-20260722.
```

### AUD-ARCH-001 — public behavior installed by package-initializer mutation

```text
Status: CONFIRMED
Priority: P2
Affected: trade_rl/simulation/__init__.py,
          trade_rl/telemetry/__init__.py,
          trade_rl/studio/__init__.py,
          trade_rl/simulation/execution.py,
          trade_rl/telemetry/training.py,
          trade_rl/studio/telemetry.py
Observed fact: package initializers use setattr() to replace public classes/functions with
               hardened subclasses/adapters from other modules.
Invariant: maintained public behavior should be locally discoverable from the public module
           and analyzable without package-import side effects.
Impact: static review sees obsolete implementations, dead compatibility code remains reachable
        as a base implementation, and refactors can accidentally bypass the maintained boundary.
Reproduction: importing the public symbols reports modules execution_adapter,
              indexed_training, and strict_telemetry rather than their declared source modules.
Recommended boundary: define/re-export the maintained implementation directly, delete superseded
                      method bodies, and keep compatibility aliases explicit and testable.
Independent remediation PR: refactor/direct-maintained-contract-exports-20260722.
```

### AUD-SIM-001 — stateful execution orchestration hotspot

```text
Status: CONFIRMED
Priority: P2
Affected: trade_rl/simulation/stateful_execution.py::execute_stateful_orders
Observed fact: one 614-line function owns input validation, bar loop, order admission/state
               transitions, path triggers, capacity allocation, fills, accounting/carry,
               margin/termination, and evidence aggregation.
Invariant: order state-machine, liquidity allocation, accounting settlement, and evidence
           packaging should be independently testable responsibilities.
Impact: a small order-lifecycle change has a wide regression surface and increases review risk.
Reproduction: AST line-span inspection of the audited source.
Recommended boundary: extract processing-bar context, order-transition processor,
                      settlement/carry coordinator, and result accumulator without changing
                      the public StatefulExecutionResult contract.
Independent remediation PR: refactor/stateful-execution-phase-services-20260722.
```

### AUD-CI-002 — sequence projection equivalence has a numerical CI flake

```text
Status: CONFIRMED
Priority: P2
Affected: tests/rl/test_sequence_policy_core.py::
          test_projection_after_selection_matches_legacy_outputs_and_gradients
Observed fact: CI run 29915846436 failed the equivalence assertion once with maximum
               absolute difference 4.112720489501953e-06 against atol=1e-6. A rerun
               with unchanged Python/TypeScript/product/document content passed the
               complete suite in run 29916374600.
Invariant: deterministic equivalence tests should distinguish meaningful policy drift
           from backend-level floating-point ordering noise.
Impact: an otherwise valid exact-head verification can fail transiently and block review.
Reproduction: compare runs 29915846436 and 29916374600; local repetition also passed.
Recommended boundary: stabilize the tested computation or use a dtype/backend-justified
                      tolerance while retaining a separate stronger semantic invariant.
Independent remediation PR: test/stabilize-sequence-projection-equivalence-20260722.
```

### AUD-RL-001 — environment facade remains a large orchestration surface

```text
Status: RISK
Priority: P2
Affected: trade_rl/rl/environment.py::ResidualMarketEnv.__init__, reset, step
Observed fact: after four focused services were extracted, environment.py remains 1,620 lines;
               __init__ spans 479 lines and step spans 223 lines.
Protected invariant: the facade must own mutable Gymnasium state and stable reset/step/info APIs.
Risk: further feature additions may re-concentrate action, risk, reward, diagnostics, and mutable
      state application in the facade.
Reproduction: AST line-span inspection; no behavioral defect was reproduced.
Recommended boundary: do not split mechanically. On the next action/risk/reward feature, extract
                      construction/wiring and transition-info assembly behind typed immutable inputs.
Independent remediation PR: defer until a behavior change provides a testable seam; track in roadmap.
```

## 12. Remediation PR roadmap

Priority order after this documentation/audit PR:

1. `fix/live-training-stream-isolation-20260722` — P1; add episode identity, environment/episode API selection, discontinuity handling, and frontend tests.
2. `test/stabilize-sequence-projection-equivalence-20260722` — P2; remove backend-sensitive numerical flakiness without weakening semantic parity.
3. `refactor/direct-maintained-contract-exports-20260722` — P2; remove package-initializer `setattr` replacement from simulation, telemetry, and Studio while preserving public imports.
4. `refactor/stateful-execution-phase-services-20260722` — P2; split the stateful executor by lifecycle, liquidity, settlement, and evidence responsibility with parity/property tests.
5. `ResidualMarketEnv` follow-up — P2 risk; only when a concrete action/risk/reward change supplies a behavior-preserving extraction seam.

No P0 remediation branch is required from this audit because no leakage, accounting corruption, sealed-test contamination, private-key exposure, or fail-open promotion path was found.

## 13. Final judgment

The architecture remediation closed the earlier P0/P1 economic-parity and telemetry-integrity findings. The current core preserves strong causality, immutable identity, sealed evaluation, conservative stateful execution, offline release, and privileged-runner boundaries.

The repository remains suitable for research workflows and eligible attested paper serving, subject to the explicit OHLCV limitations and production gates. It is not production-ready and makes no profitability claim.

The highest remaining issue is diagnostic rather than economic: Live Training can combine multiple vector environments and reset episodes into one apparent path. The main structural debt is non-local public-symbol replacement and a monolithic stateful execution coordinator. CI also has one confirmed backend-sensitive sequence-equivalence flake. These should be addressed in independent PRs, not hidden inside the documentation update.
