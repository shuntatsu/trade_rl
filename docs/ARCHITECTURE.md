# Trade RL Architecture

## Status

`trade_rl` is a research-grade portfolio-RL core with attested local/paper-serving contracts. Direct exchange trading remains unavailable. Architecture integrity, empirical profitability, and operational authorization are separate gates; passing one does not imply the others.

The current maintained system uses causal market artifacts, baseline-anchored or direct-target policies, persistent stateful order simulation, nested walk-forward selection, immutable evidence, an optional PostgreSQL metadata catalog, a local read-only research console, and fail-closed serving activation.

## Responsibility map

```text
trade_rl/
  domain/        immutable identities and standard-library canonical primitives
  telemetry/     append-only standard-library diagnostic event contracts
  artifacts/     hashing, file closure, and atomic publication
  release/       external attestation verification and offline approval contracts
  evaluation/    metrics, paired inference, folds, gates, and AUM capacity
  catalog/       framework-independent artifact metadata contracts and PostgreSQL adapters
  data/          market calendar, causal features, execution-data contracts, datasets
  strategies/    deterministic causal baselines
  simulation/    orders, bar paths, liquidity, execution, carry, margin, accounting
  risk/          pure hard/soft pre-trade and portfolio constraints
  rl/            actions, observations, normalization, rewards, environment, protocols
  learning/      framework-light teachers, behavior cloning, supervised-data contracts
  serving/       candidate bundles, registry, and fail-closed runtime
  integrations/  Stable-Baselines3 and external-system adapters
  workflows/     typed training, walk-forward, sensitivity, and publication orchestration
  studio/        local research read models, job control, and typed API surfaces
  cli/           single authoritative configuration and execution entry point
```

The enforced Import Linter layer order is exactly:

```text
cli
  -> studio
  -> workflows
  -> integrations
  -> serving
  -> learning
  -> rl
  -> risk
  -> simulation
  -> strategies
  -> data
  -> catalog
  -> evaluation
  -> release
  -> artifacts
  -> telemetry
  -> domain
```

`domain` is standard-library only. `telemetry` is explicitly placed below artifacts and is forbidden from importing NumPy, Gymnasium, model frameworks, psycopg, numerical/research layers, or upper application layers. `release` cannot depend on evaluation, data, simulation, RL, serving, integrations, workflows, Studio, or model frameworks. `learning` and the core training protocol cannot depend on Stable-Baselines3, sb3-contrib, or Torch. `workflows` may use integration interfaces but may not directly import model frameworks. Serving cannot import training workflows or Stable-Baselines3. Runtime, training, and Studio paths cannot import offline signing modules. Catalog contracts cannot depend on NumPy, model frameworks, psycopg, data, RL, learning, integrations, workflows, or CLI.

## Privileged GPU boundary

Self-hosted GPU runners never execute pull-request-controlled workflows. The maintained full-run workflow is restricted to the protected `main` ref, the repository owner, and the `gpu-full-training` GitHub Environment. External Actions in that privileged workflow are pinned to immutable commit SHAs. Detached full runs expose explicit start, status, and stop operations; failed-container logs and exit status are retained before cleanup.

## Action and baseline contracts

`portfolio_action_v3` supports the maintained baseline-residual controls and direct per-symbol `target_weight:<symbol>` controls. The environment derives the exact dimension from `ActionSpec` and binds symbol order into identity. In target-weight mode, zero action means flat. In residual mode, zero action is exact baseline identity. Training may clip with diagnostics; evaluation and serving can reject out-of-range actions fail closed.

Trend baselines distinguish time-series direction from cross-sectional ranking. `auto` uses time-series trend for a one-symbol universe and cross-sectional trend for multiple symbols. Directional modes preserve signal confidence and cash rather than always normalizing to full gross exposure.

## Market-data and metadata contracts

Bar timestamps are close times and decisions first execute after the decision close. Continuous datasets require regular cadence; session datasets use wall-clock lookup across overnight, weekend, and holiday gaps. Feature values, eligibility, availability, and staleness are point-in-time contracts. Executor-only realized tradability is not exposed to the policy before the transition.

Dataset identity v6 is recomputed from every observation, execution, and accounting scalar/array, including effective-dated instrument filters, funding-event counts, quantity semantics, maker/taker fees, spread, participation, borrow/funding schedules, mark/index prices, corporate actions, cash rates, availability/age, and contract multipliers. The maintained artifact consists of canonical `manifest.json` and deterministic `arrays.npz`. Loading checks exact file closure, rejects symlinks/root escapes, recomputes identity, and validates array allow-list, shape, dtype, ordering, and `MarketDataset` invariants.

Binance `historical_signed` metadata carries market, ordered symbols, coverage range, issue time, source URI, policy version, payload digest, and a purpose-bound signature. The resolver requires exact market, symbol-order, and research-interval coverage. `frozen_snapshot` and `conservative_static` are explicitly non-point-in-time and retain those limitations in dataset identity and reports. They cannot enter final promotion as authenticated historical metadata.

## Stateful execution contract

Normal RL environment transitions use the stateful execution API:

```text
target proposal
  -> risk projection
  -> target-to-order reconciliation
  -> persistent OrderBookState
  -> deterministic admission and OHLC path interpretation
  -> shared processing-bar liquidity allocation
  -> BookState accounting
  -> OrderEvent and capacity evidence
```

Order intents bind dataset ID, target identity, execution-policy digest, symbol, fixed decision-time quantity, type, time in force, limit/stop price, submission/eligibility/expiry indices, reference price, decision equity, and replacement linkage. Pending orders preserve remaining quantity, cumulative fills, trigger state, terminal reason, last processed index, and a monotonic evidence version.

The simulator supports market, limit, and stop-market instructions; latency; IOC, day, and GTC semantics; cancel-and-replace; partial-fill carry; deterministic rejection; gap handling; and persistent stop triggers. One explicit optimistic, neutral, or conservative OHLC path is selected per symbol and bar. Capacity uses the processing bar's volume, a configured trigger-position fraction, and one deterministic symbol-level priority pool. `BookState` remains the accounting authority for cash, quantities, fees, funding, borrow, dividends, cash interest, splits, delisting settlement, margin, and economic termination.

Final promotion requires conservative primary path mode, processing-bar volume capacity, partial-fill carry, complete order-event evidence, conservative trigger-volume fractions, and an execution-policy digest matching the experiment plan. Optimistic and neutral modes are sensitivity tools only. OHLCV cannot recover queue position, hidden liquidity, auctions, or L2 depth.

The compatibility `MarketExecutor.execute_interval` API is a facade over the same target reconciliation and stateful order engine. A residual order is carried only when the caller chains the exact returned `BookState` through the same executor; an unrelated book starts a fresh compatibility order state. The compact `ExecutionResult` retains legacy aggregate fields and does not expose the detailed `OrderEvent` sequence, so release evidence still uses the explicit stateful API.

`execute_stateful_orders()` is a bounded orchestration protocol over four invocation-local services. `StatefulExecutionRuntime` owns books, order state, evidence, and aggregate result construction; `StatefulBarLifecycle` owns corporate actions, open revaluation, carry, mark-to-market, margin, and insolvency phases; `StatefulOrderTransitionProcessor` owns expiry, latency, admission, eligibility, and remainder expiry; and `StatefulSymbolFillProcessor` owns paths, triggers, shared capacity, rounding, cost, and fill evidence. The public result and evidence schemas remain in the maintained orchestration module.

## Risk ordering

Turnover is a soft operational constraint. Concentration, gross leverage, portfolio risk, and emergency drawdown limits are hard constraints applied afterward and validated again. A hard deleveraging requirement may override turnover, and each projection exposes reasons and L1 distance.

## Observation and normalization

The maintained flat observation identity is `baseline_residual_observation_v5`. It carries feature values, feature-level availability/staleness/reasons, active/tradable state, baseline and factor inputs, current/requested portfolios, fill/cost/capacity state, cash/net/gross/margin state, previous action, and seven causal pending-order coordinates per symbol: remaining-notional ratio, order type, order status, age, eligible delay, trigger state, and expiry distance. Remaining time appears only for an explicitly finite-horizon MDP. Normalization statistics are fitted on an explicit train capability, frozen elsewhere, content-addressed, and preserve mask/categorical coordinates exactly.

The maintained multi-timeframe policy uses a structured Dict observation. Completed native sequences are 15m=96 bars, 1h=168, 4h=120, and 1d=60. Each clock has ordered values, availability, and staleness tensors. Timeframe-specific left-padded residual TCNs use per-timestep LayerNorm and dilation schedules whose receptive fields cover the declared window without reading future timesteps.

Cross-asset attention preserves one contextual token per ordered symbol. Learned symbol embeddings bind token identity; the actor receives ordered per-symbol tokens so each target weight remains attached to its symbol. Only critic/global context uses pooled portfolio representations. PPO and behavior cloning share this feature extractor.

The maintained dataset has 226 ordered point-in-time channels across four clocks, including BTC-relative return, rolling BTC correlation/beta, cross-sectional momentum rank, and dispersion. Derived features retain native source age through as-of alignment. Dataset and policy identities bind symbols, feature order, windows, availability rules, and architecture settings.

Training–Serving observation parity includes symbol and feature order, availability, staleness, hybrid/shadow books, pending target, previous action, pending-order remaining/type/status/age/eligibility/trigger/expiry state, raw observation, normalized observation, each member action, and deterministic ensemble action.

## Reward contract

Reward schema v4 prioritizes absolute log-wealth growth. Baseline-relative growth is secondary. Drawdown is penalized only on newly worsening excess drawdown beyond a dead zone. Baseline underperformance uses a fixed real-time rolling window, tolerance, and progressive hinge. Terminal penalties are continuous in equity shortfall rather than fixed jackpots. Every component is returned for audit.

Maintained finite-horizon training, behavior cloning, checkpoint validation, configuration selection, and sealed evaluation use liquidation-at-close terminal accounting. Baseline reward pre-roll and normal episode transitions share decision-time target sizing, stateful reconciliation, processing-bar capacity, latency, partial residual carry, costs, carry, and `BookState` accounting. The pre-roll creates an isolated executor and chains only its returned books, so its pending state cannot leak into the episode proper.

## Artifact store and PostgreSQL catalog

Training output is first written to `ArtifactStore/.staging/<run-id>`. `run.json` binds each declared file by relative path, byte size, and SHA-256 together with dataset, environment, training-config, ensemble, execution, and evidence identities. Only a fully validated run moves atomically to `runs/<run-id>` and updates `latest.json`; partial failures move to `failed/<run-id>`.

Canonical JSON conversion now has one standard-library implementation in `trade_rl.domain.canonical_json`. Artifact serialization and catalog cache-key hashing use the same byte representation, preventing identity drift between the filesystem and PostgreSQL metadata paths.

The optional PostgreSQL catalog stores reusable artifact metadata, canonical cache keys, dependencies, and lifecycle status. A dedicated `PostgresSealedTestReservationStore` owns the evaluation-specific one-time reservation SQL. `PostgresArtifactCatalog.reserve_sealed_test_access()` remains a temporary compatibility delegate, so existing workflow construction stays stable while responsibilities are separated. The filesystem artifact remains authoritative: the catalog stores its verified location and identity, not mutable numerical payloads.

A PostgreSQL uniqueness boundary prevents separate workflow processes from opening the same `(experiment_plan_digest, dataset_id, fold_index)` sealed test twice. When PostgreSQL is not configured, workflows retain filesystem behavior, but cross-process sealed-test uniqueness is not provided by an in-memory object alone.

## Training and nested walk-forward

`trade-rl train run` loads a validated dataset, resolves content-addressed alpha/factor artifacts independently of path, requires complete causal reward pre-roll, fits flat and sequence normalizers only on the declared training capability, constructs the real environment, trains one policy per seed, writes canonical configuration and ensemble manifests, validates the run, and publishes atomically.

Direct training without selection evidence is `research_exploratory`. A `research_selected_final` run requires an immutable `SelectionAuthorization` before normalizer fitting or model construction. It binds the walk-forward run digest, gate-evidence digest, dataset ID, selected configuration, canonical candidate-config digest, and fixed seed set.

`trade-rl walk-forward run` provides disjoint train, checkpoint-validation, configuration-selection, and sealed-test ranges. Fold train views include required trend, reward, and sequence history without treating pre-range history as fit data. Normalizers fit only the chronological fold-train capability and remain frozen thereafter.

Checkpoint validation selects a deterministic winner for each predeclared seed. Configuration selection retains per-seed score, worst-seed uplift, dispersion, success fraction, turnover, cost fraction, and drawdown evidence, then evaluates the exact deterministic mean ensemble. The same ensemble rule is used for sealed evaluation and serving. Sealed returns never flow back into training or selection.

Independent folds expose distribution summaries rather than synthetic continuous-account metrics. Continuous results require contiguous ranges and verified account-state handoff. Final `policy.zip` files remain authoritative model artifacts.

## Training telemetry and Studio boundary

Exploratory SB3 training emits seed-scoped, append-only `training_telemetry_v1` JSONL. Normal intervals are sampled; position, risk, emergency, and terminal events are retained. Telemetry is best-effort diagnostics: a telemetry write failure may disable visualization but must not stop learning.

Telemetry JSON booleans are parsed strictly rather than by truthiness. Status and cursor reads maintain an atomic sparse sidecar index bound to device, inode, and indexed byte size. Unchanged streams seek from a stored checkpoint; appends scan only newly written complete lines; replacement, truncation, malformed sidecar data, or identity mismatch triggers a full rebuild from byte zero. JSONL remains the source of truth.

Studio reads telemetry only beneath a known job's declared artifact root and run namespaces. It rejects project-root escapes, symlinks, unknown jobs, records whose seed differs from the selected stream, and multiple distinct files that claim the same seed. Studio never ranks checkpoints from replay data, retrains a policy, activates a bundle, or submits an exchange order.

A seed stream may contain records from multiple vector `environment_id` values and successive auto-reset episodes, but each new record can also carry a nullable producer-issued `episode_id`. Live Training selects one environment and its current episode before deriving the chart, cursor, price, PnL, baseline, drawdown, events, playback, and jump state. Explicit producer identity is preferred; historical null-ID records retain terminal and counter-rollback segmentation.

Telemetry writers and readers coordinate through short OS-locked append/read/index transactions. Cursors are bound to an opaque sidecar-index generation, and mixed Status/Events generations are discarded before records are published. JSONL remains the append-only source of truth and telemetry remains exploratory rather than fitting, selection, release, Serving-activation, or exchange evidence.

## Export, serving, and release

ONNX and TorchScript export a deterministic actor wrapper rather than optimizer state. Each export is compared with Stable-Baselines3 deterministic predictions on a fixed finite corpus. Requested ONNX failure rejects the run. TorchScript is best-effort and records an explicit unsupported reason when conversion cannot be proven safe.

Candidate bundle v5 identity `serving_bundle_v5` binds action size/names/spec digest, observation schema/size, environment, normalizers, alpha/factor artifacts, selected-final run, selection proposal and authorization, walk-forward/gate evidence, fresh confirmation, execution evidence, and policy-loader contract. Approval remains detached from bundle identity. An offline Ed25519 signer creates a purpose-bound `ReleaseAttestation`; registry and runtime load public verification keys only.

`StableBaselines3PolicyLoader` lives in the integration layer. It verifies `policy-loader.json`, loads every PPO, SAC, TD3, or TQC member, and executes deterministic probe observations before runtime activation. Every member must return a finite dynamic action vector inside `[-1, 1]`; the ensemble is averaged only when all members succeed. Structured prediction requires a monotonic identity-bound `ServingStateSnapshot`.

Direct exchange websocket ingestion, order submit/cancel/replace, broker reconciliation, venue kill switches, production secrets, and operational alerting remain outside this repository.

## Production gate

Production remains `NO-GO` until the maintained GPU verification, at least 180 OOS days, a strictly positive paired block-bootstrap lower bound on RL-minus-baseline daily log excess, signed fresh confirmation, conservative complete execution evidence, and paper-trading reconciliation all pass. Passing repository CI proves code and artifact integrity at one source head; it does not prove profitability or authorize capital deployment.

See [Research Status](RESEARCH_STATUS.md), the original [2026-07-22 documentation and architecture audit](verification/2026-07-22-documentation-and-architecture-audit.md), the historical [post-remediation architecture audit](verification/2026-07-22-post-merge-architecture-audit.md), and the current [2026-07-23 architecture-audit closeout](verification/2026-07-23-architecture-audit-closeout.md).
