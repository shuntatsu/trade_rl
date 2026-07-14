# Full Architecture Hardening v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dataset identity, cross-asset accounting, causal signals, walk-forward evidence, serving activation, and release approval fail-closed and reproducible.

**Architecture:** Add versioned v2 contracts while preserving explicit legacy readers. Maintained builders, workflows, validators, bundle activation, and release paths move to the v2 contracts. Pure domain helpers remain framework-independent; SB3-specific behavior stays behind adapters.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3, PyTorch, pytest, Hypothesis-compatible property tests, Ruff, Mypy, GitHub Actions.

## Global Constraints

- New published artifacts must use canonical content identities recomputed on write and load.
- Legacy artifacts may be read only through explicit compatibility functions and are not release eligible.
- Range isolation, point-in-time availability, quantity semantics, and activation validation fail closed.
- Independent-fold results must not be reported as continuous-account total return or drawdown.
- Production exchange routing remains out of scope and must be reported as unavailable.
- Tests are added alongside each contract and the full suite is run after all implementation tasks.

---

### Task 1: Complete dataset identity and exact artifact closure

**Files:**
- Modify: `trade_rl/data/identity.py`
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/data/artifact_codec.py`
- Modify: `trade_rl/data/artifacts.py`
- Modify: `trade_rl/data/dataset_artifact.py`
- Test: `tests/data/test_market_dataset_identity_v2.py`
- Test: `tests/data/test_market_dataset_artifact.py`

**Interfaces:**
- Produces: `MarketDataset.identity_arrays() -> dict[str, np.ndarray]`
- Produces: `MarketDataset.identity_metadata() -> dict[str, object]`
- Produces: `MarketDataset.recomputed_dataset_id() -> str`
- Produces: `verify_exact_artifact_files(root: Path, allowed: frozenset[str]) -> None`

- [ ] Add tests that mutate every observation/execution/accounting array and assert `dataset_id` changes.
- [ ] Add tests rejecting identity-less publication, arbitrary IDs, symlinks, and undeclared files.
- [ ] Expand `DATASET_ID_ARRAY_FIELDS` to all resolved semantic arrays and include scalar calendar, volume-unit, multiplier, feature-name, and schema metadata.
- [ ] Make `MarketDataset` recompute and compare identity whenever a canonical payload is present; add an explicit `legacy_identity_unverified` marker for compatibility construction.
- [ ] Require verified canonical identity in maintained artifact writers and recompute again after loading.
- [ ] Enforce exact file allow-lists and reject symlinks/root escapes.
- [ ] Run targeted data identity and artifact tests.

### Task 2: Unify feature availability and staleness

**Files:**
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/data/test_feature_availability_v2.py`
- Test: `tests/rl/test_observation_v2.py`

**Interfaces:**
- Produces resolved arrays `feature_age_hours`, `feature_staleness_ratio`, `global_feature_available`, `global_feature_age_hours`, and `global_feature_staleness_ratio`.

- [ ] Write tests proving stale carried features expose positive age/ratio and unavailable global features expose an unavailable mask.
- [ ] Replace ambiguous observation reads of `feature_staleness_hours` with the resolved age and ratio arrays built by `MarketDatasetBuilder`.
- [ ] Calculate feature age from timestamps rather than bar count for session data.
- [ ] Set global feature availability from actual cross-sectional samples and mark absent samples unavailable.
- [ ] Include all availability fields in identity and observation layout identity.
- [ ] Run targeted builder and observation tests.

### Task 3: Instrument-aware quantity, liquidity, and book accounting

**Files:**
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/simulation/accounting.py`
- Modify: `trade_rl/simulation/execution.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/simulation/test_instrument_quantity_semantics.py`
- Test: `tests/simulation/test_execution.py`
- Test: `tests/simulation/test_accounting_validation.py`

**Interfaces:**
- Produces: `MarketDataset.quantity_notional(index: int, quantities: np.ndarray, prices: np.ndarray | None = None) -> np.ndarray`
- Produces: `MarketDataset.notional_to_quantity(index: int, notionals: np.ndarray, prices: np.ndarray | None = None) -> np.ndarray`
- `BookState` consumes immutable `contract_multipliers: np.ndarray`.

- [ ] Add base-asset, quote-notional-volume, and contracts-volume tests covering capacity, fill, position value, margin, and settlement.
- [ ] Store contract multipliers in `BookState`; calculate position values with multipliers.
- [ ] Replace raw `price * volume` capacity with `dataset.market_notional()`.
- [ ] Convert target notionals to quantities and filled quantities to notionals using dataset helpers.
- [ ] Apply multiplier-aware valuation to borrow, dividends, liquidation, and turnover.
- [ ] Validate restored books against dataset multiplier identity.
- [ ] Run simulation and environment tests.

### Task 4: Elapsed-time session semantics and dataset build configuration

**Files:**
- Modify: `trade_rl/data/contracts.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/data/source.py`
- Modify: `trade_rl/cli/app.py`
- Modify: `trade_rl/simulation/execution.py`
- Modify: `trade_rl/simulation/accounting.py`
- Test: `tests/data/test_session_dataset_builder.py`
- Test: `tests/simulation/test_session_carry.py`

**Interfaces:**
- Produces `SessionCalendarConfig` and `MarketBuildConfig.calendar_kind`.
- Produces `MarketDataset.elapsed_year_fraction(start_index: int, end_index: int) -> float`.

- [ ] Add tests building a weekday session dataset with overnight/weekend gaps.
- [ ] Add calendar configuration and preserve source timestamps for session datasets instead of forcing a continuous grid.
- [ ] Charge borrow and cash interest by actual elapsed year fraction.
- [ ] Resolve time-based latency and feature age against timestamps.
- [ ] Add CLI configuration fields for calendar kind, volume unit, multiplier, listing, and delisting.
- [ ] Run session and CLI tests.

### Task 5: Causal signal artifact v2 and fold-local factories

**Files:**
- Modify: `trade_rl/artifacts/signals.py`
- Modify: `trade_rl/rl/signal_artifacts.py`
- Create: `trade_rl/workflows/fold_signals.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Test: `tests/artifacts/test_signal_artifact_v2.py`
- Test: `tests/workflows/test_fold_signals.py`

**Interfaces:**
- Produces `SignalArrayManifestV2` with fit/prediction ranges, generator digests, availability, and validity arrays.
- Produces `FoldSignalProviderFactory.build(fold, stage, dataset_view) -> FoldSignals`.

- [ ] Add tests rejecting future fit ranges, prediction-range escapes, delayed values, invalid masks, symbol mismatch, and arbitrary generator lineage.
- [ ] Store values, validity, and `available_at` in deterministic signal artifacts.
- [ ] Validate every prediction against the decision timestamp and authorized prediction range.
- [ ] Add fold-local provider factory interfaces; support trend-only with an explicit empty factory.
- [ ] Remove the impossible `fit_stop <= train.start == 0` global-artifact loading path from maintained walk-forward.
- [ ] Run signal and walk-forward tests.

### Task 6: Capability-separated nested walk-forward and sealed test ledger

**Files:**
- Modify: `trade_rl/data/dataset_artifact.py`
- Create: `trade_rl/evaluation/walk_forward/capabilities.py`
- Create: `trade_rl/evaluation/walk_forward/sealed_test.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Test: `tests/evaluation/walk_forward/test_capabilities.py`
- Test: `tests/evaluation/walk_forward/test_sealed_test.py`

**Interfaces:**
- Produces stage-specific immutable dataset capabilities without a public parent dataset reference.
- Produces `SealedTestAccessRecord` and `SealedTestLedger.authorize_once(...)`.

- [ ] Add tests proving each stage cannot request indices outside its assigned half-open range.
- [ ] Materialize isolated stage datasets and hide the parent dataset from trainer/evaluator interfaces.
- [ ] Bind test opening to experiment-plan, dataset, fold, selected configuration, and policy digests.
- [ ] Reject repeated sealed-test opening for the same authorization.
- [ ] Persist access records in walk-forward artifacts.
- [ ] Run capability and fold workflow tests.

### Task 7: Truthful fold evidence and independent-fold reporting

**Files:**
- Modify: `trade_rl/evaluation/walk_forward/stitching.py`
- Modify: `trade_rl/evaluation/metrics.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `trade_rl/workflows/walk_forward_evaluation.py`
- Create: `trade_rl/evaluation/fold_metrics.py`
- Test: `tests/evaluation/test_fold_metrics.py`
- Test: `tests/evaluation/walk_forward/test_stitching.py`

**Interfaces:**
- Produces `ExecutionEvidence` and `FoldPerformanceEvidence`.
- Produces `IndependentFoldSummary`; continuous metrics are available only for `CONTINUOUS_ACCOUNT`.

- [ ] Add tests showing turnover, costs, funding, borrow, fills, and terminations survive evaluation.
- [ ] Reject continuous total return/drawdown requests for independent folds.
- [ ] Report fold metric distribution, weighted mean, median, win rate, worst fold, and explicit gaps.
- [ ] Require state digest handoff and contiguous ranges for continuous mode.
- [ ] Annualize session results from elapsed time rather than an unchecked arbitrary frequency.
- [ ] Run evaluation tests.

### Task 8: Robust inference and evidence-bound gates

**Files:**
- Modify: `trade_rl/evaluation/bootstrap.py`
- Modify: `trade_rl/evaluation/comparisons.py`
- Modify: `trade_rl/domain/evaluation.py`
- Modify: `trade_rl/evaluation/gates.py`
- Test: `tests/evaluation/test_bootstrap.py`
- Test: `tests/evaluation/test_gates.py`

**Interfaces:**
- Produces corrected moving-block p-values and fold-level resampling.
- `GateCheck` records metric, observed value, comparator, threshold, evidence digest, and implementation digest.

- [ ] Add deterministic tests for finite-sample p-value correction and block resampling.
- [ ] Replace zero-possible p-values with `(extreme + 1) / (draws + 1)` and use circular blocks.
- [ ] Add fold-level comparison for independent folds.
- [ ] Make gates recomputable from immutable evaluation evidence and reject boolean-only checks in release-eligible schemas.
- [ ] Run inference and gate tests.

### Task 9: Semantic normalization and runtime observation pipeline

**Files:**
- Modify: `trade_rl/rl/normalization.py`
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Create: `trade_rl/serving/observations.py`
- Modify: `trade_rl/serving/runtime.py`
- Test: `tests/rl/test_normalization_semantics.py`
- Test: `tests/serving/test_observation_pipeline.py`

**Interfaces:**
- Produces semantic transform groups and dataset/range/layout-bound normalizer identity.
- Produces `ServingObservationPipeline.build(raw_state) -> np.ndarray`.

- [ ] Add tests that endogenous weights/actions/masks remain pass-through or fixed-scaled and exogenous features are train-fit only.
- [ ] Remove environment rollout with zero actions as the normalization estimator.
- [ ] Fit exogenous arrays directly on the train capability and use fixed transforms for endogenous fields.
- [ ] Load and verify the observation builder and normalizer in serving before policy inference.
- [ ] Run normalization and serving observation tests.

### Task 10: Candidate bundle, activation probe, and external release attestation

**Files:**
- Modify: `trade_rl/serving/bundle.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `trade_rl/serving/sb3_loader.py`
- Modify: `trade_rl/registry/serving.py`
- Modify: `trade_rl/domain/release.py`
- Create: `trade_rl/release/attestation.py`
- Test: `tests/serving/test_activation_probe.py`
- Test: `tests/release/test_attestation.py`

**Interfaces:**
- Candidate bundle digest excludes release identity.
- Produces `ReleaseAttestation.create(...)` and `verify_release_attestation(...)`.
- Produces `PolicyLoader.probe(bundle, observations) -> ProbeReport`.

- [ ] Add tests proving bundle and release hashes are non-circular.
- [ ] Add tests proving bad member observation/action shapes, non-finite output, bounds, undeclared files, symlinks, and normalizer mismatch fail before activation.
- [ ] Remove `release_digest` from candidate bundle identity.
- [ ] Bind external attestation to bundle, dataset artifact, evaluation/gate evidence, selected policy, code, dependency, approver, and time.
- [ ] Make registry/runtime require a verified attestation for production activation.
- [ ] Run serving, registry, and release tests.

### Task 11: Typed run manifests, portable references, and provenance

**Files:**
- Modify: `trade_rl/artifacts/run_manifest.py`
- Create: `trade_rl/artifacts/provenance.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Test: `tests/artifacts/test_run_manifests_v2.py`
- Test: `tests/artifacts/test_provenance.py`

**Interfaces:**
- Produces separate `TrainingRunManifestV2` and `WalkForwardRunManifestV2`.
- Produces `capture_runtime_provenance(...) -> RuntimeProvenance`.

- [ ] Add tests rejecting semantic manifest misuse and path-dependent identities.
- [ ] Store artifact digests in identity payloads and local paths only in diagnostics.
- [ ] Capture git SHA, dirty state, lock digest, runtime/library versions, platform/hardware, and seed/determinism config.
- [ ] Enforce exact file closure and run-type-specific validators.
- [ ] Run manifest and provenance tests.

### Task 12: Typed algorithm configs and resumable off-policy state

**Files:**
- Modify: `trade_rl/rl/training.py`
- Create: `trade_rl/rl/algorithm_configs.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/rl/test_algorithm_configs.py`
- Test: `tests/rl/test_training_resume.py`

**Interfaces:**
- Produces `PPOConfig`, `SACConfig`, `TD3Config`, and `TQCConfig`.
- Produces off-policy replay-buffer artifact and resume metadata.

- [ ] Add tests rejecting parameters irrelevant to the selected algorithm.
- [ ] Split common and algorithm-specific configuration.
- [ ] Save/load replay buffers and resume metadata for supported off-policy models.
- [ ] Include resume-state identity in run manifests without requiring resume for inference artifacts.
- [ ] Run training config and resume tests.

### Task 13: Pluggable portfolio risk layer

**Files:**
- Create: `trade_rl/risk/portfolio.py`
- Modify: `trade_rl/risk/pretrade.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/risk/test_portfolio_risk.py`

**Interfaces:**
- Produces `PortfolioRiskModel` protocol and deterministic `PortfolioRiskConfig`.
- Supports volatility target, liquidity cap, concentration, net/beta, and stress constraints.

- [ ] Add tests for each hard constraint and fail-closed non-finite inputs.
- [ ] Apply portfolio risk after proposal composition and before execution while preserving emergency deleveraging.
- [ ] Include risk config and implementation digest in environment identity.
- [ ] Run risk and environment tests.

### Task 14: Framework boundary and optional training dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `trade_rl/integrations/sb3_training.py`
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `.importlinter`
- Test: `tests/test_architecture_contract.py`

**Interfaces:**
- Core keeps `PolicyTrainingBackend` protocol; SB3 implementation lives in integrations and is installed through `train-sb3` extra.

- [ ] Add architecture tests rejecting framework imports in core domain/data/evaluation/serving modules.
- [ ] Move SB3 backend and checkpoint loader to integration modules.
- [ ] Split minimal core dependencies from training extras.
- [ ] Update workflows to depend on protocol/factory injection.
- [ ] Run import-linter, Mypy, and architecture tests.

### Task 15: End-to-end verification, coverage, and documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `RESEARCH_STATUS.md`
- Create: `tests/e2e/test_research_to_serving_v2.py`

**Interfaces:**
- End-to-end path produces a verified candidate bundle and release attestation, then activates and predicts.

- [ ] Add a small real `ResidualMarketEnv` PPO path from dataset construction through runtime prediction.
- [ ] Add smoke coverage for SAC, TD3, and TQC configuration/backend construction; run dependency-available models where practical.
- [ ] Raise execution, evaluation, artifacts, serving, registry, and release critical branch thresholds to at least 90%.
- [ ] Add supported Python/platform matrix and keep dead-code reporting blocking for maintained packages.
- [ ] Update documentation to distinguish research-ready, paper-serving-ready, and live-trading-unavailable states.
- [ ] Run `ruff check .`, `ruff format --check .`, `mypy trade_rl`, import-linter, full pytest with branch coverage, critical coverage checker, and CLI smoke tests.
- [ ] Review the complete diff against every section of the design spec and remove compatibility paths accidentally used by maintained workflows.
