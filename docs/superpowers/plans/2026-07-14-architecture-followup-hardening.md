# Architecture Follow-up Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the confirmed contract-multiplier, release-attestation, portfolio-risk, covariance, configuration, identity, and portability gaps on maintained paths.

**Architecture:** Preserve existing layer boundaries. Environment-created books use dataset quantity semantics; registry installations contain a bundle directory and external attestation sidecar; portfolio risk derives deterministic causal inputs from trailing dataset history and is bound into all experiment identities.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, pytest, Ruff, Mypy, import-linter, GitHub Actions.

## Global Constraints

- No future bars may be used to resolve portfolio risk inputs.
- External release approval must remain outside the candidate bundle digest.
- Portfolio constraints are hard and may override soft turnover throttles.
- Existing legacy in-bundle release manifests remain readable.
- Newly built canonical datasets use identity schema v6.
- Production exchange routing remains NO-GO.

---

### Task 1: Contract-multiplier environment integrity

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/architecture/test_architecture_followup.py`

**Interfaces:**
- `ResidualMarketEnv` creates every `BookState` with `dataset.contract_multipliers`.
- Restore mode rejects a book whose multipliers differ from the dataset.

- [ ] Write a failing environment test with a non-unit contract multiplier.
- [ ] Run the targeted test and verify the executor rejects the incorrectly initialized book.
- [ ] Pass dataset multipliers through constructor and `_make_initial_book`; validate restore mode.
- [ ] Run the targeted test and verify it passes.

### Task 2: External-attestation registry installation

**Files:**
- Modify: `trade_rl/serving/registry.py`
- Test: `tests/serving/test_registry_external_attestation.py`

**Interfaces:**
- Registry version directory contains `bundle/` and optional `bundle.release.json`.
- `ServingRegistry.activate(source) -> ServingBundle` returns the fully loaded installed bundle.
- `ServingRegistry.active_bundle() -> ServingBundle` resolves the installed bundle subdirectory.

- [ ] Write a failing test that creates a v4 candidate bundle plus external `ReleaseAttestation`.
- [ ] Verify activation fails because the sidecar is not copied.
- [ ] Stage bundle and sidecar together, validate from staging, install atomically, and return `installed`.
- [ ] Verify activation and active reload retain release and normalizer state.

### Task 3: Causal portfolio-risk integration

**Files:**
- Modify: `trade_rl/risk/portfolio.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/configuration.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/workflows/market_walk_forward_config.py`
- Modify: `trade_rl/workflows/walk_forward_evaluation.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/cli/app.py`
- Test: `tests/risk/test_portfolio_risk.py`
- Test: `tests/architecture/test_architecture_followup.py`

**Interfaces:**
- `PortfolioRiskConfig.lookback_hours: float`.
- `PortfolioRiskModel.minimum_history_for(dataset) -> int`.
- `PortfolioRiskModel.resolve_inputs(dataset, index) -> PortfolioRiskInputs`.
- `ResidualMarketEnv(..., portfolio_risk=PortfolioRiskModel(...))`.
- `TrainingRunConfig.portfolio_risk: PortfolioRiskConfig`.

- [ ] Write failing tests for environment projection, identity changes, causal history, and config parsing.
- [ ] Verify failures show the model is disconnected.
- [ ] Add deterministic trailing risk input resolution and connect it after pre-trade risk.
- [ ] Bind config to training, walk-forward, CLI, manifests, and environment digest.
- [ ] Verify targeted tests pass.

### Task 4: Covariance fail-closed validation

**Files:**
- Modify: `trade_rl/risk/portfolio.py`
- Test: `tests/risk/test_portfolio_risk.py`

- [ ] Write failing tests for asymmetric and materially indefinite covariance matrices.
- [ ] Verify the current implementation accepts them or masks negative variance.
- [ ] Add symmetry and eigenvalue validation with a numerical tolerance.
- [ ] Verify valid PSD covariance behavior remains unchanged.

### Task 5: Strict configuration closure and identity v6

**Files:**
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/data/identity.py`
- Modify: `trade_rl/data/market.py`
- Modify: `README.md`
- Test: `tests/workflows/test_training_run_config_strict.py`
- Test: `tests/data/test_market_dataset_identity_v2.py`

- [ ] Write failing tests for unknown top-level and nested keys and v6 identity emission.
- [ ] Add exact allowed-key validation helpers and apply them to maintained training config mappings.
- [ ] Advance canonical identity constants and payloads to v6; align README.
- [ ] Run targeted tests.

### Task 6: Filesystem portability and full verification

**Files:**
- Modify: `trade_rl/artifacts/store.py`
- Test: `tests/artifacts/test_store.py`

- [ ] Add a Windows-branch test for directory sync.
- [ ] Skip directory fsync on Windows consistently with serving registry behavior.
- [ ] Run Ruff, format, Mypy, import-linter, vulture, targeted tests, full tests with branch coverage, critical coverage, and CLI smoke.
- [ ] Remove temporary patch workflow files before committing the product changes.
