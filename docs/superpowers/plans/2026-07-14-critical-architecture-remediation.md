# Critical Architecture Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect contract multipliers, external release attestations and portfolio risk to the maintained training, evaluation and serving paths.

**Architecture:** Keep `ResidualMarketEnv` as the execution composition root, add portfolio risk as a deterministic post-pretrade projection, and make `ServingRegistry` own the copied external attestation beside each installed immutable bundle. Preserve default behavior with empty `PortfolioRiskConfig`.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, pytest, GitHub Actions.

## Global Constraints

- Keep direct exchange order routing `NO-GO`.
- Do not estimate covariance, beta or stress values from future data.
- Preserve legacy internal `release.json` loading.
- Missing `portfolio_risk` configuration must remain backward compatible.
- Add tests before production changes and verify the intended failure.

---

### Task 1: Contract multiplier environment path

**Files:**
- Modify: `tests/rl/test_environment.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Consumes: `MarketDataset.contract_multipliers`
- Produces: all environment-created `BookState` instances with matching multipliers

- [ ] Add a failing test that constructs a dataset with multiplier `0.1`, resets the environment and performs one step.
- [ ] Add a failing restore-mode test for a mismatched multiplier vector.
- [ ] Run the selected tests and confirm executor mismatch or missing validation failures.
- [ ] Pass `dataset.contract_multipliers` through `_make_initial_book`, constructor initialization and restore validation.
- [ ] Run the selected tests and confirm they pass.

### Task 2: External attestation registry installation

**Files:**
- Modify: `tests/serving/helpers.py`
- Modify: `tests/serving/test_registry.py`
- Modify: `trade_rl/serving/registry.py`

**Interfaces:**
- Consumes: `default_attestation_path(source)`
- Produces: installed bundle directory plus sibling external attestation

- [ ] Add a helper option that creates a real external `ReleaseAttestation` instead of internal `release.json`.
- [ ] Add a failing activation/reload test using the external form.
- [ ] Add a failure-preservation test for an invalid external attestation.
- [ ] Run selected registry tests and confirm staging loses the sibling attestation.
- [ ] Copy and validate the attestation beside staging and destination, clean partial copies on failure, and return the fully loaded installed bundle.
- [ ] Run selected registry tests and confirm they pass.

### Task 3: Portfolio risk identity and execution integration

**Files:**
- Modify: `tests/rl/test_environment.py`
- Modify: `tests/workflows/test_training_run.py`
- Modify: `tests/workflows/test_market_walk_forward_config.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/configuration.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/workflows/walk_forward_evaluation.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/workflows/market_walk_forward_config.py`

**Interfaces:**
- Consumes: `PortfolioRiskConfig`, `PortfolioRiskModel.constrain(...)`
- Produces: final deterministic target projection and identity-bound run configuration

- [ ] Add failing tests showing portfolio risk changes environment digest and executed target.
- [ ] Add failing parsing/digest tests for training and walk-forward configuration.
- [ ] Run selected tests and confirm portfolio risk is absent.
- [ ] Add `portfolio_risk` to configuration dataclasses, JSON parsing and digest payloads.
- [ ] Inject `PortfolioRiskModel` in training and evaluation environment factories.
- [ ] Derive current market notional from causal price/volume data and project the pretrade result before execution.
- [ ] Run selected tests and confirm they pass.

### Task 4: Verification and publication

**Files:**
- Modify only files required by formatting or static checks.

- [ ] Run Ruff and formatter checks.
- [ ] Run Mypy and Import Linter.
- [ ] Run focused tests.
- [ ] Run the full test suite with branch coverage.
- [ ] Run the CLI smoke test.
- [ ] Remove temporary automation files used to apply the patch.
- [ ] Push the final branch and open a pull request to `main`.