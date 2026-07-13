# Causal Market Data Final Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every remaining causal-observation, Serving-contract, artifact-atomicity, and dataset-identity gap from PR #24.

**Architecture:** Separate execution truth from policy-visible tradability, centralize causal market-input resolution, bind its identity into Serving, and move dataset identity into a shared recomputable contract. Publish immutable artifacts through a completed staging directory and one final rename.

**Tech Stack:** Python 3.12, NumPy, pathlib/tempfile, pytest, GitHub Actions.

## Global Constraints

- Execution may use realized future-bar truth; policy construction may use only information available by the decision time.
- Serving must reject every dataset, schema, market-input, or size mismatch before policy execution.
- Dataset identity must be reproducible from persisted metadata and stored arrays.
- Artifact destinations are immutable and must not be overwritten.
- Production status remains `NO-GO` pending fresh sealed Walk-Forward evaluation.

---

### Task 1: Policy-visible tradability

**Files:**
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/rl/observations.py`
- Test: `tests/data/test_information_availability.py`
- Test: `tests/rl/test_observation_causality.py`

- [x] Add failing tests proving delayed current-row tradability cannot alter observation bytes or global tradable fraction.
- [x] Add `MarketDataset.observable_tradable(index)` and use it in policy observations.
- [x] Build global tradable fraction from `tradable & information_available`.
- [x] Run focused tests and commit.

### Task 2: Causal market-input resolution and Serving contract

**Files:**
- Create: `trade_rl/rl/market_inputs.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/serving/bundle.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `tests/serving/helpers.py`
- Test: `tests/rl/test_decision_eligibility.py`
- Test: `tests/serving/test_runtime.py`
- Test: `tests/serving/test_runtime_schema_guard.py`
- Test: `tests/serving/test_shared_observation_builder.py`

- [x] Add failing tests for full-dataset Alpha access, raw vector contract bypass, and caller-supplied Trend/Alpha bypass.
- [x] Add copied `CausalMarketView` and `MarketInputResolver` with deterministic digest.
- [x] Make the environment and Serving use the resolver.
- [x] Bind resolver digest into bundle schema v3 and require dataset/schema identities for raw prediction.
- [x] Run focused tests and commit.

### Task 3: Recomputable dataset identity

**Files:**
- Create: `trade_rl/data/identity.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/data/artifact.py`
- Test: `tests/data/test_market_builder.py`
- Test: `tests/data/test_market_artifact.py`

- [x] Add failing tests for tampered dataset IDs and manifests with valid outer digests.
- [x] Centralize canonical array hashing and identity payload construction.
- [x] Hash final stored dtypes and validate identity inside `MarketDataset`.
- [x] Persist identity payload and recompute during artifact loading.
- [x] Run focused tests and commit.

### Task 4: Atomic immutable artifact publication

**Files:**
- Modify: `trade_rl/data/artifact.py`
- Modify: `tests/cli/test_data_build_cli.py`
- Test: `tests/data/test_market_artifact.py`

- [x] Add failing tests for existing-output preservation and failed-stage cleanup.
- [x] Write and validate in a sibling staging directory.
- [x] Publish by one rename into a nonexistent destination and reject overwrites.
- [x] Run focused tests and commit.

### Task 5: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `docs/ARCHITECTURE.md`

- [x] Document observable tradability, causal Alpha/Serving resolution, recomputable identity, and immutable artifact publication.
- [ ] Run Ruff, format check, Mypy, Import Linter, Vulture advisory, full pytest with branch coverage, and CLI smoke.
- [ ] Update PR #24 with final head and verification evidence.

The normal fail-fast CI runs from the commit containing this note; its result is the completion gate for the final two items.
