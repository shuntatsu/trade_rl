# Architecture Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the architecture audit findings without weakening research causality, artifact identity, or fail-closed behavior.

**Architecture:** Harden the privileged GPU workflow as a supervised default-branch operation, introduce a signed Binance history scope contract, make missing package layers enforceable, bind final training to walk-forward authorization, restrict release key purposes, remove ambiguous dataset compatibility exports, and add repository-owned workflow policy checks.

**Tech Stack:** Python 3.12, dataclasses, canonical JSON/SHA-256/HMAC, GitHub Actions, Docker Compose, pytest, import-linter, uv.

## Global Constraints

- Direct exchange order routing remains outside scope and production status remains `NO-GO`.
- No future market information may enter features, normalization, teacher data, selection, or sealed evaluation.
- No new runtime dependency is introduced in this change.
- Privileged self-hosted runners must never execute pull-request-controlled workflow code.
- All new behavior follows red-green-refactor and must pass the complete existing CI matrix.

---

### Task 1: Privileged workflow policy and supervised GPU operations

**Files:**
- Create: `.github/check_workflow_security.py`
- Modify: `.github/workflows/launch-binance-frozen-226.yml`
- Modify: `.github/workflows/ci.yml`
- Create: `tests/architecture/test_workflow_security.py`
- Create: `examples/binance-multitimeframe/full_run_supervisor.py`
- Create: `tests/examples/test_full_run_supervisor.py`

**Interfaces:**
- Produces: `validate_workflow_security(root: Path) -> tuple[str, ...]`
- Produces: supervisor operations `start`, `status`, and `stop` with JSON evidence under the shared run root.

- [ ] Write tests proving a PR-triggered self-hosted workflow, mutable action reference, non-main dispatch, and missing environment are rejected.
- [ ] Run `uv run pytest -q tests/architecture/test_workflow_security.py` and confirm failure because the checker does not exist.
- [ ] Implement the checker using `pathlib` and text/YAML-policy inspection without a new YAML dependency.
- [ ] Rewrite the GPU workflow to use `workflow_dispatch` plus scheduled status, `environment: gpu-full-training`, owner/main guards, immutable current checkout provenance, and explicit start/status/stop operations.
- [ ] Write supervisor tests for unique labels, atomic status writes, terminal exit reporting, and idempotent stop.
- [ ] Implement the supervisor wrapper and wire it into the workflow.
- [ ] Add the checker to CI and run focused tests.

### Task 2: Signed Binance history scope and positive execution rules

**Files:**
- Modify: `trade_rl/workflows/binance_metadata_modes.py`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Modify: `tests/workflows/test_binance_metadata_modes.py`
- Modify: `tests/examples/test_binance_metadata_mode_runner.py`

**Interfaces:**
- Produces: `BinanceHistoricalSignedScope`
- Changes: `resolution_from_historical_signed(..., signed_scope: BinanceHistoricalSignedScope, ...)`

- [ ] Add failing tests for zero tick/lot/minimum values, market mismatch, symbol-order mismatch, coverage mismatch, rules after coverage, and missing start coverage.
- [ ] Run the focused tests and confirm the new cases fail for the expected reason.
- [ ] Implement the immutable signed scope and strict positive Binance validation.
- [ ] Upgrade the maintained signed payload loader to `binance_instrument_rule_history_v3` and bind all scope fields from the authenticated payload.
- [ ] Run focused metadata and runner tests.

### Task 3: Enforce learning and release package boundaries

**Files:**
- Modify: `.importlinter`
- Modify: `docs/ARCHITECTURE.md`
- Create: `tests/architecture/test_declared_layers.py`

**Interfaces:**
- Produces: explicit layer entries for `trade_rl.learning` and `trade_rl.release`.

- [ ] Write a failing test asserting both packages are present in the layer contract and release cannot import serving/integrations/workflows.
- [ ] Run the focused architecture test and confirm failure.
- [ ] Add the layers in dependency order and add forbidden contracts for release and learning frameworks.
- [ ] Update the architecture responsibility/dependency text.
- [ ] Run `uv run lint-imports` and focused tests.

### Task 4: Release key-purpose separation

**Files:**
- Modify: `trade_rl/release/signing.py`
- Modify: `trade_rl/release/attestation.py`
- Create: `trade_rl/release/offline_approval.py`
- Modify: `trade_rl/release/__init__.py`
- Modify: `trade_rl/serving/registry.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `tests/release/test_signing.py`
- Modify: `tests/release/test_attestation.py`
- Modify: serving activation tests that construct trusted keys.

**Interfaces:**
- Produces: `VerificationKey(key_id: str, key: bytes, purpose: str = "release-verification", algorithm: str = "hmac-sha256")`
- Produces offline-only: `create_release_attestation(...)` in `offline_approval.py`.
- Verification consumes `Mapping[str, VerificationKey]` and rejects signing-purpose or wrong-algorithm keys.

- [ ] Write failing tests for signing-purpose keys, unknown algorithms, and absence of signing helpers from runtime-facing exports.
- [ ] Run focused release/serving tests and confirm failure.
- [ ] Implement verification key contracts and move creation helpers to the offline module while retaining explicit compatibility imports where tests require them.
- [ ] Update runtime/registry trusted-key handling and documentation.
- [ ] Run focused release and serving tests.

### Task 5: Walk-forward selection authorization for final training

**Files:**
- Create: `trade_rl/workflows/selection_authorization.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Modify: `trade_rl/artifacts/run_manifest.py` if the authorization sidecar must be declared explicitly.
- Create: `tests/workflows/test_selection_authorization.py`
- Modify: `tests/workflows/test_training_run.py`
- Modify: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Produces: `SelectionAuthorization` with canonical digest, read/write helpers, and `verify(...)`.
- Changes: `execute_training_run(..., selection_authorization_path: Path | None = None, require_selection_authorization: bool = False)`.

- [ ] Write failing tests for missing authorization, dataset mismatch, candidate digest mismatch, seed mismatch, walk-forward digest mismatch, and a valid final-training authorization.
- [ ] Run focused tests and confirm failure.
- [ ] Implement the authorization contract and canonical sidecar.
- [ ] Make final-training mode fail before normalizer fitting or model construction when authorization is absent or mismatched.
- [ ] Emit authorization from the full runner after stable selection and pass it to the final training command/CLI.
- [ ] Label direct training `research_exploratory` and authorized final training `research_selected_final` in run evidence.
- [ ] Run focused workflow and CLI tests.

### Task 6: Canonical dataset API and workflow supply-chain gate

**Files:**
- Modify: `trade_rl/data/artifacts.py`
- Modify: `trade_rl/data/__init__.py`
- Modify: tests importing the deprecated duplicate writer.
- Extend: `.github/check_workflow_security.py`
- Extend: `tests/architecture/test_workflow_security.py`

**Interfaces:**
- Public canonical API remains `write_market_dataset_files`, `publish_market_dataset_artifact`, and `load_market_dataset_artifact`.

- [ ] Write failing tests proving the ambiguous duplicate writer is not exported and the privileged workflow contains no mutable `uses:` reference.
- [ ] Run focused tests and confirm failure.
- [ ] Remove the duplicate compatibility export while retaining `MarketDatasetView` and loader contracts.
- [ ] Pin every action in the privileged workflow to a reviewed commit SHA and enforce the rule in the checker.
- [ ] Run focused tests, import architecture, and CLI smoke.

### Task 7: Full verification and review

**Files:**
- Modify documentation only if verification reveals mismatches.

- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ruff format --check --diff .`.
- [ ] Run `uv run mypy trade_rl`.
- [ ] Run `uv run lint-imports`.
- [ ] Run `uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json`.
- [ ] Run `uv run python .github/check_critical_coverage.py coverage.json pyproject.toml`.
- [ ] Run `uv run trade-rl --version`.
- [ ] Inspect the branch diff against main and confirm every audit finding is either closed or explicitly documented as a remaining cryptographic migration.
- [ ] Open a draft PR and require CI success before merge.