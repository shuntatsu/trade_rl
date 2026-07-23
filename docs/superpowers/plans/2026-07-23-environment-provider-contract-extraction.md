# Environment Provider Contract Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract deterministic trend/alpha/factor provider resolution from `ResidualMarketEnv.__init__()` into a typed immutable contract without changing validation, identities, minimum indices, environment fields, or runtime behavior.

**Architecture:** Add `trade_rl.rl.environment_provider_contract` containing provider protocols, a frozen contract dataclass, and one builder. `ResidualMarketEnv` delegates once, assigns the returned values, and keeps all risk, action, reward, execution, observation, runtime-service, and mutable-state responsibilities unchanged.

**Tech Stack:** Python 3.12, NumPy, dataclasses, Protocol typing, Gymnasium environment, pytest, Ruff, Mypy, Import Linter, GitHub Actions, PostgreSQL integration.

## Global Constraints

- Preserve the public `ResidualMarketEnv` constructor signature.
- Preserve every existing validation message and validation order.
- Preserve `float64` copied static factor-basis semantics.
- Preserve explicit/provider/static digest precedence and SHA-256 validation.
- Preserve provider minimum-index order and maximum aggregation.
- Do not change risk, action, reward, execution, observation, episode, training, or mutable-state policy.
- Keep production status `NO-GO`.
- Follow RED -> GREEN -> REFACTOR and retain exact-head evidence.

---

### Task 1: Architecture RED contract

**Files:**
- Create: `tests/architecture/test_environment_provider_contract_decomposition.py`

**Interfaces:**
- Consumes: `ResidualMarketEnv.__init__` source through `inspect.getsource`.
- Produces: permanent ownership and constructor-size guard.

- [ ] **Step 1: Write the failing architecture tests**

Require import of `trade_rl.rl.environment_provider_contract`, local definitions of `EnvironmentProviderContract` and `EnvironmentProviderContractBuilder`, constructor delegation, removal of `_resolve_provider_digest`, `_validated_static_basis`, and `_resolve_factor_count`, absence of inline `MarketInputResolver(` and provider-minimum loops, and a constructor source span of at most 245 lines.

- [ ] **Step 2: Run the architecture test**

Run:

```bash
pytest -q tests/architecture/test_environment_provider_contract_decomposition.py
```

Expected: collection/import failure because `trade_rl.rl.environment_provider_contract` does not exist.

- [ ] **Step 3: Commit RED**

```bash
git add tests/architecture/test_environment_provider_contract_decomposition.py
git commit -m "test: require environment provider contract boundary"
```

### Task 2: Provider characterization tests

**Files:**
- Create: `tests/rl/test_environment_provider_contract.py`

**Interfaces:**
- Consumes: future `EnvironmentProviderContractBuilder` constructor and `build()` method.
- Produces: behavior contract for provider resolution.

- [ ] **Step 1: Add dataset and provider fixtures**

Create a two-symbol deterministic `MarketDataset`, a causal alpha provider exposing `predict` and `identity_digest`, legacy providers exposing `artifact_digest`, factor providers exposing `n_factors`, and configurable `minimum_index` values.

- [ ] **Step 2: Add successful characterization cases**

Cover default trend, explicit resolver, compatibility causal-alpha wrapping, explicit/provider digest precedence, static factor basis copying, inferred factor count, deterministic static basis digest, and maximum provider minimum index.

- [ ] **Step 3: Add exact-error cases**

Cover resolver/trend mismatch, missing alpha provider, required/invalid artifact digests, invalid factor-basis shape/finite values, invalid factor count, static count mismatch, and bool/non-integer/negative/out-of-range provider minimum indices.

- [ ] **Step 4: Run tests and retain RED**

Run:

```bash
pytest -q tests/rl/test_environment_provider_contract.py
```

Expected: import failure because the builder is not implemented.

- [ ] **Step 5: Commit characterization RED**

```bash
git add tests/rl/test_environment_provider_contract.py
git commit -m "test: characterize environment provider contract"
```

### Task 3: Typed provider contract implementation

**Files:**
- Create: `trade_rl/rl/environment_provider_contract.py`

**Interfaces:**
- Produces: `AlphaProvider`, `FactorBasisProvider`, `EnvironmentProviderContract`, and `EnvironmentProviderContractBuilder`.

- [ ] **Step 1: Define protocols and immutable result**

Implement the exact protocol and dataclass fields from the design document. Use `np.ndarray` for copied static basis and preserve callable provider unions.

- [ ] **Step 2: Implement trend/alpha resolution**

Preserve resolver fallback, compatibility wrapping detection, mismatch validation, alpha enablement, `AlphaContract()` default, digest precedence, and exact error strings.

- [ ] **Step 3: Implement factor resolution**

Preserve `float64` basis validation/copying, provider factor-count inference, count consistency, static digest payload schema `static_factor_basis_v1`, and exact errors.

- [ ] **Step 4: Implement minimum-index aggregation**

Start from `trend_strategy.minimum_history_for(dataset)`, validate alpha then factor provider indices in that order, default absent attributes to zero, and return their maximum.

- [ ] **Step 5: Run direct tests**

Run:

```bash
pytest -q tests/rl/test_environment_provider_contract.py
```

Expected: all direct builder cases pass.

- [ ] **Step 6: Commit implementation**

```bash
git add trade_rl/rl/environment_provider_contract.py tests/rl/test_environment_provider_contract.py
git commit -m "refactor: add environment provider contract"
```

### Task 4: Delegate the environment facade

**Files:**
- Modify: `trade_rl/rl/environment.py:8-290`
- Test: `tests/architecture/test_environment_provider_contract_decomposition.py`
- Test: existing environment and identity suites.

**Interfaces:**
- Consumes: `EnvironmentProviderContractBuilder(...).build()`.
- Produces: unchanged public environment fields with reduced constructor source.

- [ ] **Step 1: Import the provider contract types**

Import `AlphaProvider`, `FactorBasisProvider`, and `EnvironmentProviderContractBuilder` from the new module. Remove local protocol declarations and imports used only by moved helpers (`Protocol`, `require_sha256`).

- [ ] **Step 2: Replace inline provider resolution**

Call the builder with the original constructor arguments, assign every returned field, set `_static_factor_basis`, and initialize `_minimum_start_index` from the result before composer/risk construction.

- [ ] **Step 3: Remove facade helper methods**

Delete `_resolve_provider_digest`, `_validated_static_basis`, and `_resolve_factor_count` from `ResidualMarketEnv`.

- [ ] **Step 4: Run focused regression**

Run:

```bash
pytest -q \
  tests/architecture/test_environment_provider_contract_decomposition.py \
  tests/rl/test_environment_provider_contract.py \
  tests/rl/test_environment.py \
  tests/rl/test_environment_identity.py \
  tests/rl/test_environment_risk_service.py
```

Expected: all tests pass and constructor span is at most 245 lines.

- [ ] **Step 5: Run focused static checks**

Run:

```bash
ruff check \
  trade_rl/rl/environment.py \
  trade_rl/rl/environment_provider_contract.py \
  tests/architecture/test_environment_provider_contract_decomposition.py \
  tests/rl/test_environment_provider_contract.py
ruff format --check \
  trade_rl/rl/environment.py \
  trade_rl/rl/environment_provider_contract.py \
  tests/architecture/test_environment_provider_contract_decomposition.py \
  tests/rl/test_environment_provider_contract.py
mypy trade_rl/rl/environment.py trade_rl/rl/environment_provider_contract.py
```

Expected: success.

- [ ] **Step 6: Commit delegation**

```bash
git add trade_rl/rl/environment.py tests/architecture/test_environment_provider_contract_decomposition.py
git commit -m "refactor: delegate environment provider contract"
```

### Task 5: Coverage ratchet and full verification

**Files:**
- Modify: `pyproject.toml` only when exact-head coverage supports a non-lowering threshold.
- Create: `docs/verification/2026-07-23-environment-provider-contract-extraction.md`
- Modify: `docs/verification/2026-07-23-architecture-audit-closeout.md`

**Interfaces:**
- Consumes: complete CI and coverage artifacts.
- Produces: permanent regression evidence and current audit disposition.

- [ ] **Step 1: Run repository-required checks**

Run exact-head GitHub Actions covering Ruff, formatting, Mypy, Import Linter, dead-code report, serving smoke, complete pytest/coverage, critical coverage, CLI smoke, Ubuntu/Windows compatibility, training image/non-root runtime, and PostgreSQL Catalog.

- [ ] **Step 2: Inspect coverage artifact**

Record total tests, skips, warnings, total statement/branch coverage, and the new module statement/branch totals. Add a critical coverage threshold only at or below the measured exact-head result and never lower an existing threshold.

- [ ] **Step 3: Document verification**

Record RED commit/run, implementation exact head, CI run IDs, PostgreSQL run ID, constructor line reduction, unchanged behavior scope, and production `NO-GO` status.

- [ ] **Step 4: Update audit disposition**

Keep `AUD-RL-001` open unless the remaining risk is independently eliminated. Describe provider resolution as extracted and identify remaining risk/action/reward/runtime/mutable wiring precisely.

- [ ] **Step 5: Final exact-head verification and merge**

Run CI after documentation changes, verify the diff contains no temporary workflow or trigger files, mark the PR ready, squash merge with expected-head protection, and verify the merged PR/main commit.
