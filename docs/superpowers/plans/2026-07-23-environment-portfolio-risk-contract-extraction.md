# Environment Portfolio-Risk Contract Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract portfolio-risk model/provider selection, provider identity validation, and minimum-index aggregation from `ResidualMarketEnv.__init__()` into one typed, behavior-preserving contract.

**Architecture:** Add a frozen `EnvironmentPortfolioRiskContract` and deterministic `EnvironmentPortfolioRiskContractBuilder`. The builder receives only the dataset, optional model/provider, and the already-validated incoming minimum index; the environment invokes it once and preserves its existing attributes.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pytest, Ruff, Mypy, Import Linter, pytest-cov.

## Global Constraints

- Preserve the public `ResidualMarketEnv` constructor signature.
- Preserve default `PortfolioRiskModel()` construction.
- Preserve automatic `RollingPortfolioRiskInputsProvider()` selection only when advanced inputs are required and no provider was supplied.
- Preserve SHA-256 validation field text and minimum-index error text.
- Preserve digest validation before reading `minimum_index`.
- Preserve supplied model/provider object identities.
- Do not move pre-trade risk, emergency-risk, action, reward, executor, observation, or mutable-state policy.
- Keep `AUD-RL-001` as `OPEN RISK, FURTHER REDUCED`.
- Keep production status `NO-GO`.

---

### Task 1: Add RED contract and architecture characterization

**Files:**
- Create: `tests/rl/test_environment_portfolio_risk_contract.py`
- Create: `tests/architecture/test_environment_portfolio_risk_contract_decomposition.py`

**Interfaces:**
- Consumes: current inline portfolio-risk construction behavior and the design spec.
- Produces: failing tests requiring `EnvironmentPortfolioRiskContract` and `EnvironmentPortfolioRiskContractBuilder`.

- [ ] **Step 1: Write direct builder tests**

Define a small dataset with `n_bars`, a provider recording whether `minimum_index` was accessed, and tests requiring:

```python
contract = EnvironmentPortfolioRiskContractBuilder(
    dataset,
    portfolio_risk=model,
    inputs_provider=provider,
).build(minimum_start_index=3)

assert contract.portfolio_risk is model
assert contract.inputs_provider is provider
assert contract.minimum_start_index == expected
```

Cover default model/no provider, supplied identities, automatic rolling provider, maximum aggregation, digest-before-minimum ordering, and invalid minimum-index values including `True`, `1.5`, `-1`, and `dataset.n_bars`.

- [ ] **Step 2: Write facade architecture tests**

Require:

```python
assert "EnvironmentPortfolioRiskContractBuilder(" in constructor_source
assert constructor_source.count("EnvironmentPortfolioRiskContractBuilder(") == 1
assert "RollingPortfolioRiskInputsProvider()" not in constructor_source
assert "portfolio_risk_inputs_provider.identity_digest" not in constructor_source
assert "portfolio risk inputs minimum_index is invalid" not in constructor_source
assert constructor_line_count <= 220
```

Also require that `trade_rl/rl/environment_portfolio_risk_contract.py` locally owns both public classes.

- [ ] **Step 3: Verify clean RED**

Run:

```bash
uv run ruff check tests/rl/test_environment_portfolio_risk_contract.py tests/architecture/test_environment_portfolio_risk_contract_decomposition.py
uv run ruff format --check tests/rl/test_environment_portfolio_risk_contract.py tests/architecture/test_environment_portfolio_risk_contract_decomposition.py
uv run pytest -q tests/rl/test_environment_portfolio_risk_contract.py tests/architecture/test_environment_portfolio_risk_contract_decomposition.py
```

Expected: Ruff and formatting pass; pytest collection fails only because `trade_rl.rl.environment_portfolio_risk_contract` does not exist.

- [ ] **Step 4: Commit RED tests**

```bash
git add tests/rl/test_environment_portfolio_risk_contract.py tests/architecture/test_environment_portfolio_risk_contract_decomposition.py
git commit -m "test: define environment portfolio risk contract"
```

### Task 2: Implement the typed contract and facade delegation

**Files:**
- Create: `trade_rl/rl/environment_portfolio_risk_contract.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Consumes: `MarketDataset`, `PortfolioRiskModel | None`, `PortfolioRiskInputsProvider | None`, and `minimum_start_index: int`.
- Produces: `EnvironmentPortfolioRiskContractBuilder.build(*, minimum_start_index: int) -> EnvironmentPortfolioRiskContract`.

- [ ] **Step 1: Add the frozen contract**

```python
@dataclass(frozen=True, slots=True)
class EnvironmentPortfolioRiskContract:
    portfolio_risk: PortfolioRiskModel
    inputs_provider: PortfolioRiskInputsProvider | None
    minimum_start_index: int
```

- [ ] **Step 2: Add the deterministic builder**

Implement `build()` in this exact order:

```python
portfolio_risk = self.portfolio_risk or PortfolioRiskModel()
provider = self.inputs_provider
if portfolio_risk.requires_advanced_inputs and provider is None:
    provider = RollingPortfolioRiskInputsProvider()
if provider is not None:
    require_sha256(
        provider.identity_digest,
        field="portfolio_risk_inputs_provider.identity_digest",
    )
    provider_minimum = provider.minimum_index
    if (
        isinstance(provider_minimum, bool)
        or not isinstance(provider_minimum, int)
        or provider_minimum < 0
        or provider_minimum >= self.dataset.n_bars
    ):
        raise ValueError("portfolio risk inputs minimum_index is invalid")
    minimum_start_index = max(minimum_start_index, provider_minimum)
return EnvironmentPortfolioRiskContract(
    portfolio_risk=portfolio_risk,
    inputs_provider=provider,
    minimum_start_index=minimum_start_index,
)
```

- [ ] **Step 3: Delegate from `ResidualMarketEnv`**

After provider-contract assignment, invoke the new builder once and assign its three returned values. Remove direct `RollingPortfolioRiskInputsProvider` and `require_sha256` imports if no longer used elsewhere in the module.

- [ ] **Step 4: Verify focused GREEN**

Run:

```bash
uv run ruff check trade_rl/rl/environment.py trade_rl/rl/environment_portfolio_risk_contract.py tests/rl/test_environment_portfolio_risk_contract.py tests/architecture/test_environment_portfolio_risk_contract_decomposition.py
uv run ruff format --check trade_rl/rl/environment.py trade_rl/rl/environment_portfolio_risk_contract.py tests/rl/test_environment_portfolio_risk_contract.py tests/architecture/test_environment_portfolio_risk_contract_decomposition.py
uv run mypy trade_rl/rl/environment.py trade_rl/rl/environment_portfolio_risk_contract.py
uv run pytest -q tests/rl/test_environment_portfolio_risk_contract.py tests/architecture/test_environment_portfolio_risk_contract_decomposition.py tests/rl
```

Expected: all commands pass.

- [ ] **Step 5: Commit implementation**

```bash
git add trade_rl/rl/environment.py trade_rl/rl/environment_portfolio_risk_contract.py
git commit -m "refactor: extract environment portfolio risk contract"
```

### Task 3: Add permanent coverage and ownership controls

**Files:**
- Modify: `tests/rl/test_environment_portfolio_risk_contract.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: full-suite branch coverage.
- Produces: complete behavior characterization and a permanent 100.0% critical-coverage ratchet.

- [ ] **Step 1: Run complete coverage**

```bash
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json:coverage.json
```

- [ ] **Step 2: Inspect extracted-module coverage**

Read `coverage.json` for `trade_rl/rl/environment_portfolio_risk_contract.py`. Add behavior-relevant cases only for any uncovered statements or branches.

- [ ] **Step 3: Add the ratchet**

Under `[tool.trade_rl.critical_coverage.files]`, add:

```toml
"trade_rl/rl/environment_portfolio_risk_contract.py" = 100.0
```

- [ ] **Step 4: Verify coverage and critical gate**

Run:

```bash
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json:coverage.json
uv run python scripts/check_critical_coverage.py coverage.json
```

Expected: extracted module is 100.0% and the critical gate passes.

- [ ] **Step 5: Commit coverage controls**

```bash
git add tests/rl/test_environment_portfolio_risk_contract.py pyproject.toml
git commit -m "test: guard environment portfolio risk contract"
```

### Task 4: Complete exact-head verification and merge

**Files:**
- Create: `docs/verification/2026-07-23-environment-portfolio-risk-contract-extraction.md`
- Modify: `docs/verification/2026-07-23-architecture-audit-closeout.md`

**Interfaces:**
- Consumes: RED evidence, exact implementation SHA, CI/PostgreSQL run IDs, test totals, coverage totals, module coverage, constructor reduction, and final diff.
- Produces: reviewable evidence and updated `AUD-RL-001` disposition.

- [ ] **Step 1: Require all maintained checks**

Require success for Studio frontend/fixed viewport, workflow security, Ruff, format, Mypy, Import Linter, dead-code report, serving smoke, complete pytest/coverage, critical coverage, CLI, Ubuntu, Windows, training-image/non-root probe, and PostgreSQL Catalog.

- [ ] **Step 2: Record exact evidence**

Document:

- clean RED head and failure reason;
- implementation and final documentation heads;
- CI and PostgreSQL run IDs;
- passed/skipped/warning totals;
- total statement and branch coverage;
- extracted-module statement and branch coverage;
- constructor source-span reduction;
- changed-file scope and absence of temporary files;
- non-goals and `NO-GO` status.

- [ ] **Step 3: Update audit closeout**

State that portfolio-risk input construction is extracted. Keep `AUD-RL-001` open because config/action/episode validation, reward/executor construction, and mutable Gymnasium initialization remain.

- [ ] **Step 4: Review final diff**

Confirm no temporary workflow, trigger, diagnostic, generated coverage file, unrelated source change, or public constructor change remains.

- [ ] **Step 5: Ready and squash merge**

Use the exact verified PR head SHA as the merge guard and squash merge into `main`.