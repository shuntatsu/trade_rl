# Simulation SHA-256 Authority Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` and `superpowers:verification-before-completion`. Execute this plan task-by-task on an isolated branch.

**Goal:** Remove duplicate lowercase-SHA-256 lexical parsers from the simulation domain while preserving each simulation boundary's existing domain-specific exception type and message.

**Architecture:** `trade_rl._validation.require_sha256` remains the repository lexical authority. `simulation.orders.model`, `simulation.orders.reconciliation`, and `simulation.liquidity` keep local boundary adapters only to translate `ValueError` into their existing domain exceptions. No new generic simulation validation module is introduced.

**Tech Stack:** Python 3.12, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** GitHub Issue #452 (`design review: LLM-first repository structure and semantic authority`), especially the 2026-09-11 re-self-review on semantic responsibility vs textual duplication.

## Global Constraints

- Do not change simulation algorithms, order IDs, persisted schemas, public facades, or state transitions.
- Do not normalize digest text with `.lower()` or `.strip()`.
- Preserve `OrderDomainError`, `OrderReconciliationError`, and `LiquidityAllocationError` at their current public boundaries.
- Do not centralize unrelated numeric tolerances merely because values are equal.
- Do not broaden the scope into `data/contracts.py`; that remains a separate read-only architecture audit.
- TDD RED must fail specifically because the three simulation adapters do not yet delegate lexical validation to `require_sha256`.
- Final exact PR head must pass the hardened repository CI and distribution/package checks before the PR is considered ready.

---

### Task 1: Add the failing authority contract

**Files:**
- Create: `tests/architecture/test_simulation_digest_authority.py`

**Interfaces:**
- Consumes: `trade_rl._validation.require_sha256` as the canonical lexical contract.
- Produces: an executable architecture contract requiring three simulation boundary adapters to delegate to that authority while preserving local exception translation.

- [ ] **Step 1: Write the failing tests**

Add tests that monkeypatch each module-level `require_sha256` dependency and call the existing private boundary adapter:

```python
from __future__ import annotations

import pytest

from trade_rl.simulation import liquidity
from trade_rl.simulation.orders import model, reconciliation


def _reject(value: str, *, field: str) -> str:
    raise ValueError(f"{field} rejected by canonical SHA authority")


def test_order_model_digest_adapter_delegates_and_preserves_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def accept(value: str, *, field: str) -> str:
        calls.append((value, field))
        return value

    monkeypatch.setattr(model, "require_sha256", accept)
    model._validate_digest("execution_policy_digest", "a" * 64)
    assert calls == [("a" * 64, "execution_policy_digest")]

    monkeypatch.setattr(model, "require_sha256", _reject)
    with pytest.raises(model.OrderDomainError, match="canonical SHA authority"):
        model._validate_digest("execution_policy_digest", "a" * 64)


def test_reconciliation_digest_adapter_delegates_and_preserves_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def accept(value: str, *, field: str) -> str:
        calls.append((value, field))
        return value

    monkeypatch.setattr(reconciliation, "require_sha256", accept)
    reconciliation._validate_digest("a" * 64)
    assert calls == [("a" * 64, "execution_policy_digest")]

    monkeypatch.setattr(reconciliation, "require_sha256", _reject)
    with pytest.raises(reconciliation.OrderReconciliationError, match="canonical SHA authority"):
        reconciliation._validate_digest("a" * 64)


def test_liquidity_digest_adapter_delegates_and_preserves_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def accept(value: str, *, field: str) -> str:
        calls.append((value, field))
        return value

    monkeypatch.setattr(liquidity, "require_sha256", accept)
    liquidity._validate_digest("order_id", "a" * 64)
    assert calls == [("a" * 64, "order_id")]

    monkeypatch.setattr(liquidity, "require_sha256", _reject)
    with pytest.raises(liquidity.LiquidityAllocationError, match="canonical SHA authority"):
        liquidity._validate_digest("order_id", "a" * 64)
```

The production change that should make these tests pass is explicit delegation from each adapter to `trade_rl._validation.require_sha256`; no other change should satisfy the contract.

- [ ] **Step 2: Run RED**

Run the full repository CI on the RED commit. Expected result: static checks pass and the new authority tests fail because the three modules do not expose/use `require_sha256` yet. No unrelated failure is acceptable.

### Task 2: Delegate lexical validation without changing domain boundaries

**Files:**
- Modify: `trade_rl/simulation/orders/model.py`
- Modify: `trade_rl/simulation/orders/reconciliation.py`
- Modify: `trade_rl/simulation/liquidity.py`
- Test: `tests/architecture/test_simulation_digest_authority.py`

**Interfaces:**
- Consumes: `require_sha256(value: str, *, field: str) -> str`.
- Produces: the same existing `_validate_digest(...) -> None` adapters and the same domain exception contracts.

- [ ] **Step 1: Implement the minimal GREEN change**

For each module, import `require_sha256` and replace the duplicated `len(...)` / hexadecimal-character parser with:

```python
try:
    require_sha256(value, field=field_name)
except ValueError as error:
    raise ExistingDomainError(str(error)) from error
```

Use the existing caller-supplied field name in `orders/model.py` and `liquidity.py`; use the fixed `execution_policy_digest` field name in `orders/reconciliation.py`.

- [ ] **Step 2: Verify targeted behavior**

Run the new authority tests plus existing order/liquidity/reconciliation suites. Confirm canonical lowercase hex64 remains accepted, malformed digest spelling remains rejected, and the existing domain exception classes remain observable.

- [ ] **Step 3: Run full verification**

Run Ruff, format check, production Mypy, architecture-tooling Mypy, full pytest, build, distribution source closure/sdist rebuild, clean installed smoke, CLI smoke, and package identity on the exact final head.

- [ ] **Step 4: Falsification review**

Temporarily reintroduce one local hexadecimal parser or bypass `require_sha256`; confirm the authority test fails. Restore the final tree and rerun the focused suite.

### Task 3: Close the temporary docs lifecycle and prepare review

**Files:**
- Delete: `docs/plans/2026-09-11-simulation-sha256-authority-plan.md`
- Modify: `docs/README.md` to remove the Active-plan entry.

**Interfaces:**
- Produces: no permanent new docs file. The durable rule remains represented by source ownership, architecture tests, and Issue #452.

- [ ] **Step 1: Remove the completed Active plan**

Delete this plan after implementation verification; do not promote a one-off SHA helper inventory into permanent architecture prose.

- [ ] **Step 2: Final diff review**

Expected durable diff: three production modules plus one architecture test, with no unrelated folder move, tolerance refactor, generated file, temporary workflow, or plan file.

- [ ] **Step 3: Prepare a review PR**

The PR must state the RED evidence, final exact-head verification, unchanged simulation behavior/public API/schema, and residual limitation that this closes only the observed simulation SHA lexical duplicates, not every possible semantic duplicate in the repository.
