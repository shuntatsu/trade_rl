# Nautilus Funding Equity Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect canonical perpetual funding settlements to the existing integer-valued execution trace so funding changes are visible in interval/equity evidence rather than only in final economic closure.

**Architecture:** Keep the funding calculation in `trade_rl.integrations.nautilus.funding_adapter` and reuse the framework-neutral `CanonicalExecutionRecord` from `trade_rl.simulation.execution_parity`. Add one pure adapter that turns an already-settled funding boundary into a canonical `event_type="funding"` record, validating tick/lot alignment and applying the signed funding minor units to the supplied pre-settlement equity. Do not add a second trace model or change authority promotion yet.

**Tech Stack:** Python 3.12, `Decimal`, pytest, Ruff, MyPy, Import Linter, GitHub Actions `CI` and `Nautilus Capability`.

## Global Constraints

- Maintained instrument remains `BTCUSDT-PERP.BINANCE`.
- Pinned Nautilus runtime remains `nautilus_trader==1.230.0`.
- Production remains `NO-GO`.
- Funding settlement remains explicit at the Trade RL integration boundary while the pinned Python BacktestEngine does not natively dispatch funding settlement.
- Canonical economic evidence uses integer minor units; price and quantity identity use exact integer tick/lot units.
- No automatic authority promotion is introduced by this slice.

---

### Task 1: Canonical funding trace record

**Files:**
- Modify: `tests/integrations/test_nautilus_funding_adapter.py`
- Modify: `trade_rl/integrations/nautilus/funding_adapter.py`

**Interfaces:**
- Consumes: `CanonicalFundingSettlement`, `CanonicalExecutionRecord`, `price_tick: Decimal`, `lot_size: Decimal`, `sequence: int`, `equity_before_minor: int`.
- Produces: `canonicalize_funding_settlement_record(...) -> CanonicalExecutionRecord`.

- [ ] **Step 1: Write the failing behavior test**

Add a test that settles a positive-rate long and then canonicalizes it with `sequence=3`, `price_tick=Decimal("0.01")`, `lot_size=Decimal("0.001")`, and `equity_before_minor=10_000_000_000`. Assert the record is a `funding` event at the settlement boundary, has `quantity_lots == 0`, `fee_minor == 0`, `funding_minor == -1_020_000`, `position_lots == 1_000`, and `equity_minor == 9_998_980_000`.

Add fail-closed tests for non-positive sequence, misaligned settlement price, misaligned signed quantity, and non-integer equity input.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q tests/integrations/test_nautilus_funding_adapter.py
```

Expected: the new canonicalization test fails because `canonicalize_funding_settlement_record` does not exist yet.

- [ ] **Step 3: Implement the minimal adapter**

Implement in `funding_adapter.py`:

```python
def canonicalize_funding_settlement_record(
    settlement: CanonicalFundingSettlement,
    *,
    sequence: int,
    price_tick: Decimal,
    lot_size: Decimal,
    equity_before_minor: int,
) -> CanonicalExecutionRecord:
    ...
```

Validate `sequence` as a positive non-bool integer, `equity_before_minor` as a non-bool integer, and tick/lot increments as finite positive decimals. Convert settlement price and signed quantity to exact integer grid units and reject off-grid values. Return a `CanonicalExecutionRecord` with `event_type="funding"`, `timestamp_ns=settlement.boundary_ns`, zero fill quantity and fee, the settlement funding minor units, preserved position lots, post-settlement equity, and no terminal reason.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
uv run pytest -q tests/integrations/test_nautilus_funding_adapter.py
```

Expected: all tests in the file pass.

- [ ] **Step 5: Run architecture-adjacent verification**

```bash
uv run pytest -q tests/simulation/test_execution_parity.py tests/integrations/test_nautilus_funding_adapter.py
uv run ruff check trade_rl/integrations/nautilus/funding_adapter.py tests/integrations/test_nautilus_funding_adapter.py
uv run mypy trade_rl/integrations/nautilus/funding_adapter.py
uv run lint-imports
```

Expected: all commands exit 0.

### Task 2: Exercise funding/equity evidence in the dedicated Nautilus capability lane

**Files:**
- Modify: `.github/workflows/nautilus-capability.yml`

**Interfaces:**
- Consumes: Task 1 focused test file.
- Produces: dedicated CI evidence that funding/equity canonicalization is tested under the exact Nautilus dependency environment.

- [ ] **Step 1: Add the focused funding adapter test to the existing funding step**

Change the funding step to execute both `tests/integrations/test_nautilus_funding_adapter.py` and `tests/integrations/test_nautilus_funding_settlement.py` in the exact-wheel environment.

- [ ] **Step 2: Validate workflow syntax through repository CI/security checks**

Run the repository's workflow security/CI checks and confirm the `Nautilus Capability` workflow completes successfully on the PR head.

### Task 3: Update migration evidence status

**Files:**
- Modify: `docs/NAUTILUS_MIGRATION.md`

**Interfaces:**
- Consumes: verified Task 1 and Task 2 results.
- Produces: migration status that distinguishes completed canonical funding/equity trace wiring from still-unimplemented full historical interval replay.

- [ ] **Step 1: Update implemented evidence**

Record that canonical funding settlements now produce integer funding/equity trace records with exact tick/lot identity.

- [ ] **Step 2: Narrow the remaining-work item**

Replace `connect canonical funding settlements into complete interval/equity evidence` with the remaining work: integrate these canonical funding records into complete historical interval replay/equity traces and downstream promotion evidence.

- [ ] **Step 3: Final verification**

Run focused tests, Ruff, MyPy, Import Linter, the full CI suite, and confirm `Nautilus Capability` succeeds on the final head before marking this slice complete.
