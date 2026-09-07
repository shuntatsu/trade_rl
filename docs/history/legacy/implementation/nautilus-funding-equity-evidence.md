# Nautilus Funding Equity Evidence Implementation Plan

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

## Task 1: Canonical funding trace record

**Files:**
- `tests/integrations/test_nautilus_funding_adapter.py`
- `trade_rl/integrations/nautilus/funding_adapter.py`

**Interface:**

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

The adapter must validate a positive non-bool sequence, a non-bool integer pre-settlement equity, finite positive tick/lot increments, and exact price/quantity grid alignment. It returns a `funding` record at the settlement boundary with zero fill quantity and fee, preserved signed position lots, the funding amount in minor units, post-settlement equity, and no terminal reason.

Verification:

```bash
uv run pytest -q tests/integrations/test_nautilus_funding_adapter.py
uv run pytest -q tests/simulation/test_execution_parity.py tests/integrations/test_nautilus_funding_adapter.py
uv run ruff check trade_rl/integrations/nautilus/funding_adapter.py tests/integrations/test_nautilus_funding_adapter.py
uv run mypy trade_rl/integrations/nautilus/funding_adapter.py
uv run lint-imports
```

## Task 2: Dedicated Nautilus capability evidence

Update `.github/workflows/nautilus-capability.yml` so the exact-wheel funding step executes both `tests/integrations/test_nautilus_funding_adapter.py` and `tests/integrations/test_nautilus_funding_settlement.py`.

## Task 3: Migration status

After Task 1 and Task 2 verify green, update `docs/NAUTILUS_MIGRATION.md` to record canonical funding/equity trace wiring as implemented. The remaining work is full historical interval replay/equity trace integration and downstream promotion evidence, not the basic canonical funding record adapter.

## Final verification

Run focused tests, Ruff, MyPy, Import Linter, full repository CI, and the dedicated `Nautilus Capability` workflow on the same final head before marking this slice complete.
