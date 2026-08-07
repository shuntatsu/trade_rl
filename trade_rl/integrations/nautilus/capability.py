"""Small, deterministic capability probe for the pinned NautilusTrader wheel."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.integrations.nautilus.runtime_identity import (
    NautilusRuntimeIdentity,
    require_nautilus_runtime,
)


@dataclass(frozen=True, slots=True)
class NautilusCapabilityReport:
    """Capabilities required before building the authoritative execution bridge."""

    runtime: NautilusRuntimeIdentity
    engine_constructed: bool
    binance_margin_venue_added: bool
    engine_disposed: bool
    errors: tuple[str, ...]


def run_nautilus_capability_probe() -> NautilusCapabilityReport:
    """Exercise only the upstream primitives the first migration slice depends on."""

    runtime = require_nautilus_runtime()
    errors: list[str] = []
    engine = None
    engine_constructed = False
    binance_margin_venue_added = False
    engine_disposed = False

    try:
        from nautilus_trader.adapters.binance import BINANCE_VENUE
        from nautilus_trader.backtest.config import BacktestEngineConfig
        from nautilus_trader.backtest.engine import BacktestEngine
        from nautilus_trader.model import Money
        from nautilus_trader.model.currencies import USDT
        from nautilus_trader.model.enums import AccountType, OmsType

        engine = BacktestEngine(
            config=BacktestEngineConfig(shutdown_on_error=True),
        )
        engine_constructed = True
        engine.add_venue(
            venue=BINANCE_VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(100_000, USDT)],
        )
        binance_margin_venue_added = True
    except Exception as exc:  # pragma: no cover - exercised by pinned-wheel CI
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if engine is not None:
            try:
                engine.dispose()
                engine_disposed = True
            except Exception as exc:  # pragma: no cover - upstream failure evidence
                errors.append(f"dispose {type(exc).__name__}: {exc}")

    return NautilusCapabilityReport(
        runtime=runtime,
        engine_constructed=engine_constructed,
        binance_margin_venue_added=binance_margin_venue_added,
        engine_disposed=engine_disposed,
        errors=tuple(errors),
    )
