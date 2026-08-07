from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from trade_rl.integrations.nautilus.trace_adapter import (
    canonicalize_nautilus_fill_events,
)


class _DecimalValue:
    def __init__(self, value: str) -> None:
        self._value = Decimal(value)

    def as_decimal(self) -> Decimal:
        return self._value


def _fill(
    *,
    side: str,
    qty: str,
    px: str,
    commission: str,
    ts_event: int,
) -> object:
    return SimpleNamespace(
        order_side=side,
        last_qty=_DecimalValue(qty),
        last_px=_DecimalValue(px),
        commission=_DecimalValue(commission),
        ts_event=ts_event,
    )


def test_nautilus_fill_adapter_tracks_signed_position_and_commission() -> None:
    result = canonicalize_nautilus_fill_events(
        (
            _fill(
                side="BUY",
                qty="1.000",
                px="100.1",
                commission="0.04004000",
                ts_event=10,
            ),
            _fill(
                side="SELL",
                qty="1.000",
                px="104.9",
                commission="0.04196000",
                ts_event=200,
            ),
        ),
        price_tick=Decimal("0.1"),
        lot_size=Decimal("0.001"),
        currency_precision=8,
    )

    assert [fill.price_ticks for fill in result.fills] == [1001, 1049]
    assert [fill.quantity_lots for fill in result.fills] == [1000, -1000]
    assert [fill.position_lots for fill in result.fills] == [1000, 0]
    assert result.fee_minor == 8_200_000


def test_nautilus_fill_adapter_rejects_off_grid_execution() -> None:
    try:
        canonicalize_nautilus_fill_events(
            (
                _fill(
                    side="BUY",
                    qty="1.000",
                    px="100.15",
                    commission="0",
                    ts_event=10,
                ),
            ),
            price_tick=Decimal("0.1"),
            lot_size=Decimal("0.001"),
            currency_precision=8,
        )
    except ValueError as exc:
        assert "price" in str(exc)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("off-grid price was accepted")
