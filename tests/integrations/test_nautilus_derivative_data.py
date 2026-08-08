from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.derivative_projection import (
    FundingPoint,
    build_funding_rate_update,
    build_index_price_update,
    build_mark_price_update,
)
from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    ProjectedMarketEvent,
)
from trade_rl.integrations.nautilus.instrument import (
    build_maintained_btcusdt_perpetual,
)


@pytest.mark.nautilus
def test_mark_index_and_funding_objects_preserve_causal_identity() -> None:
    instrument = build_maintained_btcusdt_perpetual()
    mark = build_mark_price_update(
        ProjectedMarketEvent(MarketPhase.MARK, 100, 104.5),
        instrument=instrument,
    )
    index = build_index_price_update(
        ProjectedMarketEvent(MarketPhase.INDEX, 99, 104.25),
        instrument=instrument,
    )
    funding = build_funding_rate_update(
        FundingPoint(
            rate=Decimal("0.0001"),
            observed_ns=90,
            next_funding_ns=120,
            interval_minutes=480,
        ),
        instrument=instrument,
    )

    assert str(mark.instrument_id) == "BTCUSDT-PERP.BINANCE"
    assert str(mark.value) == "104.5"
    assert mark.ts_event == 100
    assert str(index.value) == "104.25"
    assert index.ts_event == 99
    assert funding.rate == Decimal("0.0001")
    assert funding.ts_event == 90
    assert funding.next_funding_ns == 120
    assert funding.interval == 480
