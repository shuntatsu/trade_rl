from __future__ import annotations

from decimal import Decimal

import pytest

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


def test_funding_point_requires_explicit_future_settlement_boundary() -> None:
    with pytest.raises(ValueError, match="next_funding_ns"):
        FundingPoint(
            rate=Decimal("0.0001"),
            observed_ns=10,
            next_funding_ns=10,
            interval_minutes=480,
        )

    with pytest.raises(ValueError, match="interval_minutes"):
        FundingPoint(
            rate=Decimal("0.0001"),
            observed_ns=10,
            next_funding_ns=20,
            interval_minutes=0,
        )


def test_mark_and_index_builders_reject_wrong_phase_before_optional_import() -> None:
    wrong = ProjectedMarketEvent(
        phase=MarketPhase.OPEN_QUOTE,
        timestamp_ns=10,
        price=100.0,
    )

    with pytest.raises(ValueError, match="MARK"):
        build_mark_price_update(wrong, instrument=object())
    with pytest.raises(ValueError, match="INDEX"):
        build_index_price_update(wrong, instrument=object())


def test_funding_rate_builder_requires_runtime_only_after_contract_validation() -> None:
    point = FundingPoint(
        rate=Decimal("0.0001"),
        observed_ns=10,
        next_funding_ns=20,
        interval_minutes=480,
    )
    assert point.rate == Decimal("0.0001")
    assert point.observed_ns == 10
    assert point.next_funding_ns == 20
