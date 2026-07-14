from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def _dataset():
    n = 16
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(n, dtype=np.float64)
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=np.concatenate((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(n, 1_000.0),
        funding_rate=np.zeros(n),
        tradable=np.ones(n, dtype=np.bool_),
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
        )
    ).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("fee_rate", 0.001),
        ("maker_fee_rate", 0.0002),
        ("taker_fee_rate", 0.0004),
        ("spread_rate", 0.0003),
        ("max_participation_rate", 0.5),
        ("minimum_notional", 10.0),
        ("lot_size", 0.01),
        ("tick_size", 0.1),
        ("borrow_rate", 0.08),
        ("mark_price", 123.0),
        ("index_price", 122.0),
        ("dividend", 0.5),
        ("split_factor", 2.0),
        ("delisting_recovery", 0.5),
    ],
)
def test_identity_binds_every_execution_array(
    field_name: str, replacement: float
) -> None:
    dataset = _dataset()
    original = dataset.resolved_array(field_name)
    changed = original.copy()
    changed[-1, 0] = replacement

    with pytest.raises(ValueError, match="dataset_id"):
        replace(dataset, **{field_name: changed})


def test_identity_binds_cash_rate_and_direction_masks() -> None:
    dataset = _dataset()
    cash_rate = dataset.resolved_array("cash_rate").copy()
    cash_rate[-1] = 0.05
    with pytest.raises(ValueError, match="dataset_id"):
        replace(dataset, cash_rate=cash_rate)

    buy_allowed = dataset.resolved_array("buy_allowed").copy()
    buy_allowed[-1, 0] = False
    with pytest.raises(ValueError, match="dataset_id"):
        replace(dataset, buy_allowed=buy_allowed)


def test_dataset_exposes_recomputed_identity() -> None:
    dataset = _dataset()
    assert dataset.recomputed_dataset_id() == dataset.dataset_id
    assert set(dataset.identity_arrays()) >= {
        "fee_rate",
        "mark_price",
        "cash_rate",
        "contract_multipliers",
        "feature_staleness_hours",
    }
