from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT
from trade_rl.data.contracts import InstrumentContract, InstrumentExecutionRule
from trade_rl.data.economic_semantics import build_market_economic_semantics


def test_economic_semantics_are_explicit_point_in_time_and_immutable() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = np.arange(
        np.datetime64("2026-01-01T00:15:00", "ns"),
        np.datetime64("2026-01-01T01:15:00", "ns"),
        np.timedelta64(15, "m"),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=start,
        tick_size=0.1,
        lot_size=0.001,
        minimum_notional=5.0,
        execution_rules=(
            InstrumentExecutionRule(
                effective_at=start, tick_size=0.1, lot_size=0.001, minimum_notional=5.0
            ),
            InstrumentExecutionRule(
                effective_at=start + timedelta(minutes=45),
                tick_size=0.2,
                lot_size=0.002,
                minimum_notional=10.0,
            ),
        ),
    )
    shape = (len(timestamps), 1)
    semantics = build_market_economic_semantics(
        timestamps=timestamps,
        instruments=(contract,),
        row_present=np.ones(shape, dtype=np.bool_),
        raw_tradable=np.ones(shape, dtype=np.bool_),
        source_information_available=np.ones(shape, dtype=np.bool_),
        available_at=np.broadcast_to(timestamps[:, None], shape),
        close=np.full(shape, 100.0),
        funding_event_count=np.zeros(shape, dtype=np.int32),
    )
    assert semantics.tick_size[:, 0].tolist() == [0.1, 0.1, 0.2, 0.2]
    assert semantics.minimum_notional[:, 0].tolist() == [5.0, 5.0, 10.0, 10.0]
    assert set(semantics.market_dataset_kwargs()) >= {
        "fee_rate",
        "spread_rate",
        "borrow_rate",
        "mark_price",
        "index_price",
    }
    assert all(
        not value.flags.writeable
        for value in semantics.market_dataset_kwargs().values()
    )


def test_vision_and_postgres_use_the_same_constructor() -> None:
    assert (
        "build_market_economic_semantics"
        in (PYTHON_SOURCE_ROOT / "data/builder.py").read_text()
    )
    assert (
        "build_market_economic_semantics"
        in (PYTHON_SOURCE_ROOT / "integrations/postgres_market_dataset.py").read_text()
    )
