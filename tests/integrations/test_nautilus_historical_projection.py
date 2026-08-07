from __future__ import annotations

import numpy as np
import pytest

from tests.data.test_market_dataset_v2 import kwargs
from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.integrations.nautilus import historical_projection


def _single_symbol_market() -> MarketDataset:
    values = kwargs(n_symbols=1)
    close = np.asarray(values["close"], dtype=np.float64)
    values["mark_price"] = close + 0.25
    values["index_price"] = close - 0.25
    return MarketDataset(**values)


def test_project_historical_source_bar_uses_close_timestamp_and_fixed_cadence() -> None:
    project = getattr(historical_projection, "project_historical_source_bar", None)
    assert callable(project), "historical source-bar projector must exist"
    market = _single_symbol_market()
    index = 2
    close_ns = int(market.timestamps[index].astype("datetime64[ns]").astype(np.int64))
    cadence_ns = int(
        market.timestamps[1].astype("datetime64[ns]").astype(np.int64)
        - market.timestamps[0].astype("datetime64[ns]").astype(np.int64)
    )

    bar = project(market, processing_index=index)

    assert bar.open_ns == close_ns - cadence_ns
    assert bar.close_ns == close_ns
    assert bar.open_price == pytest.approx(market.open[index, 0])
    assert bar.high_price == pytest.approx(market.high[index, 0])
    assert bar.low_price == pytest.approx(market.low[index, 0])
    assert bar.close_price == pytest.approx(market.close[index, 0])
    assert bar.mark_price == pytest.approx(market.mark_price[index, 0])
    assert bar.index_price == pytest.approx(market.index_price[index, 0])


def test_project_historical_source_bar_rejects_non_single_symbol_or_session_data() -> None:
    project = getattr(historical_projection, "project_historical_source_bar", None)
    assert callable(project), "historical source-bar projector must exist"

    with pytest.raises(ValueError, match="single-symbol"):
        project(MarketDataset(**kwargs(n_symbols=2)), processing_index=1)

    values = kwargs(n_symbols=1)
    timestamps = np.asarray(values["timestamps"]).copy()
    timestamps[4:] += np.timedelta64(16, "h")
    values.update(
        timestamps=timestamps,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        periods_per_year=1_638,
    )
    with pytest.raises(ValueError, match="continuous"):
        project(MarketDataset(**values), processing_index=1)
