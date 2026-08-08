from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from tests.data.test_market_dataset_v2 import kwargs
from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.integrations.nautilus.event_projection import SourceBar

_MODULE = "trade_rl.integrations.nautilus.historical_projection"


def _projector():
    spec = importlib.util.find_spec(_MODULE)
    assert spec is not None, "historical source-bar projection module must exist"
    module = importlib.import_module(_MODULE)
    project = getattr(module, "project_historical_source_bar", None)
    assert callable(project), "historical source-bar projector must exist"
    return project


def _interval_projector():
    spec = importlib.util.find_spec(_MODULE)
    assert spec is not None, "historical source-bar projection module must exist"
    module = importlib.import_module(_MODULE)
    project = getattr(module, "project_historical_interval_source_bars", None)
    assert callable(project), "historical interval source-bar projector must exist"
    return project


def _single_symbol_market() -> MarketDataset:
    values = kwargs(n_symbols=1)
    close = np.asarray(values["close"], dtype=np.float64)
    values["mark_price"] = close + 0.25
    values["index_price"] = close - 0.25
    return MarketDataset(**values)


def test_project_historical_source_bar_uses_existing_source_bar_contract() -> None:
    project = _projector()
    market = _single_symbol_market()
    index = 2
    close_ns = int(market.timestamps[index].astype("datetime64[ns]").astype(np.int64))
    cadence_ns = int(
        market.timestamps[1].astype("datetime64[ns]").astype(np.int64)
        - market.timestamps[0].astype("datetime64[ns]").astype(np.int64)
    )

    bar = project(market, processing_index=index)

    assert isinstance(bar, SourceBar)
    assert bar.open_ns == close_ns - cadence_ns
    assert bar.close_ns == close_ns
    assert bar.open_price == pytest.approx(market.open[index, 0])
    assert bar.high_price == pytest.approx(market.high[index, 0])
    assert bar.low_price == pytest.approx(market.low[index, 0])
    assert bar.close_price == pytest.approx(market.close[index, 0])
    assert bar.mark_price == pytest.approx(market.mark_price[index, 0])
    assert bar.index_price == pytest.approx(market.index_price[index, 0])


def test_project_historical_interval_source_bars_excludes_start_boundary() -> None:
    project_interval = _interval_projector()
    market = _single_symbol_market()
    start_index = 1
    end_index = 3

    bars = project_interval(
        market,
        start_index=start_index,
        end_index=end_index,
    )

    assert len(bars) == end_index - start_index
    assert tuple(bar.close_ns for bar in bars) == tuple(
        int(market.timestamps[index].astype("datetime64[ns]").astype(np.int64))
        for index in range(start_index + 1, end_index + 1)
    )


@pytest.mark.parametrize(("start_index", "end_index"), ((2, 2), (3, 2)))
def test_project_historical_interval_source_bars_rejects_empty_or_reversed_range(
    start_index: int,
    end_index: int,
) -> None:
    project_interval = _interval_projector()

    with pytest.raises(ValueError, match="ordered"):
        project_interval(
            _single_symbol_market(),
            start_index=start_index,
            end_index=end_index,
        )


def test_project_historical_source_bar_rejects_non_single_symbol_or_session_data() -> (
    None
):
    project = _projector()

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
