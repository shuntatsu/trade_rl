import numpy as np

from mars_lite.env.market_execution_core import BookState, MarketExecutionCore
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.execution import make_execution_model


def _feature_set(n_bars: int = 10) -> FeatureSet:
    close = (1.01 ** np.arange(n_bars, dtype=np.float64)).reshape(-1, 1)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    return FeatureSet(
        symbols=["A"],
        timestamps=timestamps,
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        close=close,
        open_next=close.copy(),
        funding_rate=np.zeros_like(close),
        feature_names=["dummy"],
        global_feature_names=["dummy_global"],
    )


def test_one_target_change_charges_cost_once_for_interval() -> None:
    fs = _feature_set()
    core = MarketExecutionCore(
        fs,
        make_execution_model(fee_rate=0.001, spread_rate=0.0, impact_rate=0.0),
    )
    book = BookState.zero(1)

    result = core.execute_interval(book, np.array([1.0]), start_t=1, bars=3)

    assert result.bars_advanced == 3
    assert result.interval_cost == 0.001
    assert result.interval_turnover == 1.0
    assert result.book.n_trades == 1
    assert result.book.total_cost == 0.001


def test_tail_interval_advances_only_available_bars() -> None:
    fs = _feature_set(n_bars=8)
    core = MarketExecutionCore(
        fs,
        make_execution_model(fee_rate=0.0, spread_rate=0.0, impact_rate=0.0),
    )

    result = core.execute_interval(
        BookState.zero(1), np.array([1.0]), start_t=4, bars=8
    )

    assert result.bars_advanced == 2
    assert result.next_t == 6


def test_holding_same_target_in_next_interval_has_zero_turnover() -> None:
    fs = _feature_set()
    core = MarketExecutionCore(
        fs,
        make_execution_model(fee_rate=0.001, spread_rate=0.0, impact_rate=0.0),
    )
    first = core.execute_interval(BookState.zero(1), np.array([1.0]), start_t=0, bars=2)
    second = core.execute_interval(first.book, np.array([1.0]), start_t=2, bars=2)

    assert second.interval_turnover == 0.0
    assert second.interval_cost == 0.0
