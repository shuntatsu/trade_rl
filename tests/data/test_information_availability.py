from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import (
    CsvMarketDataSource,
    InMemoryMarketDataSource,
    RawMarketSeries,
)
from trade_rl.evaluation import run_single_symbol_replay
from trade_rl.simulation import BookState, ExecutionResult, MarketExecutor
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


def _timestamps(count: int = 24) -> np.ndarray:
    return np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        count
    ) * np.timedelta64(1, "h")


def _raw(
    *,
    delayed_index: int | None = None,
    delayed_tradable: bool = True,
) -> RawMarketSeries:
    timestamps = _timestamps()
    close = np.exp(np.arange(len(timestamps), dtype=np.float64) * 0.001)
    open_price = np.concatenate([close[:1], close[:-1]])
    available_at = timestamps.copy()
    if delayed_index is not None:
        available_at[delayed_index] += np.timedelta64(1, "h")
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=available_at,
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full(len(timestamps), 1_000.0),
        funding_rate=np.zeros(len(timestamps)),
        tradable=np.where(
            np.arange(len(timestamps)) == delayed_index,
            delayed_tradable,
            True,
        ).astype(np.bool_),
    )


def _build(raw: RawMarketSeries):
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (contract,),
    )


def test_raw_series_defaults_availability_to_event_timestamp() -> None:
    raw = _raw()

    assert raw.available_at is not None
    np.testing.assert_array_equal(raw.available_at, raw.timestamps)


def test_raw_series_rejects_availability_before_event_time() -> None:
    timestamps = _timestamps()
    close = np.ones(len(timestamps), dtype=np.float64)

    with pytest.raises(ValueError, match="available_at"):
        RawMarketSeries(
            timestamps=timestamps,
            available_at=timestamps - np.timedelta64(1, "h"),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=np.ones(len(timestamps)),
            funding_rate=np.zeros(len(timestamps)),
            tradable=np.ones(len(timestamps), dtype=np.bool_),
        )


def test_delayed_market_row_is_not_visible_to_same_time_features() -> None:
    market = _build(_raw(delayed_index=10))

    assert market.information_available is not None
    assert market.available_at is not None
    assert not market.information_available[10, 0]
    assert market.available_at[10, 0] == market.timestamps[11]
    np.testing.assert_allclose(market.features[10, 0], market.features[9, 0])
    np.testing.assert_allclose(market.features[11, 0], market.features[9, 0])
    assert market.feature_staleness is not None
    assert market.feature_staleness[10, 0, 0] > market.feature_staleness[9, 0, 0]
    assert market.feature_staleness[11, 0, 0] > market.feature_staleness[10, 0, 0]
    assert market.feature_staleness[12, 0, 0] == 0.0


def test_delayed_tradability_is_hidden_from_policy_and_global_fraction() -> None:
    delayed_true = _build(_raw(delayed_index=10, delayed_tradable=True))
    delayed_false = _build(_raw(delayed_index=10, delayed_tradable=False))

    assert delayed_true.tradable[10, 0]
    assert not delayed_false.tradable[10, 0]
    np.testing.assert_array_equal(
        delayed_true.observable_tradable(10),
        delayed_false.observable_tradable(10),
    )
    assert delayed_true.global_features[10, 1] == 0.0
    assert delayed_false.global_features[10, 1] == 0.0


class _RecordingLong:
    def __init__(self) -> None:
        self.observations: list[StrategyObservation] = []

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        self.observations.append(observation)
        return PositionIntent.LONG


def test_delayed_close_is_rejected_before_target_row_execution_and_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw(delayed_index=10)
    altered_close = raw.close.copy()
    altered_close[10] *= 1.1
    altered_high = raw.high.copy()
    altered_low = raw.low.copy()
    altered_high[10] = max(raw.open[10], altered_close[10]) * 1.001
    altered_low[10] = min(raw.open[10], altered_close[10]) * 0.999

    twins = (
        _build(raw),
        _build(
            replace(
                raw,
                close=altered_close,
                high=altered_high,
                low=altered_low,
            )
        ),
    )
    first, second = twins
    for market in twins:
        assert market.information_available is not None
        assert market.available_at is not None
        assert not market.information_available[10, 0]
        assert market.available_at[10, 0] == market.timestamps[11]
        assert market.tradable[10, 0]
        np.testing.assert_allclose(
            market.resolved_array("mark_price")[10, 0], market.close[10, 0]
        )
    np.testing.assert_array_equal(first.timestamps, second.timestamps)
    np.testing.assert_array_equal(first.available_at, second.available_at)
    np.testing.assert_array_equal(
        first.information_available, second.information_available
    )
    np.testing.assert_array_equal(first.tradable, second.tradable)
    np.testing.assert_allclose(first.features, second.features)
    np.testing.assert_allclose(first.open[:11], second.open[:11])
    np.testing.assert_allclose(first.close[:10], second.close[:10])
    assert first.close[10, 0] != second.close[10, 0]

    row_nine_states: list[tuple[float, float, float | None]] = []
    original_execute_interval = MarketExecutor.execute_interval
    for market in twins:
        strategy = _RecordingLong()
        execution_starts: list[int] = []

        def record_execution_start(
            executor: MarketExecutor,
            book: BookState,
            target_weights: np.ndarray,
            *,
            start_index: int,
            bars: int,
        ) -> ExecutionResult:
            execution_starts.append(start_index)
            return original_execute_interval(
                executor, book, target_weights, start_index=start_index, bars=bars
            )

        with monkeypatch.context() as patch:
            patch.setattr(MarketExecutor, "execute_interval", record_execution_start)
            with pytest.raises(ValueError, match="decision-time mark unavailable"):
                run_single_symbol_replay(
                    market,
                    strategy,
                    start_index=8,
                    stop_index=11,
                    gross_budget=0.5,
                )
        assert execution_starts == [8]
        assert [observation.index for observation in strategy.observations] == [8, 9]

        row_nine = next(
            observation
            for observation in strategy.observations
            if observation.index == 9
        )
        row_nine_states.append(
            (
                row_nine.current_weight,
                row_nine.current_position_quantity,
                row_nine.gross_position_return,
            )
        )

    assert row_nine_states[0] == row_nine_states[1]


def test_information_availability_changes_dataset_identity() -> None:
    on_time = _build(_raw())
    delayed = _build(_raw(delayed_index=10))

    assert on_time.dataset_id != delayed.dataset_id


def test_csv_source_parses_available_at(tmp_path: Path) -> None:
    path = tmp_path / "BTCUSDT.csv"
    path.write_text(
        "timestamp,available_at,open,high,low,close,volume\n"
        "2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,1,1,1,1,10\n"
        "2026-01-01T01:00:00Z,2026-01-01T02:00:00Z,1,1,1,1,10\n",
        encoding="utf-8",
    )

    raw = CsvMarketDataSource(tmp_path).load("BTCUSDT")

    assert raw.available_at is not None
    assert raw.available_at[1] == np.datetime64("2026-01-01T02:00:00", "ns")
