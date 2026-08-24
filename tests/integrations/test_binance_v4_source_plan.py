from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.integrations.binance_v4_source_plan import (
    build_v4_reference_decision_clock,
    plan_binance_v4_symbol_sources,
)


def _timestamps() -> np.ndarray:
    start = np.datetime64("2026-01-01T00:15", "ns")
    return start + np.arange(96) * np.timedelta64(15, "m")


def test_reference_clock_uses_existing_dataset_timestamps_exactly() -> None:
    dataset = SimpleNamespace(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=_timestamps(),
        calendar_kind="continuous_24_7",
        nominal_bar_hours=0.25,
    )
    clock = build_v4_reference_decision_clock(dataset)
    assert clock.decision_indices[0] == 0
    assert clock.decision_indices[-1] == 95
    assert len(clock.decision_indices) == 96
    np.testing.assert_array_equal(clock.decision_timestamps, _timestamps())
    assert len(clock.source_digest) == 64


def test_reference_clock_uses_validated_dataset_bar_hours_property() -> None:
    dataset = SimpleNamespace(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=_timestamps(),
        calendar_kind="continuous_24_7",
        nominal_bar_hours=None,
        bar_hours=0.25,
    )

    clock = build_v4_reference_decision_clock(dataset)

    np.testing.assert_array_equal(clock.decision_timestamps, _timestamps())


def test_reference_clock_rejects_non_btc_or_non_15m_dataset() -> None:
    wrong_symbol = SimpleNamespace(
        dataset_id="a" * 64,
        symbols=("ETHUSDT",),
        timestamps=_timestamps(),
        calendar_kind="continuous_24_7",
        nominal_bar_hours=0.25,
    )
    with pytest.raises(ValueError, match="BTCUSDT"):
        build_v4_reference_decision_clock(wrong_symbol)

    irregular = SimpleNamespace(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.asarray(
            [
                np.datetime64("2026-01-01T00:15"),
                np.datetime64("2026-01-01T00:45"),
            ],
            dtype="datetime64[ns]",
        ),
        calendar_kind="continuous_24_7",
        nominal_bar_hours=0.5,
    )
    with pytest.raises(ValueError, match="15-minute|15m|cadence"):
        build_v4_reference_decision_clock(irregular)


def test_source_plan_uses_previous_open_day_but_inclusive_metrics_day() -> None:
    plan = plan_binance_v4_symbol_sources(
        symbol="BTCUSDT",
        decision_timestamps=_timestamps(),
        include_metrics=True,
    )
    assert plan.spot_kline_urls == (
        "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/15m/"
        "BTCUSDT-15m-2026-01-01.zip",
    )
    assert plan.perp_kline_urls == (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/"
        "BTCUSDT-15m-2026-01-01.zip",
    )
    assert plan.mark_price_kline_urls == (
        "https://data.binance.vision/data/futures/um/daily/markPriceKlines/"
        "BTCUSDT/15m/BTCUSDT-15m-2026-01-01.zip",
    )
    assert plan.funding_urls == (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
        "BTCUSDT-fundingRate-2026-01.zip",
    )
    # The last decision is exactly Jan 2 00:00, so a metrics event at that exact
    # decision can live in the Jan 2 daily archive and must be considered.
    assert plan.metrics_urls == (
        "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/"
        "BTCUSDT-metrics-2026-01-01.zip",
        "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/"
        "BTCUSDT-metrics-2026-01-02.zip",
    )


def test_source_plan_omits_metrics_for_core_profile() -> None:
    plan = plan_binance_v4_symbol_sources(
        symbol="ETHUSDT",
        decision_timestamps=_timestamps(),
        include_metrics=False,
    )
    assert plan.metrics_urls == ()
    assert len(plan.source_digest) == 64


def test_source_plan_rejects_non_regular_decision_clock() -> None:
    with pytest.raises(ValueError, match="15-minute|15m|regular"):
        plan_binance_v4_symbol_sources(
            symbol="BTCUSDT",
            decision_timestamps=np.asarray(
                [
                    np.datetime64("2026-01-01T00:15"),
                    np.datetime64("2026-01-01T00:45"),
                ],
                dtype="datetime64[ns]",
            ),
            include_metrics=False,
        )
