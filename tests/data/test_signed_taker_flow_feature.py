from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec, NormalizationMode
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.source import RawMarketSeries


def _raw(*, taker: np.ndarray | None) -> RawMarketSeries:
    n = 30
    timestamps = np.datetime64("2022-01-01T00:00:00", "ns") + np.arange(
        n
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(n, dtype=np.float64)
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=timestamps,
        open=np.concatenate((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(n, 10.0, dtype=np.float64),
        taker_buy_quote_volume=taker,
        funding_rate=np.zeros(n, dtype=np.float64),
        tradable=np.ones(n, dtype=np.bool_),
    )


def _spec(**changes: object) -> FeatureSpec:
    spec = FeatureSpec(
        name="1h__signed_taker_quote_flow_24bar",
        kind=FeatureKind.SIGNED_TAKER_QUOTE_FLOW,
        lookback=24,
        normalization=NormalizationMode.NONE,
        timeframe="1h",
    )
    return replace(spec, **changes)


def _calculate(
    *,
    volume: np.ndarray | None = None,
    taker: np.ndarray | None = None,
    row_present: np.ndarray | None = None,
    active: np.ndarray | None = None,
    tradable: np.ndarray | None = None,
    spec: FeatureSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 30
    close = 100.0 + np.arange(n, dtype=np.float64)
    if volume is None:
        volume = np.full(n, 10.0, dtype=np.float64)
    if taker is None:
        taker = np.full(n, 6.0, dtype=np.float64)
    if row_present is None:
        row_present = np.ones(n, dtype=np.bool_)
    if active is None:
        active = np.ones(n, dtype=np.bool_)
    if tradable is None:
        tradable = np.ones(n, dtype=np.bool_)
    return calculate_feature_events(
        _spec() if spec is None else spec,
        open_price=np.concatenate((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
        taker_buy_quote_volume=taker,
        funding_rate=np.zeros(n, dtype=np.float64),
        funding_available=np.zeros(n, dtype=np.bool_),
        row_present=row_present,
        active=active,
        tradable=tradable,
    )


def test_raw_market_series_preserves_optional_taker_quote_volume_read_only() -> None:
    taker = np.linspace(1.0, 9.0, 30, dtype=np.float64)
    series = _raw(taker=taker)

    assert series.taker_buy_quote_volume is not None
    np.testing.assert_allclose(series.taker_buy_quote_volume, taker)
    assert not series.taker_buy_quote_volume.flags.writeable
    assert _raw(taker=None).taker_buy_quote_volume is None


@pytest.mark.parametrize(
    "taker,match",
    [
        (np.ones(29), "shape"),
        (np.concatenate((np.ones(29), [np.nan])), "finite"),
        (np.concatenate((np.ones(29), [-1.0])), "non-negative"),
        (np.full(30, 11.0), "volume"),
    ],
)
def test_raw_market_series_rejects_invalid_taker_quote_volume(
    taker: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _raw(taker=taker)


def test_signed_taker_flow_exact_24_bar_formula_and_bounds() -> None:
    values, valid, source_start = _calculate()

    assert not np.any(valid[:23])
    assert valid[23]
    assert values[23] == pytest.approx(0.2)
    assert source_start[23] == 0
    assert valid[24]
    assert source_start[24] == 1

    buy_values, buy_valid, _ = _calculate(taker=np.full(30, 10.0))
    sell_values, sell_valid, _ = _calculate(taker=np.zeros(30))
    assert buy_valid[23] and sell_valid[23]
    assert buy_values[23] == pytest.approx(1.0)
    assert sell_values[23] == pytest.approx(-1.0)


def test_signed_taker_flow_zero_denominator_is_unavailable() -> None:
    volume = np.full(30, 10.0)
    volume[:24] = 0.0
    taker = np.full(30, 6.0)
    taker[:24] = 0.0

    values, valid, source_start = _calculate(volume=volume, taker=taker)

    assert not valid[23]
    assert values[23] == 0.0
    assert source_start[23] == -1


def test_signed_taker_flow_missing_raw_field_is_unavailable() -> None:
    n = 30
    close = 100.0 + np.arange(n, dtype=np.float64)
    values, valid, source_start = calculate_feature_events(
        _spec(),
        open_price=np.concatenate((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(n, 10.0),
        taker_buy_quote_volume=None,
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=np.bool_),
        row_present=np.ones(n, dtype=np.bool_),
        active=np.ones(n, dtype=np.bool_),
        tradable=np.ones(n, dtype=np.bool_),
    )

    assert not np.any(valid)
    assert not np.any(values)
    assert np.all(source_start == -1)


def test_signed_taker_flow_requires_complete_active_tradable_window() -> None:
    row_present = np.ones(30, dtype=np.bool_)
    active = np.ones(30, dtype=np.bool_)
    tradable = np.ones(30, dtype=np.bool_)
    row_present[5] = False
    active[7] = False
    tradable[9] = False

    _, valid, _ = _calculate(
        row_present=row_present,
        active=active,
        tradable=tradable,
    )

    assert not np.any(valid[23:29])
    assert valid[29]


def test_signed_taker_flow_prefix_causality() -> None:
    baseline_values, baseline_valid, _ = _calculate()
    taker = np.full(30, 6.0)
    taker[27:] = 10.0
    changed_values, changed_valid, _ = _calculate(taker=taker)

    np.testing.assert_array_equal(changed_valid[:27], baseline_valid[:27])
    np.testing.assert_array_equal(changed_values[:27], baseline_values[:27])


def test_signed_taker_flow_rejects_alternate_lookback_and_normalization() -> None:
    with pytest.raises(ValueError, match="24"):
        _calculate(spec=_spec(lookback=23))
    with pytest.raises(ValueError, match="normalization"):
        _calculate(
            spec=_spec(
                normalization=NormalizationMode.ROLLING_ZSCORE,
                normalization_window=24,
                min_periods=24,
            )
        )
