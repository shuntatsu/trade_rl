"""Independent causal oracles for hourly observations of a forming weekly BB."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.data.market import MarketDataset
from trade_rl.data.view import MarketDatasetView

_MODULE = "trade_rl.data.features.forming_week_context"
_NAMES = (
    "forming_week_bb_hour_high_position",
    "forming_week_bb_hour_low_position",
)
_WEEK_HOURS = 168


def _transform(source: MarketDataset) -> MarketDataset:
    assert importlib.util.find_spec(_MODULE) is not None, (
        "forming-week BB API is missing: implement with_forming_week_context"
    )
    module = importlib.import_module(_MODULE)
    assert getattr(module, "FORMING_WEEK_NAMES", None) == _NAMES
    transform = getattr(module, "with_forming_week_context", None)
    assert callable(transform), "forming-week BB transform must be callable"
    return transform(source)


def _market(weeks: int = 21, *, symbols: int = 1) -> MarketDataset:
    bars = weeks * _WEEK_HOURS
    timestamps = np.datetime64("2020-01-06T01", "ns") + np.arange(
        bars
    ) * np.timedelta64(1, "h")
    close = np.full((bars, symbols), 100.0, dtype=np.float64)
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=tuple(f"ASSET{index}" for index in range(symbols)),
        timestamps=timestamps,
        features=np.full((bars, symbols, 1), 0.25, dtype=np.float32),
        feature_available=np.ones((bars, symbols, 1), dtype=np.bool_),
        feature_names=("existing",),
        global_features=np.zeros((bars, 1), dtype=np.float32),
        global_feature_names=("global",),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.full_like(close, 1_000.0),
        funding_rate=np.full_like(close, 0.0001),
        fee_rate=np.full_like(close, 0.0005),
        tradable=np.ones_like(close, dtype=np.bool_),
        information_available=np.ones_like(close, dtype=np.bool_),
        available_at=timestamps[:, None]
        + np.zeros_like(close, dtype="timedelta64[ns]"),
        periods_per_year=8_760,
    )


def _prices(
    source: MarketDataset,
    close: np.ndarray,
    *,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> MarketDataset:
    return replace(
        source,
        open=close,
        close=close,
        high=close if high is None else high,
        low=close if low is None else low,
        identity_payload_json=None,
    )


def _decimal_positions(
    prior_closes: list[float], current_close: float, high: float, low: float
) -> np.ndarray:
    # This oracle never calls production means, variance, or calendar helpers.
    with localcontext() as context:
        context.prec = 60
        sample = [Decimal.from_float(value) for value in prior_closes + [current_close]]
        mean = sum(sample) / Decimal(20)
        variance = sum((value - mean) ** 2 for value in sample) / Decimal(20)
        if variance == 0:
            return np.zeros(2, dtype=np.float32)
        width = Decimal(2) * variance.sqrt()
        return np.asarray(
            [
                float((Decimal.from_float(price) - mean) / width)
                for price in (high, low)
            ],
            dtype=np.float32,
        )


def test_forming_week_api_exists_with_exact_two_feature_names() -> None:
    assert importlib.util.find_spec(_MODULE) is not None, (
        "forming-week BB API is missing: implement with_forming_week_context"
    )
    module = importlib.import_module(_MODULE)
    assert module.FORMING_WEEK_NAMES == _NAMES
    assert callable(module.with_forming_week_context)


def test_first_forming_hour_uses_nineteen_completed_closes_and_population_bb() -> None:
    source = _market()
    row = 19 * _WEEK_HOURS
    close = source.close.copy()
    close[row, 0] = 105.0
    high, low = close.copy(), close.copy()
    high[row, 0], low[row, 0] = 106.0, 104.0
    data = _transform(_prices(source, close, high=high, low=low))

    assert not data.feature_available[:row, :, -2:].any()
    assert data.feature_available[row, 0, -2:].all()
    expected = _decimal_positions([100.0] * 19, 105.0, 106.0, 104.0)
    np.testing.assert_allclose(data.features[row, 0, -2:], expected, rtol=1e-6, atol=0)
    # Mean100.25, population width2.179449..., upper102.429449...; both wicks exceed it.
    assert data.features[row, 0, -2] > 1.0
    assert data.features[row, 0, -1] > 1.0


def test_hourly_wick_does_not_reuse_an_earlier_forming_week_extreme() -> None:
    source = _market()
    row = 19 * _WEEK_HOURS
    close = source.close.copy()
    close[row, 0], close[row + 1, 0] = 105.0, 98.0
    high, low = close.copy(), close.copy()
    high[row, 0], high[row + 1, 0] = 120.0, 100.0
    low[row + 1, 0] = 97.0
    data = _transform(_prices(source, close, high=high, low=low))

    assert data.features[row, 0, -2] >= 1.0
    assert data.features[row + 1, 0, -2] < 1.0
    expected = _decimal_positions([100.0] * 19, 98.0, 100.0, 97.0)
    np.testing.assert_allclose(data.features[row + 1, 0, -2:], expected, rtol=1e-6)


def test_monday_midnight_closes_old_week_and_next_hour_shifts_prior_sample() -> None:
    source = _market(22)
    close = source.close.copy()
    for week in range(21):
        close[(week + 1) * _WEEK_HOURS - 1, 0] = 100.0 + week
    endpoint = 20 * _WEEK_HOURS - 1
    close[endpoint, 0] = 140.0
    close[endpoint + 1, 0] = 95.0
    data = _transform(_prices(source, close))

    np.testing.assert_allclose(
        data.features[endpoint, 0, -2:],
        _decimal_positions([100.0 + week for week in range(19)], 140.0, 140.0, 140.0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        data.features[endpoint + 1, 0, -2:],
        _decimal_positions(
            [100.0 + week for week in range(1, 19)] + [140.0], 95, 95, 95
        ),
        rtol=1e-6,
    )


def test_truncated_prefix_and_future_prices_cannot_repaint_earlier_features() -> None:
    source = _market(23)
    cutoff = 20 * _WEEK_HOURS + 37
    close = source.close.copy()
    close[19 * _WEEK_HOURS :, 0] += np.arange(source.n_bars - 19 * _WEEK_HOURS) / 100
    source = _prices(source, close)
    original = _transform(source)
    prefix = _transform(MarketDatasetView(source, 0, cutoff).materialize())
    poisoned = close.copy()
    poisoned[cutoff:] *= 9
    changed = _transform(_prices(source, poisoned))
    for field in (
        "features",
        "feature_available",
        "feature_staleness",
        "feature_staleness_hours",
        "feature_missing_reason",
    ):
        expected = original.resolved_array(field)[:cutoff]
        np.testing.assert_array_equal(prefix.resolved_array(field), expected)
        np.testing.assert_array_equal(changed.resolved_array(field)[:cutoff], expected)


@pytest.mark.parametrize("kind", ["missing", "late", "inactive"])
def test_bad_current_constituent_invalidates_remaining_week_and_nineteen_prior_weeks(
    kind: str,
) -> None:
    source = _market(40)
    first = 19 * _WEEK_HOURS
    bad = first + 5
    flags = np.ones_like(source.close, dtype=np.bool_)
    flags[bad, 0] = False
    if kind == "missing":
        source = replace(source, information_available=flags)
    elif kind == "inactive":
        source = replace(
            source,
            asset_active=flags,
            symbol_active=flags,
            tradable=flags,
            information_available=flags,
        )
    else:
        release = source.resolved_array("available_at").copy()
        release[bad, 0] += np.timedelta64(1, "h")
        source = replace(source, available_at=release, information_available=flags)
    data = _transform(source)

    assert data.feature_available[first:bad, 0, -2:].all()
    assert not data.feature_available[bad : 39 * _WEEK_HOURS, 0, -2:].any()
    assert data.feature_available[39 * _WEEK_HOURS :, 0, -2:].all()
    np.testing.assert_array_equal(data.features[bad : 39 * _WEEK_HOURS, 0, -2:], 0)


def test_invalid_first_complete_week_leaves_lookback_without_backfill() -> None:
    source = _market(22)
    release = source.resolved_array("available_at").copy()
    release[5, 0] += np.timedelta64(1, "h")
    flags = source.resolved_array("information_available").copy()
    flags[5, 0] = False
    data = _transform(
        replace(source, available_at=release, information_available=flags)
    )
    assert not data.feature_available[: 20 * _WEEK_HOURS, 0, -2:].any()
    assert data.feature_available[20 * _WEEK_HOURS, 0, -2:].all()


def test_partial_first_week_is_excluded_even_with_nineteen_full_row_groups() -> None:
    source = _market(22)
    timestamps = source.timestamps + np.timedelta64(24, "h")
    data = _transform(
        replace(source, timestamps=timestamps, available_at=timestamps[:, None])
    )
    first = 20 * _WEEK_HOURS - 24
    assert not data.feature_available[:first, 0, -2:].any()
    assert data.feature_available[first, 0, -2:].all()


def test_exact_zero_variance_is_available_neutral_even_with_hourly_wicks() -> None:
    source = _market()
    row = 19 * _WEEK_HOURS
    high, low = source.high.copy(), source.low.copy()
    high[row, 0], low[row, 0] = 150.0, 50.0
    data = _transform(replace(source, high=high, low=low))
    assert data.feature_available[row, 0, -2:].all()
    np.testing.assert_array_equal(data.features[row, 0, -2:], [0, 0])


@pytest.mark.parametrize("scale", [1.0, 7.0, 1e-13, 1e-200, 1e200])
def test_positive_small_variance_keeps_ratios_without_absolute_epsilon(
    scale: float,
) -> None:
    source = _market()
    row = 19 * _WEEK_HOURS
    close = source.close.copy()
    close[row, 0] = 105.0
    high, low = close.copy(), close.copy()
    high[row, 0], low[row, 0] = 106.0, 104.0
    data = _transform(
        _prices(source, close * scale, high=high * scale, low=low * scale)
    )
    expected = _decimal_positions(
        [100.0 * scale] * 19, 105 * scale, 106 * scale, 104 * scale
    )
    np.testing.assert_allclose(data.features[row, 0, -2:], expected, rtol=1e-6, atol=0)
    assert data.feature_available[row, 0, -2:].all()
    assert data.features[row, 0, -2] > 1.0


@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200])
@pytest.mark.parametrize("ulp_steps", [1, 2])
def test_adjacent_float_variance_preserves_actual_bb_reach_at_each_price_scale(
    scale: float, ulp_steps: int
) -> None:
    source = _market()
    row = 19 * _WEEK_HOURS
    reference = 100.0 * scale
    current = reference
    for _ in range(ulp_steps):
        current = float(np.nextafter(current, np.inf))
    assert current > reference  # The scaled input retains a representable difference.
    close = np.full_like(source.close, reference)
    close[row, 0] = current
    low = np.full_like(close, reference)
    data = _transform(_prices(source, close, low=low))
    expected = _decimal_positions([reference] * 19, current, current, reference)

    np.testing.assert_allclose(data.features[row, 0, -2:], expected, rtol=1e-6, atol=0)
    assert data.feature_available[row, 0, -2:].all()
    assert data.features[row, 0, -2] > 1.0
    assert data.features[row, 0, -1] < 0.0


def test_symbol_permutation_and_per_symbol_invalidity_are_independent() -> None:
    source = _market(22, symbols=2)
    close = source.close.copy()
    close[19 * _WEEK_HOURS :, 0] = 105
    close[19 * _WEEK_HOURS :, 1] = 95
    flags = source.resolved_array("information_available").copy()
    flags[19 * _WEEK_HOURS + 1, 0] = False
    source = replace(_prices(source, close), information_available=flags)
    data = _transform(source)
    swapped = _transform(
        replace(
            source,
            symbols=source.symbols[::-1],
            open=source.open[:, ::-1],
            high=source.high[:, ::-1],
            low=source.low[:, ::-1],
            close=source.close[:, ::-1],
            information_available=flags[:, ::-1],
        )
    )
    np.testing.assert_array_equal(
        swapped.features[:, :, -2:], data.features[:, ::-1, -2:]
    )
    np.testing.assert_array_equal(
        swapped.feature_available[:, :, -2:], data.feature_available[:, ::-1, -2:]
    )
    assert not data.feature_available[19 * _WEEK_HOURS + 2, 0, -2:].any()
    assert data.feature_available[19 * _WEEK_HOURS + 2, 1, -2:].all()


def test_metadata_is_current_or_explicitly_missing_and_source_arrays_are_unchanged() -> (
    None
):
    source = _market().with_content_identity({"source_receipt": "synthetic-hourly"})
    data = _transform(source)
    assert data.identity_verified and data.dataset_id != source.dataset_id
    assert data.feature_names[-2:] == _NAMES
    appended = {
        "features",
        "feature_available",
        "feature_staleness",
        "feature_staleness_hours",
        "feature_missing_reason",
    }
    for name, original in source.identity_arrays().items():
        actual = data.identity_arrays()[name]
        np.testing.assert_array_equal(
            actual[:, :, :-2] if name in appended else actual, original
        )
    valid = data.feature_available[:, :, -2:]
    np.testing.assert_array_equal(
        data.feature_staleness_hours[:, :, -2:], np.where(valid, 0, 168)
    )
    np.testing.assert_array_equal(
        data.feature_staleness[:, :, -2:], np.where(valid, 0, 1)
    )
    np.testing.assert_array_equal(
        data.feature_missing_reason[:, :, -2:], (~valid).astype(np.int8)
    )
    payload = json.loads(data.identity_payload_json)
    assert payload["source_dataset"]["dataset_id"] == source.dataset_id
    assert payload["source_dataset"]["identity_payload"] == json.loads(
        source.identity_payload_json
    )


def test_artifact_roundtrip_preserves_identity_and_all_arrays(tmp_path: Path) -> None:
    source = _market().with_content_identity({"source_receipt": "synthetic-hourly"})
    data = _transform(source)
    write_market_dataset_files(tmp_path, data)
    restored = load_market_dataset_artifact(tmp_path)
    assert restored.identity_verified
    assert restored.dataset_id == data.dataset_id
    assert restored.identity_payload_json == data.identity_payload_json
    for name, expected in data.identity_arrays().items():
        np.testing.assert_array_equal(restored.identity_arrays()[name], expected)
    changed_fee = replace(
        source, identity_payload_json=None, fee_rate=np.full_like(source.close, 0.001)
    )
    assert (
        _transform(
            changed_fee.with_content_identity({"source_receipt": "synthetic-hourly"})
        ).dataset_id
        != data.dataset_id
    )


@pytest.mark.parametrize("kind", ["nonhour", "session"])
def test_transform_rejects_noncontinuous_utc_hour_contract(kind: str) -> None:
    source = _market()
    times = source.timestamps.copy()
    if kind == "nonhour":
        times += np.timedelta64(30, "m")
        source = replace(source, timestamps=times, available_at=times[:, None])
    else:
        source = replace(source, calendar_kind=MarketCalendarKind.SESSION)
    with pytest.raises(ValueError, match="UTC hour|continuous"):
        _transform(source)


def test_duplicate_close_endpoint_is_rejected_by_dataset_boundary() -> None:
    source = _market()
    times = source.timestamps.copy()
    times[100] = times[99]
    with pytest.raises(ValueError, match="increasing|unique|duplicate"):
        _transform(replace(source, timestamps=times, available_at=times[:, None]))


def test_hourly_gap_is_rejected_by_continuous_dataset_boundary() -> None:
    source = _market()
    times = source.timestamps.copy()
    times[100:] += np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="continuous|regular"):
        _transform(replace(source, timestamps=times, available_at=times[:, None]))


@pytest.mark.parametrize("names", [_NAMES, (_NAMES[0],), (_NAMES[1],)])
def test_existing_or_partial_forming_context_rejects_before_appending(
    names: tuple[str, ...],
) -> None:
    source = _market()
    source = replace(
        source,
        feature_names=("existing",) + names,
        features=np.zeros((source.n_bars, 1, len(names) + 1), dtype=np.float32),
        feature_available=np.ones((source.n_bars, 1, len(names) + 1), dtype=np.bool_),
        feature_staleness=None,
        feature_staleness_hours=None,
        feature_missing_reason=None,
    )
    with pytest.raises(ValueError, match="already|incomplete"):
        _transform(source)
