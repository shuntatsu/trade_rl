from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.test_shared_cash_replay import _market
from trade_rl.data.features.weekly_context import WEEKLY_NAMES, with_weekly_context
from trade_rl.evaluation.bot import BotConfig, create_strategy_instances
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


def _weekly_market(weeks: int = 80):
    weekly = 100.0 + np.arange(weeks) + 8 * np.sin(np.arange(weeks) / 3)
    prices = np.repeat(weekly, 168).reshape(-1, 1)
    source = _market(prices)
    timestamps = np.datetime64("2020-01-06T01", "ns") + np.arange(
        len(prices)
    ) * np.timedelta64(1, "h")
    return replace(source, timestamps=timestamps, available_at=timestamps[:, None])


def test_completed_calendar_week_and_independent_plotted_cloud_formula():
    source = _weekly_market()
    data = with_weekly_context(source)
    row = 78 * 168 - 1
    assert not data.feature_available[row - 1, 0, -1]
    assert data.feature_available[row, 0, -7:].all()
    closes = source.close[167::168, 0]
    highs = source.high.reshape(-1, 168).max(axis=1)
    lows = source.low.reshape(-1, 168).min(axis=1)

    def midpoint(end, length):
        return (
            max(highs[end - length + 1 : end + 1])
            + min(lows[end - length + 1 : end + 1])
        ) / 2

    mean, std = np.mean(closes[58:78]), np.std(closes[58:78])
    tenkan, kijun = midpoint(77, 9), midpoint(77, 26)
    cloud_a = (midpoint(51, 9) + midpoint(51, 26)) / 2
    cloud_b = midpoint(51, 52)
    expected = [
        (closes[77] - mean) / (2 * std),
        (highs[77] - mean) / (2 * std),
        (lows[77] - mean) / (2 * std),
        (closes[77] - tenkan) / closes[77],
        (closes[77] - kijun) / closes[77],
        closes[77] / max(cloud_a, cloud_b) - 1,
        closes[77] / min(cloud_a, cloud_b) - 1,
    ]
    np.testing.assert_array_equal(
        data.features[row, 0, -7:], np.asarray(expected, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        data.features[row : row + 168, 0, -7:],
        np.tile(data.features[row, 0, -7:], (168, 1)),
    )
    assert data.feature_staleness_hours[row + 167, 0, -1] == 167
    assert data.identity_verified
    assert data.feature_names[-7:] == WEEKLY_NAMES
    for field in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "fee_rate",
        "tradable",
        "available_at",
    ):
        np.testing.assert_array_equal(
            data.resolved_array(field), source.resolved_array(field)
        )
    np.testing.assert_array_equal(data.features[:, :, :-7], source.features)


def test_partial_week_future_poison_and_prefix_do_not_repaint():
    source = _weekly_market()
    cutoff = 79 * 168 + 20
    before = with_weekly_context(source)
    prices = source.close.copy()
    prices[cutoff:] *= 3
    after = with_weekly_context(
        replace(source, close=prices, high=np.maximum(source.high, prices))
    )
    np.testing.assert_array_equal(before.features[:cutoff], after.features[:cutoff])
    np.testing.assert_array_equal(
        before.feature_available[:cutoff], after.feature_available[:cutoff]
    )
    assert before.features[cutoff, 0, -1] == before.features[79 * 168 - 1, 0, -1]


@pytest.mark.parametrize("kind", ["missing", "late", "inactive"])
def test_unusable_constituent_invalidates_full_window_then_recovers(kind):
    source = _weekly_market()
    mask = np.ones_like(source.close, dtype=bool)
    mask[168 + 4] = False
    if kind == "late":
        release = source.resolved_array("available_at").copy()
        release[168 + 4] += np.timedelta64(1, "h")
        source = replace(source, available_at=release, information_available=mask)
    elif kind == "inactive":
        source = replace(
            source,
            asset_active=mask,
            symbol_active=mask,
            tradable=mask,
            information_available=mask,
        )
    else:
        source = replace(source, information_available=mask)
    data = with_weekly_context(source)
    assert not data.feature_available[79 * 168 - 1, 0, -7:].any()
    assert data.feature_available[80 * 168 - 1, 0, -7:].all()


def test_week_alignment_duplicates_and_nonhourly_reject():
    source = _weekly_market()
    shifted = source.timestamps + np.timedelta64(30, "m")
    with pytest.raises(ValueError, match="UTC hour"):
        with_weekly_context(
            replace(source, timestamps=shifted, available_at=shifted[:, None])
        )
    with pytest.raises(ValueError, match="already"):
        with_weekly_context(with_weekly_context(source))
    short = _market(np.arange(10.0, 20.0).reshape(-1, 1))
    assert not with_weekly_context(short).feature_available[:, :, -7:].any()


def test_zero_sigma_scale_invariance_and_partial_first_week():
    source = _weekly_market()
    constant = np.full_like(source.close, 100.0)
    flat = with_weekly_context(
        replace(source, open=constant, high=constant, low=constant, close=constant)
    )
    assert flat.feature_available[-1, 0, -7:].all()
    np.testing.assert_array_equal(flat.features[-1, 0, -7:], [0, 1, -1, 0, 0, 0, 0])
    scaled = with_weekly_context(
        replace(
            source,
            open=source.open * 3,
            high=source.high * 3,
            low=source.low * 3,
            close=source.close * 3,
        )
    )
    original = with_weekly_context(source)
    np.testing.assert_array_equal(
        scaled.features[:, :, -7:], original.features[:, :, -7:]
    )
    # Starting at Tuesday creates an incomplete first week, never a 168-row rolling week.
    timestamps = source.timestamps + np.timedelta64(24, "h")
    partial = with_weekly_context(
        replace(source, timestamps=timestamps, available_at=timestamps[:, None])
    )
    assert not partial.feature_available[78 * 168 - 1, 0, -7:].any()
    assert partial.feature_available[79 * 168 - 25, 0, -7:].all()


def test_new_invalid_week_expires_prior_permission_at_exact_endpoint():
    source = _weekly_market()
    flags = source.resolved_array("information_available").copy()
    flags[79 * 168 - 10] = False
    data = with_weekly_context(replace(source, information_available=flags))
    row = 79 * 168 - 1
    assert data.feature_available[row - 1, 0, -7:].all()
    assert not data.feature_available[row, 0, -7:].any()
    assert not data.feature_available[row + 1, 0, -7:].any()


@pytest.mark.parametrize("scale", [1.0, 7.0, 1e-13])
@pytest.mark.parametrize("zero_sigma", [False, True])
def test_reachable_rejection_wick_with_zero_or_tiny_sigma(scale, zero_sigma):
    # Independent raw-price oracle: cloud90, Tenkan115, Kijun105, C110/H125.
    # BB upper is110 at zero sigma, or111.897366... otherwise, both below high125.
    closes = np.r_[
        np.full(52, 90.0),
        np.full(6, 100.0),
        np.full(20, 110.0) if zero_sigma else [109.0, 111.0] * 9 + [110.0, 110.0],
    ]
    source = _market(np.repeat(closes, 168).reshape(-1, 1))
    timestamps = np.datetime64("2020-01-06T01", "ns") + np.arange(
        source.n_bars
    ) * np.timedelta64(1, "h")
    highs = np.repeat(np.r_[np.full(52, 90.0), np.full(25, 115.0), 125.0], 168).reshape(
        -1, 1
    )
    lows = np.repeat(
        np.r_[np.full(52, 90.0), np.full(17, 85.0), np.full(9, 105.0)], 168
    ).reshape(-1, 1)
    data = with_weekly_context(
        replace(
            source,
            timestamps=timestamps,
            available_at=timestamps[:, None],
            open=source.open * scale,
            close=source.close * scale,
            high=highs * scale,
            low=lows * scale,
            features=np.full_like(source.features, 0.03),
        )
    )
    row = data.n_bars - 1
    assert data.feature_available[row, 0, -7:].all()
    assert data.features[row, 0, -6] >= 1.0  # Reaching inclusive raw upperBB.
    observation = StrategyObservation(
        index=row,
        timestamp=data.timestamps[row],
        symbol="ANY",
        features=data.features[row, 0],
        feature_available=data.feature_available[row, 0],
        global_features=np.zeros(1),
        global_feature_available=np.ones(1, dtype=bool),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )
    strategy = create_strategy_instances(
        data,
        BotConfig(strategy_name="weekly_bb_ichimoku", volatility_regime_threshold=0),
    )[0]
    assert strategy.decide(observation) is PositionIntent.FLAT
