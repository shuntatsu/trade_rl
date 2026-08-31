from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.v4_context import (
    BARS_4H,
    CROSS_MARKET_CORE_NAMES,
    CROSS_MARKET_DERIVATIVE_NAMES,
    GLOBAL_MARKET_CORE_NAMES,
    GLOBAL_MARKET_DERIVATIVE_NAMES,
    CausalBetaConfig,
    V4CrossMarketInputs,
    V4GlobalMarketInputs,
    build_causal_beta_series,
    build_cross_market_context,
    build_funding_context_series,
    build_global_market_context,
    robust_trailing_zscore,
    spot_perp_log_basis,
    taker_quote_imbalance,
)


def _digest(value: str) -> str:
    return value * 64


def _timestamps(row_count: int) -> np.ndarray:
    return np.datetime64("2026-01-01T00:00", "ns") + np.arange(
        row_count
    ) * np.timedelta64(15, "m")


def _cross_market_inputs(
    *,
    row_count: int = 800,
    include_derivatives: bool = False,
    phase: float = 0.0,
) -> V4CrossMarketInputs:
    index = np.arange(row_count, dtype=np.float64)
    decision_indices = np.arange(row_count, dtype=np.int64)
    timestamps = _timestamps(row_count)
    spot_close = 100.0 * np.exp(0.00015 * index + 0.003 * np.sin(index / 19.0 + phase))
    perp_close = spot_close * np.exp(0.0005 + 0.0002 * np.sin(index / 23.0))
    perp_mark = perp_close * np.exp(0.00005 * np.cos(index / 17.0))
    spot_quote = 1_000_000.0 * (1.1 + 0.08 * np.sin(index / 11.0 + phase))
    perp_quote = 1_500_000.0 * (1.1 + 0.07 * np.cos(index / 13.0 + phase))
    spot_taker = spot_quote * (0.50 + 0.08 * np.sin(index / 7.0 + phase))
    perp_taker = perp_quote * (0.50 + 0.07 * np.cos(index / 9.0 + phase))
    row_available = np.ones(row_count, dtype=np.bool_)

    funding = np.zeros(row_count, dtype=np.float64)
    funding_events = np.zeros(row_count, dtype=np.bool_)
    event_rows = np.arange(0, row_count, 32)
    funding[event_rows] = 0.0001 * np.sin(np.arange(len(event_rows)) / 3.0 + phase)
    funding_events[event_rows] = True

    if include_derivatives:
        open_interest = 100_000_000.0 * np.exp(
            0.00008 * index + 0.002 * np.sin(index / 31.0 + phase)
        )
        global_ls = 1.0 + 0.1 * np.sin(index / 29.0 + phase)
        top_position_ls = 1.0 + 0.08 * np.cos(index / 37.0 + phase)
        derivative_available = np.ones(row_count, dtype=np.bool_)
        derivative_staleness = np.mod(index, 4.0) * 0.25
    else:
        open_interest = None
        global_ls = None
        top_position_ls = None
        derivative_available = None
        derivative_staleness = None

    return V4CrossMarketInputs(
        decision_indices=decision_indices,
        decision_timestamps=timestamps,
        spot_close=spot_close,
        spot_quote_volume=spot_quote,
        spot_taker_buy_quote_volume=spot_taker,
        spot_row_available=row_available,
        perp_close=perp_close,
        perp_mark_price=perp_mark,
        perp_quote_volume=perp_quote,
        perp_taker_buy_quote_volume=perp_taker,
        perp_row_available=row_available,
        funding_event_rate=funding,
        funding_event_available=funding_events,
        open_interest_value=open_interest,
        global_long_short_ratio=global_ls,
        top_position_long_short_ratio=top_position_ls,
        derivatives_available=derivative_available,
        derivatives_staleness_hours=derivative_staleness,
        source_digest=_digest("a" if phase == 0.0 else "b"),
    )


def _prices_from_log_returns(returns: np.ndarray) -> np.ndarray:
    return np.exp(np.concatenate(([0.0], np.cumsum(returns, dtype=np.float64))))


def test_v4_feature_name_contracts_are_frozen() -> None:
    assert len(CROSS_MARKET_CORE_NAMES) == 24
    assert len(CROSS_MARKET_DERIVATIVE_NAMES) == 7
    assert len(GLOBAL_MARKET_CORE_NAMES) == 38
    assert len(GLOBAL_MARKET_DERIVATIVE_NAMES) == 6
    assert len(set(CROSS_MARKET_CORE_NAMES)) == 24
    assert len(set(CROSS_MARKET_DERIVATIVE_NAMES)) == 7
    assert len(set(GLOBAL_MARKET_CORE_NAMES)) == 38
    assert len(set(GLOBAL_MARKET_DERIVATIVE_NAMES)) == 6
    assert CROSS_MARKET_CORE_NAMES[0] == "spot_log_return_1h"
    assert CROSS_MARKET_CORE_NAMES[-1] == "basis_z_x_flow_divergence_4h"
    assert CROSS_MARKET_DERIVATIVE_NAMES[0] == "open_interest_log_change_1h"
    assert GLOBAL_MARKET_CORE_NAMES[0] == "btc_spot_log_return_1h"
    assert GLOBAL_MARKET_CORE_NAMES[-1] == "btc_eth_perp_return_dispersion_4h"


def test_taker_quote_imbalance_is_signed_and_bounded() -> None:
    assert taker_quote_imbalance(75.0, 100.0) == pytest.approx(0.5)
    assert taker_quote_imbalance(25.0, 100.0) == pytest.approx(-0.5)
    assert taker_quote_imbalance(0.0, 100.0) == -1.0
    assert taker_quote_imbalance(100.0, 100.0) == 1.0
    with pytest.raises(ValueError, match="taker_buy_quote"):
        taker_quote_imbalance(-1.0, 100.0)
    with pytest.raises(ValueError, match="total_quote"):
        taker_quote_imbalance(1.0, 0.0)


def test_spot_perp_log_basis_uses_perp_over_spot() -> None:
    assert spot_perp_log_basis(spot=100.0, perp=101.0) == pytest.approx(np.log(1.01))
    with pytest.raises(ValueError, match="basis prices"):
        spot_perp_log_basis(spot=0.0, perp=101.0)


def test_robust_trailing_zscore_is_prefix_causal() -> None:
    values = np.linspace(-2.0, 3.0, 96, dtype=np.float64)
    available = np.ones(96, dtype=np.bool_)
    first_values, first_available = robust_trailing_zscore(
        values,
        available=available,
        window=64,
        minimum_support=32,
    )
    mutated = values.copy()
    mutated[80:] += 1_000.0
    second_values, second_available = robust_trailing_zscore(
        mutated,
        available=available,
        window=64,
        minimum_support=32,
    )
    np.testing.assert_array_equal(first_available[:80], second_available[:80])
    np.testing.assert_allclose(
        first_values[:80], second_values[:80], atol=0.0, rtol=0.0
    )
    assert not first_available[30]
    assert first_available[31]


def test_funding_context_carries_event_age_then_expires() -> None:
    timestamps = _timestamps(101)
    rates = np.zeros(101, dtype=np.float64)
    events = np.zeros(101, dtype=np.bool_)
    rates[0] = 0.001
    events[0] = True
    result = build_funding_context_series(
        decision_timestamps=timestamps,
        funding_event_rate=rates,
        funding_event_available=events,
    )
    assert result.available.shape == (101, 3)
    assert result.staleness_hours.shape == (101, 3)
    assert result.available[0, 0]
    assert result.staleness_hours[4, 0] == pytest.approx(1.0)
    assert result.available[96, 0]
    assert not result.available[100, 0]
    assert not result.available[0, 1]
    assert not result.available[0, 2]


def test_funding_z_counts_events_not_carried_rows() -> None:
    timestamps = _timestamps(8 * 32 + 1)
    rates = np.zeros(len(timestamps), dtype=np.float64)
    events = np.zeros(len(timestamps), dtype=np.bool_)
    event_rows = np.arange(0, 8 * 32, 32)
    rates[event_rows] = np.asarray(
        [0.001, 0.002, -0.001, 0.003, 0.0, 0.0025, -0.0005, 0.004],
        dtype=np.float64,
    )
    events[event_rows] = True
    result = build_funding_context_series(
        decision_timestamps=timestamps,
        funding_event_rate=rates,
        funding_event_available=events,
    )
    first_z_row = int(event_rows[-1])
    assert not np.any(result.available[:first_z_row, 2])
    assert result.available[first_z_row, 2]
    assert result.robust_z_7d[first_z_row] != 0.0


def test_cross_market_input_rejects_taker_volume_above_total() -> None:
    inputs = _cross_market_inputs(row_count=80)
    bad_taker = inputs.spot_taker_buy_quote_volume.copy()
    bad_taker[10] = inputs.spot_quote_volume[10] + 1.0
    with pytest.raises(ValueError, match="taker"):
        replace(inputs, spot_taker_buy_quote_volume=bad_taker)


def test_cross_market_core_builds_exact_schema_and_readonly_arrays() -> None:
    inputs = _cross_market_inputs(row_count=800)
    block = build_cross_market_context(inputs, include_derivatives=False)
    assert block.feature_names == CROSS_MARKET_CORE_NAMES
    assert block.values.shape == (800, 24)
    assert block.available.shape == (800, 24)
    assert block.staleness_hours.shape == (800, 24)
    assert not block.values.flags.writeable
    assert not block.available.flags.writeable
    assert not block.staleness_hours.flags.writeable
    assert len(block.digest) == 64


def test_cross_market_derivative_profile_builds_exact_schema() -> None:
    inputs = _cross_market_inputs(row_count=800, include_derivatives=True)
    block = build_cross_market_context(inputs, include_derivatives=True)
    assert block.feature_names == (
        *CROSS_MARKET_CORE_NAMES,
        *CROSS_MARKET_DERIVATIVE_NAMES,
    )
    assert block.values.shape == (800, 31)
    oi_index = block.feature_names.index("open_interest_log_change_4h")
    assert np.any(block.available[:, oi_index])


def test_missing_spot_bar_invalidates_four_hour_spot_return() -> None:
    inputs = _cross_market_inputs(row_count=80)
    available = inputs.spot_row_available.copy()
    available[25] = False
    changed = replace(inputs, spot_row_available=available)
    block = build_cross_market_context(changed, include_derivatives=False)
    feature_index = block.feature_names.index("spot_log_return_4h")
    assert not block.available[32, feature_index]
    assert block.values[32, feature_index] == 0.0


def test_cross_market_context_future_mutation_cannot_change_prefix() -> None:
    inputs = _cross_market_inputs(row_count=800)
    first = build_cross_market_context(inputs, include_derivatives=False)
    spot_close = inputs.spot_close.copy()
    spot_close[700:] *= 2.0
    second = build_cross_market_context(
        replace(inputs, spot_close=spot_close),
        include_derivatives=False,
    )
    np.testing.assert_array_equal(first.values[:700], second.values[:700])
    np.testing.assert_array_equal(first.available[:700], second.available[:700])
    np.testing.assert_array_equal(
        first.staleness_hours[:700], second.staleness_hours[:700]
    )


def test_global_market_context_builds_exact_core_and_derivative_widths() -> None:
    btc = _cross_market_inputs(row_count=800, include_derivatives=True)
    eth = _cross_market_inputs(row_count=800, include_derivatives=True, phase=0.7)
    source = V4GlobalMarketInputs(btc=btc, eth=eth, source_digest=_digest("c"))
    core = build_global_market_context(source, include_derivatives=False)
    derivatives = build_global_market_context(source, include_derivatives=True)
    assert core.feature_names == GLOBAL_MARKET_CORE_NAMES
    assert core.values.shape == (800, 38)
    assert derivatives.feature_names == (
        *GLOBAL_MARKET_CORE_NAMES,
        *GLOBAL_MARKET_DERIVATIVE_NAMES,
    )
    assert derivatives.values.shape == (800, 44)


def test_global_market_inputs_require_identical_decision_clock() -> None:
    btc = _cross_market_inputs(row_count=80)
    eth = _cross_market_inputs(row_count=80, phase=0.7)
    shifted = replace(eth, decision_indices=eth.decision_indices + 1)
    with pytest.raises(ValueError, match="decision"):
        V4GlobalMarketInputs(btc=btc, eth=shifted, source_digest=_digest("c"))


def test_causal_beta_recovers_known_two_beta() -> None:
    bars_per_4h = 1
    btc_returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01, 0.015, -0.005, 0.02, -0.012],
        dtype=np.float64,
    )
    btc_close = _prices_from_log_returns(btc_returns)
    target_close = _prices_from_log_returns(2.0 * btc_returns)
    decision_indices = np.arange(len(btc_close), dtype=np.int64)
    available = np.ones(len(btc_close), dtype=np.bool_)
    config = CausalBetaConfig(
        return_horizon_hours=4.0,
        lookback_hours=24.0,
        minimum_complete_samples=3,
        minimum_market_variance=1e-12,
        minimum_beta=-3.0,
        maximum_beta=3.0,
    )
    result = build_causal_beta_series(
        symbol="ETHUSDT",
        decision_indices=decision_indices,
        target_close=target_close,
        btc_close=btc_close,
        target_row_available=available,
        btc_row_available=available,
        bars_per_4h=bars_per_4h,
        config=config,
        target_source_digest=_digest("1"),
        btc_source_digest=_digest("2"),
    )
    assert np.count_nonzero(result.available) > 0
    np.testing.assert_allclose(result.beta[result.available], 2.0, atol=1e-12, rtol=0.0)


def test_causal_beta_future_mutation_does_not_change_prefix() -> None:
    btc_returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01, 0.015, -0.005, 0.02, -0.012],
        dtype=np.float64,
    )
    btc_close = _prices_from_log_returns(btc_returns)
    target_close = _prices_from_log_returns(1.5 * btc_returns)
    decision_indices = np.arange(len(btc_close), dtype=np.int64)
    available = np.ones(len(btc_close), dtype=np.bool_)
    config = CausalBetaConfig(
        return_horizon_hours=4.0,
        lookback_hours=24.0,
        minimum_complete_samples=3,
        minimum_market_variance=1e-12,
        minimum_beta=-3.0,
        maximum_beta=3.0,
    )
    common = dict(
        symbol="ETHUSDT",
        decision_indices=decision_indices,
        btc_close=btc_close,
        target_row_available=available,
        btc_row_available=available,
        bars_per_4h=1,
        config=config,
        target_source_digest=_digest("1"),
        btc_source_digest=_digest("2"),
    )
    before = build_causal_beta_series(target_close=target_close, **common)
    prefix_stop = 6
    mutated_target = target_close.copy()
    mutated_target[prefix_stop + 1 :] *= 5.0
    after = build_causal_beta_series(target_close=mutated_target, **common)
    np.testing.assert_array_equal(before.beta[:prefix_stop], after.beta[:prefix_stop])
    np.testing.assert_array_equal(
        before.available[:prefix_stop], after.available[:prefix_stop]
    )


def test_causal_beta_uses_non_overlapping_four_hour_returns() -> None:
    bars_per_4h = BARS_4H
    four_hour_returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01, 0.015, -0.005] * 31,
        dtype=np.float64,
    )
    endpoints = _prices_from_log_returns(four_hour_returns)
    row_count = (len(endpoints) - 1) * bars_per_4h + 1
    btc_left = np.empty(row_count, dtype=np.float64)
    btc_right = np.empty(row_count, dtype=np.float64)
    for block in range(len(endpoints) - 1):
        start = block * bars_per_4h
        stop = start + bars_per_4h
        linear = np.linspace(endpoints[block], endpoints[block + 1], bars_per_4h + 1)
        btc_left[start : stop + 1] = linear
        phase = np.linspace(0.0, np.pi, bars_per_4h + 1)
        curved = np.exp(0.01 * np.sin(phase))
        curved[0] = 1.0
        curved[-1] = 1.0
        btc_right[start : stop + 1] = linear * curved
    target_left = np.square(btc_left)
    target_right = np.square(btc_right)
    decision_indices = np.arange(row_count, dtype=np.int64)
    available = np.ones(row_count, dtype=np.bool_)
    config = CausalBetaConfig(
        return_horizon_hours=4.0,
        lookback_hours=720.0,
        minimum_complete_samples=90,
        minimum_market_variance=1e-12,
        minimum_beta=-3.0,
        maximum_beta=3.0,
    )
    common = dict(
        symbol="ETHUSDT",
        decision_indices=decision_indices,
        target_row_available=available,
        btc_row_available=available,
        bars_per_4h=bars_per_4h,
        config=config,
        target_source_digest=_digest("1"),
        btc_source_digest=_digest("2"),
    )
    first = build_causal_beta_series(
        target_close=target_left,
        btc_close=btc_left,
        **common,
    )
    second = build_causal_beta_series(
        target_close=target_right,
        btc_close=btc_right,
        **common,
    )
    assert np.count_nonzero(first.available) > 0
    np.testing.assert_allclose(
        first.beta[first.available],
        second.beta[second.available],
        atol=1e-12,
        rtol=0.0,
    )


def test_causal_beta_excludes_four_hour_sample_with_missing_intermediate_row() -> None:
    bars_per_4h = BARS_4H
    row_count = bars_per_4h * 100 + 1
    index = np.arange(row_count, dtype=np.float64)
    btc_close = np.exp(0.0001 * index + 0.002 * np.sin(index / 17.0))
    target_close = np.square(btc_close)
    decision_indices = np.arange(row_count, dtype=np.int64)
    target_available = np.ones(row_count, dtype=np.bool_)
    btc_available = np.ones(row_count, dtype=np.bool_)
    target_available[bars_per_4h * 50 + 3] = False
    config = CausalBetaConfig(
        return_horizon_hours=4.0,
        lookback_hours=720.0,
        minimum_complete_samples=90,
        minimum_market_variance=1e-12,
        minimum_beta=-3.0,
        maximum_beta=3.0,
    )
    result = build_causal_beta_series(
        symbol="ETHUSDT",
        decision_indices=decision_indices,
        target_close=target_close,
        btc_close=btc_close,
        target_row_available=target_available,
        btc_row_available=btc_available,
        bars_per_4h=bars_per_4h,
        config=config,
        target_source_digest=_digest("1"),
        btc_source_digest=_digest("2"),
    )
    assert not result.available[bars_per_4h * 90]
    assert result.available[-1]


def test_btc_beta_is_exactly_one_when_available() -> None:
    returns = np.asarray([0.01, -0.02, 0.03, -0.01, 0.015, 0.02], dtype=np.float64)
    close = _prices_from_log_returns(returns)
    decision_indices = np.arange(len(close), dtype=np.int64)
    available = np.ones(len(close), dtype=np.bool_)
    result = build_causal_beta_series(
        symbol="BTCUSDT",
        decision_indices=decision_indices,
        target_close=close,
        btc_close=close,
        target_row_available=available,
        btc_row_available=available,
        bars_per_4h=1,
        config=CausalBetaConfig(
            return_horizon_hours=4.0,
            lookback_hours=16.0,
            minimum_complete_samples=2,
            minimum_market_variance=1e-12,
            minimum_beta=-3.0,
            maximum_beta=3.0,
        ),
        target_source_digest=_digest("1"),
        btc_source_digest=_digest("1"),
    )
    assert np.count_nonzero(result.available) > 0
    np.testing.assert_array_equal(
        result.beta[result.available],
        np.ones(np.count_nonzero(result.available), dtype=np.float64),
    )
