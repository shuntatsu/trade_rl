from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_calibration import (
    calibrate_mean_reversion_economic_gate,
)
from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_prereg import (
    canonical_mean_reversion_economic_gate_protocol,
)


def _dataset(*, slopes: tuple[float, ...]) -> MarketDataset:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    n_bars = 17_600
    n_symbols = len(protocol.symbols)
    timestamps = np.datetime64("2021-01-01T00:00:00", "ns") + np.arange(
        n_bars, dtype=np.int64
    ) * np.timedelta64(1, "h")

    time = np.arange(n_bars, dtype=np.float64)
    open_price = np.empty((n_bars, n_symbols), dtype=np.float64)
    for symbol_index in range(n_symbols):
        log_price = (
            5.0
            + 0.00005 * time
            + 0.01
            * np.sin(2.0 * math.pi * time / 168.0 + 0.31 * symbol_index)
        )
        open_price[:, symbol_index] = np.exp(log_price)

    features = np.zeros((n_bars, n_symbols, 3), dtype=np.float32)
    for symbol_index, beta in enumerate(slopes):
        future = np.log(
            open_price[25:, symbol_index] / open_price[1 : n_bars - 24, symbol_index]
        )
        signal = future / beta
        features[: signal.size, symbol_index, protocol.signal_index] = signal.astype(
            np.float32
        )

    feature_available = np.ones_like(features, dtype=np.bool_)
    tradable = np.ones((n_bars, n_symbols), dtype=np.bool_)
    active = np.ones((n_bars, n_symbols), dtype=np.bool_)
    fee = np.full((n_bars, n_symbols), protocol.market_order_fee_rate)
    taker = np.full((n_bars, n_symbols), protocol.market_order_taker_fee_rate)
    spread = np.full((n_bars, n_symbols), protocol.market_order_spread_rate)

    # Decision-row cost is intentionally wrong for the first eligible row. The
    # calibration contract reads execution economics from t+1, so this row must
    # not contaminate the observed execution-cost range.
    fee[1, :] = 0.123

    return MarketDataset(
        dataset_id=protocol.successor_dataset_id,
        symbols=protocol.symbols,
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=open_price.copy(),
        low=open_price.copy(),
        close=open_price.copy(),
        volume=np.full((n_bars, n_symbols), 1_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols)),
        tradable=tradable,
        feature_available=feature_available,
        feature_names=("feature_0", "feature_1", protocol.signal_name),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=fee,
        taker_fee_rate=taker,
        spread_rate=spread,
        max_participation_rate=np.tile(
            np.asarray(protocol.capacity_caps, dtype=np.float64), (n_bars, 1)
        ),
        asset_active=active,
    )


def test_valid_calibration_uses_conservative_fourth_order_statistic() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    slopes = (-0.60, -0.50, -0.40, -0.30, 0.20)
    result = calibrate_mean_reversion_economic_gate(
        _dataset(slopes=slopes), protocol
    )

    assert result.status == "VALID_CALIBRATION"
    assert result.protocol_digest == protocol.digest
    assert result.dataset_id == protocol.successor_dataset_id
    assert result.symbols == protocol.symbols
    assert result.negative_slope_count == 4
    assert result.beta_gate == pytest.approx(-0.30, abs=2e-7)
    assert result.evaluation_pnl_inspected is False
    assert result.strategy_execution_performed is False
    assert result.production_eligible is False
    assert result.evaluation_execution_authorized is False
    assert len(result.digest) == 64

    assert tuple(item.symbol for item in result.symbol_results) == protocol.symbols
    for item, expected in zip(result.symbol_results, slopes, strict=True):
        assert item.eligible_observations >= 8_760
        assert item.denominator > 0.0
        assert math.isfinite(item.numerator)
        assert item.beta == pytest.approx(expected, abs=2e-7)
        assert item.fee_rate_min == pytest.approx(0.0005)
        assert item.fee_rate_max == pytest.approx(0.0005)
        assert item.taker_fee_rate_min == pytest.approx(0.0)
        assert item.taker_fee_rate_max == pytest.approx(0.0)
        assert item.spread_rate_min == pytest.approx(0.0002)
        assert item.spread_rate_max == pytest.approx(0.0002)


def test_calibration_fails_when_only_three_symbols_support_reversion() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    result = calibrate_mean_reversion_economic_gate(
        _dataset(slopes=(-0.50, -0.40, -0.30, 0.20, 0.30)), protocol
    )

    assert result.status == protocol.invalid_training_edge_status
    assert result.negative_slope_count == 3
    assert result.beta_gate is None
    assert result.evaluation_execution_authorized is False


def test_calibration_fails_closed_on_insufficient_tradable_coverage() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    dataset = _dataset(slopes=(-0.60, -0.50, -0.40, -0.30, 0.20))
    tradable = np.asarray(dataset.tradable).copy()
    tradable[1:10_500, 0] = False
    broken = dataset.with_updates(tradable=tradable)

    result = calibrate_mean_reversion_economic_gate(broken, protocol)

    assert result.status == protocol.invalid_coverage_status
    assert result.symbol_results[0].eligible_observations < 8_760
    assert result.beta_gate is None
    assert result.evaluation_execution_authorized is False


def test_calibration_requires_continuous_tradability_through_label_window() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    dataset = _dataset(slopes=(-0.60, -0.50, -0.40, -0.30, 0.20))
    baseline = calibrate_mean_reversion_economic_gate(dataset, protocol)

    tradable = np.asarray(dataset.tradable).copy()
    # Row 100 is inside the t+1..t+25 holding window for 25 decision rows.
    tradable[100, 0] = False
    changed = calibrate_mean_reversion_economic_gate(
        dataset.with_updates(tradable=tradable), protocol
    )

    assert (
        baseline.symbol_results[0].eligible_observations
        - changed.symbol_results[0].eligible_observations
        == 26
    )


def test_calibration_rejects_dataset_and_signal_identity_drift() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    dataset = _dataset(slopes=(-0.60, -0.50, -0.40, -0.30, 0.20))

    with pytest.raises(ValueError, match="dataset_id"):
        calibrate_mean_reversion_economic_gate(
            dataset.with_updates(dataset_id="0" * 64), protocol
        )

    names = list(dataset.feature_names)
    names[protocol.signal_index] = "wrong_signal"
    with pytest.raises(ValueError, match="signal"):
        calibrate_mean_reversion_economic_gate(
            dataset.with_updates(feature_names=tuple(names)), protocol
        )


def test_calibration_rejects_cost_semantic_drift_at_execution_rows() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    dataset = _dataset(slopes=(-0.60, -0.50, -0.40, -0.30, 0.20))
    fee = np.asarray(dataset.fee_rate).copy()
    fee[2, 0] = 0.001

    with pytest.raises(ValueError, match="execution cost"):
        calibrate_mean_reversion_economic_gate(dataset.with_updates(fee_rate=fee), protocol)
