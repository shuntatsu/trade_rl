from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.cross_asset_features import calculate_cross_asset_feature_events


def _spec(kind: FeatureKind, *, lookback: int = 4, min_periods: int = 2) -> FeatureSpec:
    return FeatureSpec(
        name=kind.value,
        kind=kind,
        lookback=lookback,
        min_periods=min_periods,
        max_staleness_hours=4.0,
    )


def test_cross_asset_features_are_numerically_causal() -> None:
    returns = np.asarray(
        [
            [0.01, 0.02, -0.01],
            [0.02, 0.04, 0.00],
            [-0.01, -0.02, 0.01],
            [0.03, 0.06, -0.01],
        ],
        dtype=np.float64,
    )
    available = np.ones_like(returns, dtype=np.bool_)
    age = np.zeros_like(returns)
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT")

    relative = calculate_cross_asset_feature_events(
        _spec(FeatureKind.RELATIVE_RETURN_TO_BTC, lookback=1, min_periods=1),
        aligned_returns=returns,
        return_available=available,
        return_age_hours=age,
        symbols=symbols,
        reference_symbol="BTCUSDT",
    )
    np.testing.assert_allclose(relative.values[:, 0], 0.0)
    np.testing.assert_allclose(relative.values[:, 1], returns[:, 1] - returns[:, 0])
    assert relative.valid.all()

    correlation = calculate_cross_asset_feature_events(
        _spec(FeatureKind.ROLLING_CORRELATION_TO_BTC),
        aligned_returns=returns,
        return_available=available,
        return_age_hours=age,
        symbols=symbols,
        reference_symbol="BTCUSDT",
    )
    assert correlation.valid[-1, 1]
    assert correlation.values[-1, 1] == pytest.approx(1.0)

    beta = calculate_cross_asset_feature_events(
        _spec(FeatureKind.ROLLING_BETA_TO_BTC),
        aligned_returns=returns,
        return_available=available,
        return_age_hours=age,
        symbols=symbols,
        reference_symbol="BTCUSDT",
    )
    assert beta.values[-1, 1] == pytest.approx(2.0)

    dispersion = calculate_cross_asset_feature_events(
        _spec(FeatureKind.CROSS_ASSET_DISPERSION, lookback=1, min_periods=1),
        aligned_returns=returns,
        return_available=available,
        return_age_hours=age,
        symbols=symbols,
        reference_symbol="BTCUSDT",
    )
    assert dispersion.values[-1, 0] == pytest.approx(np.std(returns[-1]))


def test_cross_asset_prefix_is_unchanged_by_future_mutation() -> None:
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0, 0.01, size=(80, 3))
    available = np.ones_like(returns, dtype=np.bool_)
    age = np.zeros_like(returns)
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
    split = 50

    for kind in (
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
        FeatureKind.CROSS_ASSET_DISPERSION,
    ):
        spec = _spec(kind, lookback=12, min_periods=4)
        original = calculate_cross_asset_feature_events(
            spec,
            aligned_returns=returns,
            return_available=available,
            return_age_hours=age,
            symbols=symbols,
            reference_symbol="BTCUSDT",
        )
        changed = returns.copy()
        changed[split:] += rng.normal(5.0, 1.0, size=changed[split:].shape)
        mutated = calculate_cross_asset_feature_events(
            spec,
            aligned_returns=changed,
            return_available=available,
            return_age_hours=age,
            symbols=symbols,
            reference_symbol="BTCUSDT",
        )
        np.testing.assert_array_equal(original.valid[:split], mutated.valid[:split])
        np.testing.assert_allclose(original.values[:split], mutated.values[:split])


def test_cross_asset_features_require_explicit_reference_symbol() -> None:
    values = np.zeros((8, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="occur exactly once"):
        calculate_cross_asset_feature_events(
            _spec(FeatureKind.RELATIVE_RETURN_TO_BTC, lookback=1, min_periods=1),
            aligned_returns=values,
            return_available=np.ones_like(values, dtype=np.bool_),
            return_age_hours=np.zeros_like(values),
            symbols=("ETHUSDT", "BNBUSDT"),
            reference_symbol="BTCUSDT",
        )


def test_cross_asset_features_preserve_delayed_source_age() -> None:
    returns = np.asarray(
        [
            [0.01, 0.00],
            [0.01, 0.03],
        ],
        dtype=np.float64,
    )
    available = np.asarray(
        [
            [True, False],
            [True, True],
        ],
        dtype=np.bool_,
    )
    age = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.75],
        ],
        dtype=np.float64,
    )

    result = calculate_cross_asset_feature_events(
        _spec(FeatureKind.RELATIVE_RETURN_TO_BTC, lookback=1, min_periods=1),
        aligned_returns=returns,
        return_available=available,
        return_age_hours=age,
        symbols=("BTCUSDT", "ETHUSDT"),
        reference_symbol="BTCUSDT",
    )

    assert result.valid[1, 1]
    assert result.source_age_hours[1, 1] == pytest.approx(0.75)
