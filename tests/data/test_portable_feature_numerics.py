from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec, MarketBuildConfig
from trade_rl.data.features.core import calculate_feature_events
from trade_rl.data.features.numerics import (
    PORTABLE_FEATURE_NUMERICS_SCHEMA,
    portable_correlation,
    portable_covariance,
    portable_dot,
    portable_log,
    portable_mean,
    portable_std,
    portable_sum,
    portable_variance,
)

_EPSILON = 1e-12

_TREND_CLOSE = np.asarray(
    [
        29015.0,
        29448.4,
        29237.06,
        29302.11,
        29237.07,
        29213.8,
        29197.3,
        29107.71,
        29025.89,
        29229.6,
        29259.29,
        29341.99,
        29257.82,
        29493.66,
        29354.58,
        29210.84,
        29324.21,
        29099.0,
        29086.62,
        29048.47,
        29214.14,
        29177.77,
        29270.89,
        29337.16,
    ],
    dtype=np.float64,
)
_CORR_CLOSE = np.asarray(
    [
        2843.24,
        2856.95,
        2850.99,
        2933.6,
        2954.63,
        2944.99,
        2935.22,
        2925.0,
        2939.76,
        2942.98,
        2939.95,
        2946.1,
        2941.37,
        2944.51,
        2931.12,
        2927.95,
        2933.63,
        2924.87,
        2923.3,
        2928.07,
        2925.4,
        2929.04,
        2893.79,
        2829.29,
        2740.65,
    ],
    dtype=np.float64,
)
_CORR_VOLUME = np.asarray(
    [
        178371479.99161,
        217700240.79219,
        170218251.66315,
        985906310.00321,
        499528188.98515,
        123546521.23818,
        90631407.76651,
        115347509.62008,
        124767163.22539,
        183858682.93128,
        72922996.68414,
        83538939.41229,
        48627626.81203,
        42328264.51671,
        102484093.06324,
        69669619.92084,
        73489528.86574,
        178626929.63437,
        100539933.68714,
        96449855.22685,
        68576812.92353,
        81692701.41202,
        292732021.44597,
        871571360.35913,
        1265830807.72211,
    ],
    dtype=np.float64,
)
_REALIZED_VOL_CLOSE = np.asarray(
    [63283.1, 63146.6, 63192.1, 63255.1, 63237.4], dtype=np.float64
)


def _mean(values: np.ndarray) -> float:
    return math.fsum(float(value) for value in values) / len(values)


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    return math.fsum(
        float(a) * float(b) for a, b in zip(left, right, strict=True)
    )


def _trend_r2_oracle(close: np.ndarray) -> float:
    sample = np.asarray([math.log(float(value)) for value in close], dtype=np.float64)
    x = np.arange(sample.size, dtype=np.float64)
    x_mean = _mean(x)
    y_mean = _mean(sample)
    x_centered = np.asarray([float(value) - x_mean for value in x], dtype=np.float64)
    y_centered = np.asarray(
        [float(value) - y_mean for value in sample], dtype=np.float64
    )
    denominator = _dot(x_centered, x_centered)
    slope = _dot(x_centered, y_centered) / denominator
    fitted = np.asarray(
        [y_mean + slope * float(value) for value in x_centered], dtype=np.float64
    )
    total = _dot(y_centered, y_centered)
    residual = math.fsum(
        (float(value) - float(predicted)) ** 2
        for value, predicted in zip(sample, fitted, strict=True)
    )
    return float(np.clip(1.0 - residual / total, 0.0, 1.0))


def _correlation_oracle(close: np.ndarray, volume: np.ndarray) -> float:
    price_log = np.asarray(
        [math.log(float(value)) for value in close], dtype=np.float64
    )
    volume_log = np.asarray(
        [math.log(max(float(value), _EPSILON)) for value in volume],
        dtype=np.float64,
    )
    price_returns = np.diff(price_log)
    volume_changes = np.diff(volume_log)
    price_mean = _mean(price_returns)
    volume_mean = _mean(volume_changes)
    left = np.asarray(
        [float(value) - price_mean for value in price_returns], dtype=np.float64
    )
    right = np.asarray(
        [float(value) - volume_mean for value in volume_changes], dtype=np.float64
    )
    denominator = math.sqrt(_dot(left, left) * _dot(right, right))
    return float(np.clip(_dot(left, right) / denominator, -1.0, 1.0))


def _realized_volatility_oracle(close: np.ndarray) -> float:
    logged = np.asarray([math.log(float(value)) for value in close], dtype=np.float64)
    returns = np.diff(logged)
    return math.sqrt(
        math.fsum(float(value) ** 2 for value in returns) / float(len(returns))
    )


def _feature_value(
    *,
    kind: FeatureKind,
    close: np.ndarray,
    lookback: int,
    volume: np.ndarray | None = None,
) -> float:
    n = len(close)
    values, valid, _ = calculate_feature_events(
        FeatureSpec(name="probe", kind=kind, lookback=lookback),
        open_price=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close.copy(),
        volume=(np.ones(n, dtype=np.float64) if volume is None else volume.copy()),
        funding_rate=np.zeros(n, dtype=np.float64),
        funding_available=np.zeros(n, dtype=np.bool_),
        row_present=np.ones(n, dtype=np.bool_),
        active=np.ones(n, dtype=np.bool_),
    )
    assert valid[-1]
    return float(values[-1])


def test_portable_reductions_have_fixed_sequence_contract() -> None:
    values = np.asarray([1e16, 1.0, -1e16, 3.0], dtype=np.float64)
    assert portable_sum(values) == 4.0
    assert portable_mean(values) == 1.0
    assert portable_dot(values, np.ones(4, dtype=np.float64)) == 4.0
    assert portable_variance(np.asarray([1.0, 2.0, 3.0])) == pytest.approx(
        2.0 / 3.0
    )
    assert portable_std(np.asarray([1.0, 2.0, 3.0])) == pytest.approx(
        math.sqrt(2.0 / 3.0)
    )
    assert portable_covariance(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, 4.0, 6.0])
    ) == pytest.approx(4.0 / 3.0)
    assert portable_correlation(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, 4.0, 6.0])
    ) == 1.0


def test_portable_log_is_positive_finite_and_ordered() -> None:
    values = np.asarray([[1.0, math.e], [math.e**2, math.e**3]], dtype=np.float64)
    expected = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    np.testing.assert_array_equal(portable_log(values), expected)
    with pytest.raises(ValueError, match="positive"):
        portable_log(np.asarray([1.0, 0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        portable_log(np.asarray([1.0, math.inf], dtype=np.float64))


def test_market_build_v3_binds_portable_numerics_schema() -> None:
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
    )
    payload = config.canonical_payload()
    assert payload["schema_version"] == "market_build_v3"
    assert payload["feature_numerics_schema"] == PORTABLE_FEATURE_NUMERICS_SCHEMA

    with pytest.raises(ValueError, match="unsupported market build schema"):
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
            schema_version="market_build_v2",
        )


@pytest.mark.parametrize(
    ("kind", "close", "lookback", "volume", "expected"),
    (
        (
            FeatureKind.TREND_R2,
            _TREND_CLOSE,
            24,
            None,
            _trend_r2_oracle(_TREND_CLOSE),
        ),
        (
            FeatureKind.PRICE_VOLUME_CORRELATION,
            _CORR_CLOSE,
            24,
            _CORR_VOLUME,
            _correlation_oracle(_CORR_CLOSE, _CORR_VOLUME),
        ),
        (
            FeatureKind.REALIZED_VOLATILITY,
            _REALIZED_VOL_CLOSE,
            4,
            None,
            _realized_volatility_oracle(_REALIZED_VOL_CLOSE),
        ),
    ),
)
def test_real_issue494_windows_match_independent_portable_oracle(
    kind: FeatureKind,
    close: np.ndarray,
    lookback: int,
    volume: np.ndarray | None,
    expected: float,
) -> None:
    actual = _feature_value(kind=kind, close=close, lookback=lookback, volume=volume)
    assert np.float32(actual).view(np.uint32) == np.float32(expected).view(np.uint32)


def test_real_issue494_oracle_bits_are_frozen() -> None:
    assert np.float32(_trend_r2_oracle(_TREND_CLOSE)).view(np.uint32) == 0x2F900424
    assert (
        np.float32(_correlation_oracle(_CORR_CLOSE, _CORR_VOLUME)).view(np.uint32)
        == 0xB6782BC0
    )
    assert (
        np.float32(_realized_volatility_oracle(_REALIZED_VOL_CLOSE)).view(np.uint32)
        == 0x3AA3DFFA
    )


def test_identity_bound_dataset_code_has_no_dispatched_numpy_numerics() -> None:
    forbidden_numpy = {
        "corrcoef",
        "cov",
        "dot",
        "log",
        "mean",
        "std",
        "sum",
        "var",
    }
    forbidden_method = {"dot", "mean", "std", "sum", "var"}
    files = (
        Path("trade_rl/data/features/core.py"),
        Path("trade_rl/data/features/cross_asset.py"),
        Path("trade_rl/data/build/builder.py"),
    )
    violations: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "np"
                and function.attr in forbidden_numpy
            ):
                violations.append(f"{path}:{node.lineno}:np.{function.attr}")
            elif (
                isinstance(function, ast.Attribute)
                and function.attr in forbidden_method
                and not (
                    isinstance(function.value, ast.Name)
                    and function.value.id.startswith("portable_")
                )
            ):
                violations.append(f"{path}:{node.lineno}:.{function.attr}")
            elif isinstance(function, ast.Name) and function.id == "sum":
                violations.append(f"{path}:{node.lineno}:sum")
    assert violations == []
