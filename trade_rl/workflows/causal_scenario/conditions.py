"""Causal train-only condition vectors for historical scenario matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256, require_unique_non_empty
from trade_rl.strategies.trend import TrendStrategy

_CONDITION_SCHEMA: Final = "causal_scenario_condition_v1"
_LAYOUT_SCHEMA: Final = "causal_scenario_condition_layout_v1"
_NORMALIZER_SCHEMA: Final = "train_robust_condition_normalizer_v1"
_BINARY_GROUPS: Final = (
    "funding_due",
    "tradable",
    "buy_allowed",
    "sell_allowed",
    "borrow_available",
    "asset_active",
)


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _readonly_float(name: str, value: object, *, ndim: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).copy(order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _readonly_bool(name: str, value: object, *, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean rank-{ndim} array")
    result = np.asarray(raw, dtype=np.bool_).copy(order="C")
    result.setflags(write=False)
    return result


def _array_payload(value: np.ndarray) -> dict[str, object]:
    return {
        "dtype": value.dtype.str,
        "shape": tuple(int(size) for size in value.shape),
        "values": value.tolist(),
    }


@dataclass(frozen=True, slots=True)
class CausalConditionConfig:
    """Versioned horizons and numerical floors for one condition vector."""

    volatility_hours: float = 24.0
    correlation_hours: float = 168.0
    scale_epsilon: float = 1e-9
    liquidity_floor: float = 1e-12
    schema_version: str = _CONDITION_SCHEMA

    def __post_init__(self) -> None:
        volatility = _finite_positive("volatility_hours", self.volatility_hours)
        correlation = _finite_positive("correlation_hours", self.correlation_hours)
        if correlation < volatility:
            raise ValueError(
                "correlation_hours must not be shorter than volatility_hours"
            )
        epsilon = _finite_positive("scale_epsilon", self.scale_epsilon)
        floor = _finite_positive("liquidity_floor", self.liquidity_floor)
        if self.schema_version != _CONDITION_SCHEMA:
            raise ValueError("unsupported causal condition schema")
        object.__setattr__(self, "volatility_hours", volatility)
        object.__setattr__(self, "correlation_hours", correlation)
        object.__setattr__(self, "scale_epsilon", epsilon)
        object.__setattr__(self, "liquidity_floor", floor)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "correlation_hours": self.correlation_hours,
                "liquidity_floor": self.liquidity_floor,
                "scale_epsilon": self.scale_epsilon,
                "schema_version": self.schema_version,
                "volatility_hours": self.volatility_hours,
            }
        )


@dataclass(frozen=True, slots=True)
class CausalConditionLayout:
    """Exact feature order and continuous/binary partition."""

    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    continuous_mask: np.ndarray
    schema_version: str = _LAYOUT_SCHEMA

    def __post_init__(self) -> None:
        symbols = require_unique_non_empty(tuple(self.symbols), field="symbols")
        names = require_unique_non_empty(
            tuple(self.feature_names), field="feature_names"
        )
        mask = _readonly_bool("continuous_mask", self.continuous_mask, ndim=1)
        if mask.shape != (len(names),):
            raise ValueError("continuous_mask shape must match feature_names")
        if self.schema_version != _LAYOUT_SCHEMA:
            raise ValueError("unsupported causal condition layout schema")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "continuous_mask", mask)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "continuous_mask": _array_payload(self.continuous_mask),
                "feature_names": self.feature_names,
                "schema_version": self.schema_version,
                "symbols": self.symbols,
            }
        )


@dataclass(frozen=True, slots=True)
class TrainRobustConditionNormalizer:
    """Median/MAD transform fitted from eligible train anchors only."""

    feature_names: tuple[str, ...]
    continuous_mask: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    train_view_digest: str
    schema_version: str = _NORMALIZER_SCHEMA

    def __post_init__(self) -> None:
        names = require_unique_non_empty(
            tuple(self.feature_names), field="feature_names"
        )
        mask = _readonly_bool("continuous_mask", self.continuous_mask, ndim=1)
        median = _readonly_float("median", self.median, ndim=1)
        scale = _readonly_float("scale", self.scale, ndim=1)
        expected = (len(names),)
        if (
            mask.shape != expected
            or median.shape != expected
            or scale.shape != expected
        ):
            raise ValueError("normalizer arrays must match feature_names")
        if np.any(scale <= 0.0):
            raise ValueError("normalizer scale must be positive")
        if np.any(median[~mask] != 0.0) or np.any(scale[~mask] != 1.0):
            raise ValueError(
                "binary normalizer dimensions must store median 0 and scale 1"
            )
        digest = require_sha256(self.train_view_digest, field="train_view_digest")
        if self.schema_version != _NORMALIZER_SCHEMA:
            raise ValueError("unsupported train condition normalizer schema")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "continuous_mask", mask)
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "train_view_digest", digest)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "continuous_mask": _array_payload(self.continuous_mask),
                "feature_names": self.feature_names,
                "median": _array_payload(self.median),
                "scale": _array_payload(self.scale),
                "schema_version": self.schema_version,
                "train_view_digest": self.train_view_digest,
            }
        )

    def transform(self, raw: np.ndarray) -> np.ndarray:
        vector = _readonly_float("raw", raw, ndim=1)
        if vector.shape != self.median.shape:
            raise ValueError("raw condition shape does not match normalizer")
        binary = vector[~self.continuous_mask]
        if np.any((binary != 0.0) & (binary != 1.0)):
            raise ValueError("binary condition dimensions must contain only 0 or 1")
        result = vector.copy()
        continuous = self.continuous_mask
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result[continuous] = (
                vector[continuous] - self.median[continuous]
            ) / self.scale[continuous]
        result[~continuous] = binary
        result[result == 0.0] = 0.0
        if not np.isfinite(result).all():
            raise ValueError("normalized condition must be finite")
        result.setflags(write=False)
        return result


def build_causal_condition_layout(symbols: tuple[str, ...]) -> CausalConditionLayout:
    resolved_symbols = require_unique_non_empty(tuple(symbols), field="symbols")
    names: list[str] = []
    continuous: list[bool] = []
    for group in ("trend_fast", "trend_base", "trend_slow", "realized_vol_24h"):
        names.extend(f"{group}:{symbol}" for symbol in resolved_symbols)
        continuous.extend(True for _ in resolved_symbols)
    for left_index, left in enumerate(resolved_symbols):
        for right in resolved_symbols[left_index + 1 :]:
            names.append(f"corr_7d:{left}|{right}")
            continuous.append(True)
    for group in (
        "spread_rate",
        "log_market_notional",
        "funding_rate",
        *_BINARY_GROUPS,
    ):
        names.extend(f"{group}:{symbol}" for symbol in resolved_symbols)
        continuous.extend(group not in _BINARY_GROUPS for _ in resolved_symbols)
    return CausalConditionLayout(
        symbols=resolved_symbols,
        feature_names=tuple(names),
        continuous_mask=np.asarray(continuous, dtype=np.bool_),
    )


def _log_returns(dataset: MarketDataset, index: int, hours: float) -> np.ndarray:
    previous = dataset.lookback_index(index, hours)
    prices = dataset.close[previous : index + 1]
    if prices.shape[0] < 2 or np.any(prices <= 0.0):
        raise ValueError("condition history contains invalid close prices")
    return np.diff(np.log(prices), axis=0)


def _condition_history_starts(
    dataset: MarketDataset,
    index: int,
    trend_strategy: TrendStrategy,
    config: CausalConditionConfig,
) -> tuple[int, ...]:
    trend = trend_strategy.config
    if trend.fast_lookback is not None:
        assert trend.base_lookback is not None
        assert trend.slow_lookback is not None
        trend_starts = (
            index - trend.fast_lookback,
            index - trend.base_lookback,
            index - trend.slow_lookback,
        )
    else:
        trend_starts = (
            dataset.lookback_index(index, trend.fast_hours),
            dataset.lookback_index(index, trend.base_hours),
            dataset.lookback_index(index, trend.slow_hours),
        )
    return (
        *trend_starts,
        dataset.lookback_index(index, config.volatility_hours),
        dataset.lookback_index(index, config.correlation_hours),
    )


def compute_raw_causal_condition(
    dataset: MarketDataset,
    index: int,
    trend_strategy: TrendStrategy,
    config: CausalConditionConfig | None = None,
    *,
    history_start: int = 0,
) -> np.ndarray:
    """Compute one condition without reading before ``history_start`` or after ``index``."""

    resolved = config or CausalConditionConfig()
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < dataset.n_bars
    ):
        raise ValueError("condition index is outside the dataset")
    if (
        isinstance(history_start, bool)
        or not isinstance(history_start, int)
        or not 0 <= history_start <= index
    ):
        raise ValueError("history_start must be inside the causal prefix")
    minimum = max(
        trend_strategy.minimum_history_for(dataset),
        dataset.minimum_index_for_history(resolved.correlation_hours),
    )
    if index < minimum:
        raise ValueError("insufficient causal history for condition vector")
    history_starts = _condition_history_starts(dataset, index, trend_strategy, resolved)
    if min(history_starts) < history_start:
        raise ValueError("causal condition history escapes assigned range")
    targets = trend_strategy.targets(dataset, index)
    vol_returns = _log_returns(dataset, index, resolved.volatility_hours)
    corr_returns = _log_returns(dataset, index, resolved.correlation_hours)
    values: list[float] = []
    values.extend(float(value) for value in targets.fast)
    values.extend(float(value) for value in targets.base)
    values.extend(float(value) for value in targets.slow)
    values.extend(float(value) for value in np.sqrt(np.sum(vol_returns**2, axis=0)))
    for left in range(dataset.n_symbols):
        for right in range(left + 1, dataset.n_symbols):
            left_values = corr_returns[:, left]
            right_values = corr_returns[:, right]
            if (
                float(np.std(left_values)) <= resolved.scale_epsilon
                or float(np.std(right_values)) <= resolved.scale_epsilon
            ):
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(left_values, right_values)[0, 1])
            values.append(correlation)
    values.extend(
        float(value) for value in dataset.resolved_array("spread_rate")[index]
    )
    notional = np.maximum(
        dataset.market_notional(index, prices=dataset.close[index]),
        resolved.liquidity_floor,
    )
    values.extend(float(value) for value in np.log(notional))
    values.extend(float(value) for value in dataset.funding_rate[index])
    for field_name in _BINARY_GROUPS:
        values.extend(
            float(value) for value in dataset.resolved_array(field_name)[index]
        )
    result = np.asarray(values, dtype=np.float64)
    layout = build_causal_condition_layout(dataset.symbols)
    if result.shape != (len(layout.feature_names),) or not np.isfinite(result).all():
        raise ValueError("causal condition vector is invalid")
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def fit_train_condition_normalizer(
    raw_anchor_conditions: np.ndarray,
    layout: CausalConditionLayout,
    train_view_digest: str,
    *,
    scale_epsilon: float = 1e-9,
) -> TrainRobustConditionNormalizer:
    matrix = _readonly_float("raw_anchor_conditions", raw_anchor_conditions, ndim=2)
    if matrix.shape[0] == 0 or matrix.shape[1] != len(layout.feature_names):
        raise ValueError("raw anchor condition matrix has an invalid shape")
    epsilon = _finite_positive("scale_epsilon", scale_epsilon)
    binary = matrix[:, ~layout.continuous_mask]
    if np.any((binary != 0.0) & (binary != 1.0)):
        raise ValueError("binary anchor conditions must contain only 0 or 1")
    median = np.zeros(matrix.shape[1], dtype=np.float64)
    scale = np.ones(matrix.shape[1], dtype=np.float64)
    continuous = layout.continuous_mask
    continuous_values = matrix[:, continuous]
    continuous_median = np.median(continuous_values, axis=0)
    continuous_scale = np.median(np.abs(continuous_values - continuous_median), axis=0)
    median[continuous] = continuous_median
    scale[continuous] = np.maximum(continuous_scale, epsilon)
    return TrainRobustConditionNormalizer(
        feature_names=layout.feature_names,
        continuous_mask=layout.continuous_mask,
        median=median,
        scale=scale,
        train_view_digest=train_view_digest,
    )


__all__ = [
    "CausalConditionConfig",
    "CausalConditionLayout",
    "TrainRobustConditionNormalizer",
    "build_causal_condition_layout",
    "compute_raw_causal_condition",
    "fit_train_condition_normalizer",
]
