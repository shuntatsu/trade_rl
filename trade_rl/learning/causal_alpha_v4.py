"""Immutable label contracts for the research-only Causal Alpha V4 lane."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.data.v4_context import V4ContextBlock
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA: Final = "causal_alpha_v4_symbol_samples_v1"
CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA: Final = "causal_alpha_v4_residual_labels_v1"
CAUSAL_ALPHA_V4_FIT_CONFIG_SCHEMA: Final = "causal_alpha_v4_fit_config_v1"
CAUSAL_ALPHA_V4_FORECAST_SCHEMA: Final = "causal_alpha_v4_forecast_v1"
CAUSAL_ALPHA_V4_HORIZONS: Final = ("4h", "24h", "72h")
CAUSAL_ALPHA_V4_UNCERTAINTY_SCHEMA: Final = "causal_alpha_v4_uncertainty_v1"
_V4_MINIMUM_STATE_ESS: Final = 30.0
CAUSAL_ALPHA_V4_TARGET_SCHEMA: Final = "causal_alpha_v4_target_v1"
_V4_TARGET_EPSILON: Final = 1e-12


def _readonly(value: object, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    array.setflags(write=False)
    return array


def _validate_labels(
    *,
    labels: np.ndarray,
    ends: np.ndarray,
    rows: int,
    horizon: str,
) -> None:
    if labels.shape != (rows,) or ends.shape != (rows,):
        raise ValueError(f"V4 {horizon} label arrays must be sample aligned")
    valid = ends >= 0
    if np.any(valid & ~np.isfinite(labels)):
        raise ValueError(f"V4 {horizon} realized labels must be finite")
    if np.any(~valid & np.isfinite(labels)):
        raise ValueError(f"V4 {horizon} unavailable labels require non-finite storage")
    if np.any(~valid & (ends != -1)):
        raise ValueError(f"V4 {horizon} unavailable label ends must be -1")


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SymbolSamples:
    """One train-symbol V4 feature/label table with persisted causal beta."""

    symbol: str
    dataset_id: str
    target_local_feature_names: tuple[str, ...]
    target_local_feature_schema_digest: str
    source_sample_digest: str
    source_context_digest: str
    decision_indices: np.ndarray
    target_local_features: np.ndarray
    target_local_available: np.ndarray
    instrument_descriptor_names: tuple[str, ...]
    instrument_descriptors: np.ndarray
    instrument_descriptor_available: np.ndarray
    local_context: V4ContextBlock
    global_context: V4ContextBlock
    beta: np.ndarray
    beta_available: np.ndarray
    labels_4h: np.ndarray
    label_end_indices_4h: np.ndarray
    labels_24h: np.ndarray
    label_end_indices_24h: np.ndarray
    labels_72h: np.ndarray
    label_end_indices_72h: np.ndarray
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V4 sample symbol must be non-empty")
        for field in (
            "dataset_id",
            "target_local_feature_schema_digest",
            "source_sample_digest",
            "source_context_digest",
        ):
            require_sha256(getattr(self, field), field=f"V4 sample {field}")
        names = tuple(self.target_local_feature_names)
        if (
            not names
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "V4 target-local feature names must be non-empty and unique"
            )
        descriptor_names = tuple(self.instrument_descriptor_names)
        if descriptor_names != UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES:
            raise ValueError(
                "V4 instrument descriptor names must match the maintained order"
            )
        if set(names).intersection(descriptor_names):
            raise ValueError("V4 target-local features must not duplicate descriptors")

        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        rows = int(decisions.size)
        if rows == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError("V4 sample decision indices must be strictly increasing")
        features = _readonly(self.target_local_features, dtype=np.float64)
        available = _readonly(self.target_local_available, dtype=np.bool_)
        if features.shape != (rows, len(names)) or available.shape != features.shape:
            raise ValueError("V4 target-local arrays do not match feature schema")
        if not np.isfinite(features).all():
            raise ValueError("V4 target-local features must be finite")
        descriptors = _readonly(self.instrument_descriptors, dtype=np.float64)
        descriptor_available = _readonly(
            self.instrument_descriptor_available, dtype=np.bool_
        )
        descriptor_shape = (rows, len(descriptor_names))
        if (
            descriptors.shape != descriptor_shape
            or descriptor_available.shape != descriptor_shape
        ):
            raise ValueError("V4 instrument descriptor arrays do not match schema")
        if not np.isfinite(descriptors).all():
            raise ValueError("V4 instrument descriptors must be finite")

        if not isinstance(self.local_context, V4ContextBlock) or not isinstance(
            self.global_context, V4ContextBlock
        ):
            raise TypeError("V4 samples require local/global context blocks")
        if not np.array_equal(self.local_context.decision_indices, decisions):
            raise ValueError("V4 local context decisions do not match samples")
        if not np.array_equal(self.global_context.decision_indices, decisions):
            raise ValueError("V4 global context decisions do not match samples")

        beta = _readonly(self.beta, dtype=np.float64).reshape(-1)
        beta_available = _readonly(self.beta_available, dtype=np.bool_).reshape(-1)
        if beta.shape != (rows,) or beta_available.shape != (rows,):
            raise ValueError("V4 persisted beta arrays must be sample aligned")
        if not np.isfinite(beta).all():
            raise ValueError("V4 persisted beta must be finite")
        if np.any(beta[beta_available] < -3.0) or np.any(beta[beta_available] > 3.0):
            raise ValueError("V4 persisted beta exceeds authored bounds")
        if self.symbol == "BTCUSDT" and np.any(beta[beta_available] != 1.0):
            raise ValueError("BTCUSDT available persisted beta must be exactly one")

        labels: dict[str, np.ndarray] = {}
        ends: dict[str, np.ndarray] = {}
        for horizon in ("4h", "24h", "72h"):
            label_values = _readonly(
                getattr(self, f"labels_{horizon}"), dtype=np.float64
            ).reshape(-1)
            label_ends = _readonly(
                getattr(self, f"label_end_indices_{horizon}"), dtype=np.int64
            ).reshape(-1)
            _validate_labels(
                labels=label_values,
                ends=label_ends,
                rows=rows,
                horizon=horizon,
            )
            labels[horizon] = label_values
            ends[horizon] = label_ends

        expected = content_and_arrays_digest(
            {
                "dataset_id": self.dataset_id,
                "global_context_digest": self.global_context.digest,
                "instrument_descriptor_names": descriptor_names,
                "local_context_digest": self.local_context.digest,
                "schema_version": CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA,
                "source_context_digest": self.source_context_digest,
                "source_sample_digest": self.source_sample_digest,
                "symbol": self.symbol,
                "target_local_feature_names": names,
                "target_local_feature_schema_digest": (
                    self.target_local_feature_schema_digest
                ),
            },
            (
                ("decision_indices", decisions),
                ("target_local_features", features),
                ("target_local_available", available),
                ("instrument_descriptors", descriptors),
                ("instrument_descriptor_available", descriptor_available),
                ("beta", beta),
                ("beta_available", beta_available),
                ("labels_4h", labels["4h"]),
                ("label_end_indices_4h", ends["4h"]),
                ("labels_24h", labels["24h"]),
                ("label_end_indices_24h", ends["24h"]),
                ("labels_72h", labels["72h"]),
                ("label_end_indices_72h", ends["72h"]),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 symbol sample digest mismatch")
        object.__setattr__(self, "target_local_feature_names", names)
        object.__setattr__(self, "instrument_descriptor_names", descriptor_names)
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "target_local_features", features)
        object.__setattr__(self, "target_local_available", available)
        object.__setattr__(self, "instrument_descriptors", descriptors)
        object.__setattr__(
            self, "instrument_descriptor_available", descriptor_available
        )
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "beta_available", beta_available)
        for horizon in ("4h", "24h", "72h"):
            object.__setattr__(self, f"labels_{horizon}", labels[horizon])
            object.__setattr__(
                self,
                f"label_end_indices_{horizon}",
                ends[horizon],
            )
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaV4ResidualLabels:
    """BTC market-proxy and beta-residual labels for all authored horizons."""

    symbol: str
    decision_indices: np.ndarray
    symbol_sample_digest: str
    market_proxy_sample_digest: str
    market_proxy_labels_4h: np.ndarray
    residual_labels_4h: np.ndarray
    available_4h: np.ndarray
    market_proxy_labels_24h: np.ndarray
    residual_labels_24h: np.ndarray
    available_24h: np.ndarray
    market_proxy_labels_72h: np.ndarray
    residual_labels_72h: np.ndarray
    available_72h: np.ndarray
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("V4 residual symbol must be non-empty")
        require_sha256(
            self.symbol_sample_digest,
            field="V4 residual symbol_sample_digest",
        )
        require_sha256(
            self.market_proxy_sample_digest,
            field="V4 residual market_proxy_sample_digest",
        )
        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        rows = int(decisions.size)
        if rows == 0 or np.any(np.diff(decisions) <= 0):
            raise ValueError("V4 residual decisions must be strictly increasing")
        arrays: dict[str, np.ndarray] = {}
        for horizon in ("4h", "24h", "72h"):
            proxy = _readonly(
                getattr(self, f"market_proxy_labels_{horizon}"), dtype=np.float64
            ).reshape(-1)
            residual = _readonly(
                getattr(self, f"residual_labels_{horizon}"), dtype=np.float64
            ).reshape(-1)
            available = _readonly(
                getattr(self, f"available_{horizon}"), dtype=np.bool_
            ).reshape(-1)
            if (
                proxy.shape != (rows,)
                or residual.shape != (rows,)
                or available.shape != (rows,)
            ):
                raise ValueError(f"V4 residual {horizon} arrays are not aligned")
            if np.any(available & (~np.isfinite(proxy) | ~np.isfinite(residual))):
                raise ValueError(f"V4 residual {horizon} available rows must be finite")
            if np.any(~available & np.isfinite(residual)):
                raise ValueError(
                    f"V4 residual {horizon} unavailable rows require NaN residuals"
                )
            arrays[f"market_proxy_labels_{horizon}"] = proxy
            arrays[f"residual_labels_{horizon}"] = residual
            arrays[f"available_{horizon}"] = available
        expected = content_and_arrays_digest(
            {
                "market_proxy_sample_digest": self.market_proxy_sample_digest,
                "schema_version": CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA,
                "symbol": self.symbol,
                "symbol_sample_digest": self.symbol_sample_digest,
            },
            (
                ("decision_indices", decisions),
                *tuple((name, value) for name, value in arrays.items()),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 residual label digest mismatch")
        object.__setattr__(self, "decision_indices", decisions)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "digest", expected)


def _residual_for_horizon(
    *,
    symbol_labels: np.ndarray,
    market_labels: np.ndarray,
    beta: np.ndarray,
    beta_available: np.ndarray,
    symbol_ends: np.ndarray,
    market_ends: np.ndarray,
    horizon: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.array_equal(symbol_ends, market_ends):
        raise ValueError(f"V4 {horizon} symbol/BTC label ends are not aligned")
    available = (
        beta_available
        & (symbol_ends >= 0)
        & np.isfinite(symbol_labels)
        & np.isfinite(market_labels)
    )
    residual = np.full(symbol_labels.shape, np.nan, dtype=np.float64)
    residual[available] = (
        symbol_labels[available] - beta[available] * market_labels[available]
    )
    reconstructed = beta[available] * market_labels[available] + residual[available]
    if reconstructed.size and np.any(
        np.abs(reconstructed - symbol_labels[available]) > 1e-15
    ):
        raise RuntimeError(f"V4 {horizon} residual reconstruction drifted")
    return market_labels.copy(), residual, available


def build_causal_alpha_v4_residual_labels(
    *,
    symbol_samples: CausalAlphaV4SymbolSamples,
    btc_market_proxy_samples: CausalAlphaV4SymbolSamples,
) -> CausalAlphaV4ResidualLabels:
    """Decompose target returns with the persisted target beta and BTC labels."""

    if not isinstance(symbol_samples, CausalAlphaV4SymbolSamples) or not isinstance(
        btc_market_proxy_samples, CausalAlphaV4SymbolSamples
    ):
        raise TypeError("V4 residual decomposition requires V4 symbol samples")
    if btc_market_proxy_samples.symbol != "BTCUSDT":
        raise ValueError("V4 market proxy samples must be BTCUSDT")
    if not np.array_equal(
        symbol_samples.decision_indices,
        btc_market_proxy_samples.decision_indices,
    ):
        raise ValueError("V4 symbol/BTC sample decisions are not aligned")
    values: dict[str, np.ndarray] = {}
    for horizon in ("4h", "24h", "72h"):
        proxy, residual, available = _residual_for_horizon(
            symbol_labels=getattr(symbol_samples, f"labels_{horizon}"),
            market_labels=getattr(btc_market_proxy_samples, f"labels_{horizon}"),
            beta=symbol_samples.beta,
            beta_available=symbol_samples.beta_available,
            symbol_ends=getattr(symbol_samples, f"label_end_indices_{horizon}"),
            market_ends=getattr(
                btc_market_proxy_samples,
                f"label_end_indices_{horizon}",
            ),
            horizon=horizon,
        )
        values[f"market_proxy_labels_{horizon}"] = proxy
        values[f"residual_labels_{horizon}"] = residual
        values[f"available_{horizon}"] = available
    return CausalAlphaV4ResidualLabels(
        symbol=symbol_samples.symbol,
        decision_indices=symbol_samples.decision_indices,
        symbol_sample_digest=symbol_samples.digest,
        market_proxy_sample_digest=btc_market_proxy_samples.digest,
        market_proxy_labels_4h=values["market_proxy_labels_4h"],
        residual_labels_4h=values["residual_labels_4h"],
        available_4h=values["available_4h"],
        market_proxy_labels_24h=values["market_proxy_labels_24h"],
        residual_labels_24h=values["residual_labels_24h"],
        available_24h=values["available_24h"],
        market_proxy_labels_72h=values["market_proxy_labels_72h"],
        residual_labels_72h=values["residual_labels_72h"],
        available_72h=values["available_72h"],
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV4FitConfig:
    """The single predeclared hierarchical ridge hypothesis for V4."""

    market_ridge_strength: float = 1.0
    residual_ridge_strength: float = 0.1
    direction_ridge_strength: float = 0.1
    schema_version: str = CAUSAL_ALPHA_V4_FIT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_ALPHA_V4_FIT_CONFIG_SCHEMA:
            raise ValueError("unsupported V4 fit config schema")
        if self.market_ridge_strength != 1.0:
            raise ValueError("V4 market ridge strength must remain 1.0")
        if self.residual_ridge_strength != 0.1:
            raise ValueError("V4 residual ridge strength must remain 0.1")
        if self.direction_ridge_strength != 0.1:
            raise ValueError("V4 direction ridge strength must remain 0.1")

    @property
    def digest(self) -> str:
        return content_digest(self)


def _canonical_horizon_arrays(
    value: Mapping[str, object],
    *,
    field_name: str,
    rows: int,
) -> dict[str, np.ndarray]:
    if set(value) != set(CAUSAL_ALPHA_V4_HORIZONS):
        raise ValueError(f"V4 forecast {field_name} horizon set is invalid")
    resolved: dict[str, np.ndarray] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        array = np.asarray(value[horizon], dtype=np.float64).reshape(-1).copy(order="C")
        if array.shape != (rows,) or not np.isfinite(array).all():
            raise ValueError(
                f"V4 forecast {field_name}[{horizon}] must be aligned and finite"
            )
        array.setflags(write=False)
        resolved[horizon] = array
    return resolved


def _canonical_horizon_digests(
    value: Mapping[str, str],
    *,
    field_name: str,
) -> dict[str, str]:
    if set(value) != set(CAUSAL_ALPHA_V4_HORIZONS):
        raise ValueError(f"V4 forecast {field_name} horizon set is invalid")
    resolved: dict[str, str] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        digest = str(value[horizon])
        require_sha256(digest, field=f"V4 forecast {field_name}[{horizon}]")
        resolved[horizon] = digest
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV4Forecast:
    """One symbol's immutable three-horizon hierarchical forecast."""

    symbol: str
    decision_indices: np.ndarray
    beta: np.ndarray
    beta_available: np.ndarray
    market_predictions: Mapping[str, np.ndarray]
    residual_predictions: Mapping[str, np.ndarray]
    direction_scores: Mapping[str, np.ndarray]
    market_model_digests: Mapping[str, str]
    residual_model_digests: Mapping[str, str]
    direction_model_digests: Mapping[str, str]
    fit_digest: str
    beta_scaled_market_contributions: Mapping[str, np.ndarray] = dataclass_field(
        init=False, default_factory=dict
    )
    final_predictions: Mapping[str, np.ndarray] = dataclass_field(
        init=False, default_factory=dict
    )
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V4 forecast symbol must be non-empty")
        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        rows = int(decisions.size)
        if rows == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError("V4 forecast decisions must be strictly increasing")
        beta = _readonly(self.beta, dtype=np.float64).reshape(-1)
        beta_available = _readonly(self.beta_available, dtype=np.bool_).reshape(-1)
        if beta.shape != (rows,) or beta_available.shape != (rows,):
            raise ValueError("V4 forecast beta arrays must be aligned")
        if not np.isfinite(beta).all():
            raise ValueError("V4 forecast beta must be finite")
        if np.any(beta[beta_available] < -3.0) or np.any(beta[beta_available] > 3.0):
            raise ValueError("V4 forecast beta exceeds authored bounds")

        market = _canonical_horizon_arrays(
            self.market_predictions,
            field_name="market_predictions",
            rows=rows,
        )
        residual = _canonical_horizon_arrays(
            self.residual_predictions,
            field_name="residual_predictions",
            rows=rows,
        )
        direction = _canonical_horizon_arrays(
            self.direction_scores,
            field_name="direction_scores",
            rows=rows,
        )
        market_digests = _canonical_horizon_digests(
            self.market_model_digests,
            field_name="market_model_digests",
        )
        residual_digests = _canonical_horizon_digests(
            self.residual_model_digests,
            field_name="residual_model_digests",
        )
        direction_digests = _canonical_horizon_digests(
            self.direction_model_digests,
            field_name="direction_model_digests",
        )
        require_sha256(self.fit_digest, field="V4 forecast fit_digest")

        beta_scaled: dict[str, np.ndarray] = {}
        final: dict[str, np.ndarray] = {}
        for horizon in CAUSAL_ALPHA_V4_HORIZONS:
            contribution = np.asarray(beta * market[horizon], dtype=np.float64)
            composed = np.asarray(contribution + residual[horizon], dtype=np.float64)
            contribution.setflags(write=False)
            composed.setflags(write=False)
            beta_scaled[horizon] = contribution
            final[horizon] = composed

        expected = content_and_arrays_digest(
            {
                "direction_model_digests": tuple(direction_digests.items()),
                "fit_digest": self.fit_digest,
                "market_model_digests": tuple(market_digests.items()),
                "residual_model_digests": tuple(residual_digests.items()),
                "schema_version": CAUSAL_ALPHA_V4_FORECAST_SCHEMA,
                "symbol": self.symbol,
            },
            (
                ("decision_indices", decisions),
                ("beta", beta),
                ("beta_available", beta_available),
                *tuple(
                    (f"market_prediction:{horizon}", market[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
                *tuple(
                    (f"residual_prediction:{horizon}", residual[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
                *tuple(
                    (f"direction_score:{horizon}", direction[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
                *tuple(
                    (f"beta_market:{horizon}", beta_scaled[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
                *tuple(
                    (f"final_prediction:{horizon}", final[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 forecast digest mismatch")
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "beta_available", beta_available)
        object.__setattr__(self, "market_predictions", MappingProxyType(market))
        object.__setattr__(self, "residual_predictions", MappingProxyType(residual))
        object.__setattr__(self, "direction_scores", MappingProxyType(direction))
        object.__setattr__(
            self, "market_model_digests", MappingProxyType(market_digests)
        )
        object.__setattr__(
            self, "residual_model_digests", MappingProxyType(residual_digests)
        )
        object.__setattr__(
            self, "direction_model_digests", MappingProxyType(direction_digests)
        )
        object.__setattr__(
            self,
            "beta_scaled_market_contributions",
            MappingProxyType(beta_scaled),
        )
        object.__setattr__(self, "final_predictions", MappingProxyType(final))
        object.__setattr__(self, "digest", expected)


def build_causal_alpha_v4_forecast(
    *,
    symbol: str,
    decision_indices: object,
    beta: object,
    beta_available: object,
    market_predictions: Mapping[str, object],
    residual_predictions: Mapping[str, object],
    direction_scores: Mapping[str, object],
    market_model_digests: Mapping[str, str],
    residual_model_digests: Mapping[str, str],
    direction_model_digests: Mapping[str, str],
    fit_digest: str,
) -> CausalAlphaV4Forecast:
    """Compose persisted-beta market and shared-residual V4 forecasts."""

    return CausalAlphaV4Forecast(
        symbol=symbol,
        decision_indices=np.asarray(decision_indices, dtype=np.int64),
        beta=np.asarray(beta, dtype=np.float64),
        beta_available=np.asarray(beta_available, dtype=np.bool_),
        market_predictions={
            horizon: np.asarray(value, dtype=np.float64)
            for horizon, value in market_predictions.items()
        },
        residual_predictions={
            horizon: np.asarray(value, dtype=np.float64)
            for horizon, value in residual_predictions.items()
        },
        direction_scores={
            horizon: np.asarray(value, dtype=np.float64)
            for horizon, value in direction_scores.items()
        },
        market_model_digests=dict(market_model_digests),
        residual_model_digests=dict(residual_model_digests),
        direction_model_digests=dict(direction_model_digests),
        fit_digest=fit_digest,
    )


class V4ForecastState(str, Enum):
    NORMAL = "normal"
    HIGH_REALIZED_VOLATILITY = "high_realized_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    BASIS_POSITIONING_STRESS = "basis_positioning_stress"


_V4_FORECAST_STATES: Final = (
    V4ForecastState.NORMAL,
    V4ForecastState.HIGH_REALIZED_VOLATILITY,
    V4ForecastState.LOW_LIQUIDITY,
    V4ForecastState.BASIS_POSITIONING_STRESS,
)


@dataclass(frozen=True, slots=True)
class CausalAlphaV4UncertaintyCell:
    state: V4ForecastState
    support: int
    effective_sample_size: float
    global_rmse: float
    state_rmse: float | None
    selected_uncertainty: float
    fallback_reason: str | None

    def __post_init__(self) -> None:
        state = V4ForecastState(self.state)
        if (
            isinstance(self.support, bool)
            or not isinstance(self.support, int)
            or self.support < 0
        ):
            raise ValueError("V4 uncertainty support must be a non-negative integer")
        for field_name in (
            "effective_sample_size",
            "global_rmse",
            "selected_uncertainty",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V4 uncertainty {field_name} must be non-negative")
        if self.state_rmse is not None and (
            not math.isfinite(self.state_rmse) or self.state_rmse < 0.0
        ):
            raise ValueError("V4 uncertainty state_rmse must be null or non-negative")
        if self.effective_sample_size < _V4_MINIMUM_STATE_ESS:
            if (
                self.fallback_reason != "insufficient_state_ess"
                or self.selected_uncertainty != self.global_rmse
            ):
                raise ValueError("V4 low-ESS state must fall back to global RMSE")
        else:
            if (
                self.fallback_reason is not None
                or self.state_rmse is None
                or self.selected_uncertainty != self.state_rmse
            ):
                raise ValueError("V4 supported state must use its state RMSE")
        object.__setattr__(self, "state", state)

    def to_payload(self) -> dict[str, object]:
        return {
            "effective_sample_size": self.effective_sample_size,
            "fallback_reason": self.fallback_reason,
            "global_rmse": self.global_rmse,
            "selected_uncertainty": self.selected_uncertainty,
            "state": self.state.value,
            "state_rmse": self.state_rmse,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV4UncertaintyModel:
    high_realized_volatility_threshold: float
    low_liquidity_threshold: float
    basis_positioning_stress_threshold: float
    threshold_digest: str
    global_rmse: Mapping[str, float]
    cells: Mapping[str, Mapping[V4ForecastState, CausalAlphaV4UncertaintyCell]]
    minimum_state_effective_sample_size: float = _V4_MINIMUM_STATE_ESS
    schema_version: str = CAUSAL_ALPHA_V4_UNCERTAINTY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "high_realized_volatility_threshold",
            "low_liquidity_threshold",
            "basis_positioning_stress_threshold",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"V4 uncertainty {field_name} must be finite")
        if self.basis_positioning_stress_threshold < 0.0:
            raise ValueError("V4 stress threshold must be non-negative")
        require_sha256(self.threshold_digest, field="V4 uncertainty threshold_digest")
        if self.minimum_state_effective_sample_size != _V4_MINIMUM_STATE_ESS:
            raise ValueError("V4 minimum state ESS must remain 30.0")
        if self.schema_version != CAUSAL_ALPHA_V4_UNCERTAINTY_SCHEMA:
            raise ValueError("unsupported V4 uncertainty schema")

        global_rmse = dict(self.global_rmse)
        if tuple(global_rmse) != CAUSAL_ALPHA_V4_HORIZONS:
            raise ValueError("V4 uncertainty global RMSE horizon order drifted")
        for horizon, value in global_rmse.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V4 uncertainty global RMSE {horizon} is invalid")

        raw_cells = dict(self.cells)
        if tuple(raw_cells) != CAUSAL_ALPHA_V4_HORIZONS:
            raise ValueError("V4 uncertainty cell horizon order drifted")
        cells: dict[str, Mapping[V4ForecastState, CausalAlphaV4UncertaintyCell]] = {}
        for horizon in CAUSAL_ALPHA_V4_HORIZONS:
            horizon_cells = dict(raw_cells[horizon])
            if tuple(horizon_cells) != _V4_FORECAST_STATES:
                raise ValueError("V4 uncertainty state order drifted")
            for state, cell in horizon_cells.items():
                if (
                    not isinstance(cell, CausalAlphaV4UncertaintyCell)
                    or cell.state is not state
                ):
                    raise TypeError("V4 uncertainty cell identity drifted")
                if cell.global_rmse != global_rmse[horizon]:
                    raise ValueError("V4 uncertainty cell/global RMSE mismatch")
            cells[horizon] = MappingProxyType(horizon_cells)

        payload = {
            "basis_positioning_stress_threshold": self.basis_positioning_stress_threshold,
            "cells": tuple(
                (
                    horizon,
                    tuple(
                        cells[horizon][state].to_payload()
                        for state in _V4_FORECAST_STATES
                    ),
                )
                for horizon in CAUSAL_ALPHA_V4_HORIZONS
            ),
            "global_rmse": tuple(global_rmse.items()),
            "high_realized_volatility_threshold": self.high_realized_volatility_threshold,
            "low_liquidity_threshold": self.low_liquidity_threshold,
            "minimum_state_effective_sample_size": self.minimum_state_effective_sample_size,
            "schema_version": self.schema_version,
            "threshold_digest": self.threshold_digest,
        }
        expected = content_digest(payload)
        if self.digest and self.digest != expected:
            raise ValueError("V4 uncertainty model digest mismatch")
        object.__setattr__(self, "global_rmse", MappingProxyType(global_rmse))
        object.__setattr__(self, "cells", MappingProxyType(cells))
        object.__setattr__(self, "digest", expected)

    def resolve_states(
        self,
        *,
        realized_volatility: object,
        liquidity: object,
        basis_positioning_stress: object,
    ) -> np.ndarray:
        volatility = np.asarray(realized_volatility, dtype=np.float64).reshape(-1)
        liquidity_values = np.asarray(liquidity, dtype=np.float64).reshape(-1)
        stress = np.asarray(basis_positioning_stress, dtype=np.float64).reshape(-1)
        if (
            volatility.size == 0
            or liquidity_values.shape != volatility.shape
            or stress.shape != volatility.shape
            or not np.isfinite(volatility).all()
            or not np.isfinite(liquidity_values).all()
            or not np.isfinite(stress).all()
        ):
            raise ValueError("V4 uncertainty state inputs must be aligned and finite")
        states = np.empty(volatility.shape, dtype=object)
        for row in range(volatility.size):
            states[row] = V4ForecastState.NORMAL
        high_volatility = volatility >= self.high_realized_volatility_threshold
        low_liquidity = liquidity_values <= self.low_liquidity_threshold
        positioning_stress = np.abs(stress) >= self.basis_positioning_stress_threshold
        for row in np.flatnonzero(high_volatility):
            states[row] = V4ForecastState.HIGH_REALIZED_VOLATILITY
        for row in np.flatnonzero(low_liquidity):
            states[row] = V4ForecastState.LOW_LIQUIDITY
        for row in np.flatnonzero(positioning_stress):
            states[row] = V4ForecastState.BASIS_POSITIONING_STRESS
        return states

    def resolve_uncertainty(
        self,
        *,
        horizon: str,
        realized_volatility: object,
        liquidity: object,
        basis_positioning_stress: object,
    ) -> np.ndarray:
        if horizon not in CAUSAL_ALPHA_V4_HORIZONS:
            raise ValueError("unsupported V4 uncertainty horizon")
        states = self.resolve_states(
            realized_volatility=realized_volatility,
            liquidity=liquidity,
            basis_positioning_stress=basis_positioning_stress,
        )
        return np.asarray(
            [self.cells[horizon][state].selected_uncertainty for state in states],
            dtype=np.float64,
        )


def _uncertainty_effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights, dtype=np.float64))
    squared = float(np.sum(np.square(weights), dtype=np.float64))
    if total <= 0.0 or squared <= 0.0:
        return 0.0
    return float(total * total / squared)


def _uncertainty_weighted_rmse(
    prediction: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = float(np.sum(weights, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("V4 uncertainty RMSE requires positive weight")
    residual = labels - prediction
    value = math.sqrt(
        float(np.sum(weights * np.square(residual), dtype=np.float64) / total)
    )
    if not math.isfinite(value):
        raise ValueError("V4 uncertainty RMSE became non-finite")
    return value


def _uncertainty_horizon_map(
    value: Mapping[str, object], *, rows: int, field_name: str, weights: bool = False
) -> dict[str, np.ndarray]:
    if tuple(value) != CAUSAL_ALPHA_V4_HORIZONS:
        raise ValueError(f"V4 uncertainty {field_name} horizon order drifted")
    result: dict[str, np.ndarray] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        array = np.asarray(value[horizon], dtype=np.float64).reshape(-1).copy(order="C")
        if array.shape != (rows,):
            raise ValueError(f"V4 uncertainty {field_name}[{horizon}] is misaligned")
        if weights and (not np.isfinite(array).all() or np.any(array < 0.0)):
            raise ValueError(f"V4 uncertainty {field_name}[{horizon}] is invalid")
        array.setflags(write=False)
        result[horizon] = array
    return result


def fit_causal_alpha_v4_uncertainty(
    *,
    final_predictions: Mapping[str, object],
    labels: Mapping[str, object],
    weights: Mapping[str, object],
    state_eligible: object,
    realized_volatility: object,
    liquidity: object,
    basis_positioning_stress: object,
) -> CausalAlphaV4UncertaintyModel:
    """Fit train-prefix state RMSE from final hierarchical forecast residuals."""

    eligible = np.asarray(state_eligible, dtype=np.bool_).reshape(-1)
    rows = int(eligible.size)
    if rows == 0 or not np.any(eligible):
        raise ValueError("V4 uncertainty requires eligible train-prefix state rows")
    volatility = np.asarray(realized_volatility, dtype=np.float64).reshape(-1)
    liquidity_values = np.asarray(liquidity, dtype=np.float64).reshape(-1)
    stress = np.asarray(basis_positioning_stress, dtype=np.float64).reshape(-1)
    if (
        volatility.shape != (rows,)
        or liquidity_values.shape != (rows,)
        or stress.shape != (rows,)
        or not np.isfinite(volatility[eligible]).all()
        or not np.isfinite(liquidity_values[eligible]).all()
        or not np.isfinite(stress[eligible]).all()
    ):
        raise ValueError("V4 uncertainty eligible state variables must be finite")

    high_volatility_threshold = float(np.quantile(volatility[eligible], 0.80))
    low_liquidity_threshold = float(np.quantile(liquidity_values[eligible], 0.20))
    stress_threshold = float(np.quantile(np.abs(stress[eligible]), 0.80))
    threshold_digest = content_digest(
        {
            "basis_positioning_stress_quantile": 0.80,
            "basis_positioning_stress_threshold": stress_threshold,
            "high_realized_volatility_quantile": 0.80,
            "high_realized_volatility_threshold": high_volatility_threshold,
            "low_liquidity_quantile": 0.20,
            "low_liquidity_threshold": low_liquidity_threshold,
            "schema_version": "causal_alpha_v4_uncertainty_thresholds_v1",
        }
    )

    prediction_map = _uncertainty_horizon_map(
        final_predictions, rows=rows, field_name="final_predictions"
    )
    label_map = _uncertainty_horizon_map(labels, rows=rows, field_name="labels")
    weight_map = _uncertainty_horizon_map(
        weights, rows=rows, field_name="weights", weights=True
    )

    state_model = CausalAlphaV4UncertaintyModel.__new__(CausalAlphaV4UncertaintyModel)
    object.__setattr__(
        state_model, "high_realized_volatility_threshold", high_volatility_threshold
    )
    object.__setattr__(state_model, "low_liquidity_threshold", low_liquidity_threshold)
    object.__setattr__(
        state_model, "basis_positioning_stress_threshold", stress_threshold
    )
    states = CausalAlphaV4UncertaintyModel.resolve_states(
        state_model,
        realized_volatility=np.where(eligible, volatility, high_volatility_threshold),
        liquidity=np.where(eligible, liquidity_values, low_liquidity_threshold + 1.0),
        basis_positioning_stress=np.where(eligible, stress, 0.0),
    )

    global_rmse: dict[str, float] = {}
    cells: dict[str, dict[V4ForecastState, CausalAlphaV4UncertaintyCell]] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        prediction = prediction_map[horizon]
        target = label_map[horizon]
        horizon_weights = weight_map[horizon]
        if np.any((horizon_weights > 0.0) & ~eligible):
            raise ValueError(
                "V4 uncertainty weights cannot cross the train-prefix state scope"
            )
        positive = horizon_weights > 0.0
        if not np.any(positive):
            raise ValueError(f"V4 uncertainty {horizon} has no positive weight")
        if (
            not np.isfinite(prediction[positive]).all()
            or not np.isfinite(target[positive]).all()
        ):
            raise ValueError(f"V4 uncertainty {horizon} weighted rows must be finite")
        global_value = _uncertainty_weighted_rmse(
            prediction[positive], target[positive], horizon_weights[positive]
        )
        global_rmse[horizon] = global_value
        horizon_cells: dict[V4ForecastState, CausalAlphaV4UncertaintyCell] = {}
        for state in _V4_FORECAST_STATES:
            state_mask = np.fromiter(
                (value is state for value in states),
                dtype=np.bool_,
                count=rows,
            )
            mask = positive & state_mask
            selected_weights = horizon_weights[mask]
            support = int(np.count_nonzero(mask))
            ess = _uncertainty_effective_sample_size(selected_weights)
            state_rmse = (
                None
                if support == 0
                else _uncertainty_weighted_rmse(
                    prediction[mask], target[mask], selected_weights
                )
            )
            fallback = ess < _V4_MINIMUM_STATE_ESS
            if fallback:
                selected_uncertainty = global_value
                fallback_reason = "insufficient_state_ess"
            else:
                assert state_rmse is not None
                selected_uncertainty = state_rmse
                fallback_reason = None
            horizon_cells[state] = CausalAlphaV4UncertaintyCell(
                state=state,
                support=support,
                effective_sample_size=ess,
                global_rmse=global_value,
                state_rmse=state_rmse,
                selected_uncertainty=selected_uncertainty,
                fallback_reason=fallback_reason,
            )
        cells[horizon] = horizon_cells

    return CausalAlphaV4UncertaintyModel(
        high_realized_volatility_threshold=high_volatility_threshold,
        low_liquidity_threshold=low_liquidity_threshold,
        basis_positioning_stress_threshold=stress_threshold,
        threshold_digest=threshold_digest,
        global_rmse=global_rmse,
        cells=cells,
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV4TargetConfig:
    """The frozen first V4 slow-anchor/fast-impulse target hypothesis."""

    slow_target_magnitudes: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.25)
    fast_deviation_magnitudes: tuple[float, ...] = (0.0, 0.025, 0.05)
    uncertainty_multiplier: float = 1.0
    execution_cost_multiplier: float = 1.5
    edge_margin: float = 0.001
    slow_rebalance_decisions: int = 16
    fast_rebalance_decisions: int = 4
    maximum_final_target_delta: float = 0.125
    maximum_fast_absolute_deviation: float = 0.05
    schema_version: str = CAUSAL_ALPHA_V4_TARGET_SCHEMA

    def __post_init__(self) -> None:
        slow = tuple(float(value) for value in self.slow_target_magnitudes)
        fast = tuple(float(value) for value in self.fast_deviation_magnitudes)
        if slow != (0.0, 0.025, 0.05, 0.10, 0.25):
            raise ValueError("V4 slow target magnitudes must remain frozen")
        if fast != (0.0, 0.025, 0.05):
            raise ValueError("V4 fast deviation magnitudes must remain frozen")
        if self.uncertainty_multiplier != 1.0:
            raise ValueError("V4 uncertainty multiplier must remain 1.0")
        if self.execution_cost_multiplier != 1.5:
            raise ValueError("V4 execution cost multiplier must remain 1.5")
        if self.edge_margin != 0.001:
            raise ValueError("V4 edge margin must remain 0.001")
        if self.slow_rebalance_decisions != 16:
            raise ValueError("V4 slow cadence must remain 16 decisions")
        if self.fast_rebalance_decisions != 4:
            raise ValueError("V4 fast cadence must remain 4 decisions")
        if self.maximum_final_target_delta != 0.125:
            raise ValueError("V4 maximum final target delta must remain 0.125")
        if self.maximum_fast_absolute_deviation != 0.05:
            raise ValueError("V4 maximum fast deviation must remain 0.05")
        if self.schema_version != CAUSAL_ALPHA_V4_TARGET_SCHEMA:
            raise ValueError("unsupported V4 target config schema")
        object.__setattr__(self, "slow_target_magnitudes", slow)
        object.__setattr__(self, "fast_deviation_magnitudes", fast)

    @property
    def digest(self) -> str:
        return content_digest(self)


def _v4_direct_objective(
    *,
    target: float,
    previous: float,
    expected_return: float,
    uncertainty: float,
    one_way_cost_rate: float,
    config: CausalAlphaV4TargetConfig,
) -> float:
    delta = target - previous
    turnover = abs(delta)
    return (
        delta * expected_return
        - config.uncertainty_multiplier * turnover * uncertainty
        - turnover
        * (one_way_cost_rate * config.execution_cost_multiplier + config.edge_margin)
    )


def _v4_staged_objective(
    *,
    previous: float,
    anchor: float,
    final: float,
    slow_expected_return: float,
    slow_uncertainty: float,
    fast_expected_return: float,
    fast_uncertainty: float,
    one_way_cost_rate: float,
    config: CausalAlphaV4TargetConfig,
) -> tuple[float, float, float]:
    slow = _v4_direct_objective(
        target=anchor,
        previous=previous,
        expected_return=slow_expected_return,
        uncertainty=slow_uncertainty,
        one_way_cost_rate=one_way_cost_rate,
        config=config,
    )
    fast_final = _v4_direct_objective(
        target=final,
        previous=previous,
        expected_return=fast_expected_return,
        uncertainty=fast_uncertainty,
        one_way_cost_rate=one_way_cost_rate,
        config=config,
    )
    fast_anchor = _v4_direct_objective(
        target=anchor,
        previous=previous,
        expected_return=fast_expected_return,
        uncertainty=fast_uncertainty,
        one_way_cost_rate=one_way_cost_rate,
        config=config,
    )
    fast_improvement = fast_final - fast_anchor
    return slow, fast_improvement, slow + fast_improvement


def _v4_is_risk_reduction(previous: float, target: float) -> bool:
    if abs(target - previous) <= _V4_TARGET_EPSILON:
        return True
    if abs(previous) <= _V4_TARGET_EPSILON:
        return abs(target) <= _V4_TARGET_EPSILON
    return (
        previous * target >= -_V4_TARGET_EPSILON
        and abs(target) <= abs(previous) + _V4_TARGET_EPSILON
    )


def _v4_consensus_allows(
    *,
    previous: float,
    target: float,
    fast_expected_return: float,
    direction_score: float,
) -> bool:
    if _v4_is_risk_reduction(previous, target):
        return True
    if (
        abs(fast_expected_return) <= _V4_TARGET_EPSILON
        or abs(direction_score) <= _V4_TARGET_EPSILON
        or fast_expected_return * direction_score <= 0.0
    ):
        return False
    return target * fast_expected_return > 0.0


def _v4_slow_candidates(
    *,
    previous: float,
    current_anchor: float,
    cap: float,
    config: CausalAlphaV4TargetConfig,
) -> tuple[float, ...]:
    values = {
        0.0,
        float(np.clip(previous, -cap, cap)),
        float(np.clip(current_anchor, -cap, cap)),
        -cap,
        cap,
    }
    for magnitude in config.slow_target_magnitudes:
        bounded = min(magnitude, cap)
        values.add(bounded)
        values.add(-bounded)
    return tuple(
        value
        for value in sorted(values)
        if abs(value - previous)
        <= config.maximum_final_target_delta + _V4_TARGET_EPSILON
    )


def _v4_fast_candidates(
    *, previous: float, anchor: float, cap: float, config: CausalAlphaV4TargetConfig
) -> tuple[float, ...]:
    values = {float(np.clip(anchor, -cap, cap))}
    for magnitude in config.fast_deviation_magnitudes:
        values.add(float(np.clip(anchor + magnitude, -cap, cap)))
        values.add(float(np.clip(anchor - magnitude, -cap, cap)))
    return tuple(
        value
        for value in sorted(values)
        if abs(value - anchor)
        <= config.maximum_fast_absolute_deviation + _V4_TARGET_EPSILON
        and abs(value - previous)
        <= config.maximum_final_target_delta + _V4_TARGET_EPSILON
    )


def _v4_choose_best(
    candidates: tuple[float, ...],
    scores: tuple[float, ...],
    *,
    previous: float,
) -> tuple[float, float]:
    if not candidates or len(candidates) != len(scores):
        raise ValueError("V4 target candidate scores are invalid")
    maximum = max(scores)
    tied = tuple(
        (value, score)
        for value, score in zip(candidates, scores, strict=True)
        if score >= maximum - 1e-15
    )
    return min(
        tied,
        key=lambda item: (
            abs(item[0] - previous),
            abs(item[0]),
            item[0],
        ),
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV4TargetPath:
    initial_weight: float
    slow_anchors: np.ndarray
    fast_deviations: np.ndarray
    targets: np.ndarray
    slow_expected_returns: np.ndarray
    fast_expected_returns: np.ndarray
    slow_uncertainties: np.ndarray
    fast_uncertainties: np.ndarray
    liquidity_weight_caps: np.ndarray
    slow_objectives: np.ndarray
    fast_objective_improvements: np.ndarray
    final_objectives: np.ndarray
    reasons: tuple[str, ...]
    slow_anchor_change_count: int
    fast_impulse_change_count: int
    submitted_change_count: int
    liquidity_deleveraging_count: int
    sign_flip_count: int
    config_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_weight):
            raise ValueError("V4 target initial weight must be finite")
        arrays: dict[str, np.ndarray] = {}
        shape: tuple[int, ...] | None = None
        for field_name in (
            "slow_anchors",
            "fast_deviations",
            "targets",
            "slow_expected_returns",
            "fast_expected_returns",
            "slow_uncertainties",
            "fast_uncertainties",
            "liquidity_weight_caps",
            "slow_objectives",
            "fast_objective_improvements",
            "final_objectives",
        ):
            array = (
                np.asarray(getattr(self, field_name), dtype=np.float64)
                .reshape(-1)
                .copy()
            )
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError(
                    f"V4 target path {field_name} must be finite and non-empty"
                )
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError("V4 target path arrays must align")
            array.setflags(write=False)
            arrays[field_name] = array
        if np.any(arrays["slow_uncertainties"] < 0.0) or np.any(
            arrays["fast_uncertainties"] < 0.0
        ):
            raise ValueError("V4 target uncertainty must be non-negative")
        if np.any(arrays["liquidity_weight_caps"] < 0.0):
            raise ValueError("V4 target liquidity caps must be non-negative")
        reasons = tuple(self.reasons)
        if len(reasons) != len(arrays["targets"]) or any(
            not reason for reason in reasons
        ):
            raise ValueError("V4 target reasons must cover every decision")
        for field_name in (
            "slow_anchor_change_count",
            "fast_impulse_change_count",
            "submitted_change_count",
            "liquidity_deleveraging_count",
            "sign_flip_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V4 target path {field_name} is invalid")
        require_sha256(self.config_digest, field="V4 target config_digest")
        expected = content_and_arrays_digest(
            {
                "config_digest": self.config_digest,
                "initial_weight": self.initial_weight,
                "reasons": reasons,
                "schema_version": CAUSAL_ALPHA_V4_TARGET_SCHEMA,
                "slow_anchor_change_count": self.slow_anchor_change_count,
                "fast_impulse_change_count": self.fast_impulse_change_count,
                "submitted_change_count": self.submitted_change_count,
                "liquidity_deleveraging_count": self.liquidity_deleveraging_count,
                "sign_flip_count": self.sign_flip_count,
            },
            tuple((field_name, array) for field_name, array in arrays.items()),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 target path digest mismatch")
        for field_name, array in arrays.items():
            object.__setattr__(self, field_name, array)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "digest", expected)


def causal_alpha_v4_target_path(
    prediction_4h: object,
    prediction_24h: object,
    prediction_72h: object,
    *,
    direction_score_4h: object,
    uncertainty_4h: object,
    uncertainty_24h: object,
    uncertainty_72h: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    config: CausalAlphaV4TargetConfig,
    initial_weight: float,
    actionable_mask: object | None = None,
) -> CausalAlphaV4TargetPath:
    """Compile one slow anchor plus one bounded 4h impulse without double cost."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (
            prediction_4h,
            prediction_24h,
            prediction_72h,
            direction_score_4h,
            uncertainty_4h,
            uncertainty_24h,
            uncertainty_72h,
            one_way_cost_rates,
            liquidity_weight_caps,
        )
    )
    rows = int(arrays[0].size)
    if rows == 0 or any(array.shape != (rows,) for array in arrays):
        raise ValueError("V4 target inputs must be non-empty and aligned")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("V4 target inputs must be finite")
    (
        fast_mu,
        prediction_24,
        prediction_72,
        direction,
        fast_sigma,
        uncertainty_24_values,
        uncertainty_72_values,
        costs,
        caps,
    ) = arrays
    if (
        np.any(fast_sigma < 0.0)
        or np.any(uncertainty_24_values < 0.0)
        or np.any(uncertainty_72_values < 0.0)
        or np.any(costs < 0.0)
        or np.any(caps < 0.0)
    ):
        raise ValueError("V4 target uncertainty/cost/cap inputs must be non-negative")
    if not isinstance(config, CausalAlphaV4TargetConfig):
        raise TypeError("V4 target compiler requires CausalAlphaV4TargetConfig")
    if not math.isfinite(initial_weight):
        raise ValueError("V4 target initial weight must be finite")
    actionable = (
        np.ones(rows, dtype=np.bool_)
        if actionable_mask is None
        else np.asarray(actionable_mask, dtype=np.bool_).reshape(-1)
    )
    if actionable.shape != (rows,):
        raise ValueError("V4 target actionable mask must align")

    prediction_72_equivalent = prediction_72 / 3.0
    slow_mu = 0.5 * (prediction_24 + prediction_72_equivalent)
    slow_disagreement = 0.5 * np.abs(prediction_24 - prediction_72_equivalent)
    slow_sigma = np.sqrt(
        0.25
        * (np.square(uncertainty_24_values) + np.square(uncertainty_72_values / 3.0))
        + np.square(slow_disagreement)
    )

    slow_anchors = np.empty(rows, dtype=np.float64)
    fast_deviations = np.empty(rows, dtype=np.float64)
    targets = np.empty(rows, dtype=np.float64)
    slow_objectives = np.empty(rows, dtype=np.float64)
    fast_improvements = np.empty(rows, dtype=np.float64)
    final_objectives = np.empty(rows, dtype=np.float64)
    reasons: list[str] = []
    previous = float(initial_weight)
    current_anchor = float(initial_weight)
    slow_changes = 0
    fast_changes = 0
    submitted = 0
    liquidity_deleveraging = 0
    sign_flips = 0

    for index in range(rows):
        cap = min(float(caps[index]), 1.0)
        old_anchor = current_anchor
        selected_anchor = float(np.clip(current_anchor, -cap, cap))
        selected = previous
        direction_blocked = False

        if abs(previous) > cap + _V4_TARGET_EPSILON:
            selected = float(np.clip(previous, -cap, cap))
            selected_anchor = selected
            reason = "liquidity_deleverage"
            liquidity_deleveraging += 1
            slow_score, fast_improvement, final_score = _v4_staged_objective(
                previous=previous,
                anchor=selected_anchor,
                final=selected,
                slow_expected_return=float(slow_mu[index]),
                slow_uncertainty=float(slow_sigma[index]),
                fast_expected_return=float(fast_mu[index]),
                fast_uncertainty=float(fast_sigma[index]),
                one_way_cost_rate=float(costs[index]),
                config=config,
            )
        elif not bool(actionable[index]):
            selected_anchor = current_anchor
            selected = previous
            reason = "unactionable_hold"
            slow_score, fast_improvement, final_score = _v4_staged_objective(
                previous=previous,
                anchor=selected_anchor,
                final=selected,
                slow_expected_return=float(slow_mu[index]),
                slow_uncertainty=float(slow_sigma[index]),
                fast_expected_return=float(fast_mu[index]),
                fast_uncertainty=float(fast_sigma[index]),
                one_way_cost_rate=float(costs[index]),
                config=config,
            )
        else:
            if index % config.slow_rebalance_decisions == 0:
                candidates = _v4_slow_candidates(
                    previous=previous,
                    current_anchor=current_anchor,
                    cap=cap,
                    config=config,
                )
                unrestricted_scores = tuple(
                    _v4_direct_objective(
                        target=value,
                        previous=previous,
                        expected_return=float(slow_mu[index]),
                        uncertainty=float(slow_sigma[index]),
                        one_way_cost_rate=float(costs[index]),
                        config=config,
                    )
                    for value in candidates
                )
                unrestricted_anchor, _ = _v4_choose_best(
                    candidates, unrestricted_scores, previous=previous
                )
                allowed_pairs = tuple(
                    (value, score)
                    for value, score in zip(
                        candidates, unrestricted_scores, strict=True
                    )
                    if _v4_consensus_allows(
                        previous=previous,
                        target=value,
                        fast_expected_return=float(fast_mu[index]),
                        direction_score=float(direction[index]),
                    )
                )
                if not allowed_pairs:
                    allowed_pairs = ((previous, 0.0),)
                selected_anchor, _ = _v4_choose_best(
                    tuple(value for value, _ in allowed_pairs),
                    tuple(score for _, score in allowed_pairs),
                    previous=previous,
                )
                direction_blocked = (
                    abs(unrestricted_anchor - selected_anchor) > _V4_TARGET_EPSILON
                )
            else:
                selected_anchor = float(np.clip(current_anchor, -cap, cap))

            if index % config.fast_rebalance_decisions != 0:
                selected = previous
                slow_score, fast_improvement, final_score = _v4_staged_objective(
                    previous=previous,
                    anchor=selected_anchor,
                    final=selected,
                    slow_expected_return=float(slow_mu[index]),
                    slow_uncertainty=float(slow_sigma[index]),
                    fast_expected_return=float(fast_mu[index]),
                    fast_uncertainty=float(fast_sigma[index]),
                    one_way_cost_rate=float(costs[index]),
                    config=config,
                )
                reason = "cadence_hold"
            else:
                candidates = _v4_fast_candidates(
                    previous=previous,
                    anchor=selected_anchor,
                    cap=cap,
                    config=config,
                )
                scored = tuple(
                    (
                        value,
                        _v4_staged_objective(
                            previous=previous,
                            anchor=selected_anchor,
                            final=value,
                            slow_expected_return=float(slow_mu[index]),
                            slow_uncertainty=float(slow_sigma[index]),
                            fast_expected_return=float(fast_mu[index]),
                            fast_uncertainty=float(fast_sigma[index]),
                            one_way_cost_rate=float(costs[index]),
                            config=config,
                        ),
                    )
                    for value in candidates
                    if _v4_consensus_allows(
                        previous=previous,
                        target=value,
                        fast_expected_return=float(fast_mu[index]),
                        direction_score=float(direction[index]),
                    )
                )
                if not scored:
                    selected = previous
                    selected_anchor = previous
                    slow_score, fast_improvement, final_score = _v4_staged_objective(
                        previous=previous,
                        anchor=selected_anchor,
                        final=selected,
                        slow_expected_return=float(slow_mu[index]),
                        slow_uncertainty=float(slow_sigma[index]),
                        fast_expected_return=float(fast_mu[index]),
                        fast_uncertainty=float(fast_sigma[index]),
                        one_way_cost_rate=float(costs[index]),
                        config=config,
                    )
                    direction_blocked = True
                else:
                    selected, final_score = _v4_choose_best(
                        tuple(value for value, _ in scored),
                        tuple(values[2] for _, values in scored),
                        previous=previous,
                    )
                    selected_values = next(
                        values for value, values in scored if value == selected
                    )
                    slow_score, fast_improvement, final_score = selected_values
                if abs(selected - previous) <= _V4_TARGET_EPSILON:
                    reason = (
                        "direction_disagreement_hold" if direction_blocked else "hold"
                    )
                elif abs(selected - selected_anchor) > _V4_TARGET_EPSILON:
                    reason = "fast_impulse"
                elif abs(selected_anchor - old_anchor) > _V4_TARGET_EPSILON:
                    reason = "slow_rebalance"
                else:
                    reason = "rebalance"

        if abs(selected_anchor - old_anchor) > _V4_TARGET_EPSILON:
            slow_changes += 1
        current_anchor = selected_anchor
        deviation = selected - selected_anchor
        if abs(deviation) > config.maximum_fast_absolute_deviation + _V4_TARGET_EPSILON:
            if reason not in {"cadence_hold", "unactionable_hold"}:
                raise RuntimeError("V4 fast deviation exceeded authored bound")
        if abs(deviation) > _V4_TARGET_EPSILON and reason == "fast_impulse":
            fast_changes += 1
        if abs(selected - previous) > _V4_TARGET_EPSILON:
            submitted += 1
        if previous * selected < 0.0:
            sign_flips += 1

        slow_anchors[index] = selected_anchor
        fast_deviations[index] = deviation
        targets[index] = selected
        slow_objectives[index] = slow_score
        fast_improvements[index] = fast_improvement
        final_objectives[index] = final_score
        reasons.append(reason)
        previous = float(selected)

    return CausalAlphaV4TargetPath(
        initial_weight=float(initial_weight),
        slow_anchors=slow_anchors,
        fast_deviations=fast_deviations,
        targets=targets,
        slow_expected_returns=slow_mu,
        fast_expected_returns=fast_mu,
        slow_uncertainties=slow_sigma,
        fast_uncertainties=fast_sigma,
        liquidity_weight_caps=caps,
        slow_objectives=slow_objectives,
        fast_objective_improvements=fast_improvements,
        final_objectives=final_objectives,
        reasons=tuple(reasons),
        slow_anchor_change_count=slow_changes,
        fast_impulse_change_count=fast_changes,
        submitted_change_count=submitted,
        liquidity_deleveraging_count=liquidity_deleveraging,
        sign_flip_count=sign_flips,
        config_digest=config.digest,
    )


__all__ = [
    "CAUSAL_ALPHA_V4_FIT_CONFIG_SCHEMA",
    "CAUSAL_ALPHA_V4_FORECAST_SCHEMA",
    "CAUSAL_ALPHA_V4_HORIZONS",
    "CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA",
    "CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA",
    "CAUSAL_ALPHA_V4_TARGET_SCHEMA",
    "CAUSAL_ALPHA_V4_UNCERTAINTY_SCHEMA",
    "CausalAlphaV4FitConfig",
    "CausalAlphaV4Forecast",
    "CausalAlphaV4ResidualLabels",
    "CausalAlphaV4SymbolSamples",
    "CausalAlphaV4TargetConfig",
    "CausalAlphaV4TargetPath",
    "CausalAlphaV4UncertaintyCell",
    "CausalAlphaV4UncertaintyModel",
    "V4ForecastState",
    "build_causal_alpha_v4_forecast",
    "build_causal_alpha_v4_residual_labels",
    "causal_alpha_v4_target_path",
    "fit_causal_alpha_v4_uncertainty",
]
