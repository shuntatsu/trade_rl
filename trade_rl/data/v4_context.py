"""Causal cross-market context primitives for the research-only V4 teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256

BARS_1H = 4
BARS_4H = 16
BARS_24H = 96
BARS_7D = 672
_NS_PER_HOUR = 3_600_000_000_000
_EPSILON = 1e-12
_UNAVAILABLE_STALENESS_HOURS = 24.0

CROSS_MARKET_CORE_NAMES = (
    "spot_log_return_1h",
    "spot_log_return_4h",
    "spot_log_return_24h",
    "spot_log_quote_volume_robust_z_4h",
    "spot_log_quote_volume_robust_z_24h",
    "spot_perp_log_basis",
    "spot_perp_basis_change_1h",
    "spot_perp_basis_change_4h",
    "spot_perp_basis_robust_z_7d",
    "spot_minus_perp_log_return_1h",
    "spot_minus_perp_log_return_4h",
    "spot_to_perp_log_quote_volume_ratio_1h",
    "spot_to_perp_log_quote_volume_ratio_4h",
    "spot_to_perp_log_quote_volume_ratio_24h",
    "spot_taker_quote_imbalance_1h",
    "spot_taker_quote_imbalance_4h",
    "perp_taker_quote_imbalance_1h",
    "perp_taker_quote_imbalance_4h",
    "spot_minus_perp_taker_imbalance_1h",
    "spot_minus_perp_taker_imbalance_4h",
    "funding_rate",
    "funding_rate_change",
    "funding_rate_robust_z_7d",
    "basis_z_x_flow_divergence_4h",
)

CROSS_MARKET_DERIVATIVE_NAMES = (
    "open_interest_log_change_1h",
    "open_interest_log_change_4h",
    "open_interest_log_change_24h",
    "global_long_short_ratio_robust_z_4h",
    "top_position_long_short_ratio_robust_z_4h",
    "basis_z_x_open_interest_change_4h",
    "funding_z_x_open_interest_change_4h",
)

GLOBAL_MARKET_CORE_NAMES = (
    "btc_spot_log_return_1h",
    "btc_spot_log_return_4h",
    "btc_spot_log_return_24h",
    "btc_perp_log_return_1h",
    "btc_perp_log_return_4h",
    "btc_perp_log_return_24h",
    "btc_spot_perp_log_basis",
    "btc_spot_perp_basis_change_4h",
    "btc_spot_perp_basis_robust_z_7d",
    "btc_spot_taker_quote_imbalance_1h",
    "btc_spot_taker_quote_imbalance_4h",
    "btc_perp_taker_quote_imbalance_1h",
    "btc_perp_taker_quote_imbalance_4h",
    "btc_spot_to_perp_log_quote_volume_ratio_4h",
    "btc_spot_to_perp_log_quote_volume_ratio_24h",
    "btc_funding_rate",
    "btc_funding_rate_robust_z_7d",
    "eth_spot_log_return_1h",
    "eth_spot_log_return_4h",
    "eth_spot_log_return_24h",
    "eth_perp_log_return_1h",
    "eth_perp_log_return_4h",
    "eth_perp_log_return_24h",
    "eth_spot_perp_log_basis",
    "eth_spot_perp_basis_change_4h",
    "eth_spot_perp_basis_robust_z_7d",
    "eth_spot_taker_quote_imbalance_1h",
    "eth_spot_taker_quote_imbalance_4h",
    "eth_perp_taker_quote_imbalance_1h",
    "eth_perp_taker_quote_imbalance_4h",
    "eth_spot_to_perp_log_quote_volume_ratio_4h",
    "eth_spot_to_perp_log_quote_volume_ratio_24h",
    "eth_funding_rate",
    "eth_funding_rate_robust_z_7d",
    "btc_minus_eth_perp_return_4h",
    "btc_minus_eth_perp_return_24h",
    "btc_minus_eth_basis",
    "btc_eth_perp_return_dispersion_4h",
)

GLOBAL_MARKET_DERIVATIVE_NAMES = (
    "btc_open_interest_log_change_4h",
    "btc_open_interest_log_change_24h",
    "btc_global_long_short_ratio_robust_z_4h",
    "eth_open_interest_log_change_4h",
    "eth_open_interest_log_change_24h",
    "eth_global_long_short_ratio_robust_z_4h",
)


def _readonly(value: object, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy(order="C")
    result.setflags(write=False)
    return result


def _readonly_float(value: object) -> np.ndarray:
    return _readonly(value, dtype=np.dtype(np.float64))


def _readonly_bool(value: object) -> np.ndarray:
    return _readonly(value, dtype=np.dtype(np.bool_))


def _readonly_int(value: object) -> np.ndarray:
    return _readonly(value, dtype=np.dtype(np.int64))


def _validate_decision_clock(
    decision_indices: object, decision_timestamps: object
) -> tuple[np.ndarray, np.ndarray]:
    indices = _readonly_int(decision_indices).reshape(-1)
    timestamps = np.asarray(decision_timestamps).copy(order="C")
    if indices.size == 0 or timestamps.ndim != 1 or timestamps.shape != indices.shape:
        raise ValueError("V4 decision clock must be non-empty and aligned")
    if np.any(indices < 0) or np.any(np.diff(indices) != 1):
        raise ValueError("V4 decision indices must be contiguous and increasing")
    if not np.issubdtype(timestamps.dtype, np.datetime64):
        raise ValueError("V4 decision timestamps must be datetime64")
    timestamps = timestamps.astype("datetime64[ns]")
    timestamp_ns = timestamps.astype(np.int64)
    if np.any(timestamp_ns == np.iinfo(np.int64).min):
        raise ValueError("V4 decision timestamps must not contain NaT")
    if timestamp_ns.size > 1:
        expected = 15 * 60 * 1_000_000_000
        if np.any(np.diff(timestamp_ns) != expected):
            raise ValueError("V4 decision timestamps must use the maintained 15m clock")
    timestamps.setflags(write=False)
    return indices, timestamps


def _aligned_float(value: object, *, rows: int, field: str) -> np.ndarray:
    array = _readonly_float(value).reshape(-1)
    if array.shape != (rows,) or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a finite decision-aligned vector")
    return array


def _aligned_bool(value: object, *, rows: int, field: str) -> np.ndarray:
    array = _readonly_bool(value).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"{field} must be a decision-aligned boolean vector")
    return array


def _validate_sha256(value: str, *, field: str) -> str:
    return require_sha256(value, field=field)


@dataclass(frozen=True, slots=True)
class V4CrossMarketInputs:
    decision_indices: np.ndarray
    decision_timestamps: np.ndarray
    spot_close: np.ndarray
    spot_quote_volume: np.ndarray
    spot_taker_buy_quote_volume: np.ndarray
    spot_row_available: np.ndarray
    perp_close: np.ndarray
    perp_mark_price: np.ndarray
    perp_quote_volume: np.ndarray
    perp_taker_buy_quote_volume: np.ndarray
    perp_row_available: np.ndarray
    funding_event_rate: np.ndarray
    funding_event_available: np.ndarray
    open_interest_value: np.ndarray | None
    global_long_short_ratio: np.ndarray | None
    top_position_long_short_ratio: np.ndarray | None
    derivatives_available: np.ndarray | None
    derivatives_staleness_hours: np.ndarray | None
    source_digest: str

    def __post_init__(self) -> None:
        indices, timestamps = _validate_decision_clock(
            self.decision_indices, self.decision_timestamps
        )
        rows = len(indices)
        arrays = {
            "spot_close": _aligned_float(self.spot_close, rows=rows, field="spot_close"),
            "spot_quote_volume": _aligned_float(
                self.spot_quote_volume, rows=rows, field="spot_quote_volume"
            ),
            "spot_taker_buy_quote_volume": _aligned_float(
                self.spot_taker_buy_quote_volume,
                rows=rows,
                field="spot_taker_buy_quote_volume",
            ),
            "perp_close": _aligned_float(self.perp_close, rows=rows, field="perp_close"),
            "perp_mark_price": _aligned_float(
                self.perp_mark_price, rows=rows, field="perp_mark_price"
            ),
            "perp_quote_volume": _aligned_float(
                self.perp_quote_volume, rows=rows, field="perp_quote_volume"
            ),
            "perp_taker_buy_quote_volume": _aligned_float(
                self.perp_taker_buy_quote_volume,
                rows=rows,
                field="perp_taker_buy_quote_volume",
            ),
            "funding_event_rate": _aligned_float(
                self.funding_event_rate, rows=rows, field="funding_event_rate"
            ),
        }
        masks = {
            "spot_row_available": _aligned_bool(
                self.spot_row_available, rows=rows, field="spot_row_available"
            ),
            "perp_row_available": _aligned_bool(
                self.perp_row_available, rows=rows, field="perp_row_available"
            ),
            "funding_event_available": _aligned_bool(
                self.funding_event_available,
                rows=rows,
                field="funding_event_available",
            ),
        }
        for price_field, availability_field in (
            ("spot_close", "spot_row_available"),
            ("perp_close", "perp_row_available"),
            ("perp_mark_price", "perp_row_available"),
        ):
            if np.any(arrays[price_field][masks[availability_field]] <= 0.0):
                raise ValueError(f"{price_field} must be positive when available")
        for volume_field, availability_field in (
            ("spot_quote_volume", "spot_row_available"),
            ("spot_taker_buy_quote_volume", "spot_row_available"),
            ("perp_quote_volume", "perp_row_available"),
            ("perp_taker_buy_quote_volume", "perp_row_available"),
        ):
            if np.any(arrays[volume_field][masks[availability_field]] < 0.0):
                raise ValueError(f"{volume_field} must be non-negative when available")
        if np.any(
            arrays["spot_taker_buy_quote_volume"][masks["spot_row_available"]]
            > arrays["spot_quote_volume"][masks["spot_row_available"]] + _EPSILON
        ):
            raise ValueError("spot taker buy quote volume exceeds total quote volume")
        if np.any(
            arrays["perp_taker_buy_quote_volume"][masks["perp_row_available"]]
            > arrays["perp_quote_volume"][masks["perp_row_available"]] + _EPSILON
        ):
            raise ValueError("perp taker buy quote volume exceeds total quote volume")

        derivative_values = (
            self.open_interest_value,
            self.global_long_short_ratio,
            self.top_position_long_short_ratio,
            self.derivatives_available,
            self.derivatives_staleness_hours,
        )
        present = tuple(value is not None for value in derivative_values)
        if any(present) and not all(present):
            raise ValueError("V4 derivative inputs must be all present or all absent")
        if all(present):
            assert self.open_interest_value is not None
            assert self.global_long_short_ratio is not None
            assert self.top_position_long_short_ratio is not None
            assert self.derivatives_available is not None
            assert self.derivatives_staleness_hours is not None
            open_interest = _aligned_float(
                self.open_interest_value, rows=rows, field="open_interest_value"
            )
            global_ratio = _aligned_float(
                self.global_long_short_ratio,
                rows=rows,
                field="global_long_short_ratio",
            )
            top_ratio = _aligned_float(
                self.top_position_long_short_ratio,
                rows=rows,
                field="top_position_long_short_ratio",
            )
            derivative_available = _aligned_bool(
                self.derivatives_available,
                rows=rows,
                field="derivatives_available",
            )
            derivative_staleness = _aligned_float(
                self.derivatives_staleness_hours,
                rows=rows,
                field="derivatives_staleness_hours",
            )
            if np.any(open_interest[derivative_available] <= 0.0):
                raise ValueError("open_interest_value must be positive when available")
            if np.any(global_ratio[derivative_available] <= 0.0) or np.any(
                top_ratio[derivative_available] <= 0.0
            ):
                raise ValueError("long/short ratios must be positive when available")
            if np.any(derivative_staleness[derivative_available] < 0.0):
                raise ValueError("derivatives staleness must be non-negative")
            object.__setattr__(self, "open_interest_value", open_interest)
            object.__setattr__(self, "global_long_short_ratio", global_ratio)
            object.__setattr__(self, "top_position_long_short_ratio", top_ratio)
            object.__setattr__(self, "derivatives_available", derivative_available)
            object.__setattr__(
                self, "derivatives_staleness_hours", derivative_staleness
            )
        else:
            for field in (
                "open_interest_value",
                "global_long_short_ratio",
                "top_position_long_short_ratio",
                "derivatives_available",
                "derivatives_staleness_hours",
            ):
                object.__setattr__(self, field, None)

        _validate_sha256(self.source_digest, field="V4 cross-market source_digest")
        object.__setattr__(self, "decision_indices", indices)
        object.__setattr__(self, "decision_timestamps", timestamps)
        for field, value in arrays.items():
            object.__setattr__(self, field, value)
        for field, value in masks.items():
            object.__setattr__(self, field, value)


@dataclass(frozen=True, slots=True)
class V4GlobalMarketInputs:
    btc: V4CrossMarketInputs
    eth: V4CrossMarketInputs
    source_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.btc, V4CrossMarketInputs) or not isinstance(
            self.eth, V4CrossMarketInputs
        ):
            raise TypeError("V4 global inputs require BTC and ETH cross-market inputs")
        if not np.array_equal(self.btc.decision_indices, self.eth.decision_indices):
            raise ValueError("V4 global decision indices must match")
        if not np.array_equal(
            self.btc.decision_timestamps, self.eth.decision_timestamps
        ):
            raise ValueError("V4 global decision timestamps must match")
        _validate_sha256(self.source_digest, field="V4 global source_digest")


@dataclass(frozen=True, slots=True)
class FundingContextSeries:
    rate: np.ndarray
    change: np.ndarray
    robust_z_7d: np.ndarray
    available: np.ndarray
    staleness_hours: np.ndarray

    def __post_init__(self) -> None:
        rate = _readonly_float(self.rate).reshape(-1)
        change = _readonly_float(self.change).reshape(-1)
        robust_z = _readonly_float(self.robust_z_7d).reshape(-1)
        available = _readonly_bool(self.available)
        staleness = _readonly_float(self.staleness_hours)
        if change.shape != rate.shape or robust_z.shape != rate.shape:
            raise ValueError("funding context value arrays must align")
        if available.shape != (len(rate), 3) or staleness.shape != available.shape:
            raise ValueError("funding availability/staleness arrays must be (row, 3)")
        if (
            not np.isfinite(rate).all()
            or not np.isfinite(change).all()
            or not np.isfinite(robust_z).all()
        ):
            raise ValueError("funding context values must be finite")
        if np.any(staleness < 0.0) or not np.isfinite(staleness).all():
            raise ValueError("funding context staleness must be finite and non-negative")
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "change", change)
        object.__setattr__(self, "robust_z_7d", robust_z)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "staleness_hours", staleness)


@dataclass(frozen=True, slots=True)
class V4ContextBlock:
    feature_names: tuple[str, ...]
    decision_indices: np.ndarray
    values: np.ndarray
    available: np.ndarray
    staleness_hours: np.ndarray
    source_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("V4 context feature names must be non-empty and unique")
        indices = _readonly_int(self.decision_indices).reshape(-1)
        values = _readonly_float(self.values)
        available = _readonly_bool(self.available)
        staleness = _readonly_float(self.staleness_hours)
        expected_shape = (len(indices), len(names))
        if values.shape != expected_shape or available.shape != expected_shape:
            raise ValueError("V4 context values and availability must match schema")
        if staleness.shape != expected_shape:
            raise ValueError("V4 context staleness must match schema")
        if not np.isfinite(values).all() or not np.isfinite(staleness).all():
            raise ValueError("V4 context values/staleness must be finite")
        if np.any(staleness < 0.0):
            raise ValueError("V4 context staleness must be non-negative")
        if np.any(values[~available] != 0.0):
            raise ValueError("unavailable V4 context values must use inert zero storage")
        if indices.size == 0 or np.any(indices < 0) or np.any(np.diff(indices) != 1):
            raise ValueError("V4 context decision indices must be contiguous")
        _validate_sha256(self.source_digest, field="V4 context source_digest")
        expected = content_and_arrays_digest(
            {
                "feature_names": names,
                "schema_version": "causal_alpha_v4_context_block_v1",
                "source_digest": self.source_digest,
            },
            (
                ("decision_indices", indices),
                ("values", values),
                ("available", available),
                ("staleness_hours", staleness),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 context block digest mismatch")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "decision_indices", indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "staleness_hours", staleness)
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalBetaConfig:
    return_horizon_hours: float = 4.0
    lookback_hours: float = 720.0
    minimum_complete_samples: int = 90
    minimum_market_variance: float = 1e-12
    minimum_beta: float = -3.0
    maximum_beta: float = 3.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.return_horizon_hours)
            or self.return_horizon_hours <= 0.0
            or not math.isfinite(self.lookback_hours)
            or self.lookback_hours < self.return_horizon_hours
        ):
            raise ValueError("causal beta horizons must be finite and positive")
        ratio = self.lookback_hours / self.return_horizon_hours
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("causal beta lookback must contain complete return horizons")
        if (
            isinstance(self.minimum_complete_samples, bool)
            or not isinstance(self.minimum_complete_samples, int)
            or self.minimum_complete_samples < 2
            or self.minimum_complete_samples > int(round(ratio))
        ):
            raise ValueError("causal beta minimum sample count is invalid")
        if (
            not math.isfinite(self.minimum_market_variance)
            or self.minimum_market_variance <= 0.0
        ):
            raise ValueError("causal beta minimum variance must be positive")
        if (
            not math.isfinite(self.minimum_beta)
            or not math.isfinite(self.maximum_beta)
            or self.minimum_beta >= self.maximum_beta
            or not self.minimum_beta <= 1.0 <= self.maximum_beta
        ):
            raise ValueError("causal beta clip bounds are invalid")

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class CausalBetaSeries:
    decision_indices: np.ndarray
    beta: np.ndarray
    available: np.ndarray
    source_start_indices: np.ndarray
    source_end_indices: np.ndarray
    config: CausalBetaConfig
    source_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        indices = _readonly_int(self.decision_indices).reshape(-1)
        beta = _readonly_float(self.beta).reshape(-1)
        available = _readonly_bool(self.available).reshape(-1)
        source_start = _readonly_int(self.source_start_indices).reshape(-1)
        source_end = _readonly_int(self.source_end_indices).reshape(-1)
        expected_shape = indices.shape
        if any(
            value.shape != expected_shape
            for value in (beta, available, source_start, source_end)
        ):
            raise ValueError("causal beta arrays must be decision aligned")
        if not np.isfinite(beta).all():
            raise ValueError("causal beta values must be finite")
        if np.any(beta[available] < self.config.minimum_beta) or np.any(
            beta[available] > self.config.maximum_beta
        ):
            raise ValueError("causal beta value exceeds configured bounds")
        if np.any(source_start[available] < 0) or np.any(source_end[available] < 0):
            raise ValueError("available causal beta rows require source ranges")
        if np.any(source_start[~available] != -1) or np.any(
            source_end[~available] != -1
        ):
            raise ValueError("unavailable causal beta rows must use -1 source ranges")
        _validate_sha256(self.source_digest, field="causal beta source_digest")
        expected = content_and_arrays_digest(
            {
                "config_digest": self.config.digest,
                "schema_version": "causal_alpha_v4_beta_series_v1",
                "source_digest": self.source_digest,
            },
            (
                ("decision_indices", indices),
                ("beta", beta),
                ("available", available),
                ("source_start_indices", source_start),
                ("source_end_indices", source_end),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal beta series digest mismatch")
        object.__setattr__(self, "decision_indices", indices)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "source_start_indices", source_start)
        object.__setattr__(self, "source_end_indices", source_end)
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class V4TargetContext:
    symbol: str
    local: V4ContextBlock
    global_market: V4ContextBlock
    beta: np.ndarray
    beta_available: np.ndarray
    beta_source_digest: str
    profile_name: str
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or not self.profile_name:
            raise ValueError("V4 target context identity must be non-empty")
        if not isinstance(self.local, V4ContextBlock) or not isinstance(
            self.global_market, V4ContextBlock
        ):
            raise TypeError("V4 target context requires local/global blocks")
        if not np.array_equal(
            self.local.decision_indices, self.global_market.decision_indices
        ):
            raise ValueError("V4 target local/global decision indices must align")
        beta = _readonly_float(self.beta).reshape(-1)
        beta_available = _readonly_bool(self.beta_available).reshape(-1)
        if (
            beta.shape != self.local.decision_indices.shape
            or beta_available.shape != beta.shape
        ):
            raise ValueError("V4 target beta must align to context rows")
        if not np.isfinite(beta).all() or np.any(beta[beta_available] < -3.0) or np.any(
            beta[beta_available] > 3.0
        ):
            raise ValueError("V4 target beta values are invalid")
        if self.symbol == "BTCUSDT" and np.any(beta[beta_available] != 1.0):
            raise ValueError("BTCUSDT available beta must be exactly one")
        _validate_sha256(self.beta_source_digest, field="V4 beta source_digest")
        expected = content_and_arrays_digest(
            {
                "beta_source_digest": self.beta_source_digest,
                "global_context_digest": self.global_market.digest,
                "local_context_digest": self.local.digest,
                "profile_name": self.profile_name,
                "schema_version": "causal_alpha_v4_target_context_v1",
                "symbol": self.symbol,
            },
            (("beta", beta), ("beta_available", beta_available)),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 target context digest mismatch")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "beta_available", beta_available)
        object.__setattr__(self, "digest", expected)

    def policy_row_digest(self, row: int) -> str:
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("row must be an integer")
        if not 0 <= row < len(self.beta):
            raise IndexError("row is outside V4 target context")
        return content_and_arrays_digest(
            {
                "context_digest": self.digest,
                "decision_index": int(self.local.decision_indices[row]),
                "profile_name": self.profile_name,
                "schema_version": "causal_alpha_v4_policy_context_row_v1",
                "symbol": self.symbol,
            },
            (
                ("local_values", self.local.values[row : row + 1]),
                ("local_available", self.local.available[row : row + 1]),
                (
                    "local_staleness_hours",
                    self.local.staleness_hours[row : row + 1],
                ),
                ("global_values", self.global_market.values[row : row + 1]),
                ("global_available", self.global_market.available[row : row + 1]),
                (
                    "global_staleness_hours",
                    self.global_market.staleness_hours[row : row + 1],
                ),
                ("beta", self.beta[row : row + 1]),
                ("beta_available", self.beta_available[row : row + 1]),
            ),
        )


def taker_quote_imbalance(taker_buy_quote: float, total_quote: float) -> float:
    if not math.isfinite(taker_buy_quote) or taker_buy_quote < 0.0:
        raise ValueError("taker_buy_quote must be finite and non-negative")
    if not math.isfinite(total_quote) or total_quote <= 0.0:
        raise ValueError("total_quote must be finite and positive")
    if taker_buy_quote > total_quote + _EPSILON:
        raise ValueError("taker_buy_quote cannot exceed total_quote")
    return float(np.clip(2.0 * taker_buy_quote / total_quote - 1.0, -1.0, 1.0))


def spot_perp_log_basis(*, spot: float, perp: float) -> float:
    if (
        not math.isfinite(spot)
        or not math.isfinite(perp)
        or spot <= 0.0
        or perp <= 0.0
    ):
        raise ValueError("basis prices must be finite and positive")
    return math.log(perp / spot)


def robust_trailing_zscore(
    values: object,
    *,
    available: object,
    window: int,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(available, dtype=np.bool_).reshape(-1)
    if raw.size == 0 or mask.shape != raw.shape or not np.isfinite(raw).all():
        raise ValueError("robust z-score inputs must be aligned finite vectors")
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or window <= 0
        or isinstance(minimum_support, bool)
        or not isinstance(minimum_support, int)
        or not 1 <= minimum_support <= window
    ):
        raise ValueError("robust z-score window/support is invalid")
    result = np.zeros(raw.shape, dtype=np.float64)
    result_available = np.zeros(raw.shape, dtype=np.bool_)
    for index in range(len(raw)):
        start = max(0, index - window + 1)
        selected = raw[start : index + 1][mask[start : index + 1]]
        if selected.size < minimum_support or not mask[index]:
            continue
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        scale = 1.4826 * mad
        result[index] = 0.0 if scale <= _EPSILON else (raw[index] - median) / scale
        result_available[index] = True
    return result, result_available


def build_funding_context_series(
    *,
    decision_timestamps: object,
    funding_event_rate: object,
    funding_event_available: object,
    maximum_staleness_hours: float = 24.0,
    z_window_hours: float = 168.0,
    minimum_z_events: int = 8,
) -> FundingContextSeries:
    timestamps = np.asarray(decision_timestamps)
    rates = np.asarray(funding_event_rate, dtype=np.float64).reshape(-1)
    events = np.asarray(funding_event_available, dtype=np.bool_).reshape(-1)
    if (
        timestamps.ndim != 1
        or not np.issubdtype(timestamps.dtype, np.datetime64)
        or rates.shape != timestamps.shape
        or events.shape != timestamps.shape
        or not np.isfinite(rates).all()
    ):
        raise ValueError("funding context inputs must be aligned and finite")
    timestamps = timestamps.astype("datetime64[ns]")
    timestamp_ns = timestamps.astype(np.int64)
    if np.any(timestamp_ns == np.iinfo(np.int64).min) or np.any(
        np.diff(timestamp_ns) <= 0
    ):
        raise ValueError("funding timestamps must be valid and increasing")
    if (
        not math.isfinite(maximum_staleness_hours)
        or maximum_staleness_hours <= 0.0
        or not math.isfinite(z_window_hours)
        or z_window_hours <= 0.0
        or isinstance(minimum_z_events, bool)
        or not isinstance(minimum_z_events, int)
        or minimum_z_events < 2
    ):
        raise ValueError("funding context configuration is invalid")

    out_rate = np.zeros(rates.shape, dtype=np.float64)
    out_change = np.zeros(rates.shape, dtype=np.float64)
    out_z = np.zeros(rates.shape, dtype=np.float64)
    available = np.zeros((len(rates), 3), dtype=np.bool_)
    staleness = np.full((len(rates), 3), maximum_staleness_hours, dtype=np.float64)
    event_indices: list[int] = []
    z_window_ns = int(round(z_window_hours * _NS_PER_HOUR))
    for index in range(len(rates)):
        if events[index]:
            event_indices.append(index)
        if not event_indices:
            continue
        latest = event_indices[-1]
        age_hours = float(timestamp_ns[index] - timestamp_ns[latest]) / _NS_PER_HOUR
        if age_hours > maximum_staleness_hours + _EPSILON:
            continue
        out_rate[index] = rates[latest]
        available[index, 0] = True
        staleness[index, 0] = age_hours
        if len(event_indices) >= 2:
            previous = event_indices[-2]
            out_change[index] = rates[latest] - rates[previous]
            available[index, 1] = True
            staleness[index, 1] = age_hours
        minimum_time = timestamp_ns[index] - z_window_ns
        window_indices = [
            event_index
            for event_index in event_indices
            if timestamp_ns[event_index] >= minimum_time
        ]
        if len(window_indices) >= minimum_z_events:
            sample = rates[np.asarray(window_indices, dtype=np.int64)]
            median = float(np.median(sample))
            mad = float(np.median(np.abs(sample - median)))
            scale = 1.4826 * mad
            out_z[index] = (
                0.0 if scale <= _EPSILON else (rates[latest] - median) / scale
            )
            available[index, 2] = True
            staleness[index, 2] = age_hours
    return FundingContextSeries(
        rate=out_rate,
        change=out_change,
        robust_z_7d=out_z,
        available=available,
        staleness_hours=staleness,
    )


def _window_complete(mask: np.ndarray, *, bars: int) -> np.ndarray:
    if bars <= 0:
        raise ValueError("window bars must be positive")
    invalid = (~mask).astype(np.int64)
    cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(invalid)))
    result = np.zeros(mask.shape, dtype=np.bool_)
    for index in range(bars - 1, len(mask)):
        start = index - bars + 1
        result[index] = cumulative[index + 1] - cumulative[start] == 0
    return result


def _log_return(
    values: np.ndarray, available: np.ndarray, *, bars: int
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros(values.shape, dtype=np.float64)
    complete = _window_complete(available, bars=bars + 1)
    indices = np.flatnonzero(complete & (np.arange(len(values)) >= bars))
    if indices.size:
        starts = indices - bars
        result[indices] = np.log(values[indices] / values[starts])
    return result, complete & (np.arange(len(values)) >= bars)


def _window_sum(
    values: np.ndarray, available: np.ndarray, *, bars: int
) -> tuple[np.ndarray, np.ndarray]:
    complete = _window_complete(available, bars=bars)
    prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values)))
    result = np.zeros(values.shape, dtype=np.float64)
    indices = np.flatnonzero(complete)
    if indices.size:
        starts = indices - bars + 1
        result[indices] = prefix[indices + 1] - prefix[starts]
    return result, complete


def _window_max(
    values: np.ndarray, available: np.ndarray, *, bars: int
) -> tuple[np.ndarray, np.ndarray]:
    complete = _window_complete(available, bars=bars)
    result = np.full(
        values.shape, _UNAVAILABLE_STALENESS_HOURS, dtype=np.float64
    )
    for index in np.flatnonzero(complete):
        start = index - bars + 1
        result[index] = float(np.max(values[start : index + 1]))
    return result, complete


def _difference(
    values: np.ndarray, available: np.ndarray, *, bars: int
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros(values.shape, dtype=np.float64)
    complete = _window_complete(available, bars=bars + 1)
    indices = np.flatnonzero(complete & (np.arange(len(values)) >= bars))
    if indices.size:
        result[indices] = values[indices] - values[indices - bars]
    return result, complete & (np.arange(len(values)) >= bars)


def _rolling_imbalance(
    taker_buy: np.ndarray,
    total_quote: np.ndarray,
    available: np.ndarray,
    *,
    bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    taker_sum, complete = _window_sum(taker_buy, available, bars=bars)
    quote_sum, quote_complete = _window_sum(total_quote, available, bars=bars)
    usable = complete & quote_complete & (quote_sum > _EPSILON)
    result = np.zeros(total_quote.shape, dtype=np.float64)
    result[usable] = np.clip(
        2.0 * taker_sum[usable] / quote_sum[usable] - 1.0,
        -1.0,
        1.0,
    )
    return result, usable


def _volume_ratio(
    left_volume: np.ndarray,
    left_available: np.ndarray,
    right_volume: np.ndarray,
    right_available: np.ndarray,
    *,
    bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    left, left_ok = _window_sum(left_volume, left_available, bars=bars)
    right, right_ok = _window_sum(right_volume, right_available, bars=bars)
    usable = left_ok & right_ok & (left > _EPSILON) & (right > _EPSILON)
    result = np.zeros(left.shape, dtype=np.float64)
    result[usable] = np.log(left[usable] / right[usable])
    return result, usable


def _feature(
    values: np.ndarray,
    available: np.ndarray,
    *,
    staleness: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_out = np.where(available, values, 0.0).astype(np.float64, copy=False)
    if isinstance(staleness, np.ndarray):
        staleness_values = np.asarray(staleness, dtype=np.float64)
    else:
        staleness_values = np.full(values.shape, float(staleness), dtype=np.float64)
    stale_out = np.where(
        available, staleness_values, _UNAVAILABLE_STALENESS_HOURS
    ).astype(np.float64, copy=False)
    return values_out, available.astype(np.bool_, copy=False), stale_out


def _local_feature_map(
    inputs: V4CrossMarketInputs,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    spot_available = np.asarray(inputs.spot_row_available, dtype=np.bool_)
    perp_available = np.asarray(inputs.perp_row_available, dtype=np.bool_)
    spot_return_1h, spot_return_1h_ok = _log_return(
        inputs.spot_close, spot_available, bars=BARS_1H
    )
    spot_return_4h, spot_return_4h_ok = _log_return(
        inputs.spot_close, spot_available, bars=BARS_4H
    )
    spot_return_24h, spot_return_24h_ok = _log_return(
        inputs.spot_close, spot_available, bars=BARS_24H
    )
    perp_return_1h, perp_return_1h_ok = _log_return(
        inputs.perp_close, perp_available, bars=BARS_1H
    )
    perp_return_4h, perp_return_4h_ok = _log_return(
        inputs.perp_close, perp_available, bars=BARS_4H
    )
    perp_return_24h, perp_return_24h_ok = _log_return(
        inputs.perp_close, perp_available, bars=BARS_24H
    )

    spot_volume_4h, spot_volume_4h_ok = _window_sum(
        inputs.spot_quote_volume, spot_available, bars=BARS_4H
    )
    spot_volume_24h, spot_volume_24h_ok = _window_sum(
        inputs.spot_quote_volume, spot_available, bars=BARS_24H
    )
    spot_log_volume_4h = np.zeros_like(spot_volume_4h)
    spot_log_volume_24h = np.zeros_like(spot_volume_24h)
    positive_4h = spot_volume_4h_ok & (spot_volume_4h > _EPSILON)
    positive_24h = spot_volume_24h_ok & (spot_volume_24h > _EPSILON)
    spot_log_volume_4h[positive_4h] = np.log(spot_volume_4h[positive_4h])
    spot_log_volume_24h[positive_24h] = np.log(spot_volume_24h[positive_24h])
    spot_volume_z_4h, spot_volume_z_4h_ok = robust_trailing_zscore(
        spot_log_volume_4h,
        available=positive_4h,
        window=BARS_7D,
        minimum_support=32,
    )
    spot_volume_z_24h, spot_volume_z_24h_ok = robust_trailing_zscore(
        spot_log_volume_24h,
        available=positive_24h,
        window=BARS_7D,
        minimum_support=32,
    )

    basis_available = spot_available & perp_available
    basis = np.zeros(len(spot_available), dtype=np.float64)
    basis[basis_available] = np.log(
        inputs.perp_mark_price[basis_available] / inputs.spot_close[basis_available]
    )
    basis_change_1h, basis_change_1h_ok = _difference(
        basis, basis_available, bars=BARS_1H
    )
    basis_change_4h, basis_change_4h_ok = _difference(
        basis, basis_available, bars=BARS_4H
    )
    basis_z, basis_z_ok = robust_trailing_zscore(
        basis, available=basis_available, window=BARS_7D, minimum_support=32
    )

    spot_minus_perp_1h_ok = spot_return_1h_ok & perp_return_1h_ok
    spot_minus_perp_1h = spot_return_1h - perp_return_1h
    spot_minus_perp_4h_ok = spot_return_4h_ok & perp_return_4h_ok
    spot_minus_perp_4h = spot_return_4h - perp_return_4h

    volume_ratio_1h, volume_ratio_1h_ok = _volume_ratio(
        inputs.spot_quote_volume,
        spot_available,
        inputs.perp_quote_volume,
        perp_available,
        bars=BARS_1H,
    )
    volume_ratio_4h, volume_ratio_4h_ok = _volume_ratio(
        inputs.spot_quote_volume,
        spot_available,
        inputs.perp_quote_volume,
        perp_available,
        bars=BARS_4H,
    )
    volume_ratio_24h, volume_ratio_24h_ok = _volume_ratio(
        inputs.spot_quote_volume,
        spot_available,
        inputs.perp_quote_volume,
        perp_available,
        bars=BARS_24H,
    )

    spot_imbalance_1h, spot_imbalance_1h_ok = _rolling_imbalance(
        inputs.spot_taker_buy_quote_volume,
        inputs.spot_quote_volume,
        spot_available,
        bars=BARS_1H,
    )
    spot_imbalance_4h, spot_imbalance_4h_ok = _rolling_imbalance(
        inputs.spot_taker_buy_quote_volume,
        inputs.spot_quote_volume,
        spot_available,
        bars=BARS_4H,
    )
    perp_imbalance_1h, perp_imbalance_1h_ok = _rolling_imbalance(
        inputs.perp_taker_buy_quote_volume,
        inputs.perp_quote_volume,
        perp_available,
        bars=BARS_1H,
    )
    perp_imbalance_4h, perp_imbalance_4h_ok = _rolling_imbalance(
        inputs.perp_taker_buy_quote_volume,
        inputs.perp_quote_volume,
        perp_available,
        bars=BARS_4H,
    )
    flow_divergence_1h = spot_imbalance_1h - perp_imbalance_1h
    flow_divergence_1h_ok = spot_imbalance_1h_ok & perp_imbalance_1h_ok
    flow_divergence_4h = spot_imbalance_4h - perp_imbalance_4h
    flow_divergence_4h_ok = spot_imbalance_4h_ok & perp_imbalance_4h_ok

    funding = build_funding_context_series(
        decision_timestamps=inputs.decision_timestamps,
        funding_event_rate=inputs.funding_event_rate,
        funding_event_available=inputs.funding_event_available,
    )
    basis_flow = basis_z * flow_divergence_4h
    basis_flow_ok = basis_z_ok & flow_divergence_4h_ok

    feature_map = {
        "spot_log_return_1h": _feature(spot_return_1h, spot_return_1h_ok),
        "spot_log_return_4h": _feature(spot_return_4h, spot_return_4h_ok),
        "spot_log_return_24h": _feature(spot_return_24h, spot_return_24h_ok),
        "spot_log_quote_volume_robust_z_4h": _feature(
            spot_volume_z_4h, spot_volume_z_4h_ok
        ),
        "spot_log_quote_volume_robust_z_24h": _feature(
            spot_volume_z_24h, spot_volume_z_24h_ok
        ),
        "spot_perp_log_basis": _feature(basis, basis_available),
        "spot_perp_basis_change_1h": _feature(
            basis_change_1h, basis_change_1h_ok
        ),
        "spot_perp_basis_change_4h": _feature(
            basis_change_4h, basis_change_4h_ok
        ),
        "spot_perp_basis_robust_z_7d": _feature(basis_z, basis_z_ok),
        "spot_minus_perp_log_return_1h": _feature(
            spot_minus_perp_1h, spot_minus_perp_1h_ok
        ),
        "spot_minus_perp_log_return_4h": _feature(
            spot_minus_perp_4h, spot_minus_perp_4h_ok
        ),
        "spot_to_perp_log_quote_volume_ratio_1h": _feature(
            volume_ratio_1h, volume_ratio_1h_ok
        ),
        "spot_to_perp_log_quote_volume_ratio_4h": _feature(
            volume_ratio_4h, volume_ratio_4h_ok
        ),
        "spot_to_perp_log_quote_volume_ratio_24h": _feature(
            volume_ratio_24h, volume_ratio_24h_ok
        ),
        "spot_taker_quote_imbalance_1h": _feature(
            spot_imbalance_1h, spot_imbalance_1h_ok
        ),
        "spot_taker_quote_imbalance_4h": _feature(
            spot_imbalance_4h, spot_imbalance_4h_ok
        ),
        "perp_taker_quote_imbalance_1h": _feature(
            perp_imbalance_1h, perp_imbalance_1h_ok
        ),
        "perp_taker_quote_imbalance_4h": _feature(
            perp_imbalance_4h, perp_imbalance_4h_ok
        ),
        "spot_minus_perp_taker_imbalance_1h": _feature(
            flow_divergence_1h, flow_divergence_1h_ok
        ),
        "spot_minus_perp_taker_imbalance_4h": _feature(
            flow_divergence_4h, flow_divergence_4h_ok
        ),
        "funding_rate": _feature(
            funding.rate,
            funding.available[:, 0],
            staleness=funding.staleness_hours[:, 0],
        ),
        "funding_rate_change": _feature(
            funding.change,
            funding.available[:, 1],
            staleness=funding.staleness_hours[:, 1],
        ),
        "funding_rate_robust_z_7d": _feature(
            funding.robust_z_7d,
            funding.available[:, 2],
            staleness=funding.staleness_hours[:, 2],
        ),
        "basis_z_x_flow_divergence_4h": _feature(basis_flow, basis_flow_ok),
        "_perp_return_1h": _feature(perp_return_1h, perp_return_1h_ok),
        "_perp_return_4h": _feature(perp_return_4h, perp_return_4h_ok),
        "_perp_return_24h": _feature(perp_return_24h, perp_return_24h_ok),
    }

    if inputs.open_interest_value is not None:
        assert inputs.derivatives_available is not None
        assert inputs.derivatives_staleness_hours is not None
        assert inputs.global_long_short_ratio is not None
        assert inputs.top_position_long_short_ratio is not None
        derivative_available = np.asarray(inputs.derivatives_available, dtype=np.bool_)
        derivative_age = np.asarray(
            inputs.derivatives_staleness_hours, dtype=np.float64
        )
        oi_1h, oi_1h_ok = _log_return(
            inputs.open_interest_value, derivative_available, bars=BARS_1H
        )
        oi_4h, oi_4h_ok = _log_return(
            inputs.open_interest_value, derivative_available, bars=BARS_4H
        )
        oi_24h, oi_24h_ok = _log_return(
            inputs.open_interest_value, derivative_available, bars=BARS_24H
        )
        oi_age_1h, _ = _window_max(
            derivative_age, derivative_available, bars=BARS_1H + 1
        )
        oi_age_4h, _ = _window_max(
            derivative_age, derivative_available, bars=BARS_4H + 1
        )
        oi_age_24h, _ = _window_max(
            derivative_age, derivative_available, bars=BARS_24H + 1
        )
        global_ls_z, global_ls_z_ok = robust_trailing_zscore(
            inputs.global_long_short_ratio,
            available=derivative_available,
            window=BARS_4H,
            minimum_support=8,
        )
        top_ls_z, top_ls_z_ok = robust_trailing_zscore(
            inputs.top_position_long_short_ratio,
            available=derivative_available,
            window=BARS_4H,
            minimum_support=8,
        )
        ratio_age, _ = _window_max(
            derivative_age, derivative_available, bars=BARS_4H
        )
        basis_oi_ok = basis_z_ok & oi_4h_ok
        funding_oi_ok = funding.available[:, 2] & oi_4h_ok
        feature_map.update(
            {
                "open_interest_log_change_1h": _feature(
                    oi_1h, oi_1h_ok, staleness=oi_age_1h
                ),
                "open_interest_log_change_4h": _feature(
                    oi_4h, oi_4h_ok, staleness=oi_age_4h
                ),
                "open_interest_log_change_24h": _feature(
                    oi_24h, oi_24h_ok, staleness=oi_age_24h
                ),
                "global_long_short_ratio_robust_z_4h": _feature(
                    global_ls_z, global_ls_z_ok, staleness=ratio_age
                ),
                "top_position_long_short_ratio_robust_z_4h": _feature(
                    top_ls_z, top_ls_z_ok, staleness=ratio_age
                ),
                "basis_z_x_open_interest_change_4h": _feature(
                    basis_z * oi_4h,
                    basis_oi_ok,
                    staleness=np.maximum(0.0, oi_age_4h),
                ),
                "funding_z_x_open_interest_change_4h": _feature(
                    funding.robust_z_7d * oi_4h,
                    funding_oi_ok,
                    staleness=np.maximum(funding.staleness_hours[:, 2], oi_age_4h),
                ),
            }
        )
    return feature_map


def _assemble_block(
    *,
    names: tuple[str, ...],
    decision_indices: np.ndarray,
    feature_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_digest: str,
) -> V4ContextBlock:
    missing = tuple(name for name in names if name not in feature_map)
    if missing:
        raise ValueError(f"V4 context source cannot build features: {missing}")
    values = np.column_stack(tuple(feature_map[name][0] for name in names))
    available = np.column_stack(tuple(feature_map[name][1] for name in names))
    staleness = np.column_stack(tuple(feature_map[name][2] for name in names))
    return V4ContextBlock(
        feature_names=names,
        decision_indices=decision_indices,
        values=values,
        available=available,
        staleness_hours=staleness,
        source_digest=source_digest,
    )


def build_cross_market_context(
    inputs: V4CrossMarketInputs, *, include_derivatives: bool
) -> V4ContextBlock:
    if not isinstance(inputs, V4CrossMarketInputs):
        raise TypeError("V4 cross-market context requires V4CrossMarketInputs")
    if include_derivatives and inputs.open_interest_value is None:
        raise ValueError("V4 derivative context requires complete derivative inputs")
    names = (
        (*CROSS_MARKET_CORE_NAMES, *CROSS_MARKET_DERIVATIVE_NAMES)
        if include_derivatives
        else CROSS_MARKET_CORE_NAMES
    )
    return _assemble_block(
        names=tuple(names),
        decision_indices=inputs.decision_indices,
        feature_map=_local_feature_map(inputs),
        source_digest=inputs.source_digest,
    )


def _global_anchor_map(
    inputs: V4CrossMarketInputs, *, prefix: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    local = _local_feature_map(inputs)
    result = {
        f"{prefix}_spot_log_return_1h": local["spot_log_return_1h"],
        f"{prefix}_spot_log_return_4h": local["spot_log_return_4h"],
        f"{prefix}_spot_log_return_24h": local["spot_log_return_24h"],
        f"{prefix}_perp_log_return_1h": local["_perp_return_1h"],
        f"{prefix}_perp_log_return_4h": local["_perp_return_4h"],
        f"{prefix}_perp_log_return_24h": local["_perp_return_24h"],
        f"{prefix}_spot_perp_log_basis": local["spot_perp_log_basis"],
        f"{prefix}_spot_perp_basis_change_4h": local["spot_perp_basis_change_4h"],
        f"{prefix}_spot_perp_basis_robust_z_7d": local[
            "spot_perp_basis_robust_z_7d"
        ],
        f"{prefix}_spot_taker_quote_imbalance_1h": local[
            "spot_taker_quote_imbalance_1h"
        ],
        f"{prefix}_spot_taker_quote_imbalance_4h": local[
            "spot_taker_quote_imbalance_4h"
        ],
        f"{prefix}_perp_taker_quote_imbalance_1h": local[
            "perp_taker_quote_imbalance_1h"
        ],
        f"{prefix}_perp_taker_quote_imbalance_4h": local[
            "perp_taker_quote_imbalance_4h"
        ],
        f"{prefix}_spot_to_perp_log_quote_volume_ratio_4h": local[
            "spot_to_perp_log_quote_volume_ratio_4h"
        ],
        f"{prefix}_spot_to_perp_log_quote_volume_ratio_24h": local[
            "spot_to_perp_log_quote_volume_ratio_24h"
        ],
        f"{prefix}_funding_rate": local["funding_rate"],
        f"{prefix}_funding_rate_robust_z_7d": local["funding_rate_robust_z_7d"],
    }
    if inputs.open_interest_value is not None:
        result.update(
            {
                f"{prefix}_open_interest_log_change_4h": local[
                    "open_interest_log_change_4h"
                ],
                f"{prefix}_open_interest_log_change_24h": local[
                    "open_interest_log_change_24h"
                ],
                f"{prefix}_global_long_short_ratio_robust_z_4h": local[
                    "global_long_short_ratio_robust_z_4h"
                ],
            }
        )
    return result


def _combine_two(
    left: tuple[np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    operation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_value, left_available, left_staleness = left
    right_value, right_available, right_staleness = right
    available = left_available & right_available
    if operation == "difference":
        value = left_value - right_value
    elif operation == "dispersion":
        value = 0.5 * np.abs(left_value - right_value)
    else:
        raise ValueError("unsupported V4 two-anchor operation")
    return _feature(value, available, staleness=np.maximum(left_staleness, right_staleness))


def build_global_market_context(
    inputs: V4GlobalMarketInputs, *, include_derivatives: bool
) -> V4ContextBlock:
    if not isinstance(inputs, V4GlobalMarketInputs):
        raise TypeError("V4 global context requires V4GlobalMarketInputs")
    if include_derivatives and (
        inputs.btc.open_interest_value is None
        or inputs.eth.open_interest_value is None
    ):
        raise ValueError("V4 global derivative context requires BTC and ETH derivatives")
    btc = _global_anchor_map(inputs.btc, prefix="btc")
    eth = _global_anchor_map(inputs.eth, prefix="eth")
    feature_map = {**btc, **eth}
    feature_map["btc_minus_eth_perp_return_4h"] = _combine_two(
        btc["btc_perp_log_return_4h"],
        eth["eth_perp_log_return_4h"],
        operation="difference",
    )
    feature_map["btc_minus_eth_perp_return_24h"] = _combine_two(
        btc["btc_perp_log_return_24h"],
        eth["eth_perp_log_return_24h"],
        operation="difference",
    )
    feature_map["btc_minus_eth_basis"] = _combine_two(
        btc["btc_spot_perp_log_basis"],
        eth["eth_spot_perp_log_basis"],
        operation="difference",
    )
    feature_map["btc_eth_perp_return_dispersion_4h"] = _combine_two(
        btc["btc_perp_log_return_4h"],
        eth["eth_perp_log_return_4h"],
        operation="dispersion",
    )
    names = (
        (*GLOBAL_MARKET_CORE_NAMES, *GLOBAL_MARKET_DERIVATIVE_NAMES)
        if include_derivatives
        else GLOBAL_MARKET_CORE_NAMES
    )
    source_digest = content_digest(
        {
            "btc_source_digest": inputs.btc.source_digest,
            "eth_source_digest": inputs.eth.source_digest,
            "global_source_digest": inputs.source_digest,
            "schema_version": "causal_alpha_v4_global_source_v1",
        }
    )
    return _assemble_block(
        names=tuple(names),
        decision_indices=inputs.btc.decision_indices,
        feature_map=feature_map,
        source_digest=source_digest,
    )


def build_causal_beta_series(
    *,
    symbol: str,
    decision_indices: object,
    target_close: object,
    btc_close: object,
    target_row_available: object,
    btc_row_available: object,
    bars_per_4h: int,
    config: CausalBetaConfig,
    target_source_digest: str,
    btc_source_digest: str,
) -> CausalBetaSeries:
    if not symbol:
        raise ValueError("causal beta symbol must be non-empty")
    indices = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    target = np.asarray(target_close, dtype=np.float64).reshape(-1)
    btc = np.asarray(btc_close, dtype=np.float64).reshape(-1)
    target_available = np.asarray(target_row_available, dtype=np.bool_).reshape(-1)
    btc_available = np.asarray(btc_row_available, dtype=np.bool_).reshape(-1)
    if (
        indices.size == 0
        or target.shape != indices.shape
        or btc.shape != indices.shape
        or target_available.shape != indices.shape
        or btc_available.shape != indices.shape
        or not np.isfinite(target).all()
        or not np.isfinite(btc).all()
        or np.any(indices < 0)
        or np.any(np.diff(indices) != 1)
    ):
        raise ValueError("causal beta source arrays must be finite and aligned")
    if np.any(target[target_available] <= 0.0) or np.any(btc[btc_available] <= 0.0):
        raise ValueError("causal beta prices must be positive when available")
    if (
        isinstance(bars_per_4h, bool)
        or not isinstance(bars_per_4h, int)
        or bars_per_4h <= 0
    ):
        raise ValueError("bars_per_4h must be a positive integer")
    if not isinstance(config, CausalBetaConfig):
        raise TypeError("causal beta config is invalid")
    _validate_sha256(target_source_digest, field="causal beta target source digest")
    _validate_sha256(btc_source_digest, field="causal beta BTC source digest")

    first_decision = int(indices[0])
    relative = indices - first_decision
    sample_rows = np.flatnonzero(
        (relative >= bars_per_4h) & (relative % bars_per_4h == 0)
    ).astype(np.int64)
    paired_available = target_available & btc_available
    complete_intervals = _window_complete(paired_available, bars=bars_per_4h + 1)
    sample_valid = complete_intervals[sample_rows]
    sample_starts = sample_rows - bars_per_4h
    target_returns = np.zeros(len(sample_rows), dtype=np.float64)
    btc_returns = np.zeros(len(sample_rows), dtype=np.float64)
    if sample_rows.size:
        target_returns = np.log(target[sample_rows] / target[sample_starts])
        btc_returns = np.log(btc[sample_rows] / btc[sample_starts])

    maximum_samples = int(round(config.lookback_hours / config.return_horizon_hours))
    beta = np.zeros(indices.shape, dtype=np.float64)
    available = np.zeros(indices.shape, dtype=np.bool_)
    source_start = np.full(indices.shape, -1, dtype=np.int64)
    source_end = np.full(indices.shape, -1, dtype=np.int64)
    for row in range(len(indices)):
        stop = int(np.searchsorted(sample_rows, row, side="right"))
        start = max(0, stop - maximum_samples)
        if stop <= start:
            continue
        local_valid = sample_valid[start:stop]
        valid_positions = np.flatnonzero(local_valid) + start
        if valid_positions.size < config.minimum_complete_samples:
            continue
        x = btc_returns[valid_positions]
        y = target_returns[valid_positions]
        if symbol == "BTCUSDT":
            resolved = 1.0
        else:
            x_centered = x - float(x.mean(dtype=np.float64))
            y_centered = y - float(y.mean(dtype=np.float64))
            market_variance = float(np.mean(np.square(x_centered), dtype=np.float64))
            if market_variance < config.minimum_market_variance:
                continue
            covariance = float(np.mean(x_centered * y_centered, dtype=np.float64))
            resolved = float(
                np.clip(
                    covariance / market_variance,
                    config.minimum_beta,
                    config.maximum_beta,
                )
            )
        first_sample = int(valid_positions[0])
        last_sample = int(valid_positions[-1])
        beta[row] = resolved
        available[row] = True
        source_start[row] = int(indices[sample_starts[first_sample]])
        source_end[row] = int(indices[sample_rows[last_sample]])

    source_digest = content_and_arrays_digest(
        {
            "bars_per_4h": bars_per_4h,
            "btc_source_digest": btc_source_digest,
            "config_digest": config.digest,
            "maximum_sample_count": maximum_samples,
            "schema_version": "causal_alpha_v4_beta_source_v1",
            "symbol": symbol,
            "target_source_digest": target_source_digest,
        },
        (
            ("decision_indices", indices),
            ("target_close", target),
            ("btc_close", btc),
            ("target_row_available", target_available),
            ("btc_row_available", btc_available),
        ),
    )
    return CausalBetaSeries(
        decision_indices=indices,
        beta=beta,
        available=available,
        source_start_indices=source_start,
        source_end_indices=source_end,
        config=config,
        source_digest=source_digest,
    )


__all__ = [
    "BARS_1H",
    "BARS_4H",
    "BARS_24H",
    "BARS_7D",
    "CROSS_MARKET_CORE_NAMES",
    "CROSS_MARKET_DERIVATIVE_NAMES",
    "GLOBAL_MARKET_CORE_NAMES",
    "GLOBAL_MARKET_DERIVATIVE_NAMES",
    "CausalBetaConfig",
    "CausalBetaSeries",
    "FundingContextSeries",
    "V4ContextBlock",
    "V4CrossMarketInputs",
    "V4GlobalMarketInputs",
    "V4TargetContext",
    "build_causal_beta_series",
    "build_cross_market_context",
    "build_funding_context_series",
    "build_global_market_context",
    "robust_trailing_zscore",
    "spot_perp_log_basis",
    "taker_quote_imbalance",
]
