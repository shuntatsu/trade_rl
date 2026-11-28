"""Causal rolling inputs for advanced portfolio-risk constraints."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Protocol, cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data._timestamps import (
    _NANOSECONDS_PER_DAY,
    _datetime64_values_to_ns_exact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_clock import NativeBarClock, _native_unmasked_array


class PortfolioRiskPriceHistory(Protocol):
    """Completed close prices and availability on the caller-owned clock."""

    @property
    def n_bars(self) -> int: ...

    @property
    def n_symbols(self) -> int: ...

    @property
    def close(self) -> np.ndarray: ...

    @property
    def timestamps(self) -> np.ndarray: ...

    @property
    def available_at(self) -> np.ndarray | None: ...


class PortfolioRiskDataset(PortfolioRiskPriceHistory, Protocol):
    @property
    def periods_per_year(self) -> int | Fraction: ...


class PortfolioRiskInputsProvider(Protocol):
    @property
    def identity_digest(self) -> str: ...

    @property
    def minimum_index(self) -> int: ...

    def inputs(
        self, dataset: PortfolioRiskDataset, *, index: int
    ) -> PortfolioRiskInputs: ...


def _readonly_array(
    value: np.ndarray, *, shape: tuple[int, ...], field: str
) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if contiguous.shape != shape or not np.isfinite(contiguous).all():
        raise ValueError(f"{field} must be finite with shape {shape}")
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class PortfolioRiskInputs:
    covariance: np.ndarray
    beta: np.ndarray
    stress_losses: np.ndarray
    as_of_index: int
    provider_digest: str
    observation_count: int
    schema_version: str = "portfolio_risk_inputs_v1"
    digest: str = ""

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta, dtype=np.float64).reshape(-1)
        n_symbols = int(beta.size)
        if n_symbols <= 0:
            raise ValueError("portfolio risk inputs require at least one symbol")
        covariance = _readonly_array(
            self.covariance,
            shape=(n_symbols, n_symbols),
            field="covariance",
        )
        beta = _readonly_array(beta, shape=(n_symbols,), field="beta")
        stress = _readonly_array(
            self.stress_losses,
            shape=(n_symbols,),
            field="stress_losses",
        )
        if np.any(stress > 0.0):
            raise ValueError("stress_losses must be non-positive return shocks")
        if isinstance(self.as_of_index, bool) or not isinstance(self.as_of_index, int):
            raise ValueError("as_of_index must be an integer")
        if self.as_of_index < 0:
            raise ValueError("as_of_index must be non-negative")
        if (
            isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count < 2
        ):
            raise ValueError("observation_count must be at least two")
        require_sha256(self.provider_digest, field="provider_digest")
        if self.schema_version != "portfolio_risk_inputs_v1":
            raise ValueError("unsupported portfolio risk inputs schema")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "stress_losses", stress)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("portfolio risk inputs digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "as_of_index": self.as_of_index,
            "beta": tuple(float(value) for value in self.beta),
            "covariance": tuple(
                tuple(float(value) for value in row) for row in self.covariance
            ),
            "observation_count": self.observation_count,
            "provider_digest": self.provider_digest,
            "schema_version": self.schema_version,
            "stress_losses": tuple(float(value) for value in self.stress_losses),
        }


@dataclass(frozen=True, slots=True)
class RollingPortfolioRiskInputsConfig:
    lookback_bars: int = 96
    minimum_observations: int = 20
    benchmark_index: int = 0
    stress_quantile: float = 0.05
    annualize_covariance: bool = True
    schema_version: str = "rolling_portfolio_risk_inputs_v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("lookback_bars", self.lookback_bars),
            ("minimum_observations", self.minimum_observations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if self.minimum_observations > self.lookback_bars:
            raise ValueError("minimum_observations cannot exceed lookback_bars")
        if (
            isinstance(self.benchmark_index, bool)
            or not isinstance(self.benchmark_index, int)
            or self.benchmark_index < 0
        ):
            raise ValueError("benchmark_index must be a non-negative integer")
        if (
            not math.isfinite(self.stress_quantile)
            or not 0.0 < self.stress_quantile < 0.5
        ):
            raise ValueError("stress_quantile must be within (0, 0.5)")
        if not isinstance(self.annualize_covariance, bool):
            raise ValueError("annualize_covariance must be a boolean")
        if self.schema_version != "rolling_portfolio_risk_inputs_v1":
            raise ValueError("unsupported rolling portfolio risk inputs schema")

    @property
    def digest(self) -> str:
        return content_digest(asdict(self))


def _sample_covariance(log_returns: np.ndarray) -> np.ndarray:
    centered = log_returns - np.mean(log_returns, axis=0, keepdims=True)
    return centered.T @ centered / float(log_returns.shape[0] - 1)


def _native_annual_covariance(
    log_returns: np.ndarray, durations_ns: tuple[int, ...]
) -> np.ndarray:
    """Annual sample covariance under a linear elapsed-time drift model.

    For intervals t_i, subtract their shares of the total log return.
    Effective exposure T - sum(t_i**2)/T corrects for fitting the drift
    under an independent-increment constant covariance model. Equal durations
    recover ordinary sample covariance at the unrounded annual frequency.
    This model does not establish independence or equal-duration stress.
    """
    year_ns = 365 * _NANOSECONDS_PER_DAY
    if all(duration == durations_ns[0] for duration in durations_ns):
        return _sample_covariance(log_returns) * float(
            Fraction(year_ns, durations_ns[0])
        )
    total_ns = sum(durations_ns)
    fractions = np.asarray(durations_ns, dtype=np.float64) / float(total_ns)
    centered = log_returns - fractions[:, None] * np.sum(log_returns, axis=0)
    effective_ns = Fraction(total_ns) - Fraction(
        sum(d * d for d in durations_ns), total_ns
    )
    return centered.T @ centered / float(effective_ns / year_ns)


def _native_risk_window(
    dataset: PortfolioRiskPriceHistory,
    *,
    clock: NativeBarClock,
    window: slice,
    observations: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Read only the completed close window and its publication timestamps."""
    timestamps = _datetime64_values_to_ns_exact(
        _native_unmasked_array(dataset.timestamps, field_name="risk timestamps")[window]
    )
    if timestamps.shape != (observations + 1,):
        raise ValueError("portfolio risk native timestamps must match history")
    starts, ends = clock.interval_bounds(timestamps[1:])
    if timestamps[0] != starts[0]:
        raise ValueError("portfolio risk native closes must be consecutive")
    available_at = dataset.available_at
    if available_at is None:
        raise ValueError("portfolio risk prices not available at decision time")
    publication = _datetime64_values_to_ns_exact(
        _native_unmasked_array(available_at, field_name="risk publication")[window]
    )
    if (
        publication.shape != (observations + 1, dataset.n_symbols)
        or np.any(publication < timestamps[:, None])
        or np.any(publication > ends[-1])
    ):
        raise ValueError("portfolio risk prices not available at decision time")
    raw_prices = _native_unmasked_array(dataset.close, field_name="risk closes")[window]
    if raw_prices.dtype.kind not in "fiu":
        raise ValueError("portfolio risk native closes must be real numbers")
    # Do not let narrowing erase a distinct source price or hide invalid
    # history. Inspect only the consumed window, never future row values.
    if raw_prices.dtype.itemsize > 8 or (
        raw_prices.dtype.kind in "iu"
        and raw_prices.dtype.itemsize == 8
        and (
            np.any(raw_prices > 2**53)
            or (raw_prices.dtype.kind == "i" and np.any(raw_prices < -(2**53)))
        )
    ):
        raise ValueError("portfolio risk native closes must be binary64-compatible")
    prices = np.asarray(raw_prices, dtype=np.float64)
    if prices.shape != (observations + 1, dataset.n_symbols):
        raise ValueError("portfolio risk closes must match native history")
    if not np.isfinite(prices).all():
        raise ValueError("portfolio risk native closes must be finite")
    durations_ns = tuple(
        int(end) - int(begin)
        for begin, end in zip(starts.view(np.int64), ends.view(np.int64), strict=True)
    )
    return prices, durations_ns


class RollingPortfolioRiskInputsProvider:
    """Compute risk inputs from completed close-to-close returns through one index.

    With an explicit native clock, validate the consumed close/publication grid
    and estimate annual covariance using actual ACT/365F interval durations.
    The legacy provider identity and nominal-frequency calculation are unchanged.
    Nonannualized covariance and stress shocks remain per observed bar, not
    fixed-duration shocks. No Dataset or execution-clock admission is implied.
    """

    def __init__(
        self,
        config: RollingPortfolioRiskInputsConfig | None = None,
        *,
        native_clock: NativeBarClock | None = None,
    ) -> None:
        if native_clock is not None and type(native_clock) is not NativeBarClock:
            raise ValueError("native_clock must be a NativeBarClock")
        self.config = config or RollingPortfolioRiskInputsConfig()
        self._native_clock = native_clock

    @property
    def native_clock(self) -> NativeBarClock | None:
        return self._native_clock

    @property
    def identity_digest(self) -> str:
        clock = self.native_clock
        if clock is None:
            return self.config.digest
        return content_digest(
            {
                "schema_version": "rolling_portfolio_risk_native_clock_v1",
                "config_digest": self.config.digest,
                "kind": clock.kind.value,
                "fixed_seconds": clock.fixed_seconds,
                "fixed_anchor_ns": int(clock.fixed_anchor.astype(np.int64))
                if clock.fixed_anchor is not None
                else None,
            }
        )

    @property
    def minimum_index(self) -> int:
        return self.config.minimum_observations

    def inputs(
        self,
        dataset: PortfolioRiskPriceHistory,
        *,
        index: int,
    ) -> PortfolioRiskInputs:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("portfolio risk index must be an integer")
        if not 0 <= index < dataset.n_bars:
            raise ValueError("portfolio risk index is outside the dataset")
        if not 0 <= self.config.benchmark_index < dataset.n_symbols:
            raise ValueError("benchmark_index is outside dataset symbols")
        start = max(1, index - self.config.lookback_bars + 1)
        durations_ns: tuple[int, ...] | None = None
        clock = self.native_clock
        if (
            isinstance(dataset, MarketDataset)
            and dataset.native_calendar_month
            and clock != dataset.native_clock
        ):
            raise ValueError(
                "calendar month risk inputs require the matching native clock"
            )
        if clock is None:
            available_at = dataset.available_at
            if available_at is None or np.any(
                available_at[start - 1 : index + 1] > dataset.timestamps[index]
            ):
                raise ValueError("portfolio risk prices not available at decision time")
            prices = np.asarray(dataset.close[start - 1 : index + 1], dtype=np.float64)
        else:
            observations = index - start + 1
            if observations < self.config.minimum_observations:
                raise ValueError(
                    "portfolio risk inputs have insufficient finite history"
                )
            prices, durations_ns = _native_risk_window(
                dataset,
                clock=clock,
                window=slice(start - 1, index + 1),
                observations=observations,
            )
        if prices.shape[1:] != (dataset.n_symbols,) or np.any(prices <= 0.0):
            raise ValueError(
                "portfolio risk closes must be positive and symbol aligned"
            )
        log_returns = np.diff(np.log(prices), axis=0)
        if (
            log_returns.shape[0] < self.config.minimum_observations
            or not np.isfinite(log_returns).all()
        ):
            raise ValueError("portfolio risk inputs have insufficient finite history")
        if self.config.annualize_covariance and durations_ns is not None:
            covariance = _native_annual_covariance(log_returns, durations_ns)
        else:
            covariance = _sample_covariance(log_returns)
            if self.config.annualize_covariance:
                # Only the legacy provider requires the nominal frequency.
                try:
                    periods = cast(PortfolioRiskDataset, dataset).periods_per_year
                except AttributeError as exc:
                    raise ValueError(
                        "legacy annualization requires periods_per_year"
                    ) from exc
                covariance = covariance * float(periods)
        benchmark_variance = float(
            covariance[self.config.benchmark_index, self.config.benchmark_index]
        )
        if benchmark_variance <= 1e-18:
            raise ValueError("benchmark variance is insufficient for beta")
        beta = covariance[:, self.config.benchmark_index] / benchmark_variance
        stress = np.minimum(
            np.quantile(log_returns, self.config.stress_quantile, axis=0),
            0.0,
        )
        return PortfolioRiskInputs(
            covariance=covariance,
            beta=beta,
            stress_losses=stress,
            as_of_index=index,
            provider_digest=self.identity_digest,
            observation_count=int(log_returns.shape[0]),
        )


__all__ = [
    "PortfolioRiskInputs",
    "PortfolioRiskInputsProvider",
    "RollingPortfolioRiskInputsConfig",
    "RollingPortfolioRiskInputsProvider",
]
