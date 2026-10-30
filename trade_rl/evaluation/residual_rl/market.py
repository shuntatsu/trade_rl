"""Deterministic synthetic multi-asset markets with an explicitly planted alpha.

A market is a fully pre-generated set of arrays so causality can be audited by
perturbing only the part of the arrays that lies after a decision time.

Timeline (day ``t``): the signal ``z[t]`` and the allocator forecast ``mu[t]`` are
known at the decision at the start of day ``t``; the realised simple return of day
``t`` (``returns[t]``, mark ``t`` to mark ``t + 1``) is generated from the planted
alpha ``alpha[t]`` and a shock ``shock[t]`` that is independent of everything known
at ``t``.

Regimes:

* ``PLANTED_ALPHA``: ``alpha = a * z`` and ``mu = a * z + nu`` (a noisy view of the
  true expected return).
* ``ZERO_ALPHA``: ``alpha = 0``. ``mu`` has the same distribution as in the planted
  regime but is independent of the returns, so the allocator trades on a signal
  that is worth nothing and every unit of cost is a pure loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

NS_PER_DAY = 86_400 * 1_000_000_000


class Regime(str, Enum):
    PLANTED_ALPHA = "planted_alpha"
    ZERO_ALPHA = "zero_alpha"


@dataclass(frozen=True, slots=True)
class MarketSpec:
    """Generator constants. Every field is part of the frozen protocol."""

    n_assets: int = 4
    warmup_days: int = 40
    horizon_days: int = 252
    daily_vol: tuple[float, ...] = (0.008, 0.010, 0.012, 0.014)
    common_factor_share: float = 0.30
    signal_scale: float = 0.0012
    signal_persistence: float = 0.85
    forecast_noise: float = 0.0008
    initial_price: float = 100.0

    def __post_init__(self) -> None:
        if type(self.n_assets) is not int or not 3 <= self.n_assets <= 5:
            raise ValueError("n_assets must be an integer within [3, 5]")
        if type(self.warmup_days) is not int or self.warmup_days < 31:
            raise ValueError("warmup_days must cover the allocator history minimum")
        if type(self.horizon_days) is not int or self.horizon_days < 2:
            raise ValueError("horizon_days must be an integer of at least two")
        if len(self.daily_vol) != self.n_assets or any(
            not math.isfinite(v) or v <= 0.0 for v in self.daily_vol
        ):
            raise ValueError("daily_vol must be positive per asset")
        if not 0.0 <= self.common_factor_share < 1.0:
            raise ValueError("common_factor_share must be within [0, 1)")
        if not 0.0 <= self.signal_persistence < 1.0:
            raise ValueError("signal_persistence must be within [0, 1)")
        for name in ("signal_scale", "forecast_noise", "initial_price"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def total_days(self) -> int:
        return self.warmup_days + self.horizon_days

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(f"S{i}" for i in range(self.n_assets))


@dataclass(frozen=True, slots=True)
class SyntheticMarket:
    """Pre-generated arrays; row ``t`` is day ``t`` of ``total_days``."""

    regime: Regime
    seed: int
    spec: MarketSpec
    signal: np.ndarray  # z[t]: the AR(1) signal known at decision t, (T, K)
    forecast_noise: np.ndarray  # nu[t]: allocator forecast noise, (T, K)
    shock: np.ndarray  # return noise of day t, independent of anything known at t
    alpha: np.ndarray  # true expected simple return of day t, (T, K)
    mu: np.ndarray  # the forecast handed to the allocator at decision t, (T, K)
    returns: np.ndarray  # realised simple return of day t, (T, K)
    prices: np.ndarray  # marks, (T + 1, K); prices[t] is the mark at decision t

    def replace_from(
        self,
        day: int,
        *,
        shock: np.ndarray | None = None,
        signal: np.ndarray | None = None,
        forecast_noise: np.ndarray | None = None,
    ) -> SyntheticMarket:
        """Rebuild the market with different arrays from ``day`` onwards.

        Used by the causality oracle: the rebuilt market shares every input of
        decisions ``< day`` with the original and differs only afterwards.
        """

        if not 0 <= day <= self.spec.total_days:
            raise ValueError("day out of range")
        new_signal = self.signal.copy()
        new_noise = self.forecast_noise.copy()
        new_shock = self.shock.copy()
        if signal is not None:
            new_signal[day:] = signal[day:]
        if forecast_noise is not None:
            new_noise[day:] = forecast_noise[day:]
        if shock is not None:
            new_shock[day:] = shock[day:]
        return _assemble(
            self.regime, self.seed, self.spec, new_signal, new_noise, new_shock
        )


def _assemble(
    regime: Regime,
    seed: int,
    spec: MarketSpec,
    signal: np.ndarray,
    forecast_noise: np.ndarray,
    shock: np.ndarray,
) -> SyntheticMarket:
    scale = spec.signal_scale
    forecast = scale * signal + forecast_noise
    alpha = scale * signal if regime is Regime.PLANTED_ALPHA else np.zeros_like(signal)
    vol = np.asarray(spec.daily_vol, dtype=np.float64)
    returns = alpha + vol * shock
    if np.any(returns <= -0.9):  # unreachable for the protocol constants
        raise ValueError("generated return is not a valid simple return")
    prices = np.empty((spec.total_days + 1, spec.n_assets), dtype=np.float64)
    prices[0] = spec.initial_price
    for t in range(spec.total_days):
        prices[t + 1] = prices[t] * (1.0 + returns[t])
    return SyntheticMarket(
        regime=regime,
        seed=seed,
        spec=spec,
        signal=signal,
        forecast_noise=forecast_noise,
        shock=shock,
        alpha=alpha,
        mu=forecast,
        returns=returns,
        prices=prices,
    )


def generate_market(spec: MarketSpec, regime: Regime, seed: int) -> SyntheticMarket:
    """Return the deterministic market for ``(spec, regime, seed)``.

    The regime does not enter the random stream: the planted and zero-alpha
    markets of the same seed share signal, forecast noise and shocks and differ
    only in whether the signal is worth anything.
    """

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 810_001]))
    days, assets = spec.total_days, spec.n_assets
    innovations = rng.standard_normal((days, assets))
    phi = spec.signal_persistence
    signal = np.empty((days, assets), dtype=np.float64)
    signal[0] = innovations[0]
    drive = math.sqrt(1.0 - phi * phi)
    for t in range(1, days):
        signal[t] = phi * signal[t - 1] + drive * innovations[t]
    forecast_noise = spec.forecast_noise * rng.standard_normal((days, assets))
    common = rng.standard_normal((days, 1))
    idiosyncratic = rng.standard_normal((days, assets))
    share = spec.common_factor_share
    shock = math.sqrt(share) * common + math.sqrt(1.0 - share) * idiosyncratic
    return _assemble(regime, seed, spec, signal, forecast_noise, shock)


__all__ = [
    "NS_PER_DAY",
    "MarketSpec",
    "Regime",
    "SyntheticMarket",
    "generate_market",
]
