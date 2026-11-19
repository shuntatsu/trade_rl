"""Seeded synthetic markets with known truth for the SD1 oracles (Issue #1146 section 7).

No market value is read anywhere: every array is drawn from a ``numpy`` generator with
a fixed seed. Panels are flattened row-major, ``(hour, symbol)`` to one row per pair,
so "the rows of one UTC day" are ``24 * symbols`` consecutive rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOURS_PER_DAY = 24


def day_ids_for(days: int, symbols: int) -> np.ndarray:
    return np.repeat(np.arange(days * HOURS_PER_DAY) // HOURS_PER_DAY, symbols).astype(
        np.int64
    )


@dataclass(frozen=True)
class PlantedPanel:
    day_ids: np.ndarray
    target: np.ndarray
    signal: np.ndarray
    nulls: tuple[np.ndarray, ...]
    true_rank_ic: float


def planted_signal_panel(
    seed: int, *, days: int, symbols: int, null_features: int, strength: float
) -> PlantedPanel:
    """``y = strength * signal + N(0, 1)`` with iid Gaussian signal and null features.

    For a bivariate Gaussian pair with Pearson rho, the Spearman correlation is
    ``(6 / pi) * asin(rho / 2)``; that closed form is the oracle for the recovery.
    """

    rng = np.random.default_rng(seed)
    rows = days * HOURS_PER_DAY * symbols
    signal = rng.standard_normal(rows)
    target = strength * signal + rng.standard_normal(rows)
    nulls = tuple(rng.standard_normal(rows) for _ in range(null_features))
    rho = strength / np.sqrt(1.0 + strength**2)
    return PlantedPanel(
        day_ids=day_ids_for(days, symbols),
        target=target,
        signal=signal,
        nulls=nulls,
        true_rank_ic=float((6.0 / np.pi) * np.arcsin(rho / 2.0)),
    )


def shuffle_days(
    target: np.ndarray, *, rows_per_day: int, rng: np.random.Generator
) -> np.ndarray:
    """Permute whole days of the target: the future is shuffled, its law is kept."""

    blocks = target.reshape(-1, rows_per_day)
    return blocks[rng.permutation(blocks.shape[0])].reshape(-1)


@dataclass(frozen=True)
class CommonLevelPanel:
    returns: np.ndarray  # (hours, symbols) realised labels
    market_forecast: np.ndarray  # (hours,) noisy view of the common factor
    betas: np.ndarray
    day_ids: np.ndarray


def common_level_panel(
    seed: int,
    *,
    days: int,
    betas: tuple[float, ...],
    idiosyncratic_sd: float,
    forecast_noise_sd: float,
) -> CommonLevelPanel:
    """``r[t, i] = beta_i * M[t] + e[t, i]`` with a feature that sees ``M`` noisily."""

    rng = np.random.default_rng(seed)
    hours = days * HOURS_PER_DAY
    factor = rng.standard_normal(hours)
    beta = np.asarray(betas, dtype=np.float64)
    noise = idiosyncratic_sd * rng.standard_normal((hours, beta.shape[0]))
    returns = factor[:, None] * beta[None, :] + noise
    forecast = factor + forecast_noise_sd * rng.standard_normal(hours)
    return CommonLevelPanel(returns, forecast, beta, day_ids_for(days, beta.shape[0]))


def tail_only_panel(
    seed: int, *, rows: int, tail_share: float, jump: float
) -> tuple[np.ndarray, np.ndarray]:
    """Feature ``f ~ N(0,1)``; the target relates to it only in the extreme tail."""

    rng = np.random.default_rng(seed)
    feature = rng.standard_normal(rows)
    target = rng.standard_normal(rows)
    cutoff = np.quantile(np.abs(feature), 1.0 - tail_share)
    tail = np.abs(feature) > cutoff
    target[tail] += jump * np.sign(feature[tail])
    return feature, target
