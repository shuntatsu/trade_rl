"""Opt-in ACT/365F annualization of complete native-bar simple returns.

This module does not alter legacy ReturnSeries/PerformanceMetrics identities,
infer missing returns, admit a Dataset or assert that a venue published a bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import fmean

import numpy as np

from trade_rl._validation import _ordered_ns_durations
from trade_rl.data._timestamps import _NANOSECONDS_PER_DAY
from trade_rl.data.native_clock import NativeBarClock
from trade_rl.evaluation.series import _normalize_duration_returns


@dataclass(frozen=True, slots=True)
class NativeAnnualization:
    """Exact temporal metadata and descriptive zero-target risk metrics."""

    elapsed_years: Fraction
    periods_per_year: Fraction
    annualized_arithmetic_return: float
    annualized_volatility: float
    annualized_downside_deviation: float
    sharpe: float
    sortino: float


def evaluate_native_bar_annualization(
    values: tuple[float, ...],
    *,
    clock: NativeBarClock,
    closes: np.ndarray,
) -> NativeAnnualization:
    """Evaluate one simple return per complete consecutive native interval.

    Let t_i be ACT/365F interval years, T=sum(t_i), and r_i simple returns.
    Arithmetic drift is mu=sum(r_i)/T; annual volatility is
    sqrt(sum((r_i-mu*t_i)**2)/T); zero-target downside deviation is
    sqrt(sum(min(r_i,0)**2)/T). Ratios use mu over these respective
    deviations, with zero denominator yielding zero as in legacy metrics.
    For equal durations this reduces to the legacy population-moment
    Sharpe/Sortino convention with the *unrounded* observed frequency.

    Unequal-duration residuals describe a linear arithmetic drift and
    variance-per-unit-time model, not a compounding return, a nominal monthly
    Sharpe, or evidence of independent increments. No serial-correlation
    adjustment is inferred. Caller owns return provenance and must supply
    one return from the validated interval start through its close.
    """
    if not isinstance(clock, NativeBarClock):
        raise ValueError("clock must be a NativeBarClock")
    starts, ends = clock.interval_bounds(closes)
    # Each interval fits signed ns, but their total may not.
    durations_ns = ends.view(np.int64) - starts.view(np.int64)
    total_ns = int(ends[-1].astype(np.int64)) - int(starts[0].astype(np.int64))
    return _evaluate_duration_annualization(values, durations_ns, total_ns)


def evaluate_session_elapsed_annualization(
    values: tuple[float, ...], *, boundaries_ns: tuple[int, ...]
) -> NativeAnnualization:
    """Use actual close-to-close durations of an explicitly supplied grid.

    Same arithmetic-drift model as native calendar annualization, including
    all elapsed time between boundaries. No venue schedule or return
    provenance is inferred; callers supply a complete, pinned sequence.
    """
    durations = _ordered_ns_durations(boundaries_ns)
    return _evaluate_duration_annualization(
        values, np.asarray(durations, dtype=np.int64), sum(durations)
    )


def _evaluate_duration_annualization(
    values: tuple[float, ...], durations_ns: np.ndarray, total_ns: int
) -> NativeAnnualization:
    if not isinstance(values, tuple) or len(values) != durations_ns.size:
        raise ValueError("one return per native interval is required")
    normalized = _normalize_duration_returns(values, label="native")
    elapsed = Fraction(total_ns, 365 * _NANOSECONDS_PER_DAY)
    years = float(elapsed)
    try:
        total = math.fsum(normalized)
        drift = total / years
        if np.all(durations_ns == durations_ns[0]):
            # Center the small differences again: a rounded mean can land
            # on an endpoint when returns differ by only one floating-point ULP.
            mean = (
                normalized[0]
                if all(value == normalized[0] for value in normalized)
                else fmean(normalized)
            )
            offsets = tuple(value - mean for value in normalized)
            correction = fmean(offsets)
            residuals = (offset - correction for offset in offsets)
        else:
            # Exact binary-rational centering avoids fabricating risk when
            # the input returns are proportional to elapsed time.
            exact = tuple(Fraction.from_float(value) for value in normalized)
            exact_total = sum(exact, Fraction(0))
            residuals = (
                float(value - exact_total * int(duration) / total_ns)
                for value, duration in zip(exact, durations_ns, strict=True)
            )
        scale = math.sqrt(years)
        volatility = math.hypot(*residuals) / scale
        downside = math.hypot(*(min(value, 0.0) for value in normalized)) / scale
        sharpe = drift / volatility if volatility > 0.0 else 0.0
        sortino = drift / downside if downside > 0.0 else 0.0
    except OverflowError as exc:
        raise ValueError("native annualization is not representable") from exc
    if not all(
        math.isfinite(x) for x in (drift, volatility, downside, sharpe, sortino)
    ):
        raise ValueError("native annualization is not representable")

    return NativeAnnualization(
        elapsed_years=elapsed,
        periods_per_year=len(normalized) / elapsed,
        annualized_arithmetic_return=drift,
        annualized_volatility=volatility,
        annualized_downside_deviation=downside,
        sharpe=sharpe,
        sortino=sortino,
    )
