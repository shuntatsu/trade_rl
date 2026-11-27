"""Multiple-testing-aware Sharpe ratio statistics.

Pure standard-library implementations of the Probabilistic Sharpe Ratio
(Bailey and Lopez de Prado, 2012), the expected maximum Sharpe ratio of ``N``
unskilled trials and the Deflated Sharpe Ratio (Bailey and Lopez de Prado,
2014).

Every Sharpe ratio here is **per period** (the mean return divided by the
return standard deviation of one observation interval), never annualised.
``skew`` is the third standardised moment and ``kurtosis`` is the
**non-excess** fourth standardised moment (a normal distribution has
kurtosis 3, not 0). Annualised figures must be converted by the caller before
they are passed in; the functions cannot detect the mistake.

The functions are generic. They carry no research roster, threshold or claim,
and a high value is a statistic about the supplied inputs only. In particular
a high Deflated Sharpe Ratio computed on data that was already used for
development is not out-of-sample evidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from statistics import NormalDist

EULER_MASCHERONI = 0.5772156649015329
"""Euler-Mascheroni constant used by the expected-maximum approximation."""

MAX_TRIAL_COUNT = 10**15
"""Largest trial count accepted, keeping ``1/N`` well inside float range."""

MAX_OBSERVATIONS = 10**15
"""Largest ``n_obs`` accepted, keeping ``sqrt(n - 1)`` exact enough and finite."""

_NORMAL = NormalDist()
_KURTOSIS_RELATIVE_TOLERANCE = 1e-9


def require_finite_real(value: object, *, name: str) -> float:
    """Return ``value`` as a float if it is a finite ``numbers.Real``.

    This is the single numeric-acceptance rule shared with the trial ledger:
    any ``numbers.Real`` except ``bool`` (Python and NumPy floats and ints,
    ``Fraction``) is accepted; strings, ``None``, complex values, NaN, infinity
    and values too large for a float are refused with ``ValueError``.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _integer(value: object, *, name: str, minimum: int, maximum: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        bound = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be {bound}")
    return result


def validate_sharpe_moments(
    sharpe: object, n_obs: object, skew: object, kurtosis: object
) -> tuple[float, int, float, float]:
    """Validate and normalise per-period Sharpe summary statistics.

    ``n_obs`` must be an integer in ``[2, MAX_OBSERVATIONS]``. ``kurtosis`` is the
    non-excess definition and, by the Pearson inequality, cannot be smaller
    than ``1 + skew**2`` (a skew whose square overflows is refused, so the
    bound is never evaluated as ``inf`` or ``NaN``); an excess-kurtosis value near zero is therefore
    refused instead of being silently misread.
    """

    resolved_sharpe = require_finite_real(sharpe, name="sharpe")
    resolved_n = _integer(n_obs, name="n_obs", minimum=2, maximum=MAX_OBSERVATIONS)
    resolved_skew = require_finite_real(skew, name="skew")
    resolved_kurtosis = require_finite_real(kurtosis, name="kurtosis")
    floor = 1.0 + resolved_skew * resolved_skew
    if not math.isfinite(floor) or (
        resolved_kurtosis < floor - _KURTOSIS_RELATIVE_TOLERANCE * floor
    ):
        raise ValueError(
            "kurtosis must be the non-excess definition and at least 1 + skew**2"
        )
    return resolved_sharpe, resolved_n, resolved_skew, resolved_kurtosis


@dataclass(frozen=True, slots=True)
class SharpeMoments:
    """Per-period Sharpe ratio with the moments the PSR needs."""

    sharpe: float
    n_obs: int
    skew: float
    kurtosis: float

    def __post_init__(self) -> None:
        sharpe, n_obs, skew, kurtosis = validate_sharpe_moments(
            self.sharpe, self.n_obs, self.skew, self.kurtosis
        )
        object.__setattr__(self, "sharpe", sharpe)
        object.__setattr__(self, "n_obs", n_obs)
        object.__setattr__(self, "skew", skew)
        object.__setattr__(self, "kurtosis", kurtosis)


def sharpe_moments_from_returns(returns: Sequence[float]) -> SharpeMoments:
    """Compute the per-period SR, skewness and non-excess kurtosis of returns.

    Bias convention (explicit, and not interchangeable with other libraries):

    * ``sharpe`` is ``mean / s`` where ``s`` is the **sample** standard
      deviation with ``n - 1`` in the denominator;
    * ``skew`` is ``m3 / m2**1.5`` and ``kurtosis`` is ``m4 / m2**2`` where
      ``mk`` is the **population** central moment (divide by ``n``). No
      small-sample bias correction is applied and kurtosis is *not* excess.

    The moments are computed on the series divided by its largest absolute
    value, so every finite series (from subnormal to near-maximum scale) gives
    the same dimensionless result as its rescaled copy, up to rounding. At least
    two finite observations are required, and the population standard deviation
    must exceed ``1e-12`` of the largest absolute value; constant, all-zero and
    near-constant series are refused with ``ValueError``.
    """

    if isinstance(returns, (str, bytes)) or not isinstance(returns, Sequence):
        raise ValueError("returns must be a sequence of finite real numbers")
    values = [require_finite_real(item, name="return") for item in returns]
    n = len(values)
    if n < 2:
        raise ValueError("at least two return observations are required")
    # All returned moments are dimensionless. Scaling by max|x| first avoids
    # intermediate overflow/underflow for any finite series and keeps the
    # dispersion cutoff relative to the series' own scale.
    scale = max(abs(item) for item in values)
    if scale == 0.0:
        raise ValueError("return series must have non-zero finite dispersion")
    normalized = [item / scale for item in values]
    mean = math.fsum(normalized) / n
    deviations = [item - mean for item in normalized]
    m2 = math.fsum(d * d for d in deviations) / n
    if m2 <= 0.0 or not math.isfinite(m2) or math.sqrt(m2) <= 1e-12:
        raise ValueError("return series must have non-zero finite dispersion")
    m3 = math.fsum(d**3 for d in deviations) / n
    m4 = math.fsum(d**4 for d in deviations) / n
    sample_std = math.sqrt(m2 * (n / (n - 1)))
    skew = m3 / m2**1.5
    kurtosis = m4 / (m2 * m2)
    if not all(math.isfinite(x) for x in (mean, sample_std, skew, kurtosis)):
        raise ValueError("return moments are not finite")
    # Rounding can leave kurtosis a few ulps below its exact lower bound.
    kurtosis = max(kurtosis, 1.0 + skew * skew)
    return SharpeMoments(
        sharpe=mean / sample_std, n_obs=n, skew=skew, kurtosis=kurtosis
    )


def sharpe_z_statistic(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    benchmark_sr: float = 0.0,
) -> float:
    """The standardised statistic whose normal CDF is the PSR.

    Unlike the PSR it does not saturate at exactly 0.0 or 1.0 for large
    magnitudes, so it keeps distinguishing strongly separated candidates.
    Inputs and refusals are those of :func:`probabilistic_sharpe_ratio`.
    """

    sharpe, n, skew_value, kurtosis_value = validate_sharpe_moments(
        sr, n_obs, skew, kurtosis
    )
    benchmark = require_finite_real(benchmark_sr, name="benchmark_sr")
    radicand = (
        1.0 - skew_value * sharpe + (kurtosis_value - 1.0) / 4.0 * sharpe * sharpe
    )
    if not math.isfinite(radicand) or radicand <= 0.0:
        raise ValueError("PSR variance term is not positive for these inputs")
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(radicand)
    if not math.isfinite(z):
        raise ValueError("PSR statistic is not finite")
    return z


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    benchmark_sr: float = 0.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sr``.

    Bailey and Lopez de Prado (2012)::

        PSR = Phi((SR - SR*) * sqrt(n - 1)
                  / sqrt(1 - skew * SR + (kurtosis - 1) / 4 * SR**2))

    ``sr`` and ``benchmark_sr`` are per-period; ``kurtosis`` is non-excess.
    The inputs are refused when the radicand is not strictly positive, which
    the formula cannot represent. The result describes the supplied sample
    only; it does not correct for how many strategies were tried (see
    :func:`deflated_sharpe_ratio`).

    The result is a float CDF value and saturates to exactly ``1.0`` or
    ``0.0`` once ``|z|`` exceeds roughly 8; use :func:`sharpe_z_statistic` to
    order values in that regime.
    """

    return _NORMAL.cdf(sharpe_z_statistic(sr, n_obs, skew, kurtosis, benchmark_sr))


def expected_maximum_sharpe(n_trials: int, trial_sr_variance: float) -> float:
    """Expected maximum per-period SR among ``n_trials`` skill-free trials.

    Bailey and Lopez de Prado (2014)::

        E[max] ~= sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N)
                             + g * Phi^-1(1 - 1/(N * e)))

    with ``g`` the Euler-Mascheroni constant and ``V`` the cross-trial
    variance of per-period Sharpe ratios.

    For ``n_trials == 1`` the formula's ``Phi^-1(0)`` is undefined; this
    function returns ``0.0`` by convention because a single trial involves no
    selection, which makes the deflated ratio equal the plain PSR against zero.
    A zero ``trial_sr_variance`` likewise returns ``0.0``.
    """

    n = _integer(n_trials, name="n_trials", minimum=1, maximum=MAX_TRIAL_COUNT)
    variance = require_finite_real(trial_sr_variance, name="trial_sr_variance")
    if variance < 0.0:
        raise ValueError("trial_sr_variance must be non-negative")
    if n == 1 or variance == 0.0:
        return 0.0
    # Phi^-1(1 - p) = -Phi^-1(p); the lower tail stays accurate for tiny p.
    first = -_NORMAL.inv_cdf(1.0 / n)
    second = -_NORMAL.inv_cdf(1.0 / (n * math.e))
    result = math.sqrt(variance) * (
        (1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second
    )
    if not math.isfinite(result):
        raise ValueError("expected maximum Sharpe ratio is not finite")
    return result


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    trial_sr_variance: float,
) -> float:
    """PSR against the expected maximum SR of ``n_trials`` skill-free trials.

    Bailey and Lopez de Prado (2014). ``n_trials`` must count every trial that
    competed for selection, including failed or abandoned ones, and
    ``trial_sr_variance`` is the cross-trial variance of per-period SRs. With
    ``n_trials == 1`` the benchmark is zero and the result equals
    ``probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, 0.0)``.

    A high value on data already used for development is not out-of-sample
    evidence and authorises nothing.
    """

    benchmark = expected_maximum_sharpe(n_trials, trial_sr_variance)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, benchmark)


__all__ = [
    "EULER_MASCHERONI",
    "MAX_OBSERVATIONS",
    "MAX_TRIAL_COUNT",
    "SharpeMoments",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "probabilistic_sharpe_ratio",
    "require_finite_real",
    "sharpe_moments_from_returns",
    "sharpe_z_statistic",
    "validate_sharpe_moments",
]
