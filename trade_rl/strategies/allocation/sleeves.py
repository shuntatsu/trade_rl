"""Causal combination of independent strategy sleeves into one portfolio target.

A *sleeve* is one strategy's target weight vector over a shared, ordered symbol
tuple. :func:`combine_sleeves` turns ``K`` sleeve targets plus each sleeve's
trailing per-period return history into one symbol-weight vector. It is a pure
function of its arguments: no clock, no hidden state, no randomness, no account.
Which sleeves exist, their parameters and every default are the caller's
(research's) choice; this module only fixes the reference arithmetic.

The combiner cannot attest the chronology of supplied rows. The history is a bare ``(T, K)`` array of
the rows the caller passes, oldest first; the function reads nothing else and
only the most recent ``lookback`` rows. The caller owns the contract that every
passed row is a realized sleeve return strictly before the decision time.

Order of operations (exact):

1. Validate sleeves, symbols, history and config. Refuse, never guess.
2. Take the last ``lookback`` history rows. Sample covariance with the explicit
   ``ddof``: ``S = C'C / (n - ddof)`` where ``C`` is the column-demeaned window.
   Internally each column is scaled to unit peak before the products, and the exact
   scales and individual variance-contribution factors are recombined with
   exponent-tracking arithmetic. Nonrepresentable normalized nonzero risk
   weights and final nonzero products that underflow to zero or overflow are
   refused with ``SleeveCombinerError`` (``non_representable`` /
   ``non_finite``). Other matrix operations use binary64, not exact rational
   arithmetic; this is not an arbitrary-scale accuracy guarantee.
3. Shrink toward the diagonal: ``Sigma = (1 - d) S + d diag(S)``, ``d`` in [0, 1].
4. Apply the zero-variance policy (a sleeve whose window is constant): refuse by
   default, or ``EXCLUDE`` it (weight 0; the remaining ``FIXED`` weights or ERC
   budgets are renormalized to sum to 1; all sleeves excluded is refused).
5. Sleeve weights ``w`` (non-negative, sum to 1) by ``FIXED``, ``INVERSE_VOL``
   (``w_k`` proportional to ``1 / sqrt(Sigma_kk)``) or ``EQUAL_RISK_CONTRIBUTION``
   (positive budgets ``b``; ``w_k (Sigma w)_k / (w' Sigma w) = b_k``).
6. Net the sleeves: ``netted_j = sum_k w_k * target_kj``.
7. Optional volatility target: ``scale = min(max_leverage, target / predicted_vol)``
   with ``predicted_vol = sqrt(periods_per_year * w' Sigma w)``; ``scale = 1`` with
   no target. ``scaled = scale * netted``.
8. Optional per-symbol absolute caps: ``clip(scaled_j, -cap_j, cap_j)``.
9. Optional gross cap: if ``sum |.| > max_gross`` multiply every weight by
   ``max_gross / sum |.|`` (which cannot break the symbol caps).

The result reports the weights and arithmetic only; it makes no claim about
profitability and is not a ``PortfolioPolicy``. The existing policy contract
hands a policy only one account-level snapshot and no per-sleeve return series,
so a policy-wrapping adapter would need a shadow book per sleeve; that is
account/P&L state this package must not own, hence no adapter here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

_SINGULAR_RELATIVE_EIGENVALUE = 1e-12
_ZERO_VARIANCE_RELATIVE = 1e-24
_WEIGHT_SUM_TOLERANCE = 1e-9


class SleeveCombinerError(ValueError):
    """The inputs or configuration are invalid, or the combination was refused."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason


class SleeveMethod(str, Enum):
    FIXED = "fixed"
    INVERSE_VOL = "inverse_vol"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"


class ZeroVariancePolicy(str, Enum):
    """What to do with a sleeve whose return window is constant."""

    REFUSE = "refuse"
    EXCLUDE = "exclude"


@dataclass(frozen=True, slots=True)
class SleeveTarget:
    """One sleeve's target weights over the shared ordered symbol tuple."""

    name: str
    symbols: tuple[str, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("sleeve name must be a non-empty string")
        if type(self.symbols) is not tuple or not self.symbols:
            raise TypeError("sleeve symbols must be a non-empty tuple")
        if any(not isinstance(value, str) or not value for value in self.symbols):
            raise TypeError("sleeve symbols must be non-empty strings")
        if type(self.weights) is not tuple:
            raise TypeError("sleeve weights must be a tuple")
        if len(self.weights) != len(self.symbols):
            raise SleeveCombinerError(
                "shape", f"sleeve {self.name!r} weights must match its symbols"
            )
        object.__setattr__(
            self,
            "weights",
            tuple(
                _real(value, name=f"sleeve {self.name!r} weight")
                for value in self.weights
            ),
        )


@dataclass(frozen=True, slots=True)
class VolatilityTarget:
    """Portfolio volatility target ``scale = min(max_leverage, target / vol)``.

    ``target_vol`` is in the units of ``sqrt(periods_per_year) * period vol``, so
    leave ``periods_per_year = 1`` for a per-period target.
    """

    target_vol: float
    max_leverage: float
    periods_per_year: float = 1.0

    def __post_init__(self) -> None:
        for name in ("target_vol", "max_leverage", "periods_per_year"):
            value = _real(getattr(self, name), name=name)
            if value <= 0.0:
                raise SleeveCombinerError("config", f"{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SleeveCombinerConfig:
    """Explicit method configuration. There are no tuned defaults.

    ``fixed_weights`` is required for (and only allowed with) ``FIXED``.
    ``risk_budgets`` is only allowed with ``EQUAL_RISK_CONTRIBUTION`` and defaults
    to equal budgets. ``max_abs_weight`` is one cap for every symbol or one cap
    per symbol. ``erc_tolerance`` bounds the largest absolute deviation of a risk
    contribution fraction from its budget.
    """

    method: SleeveMethod
    lookback: int
    ddof: int
    shrinkage: float
    zero_variance: ZeroVariancePolicy = ZeroVariancePolicy.REFUSE
    fixed_weights: tuple[float, ...] | None = None
    risk_budgets: tuple[float, ...] | None = None
    erc_tolerance: float = 1e-10
    erc_max_iterations: int = 10_000
    vol_target: VolatilityTarget | None = None
    max_abs_weight: float | tuple[float, ...] | None = None
    max_gross: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, SleeveMethod):
            raise TypeError("method must be a SleeveMethod")
        if not isinstance(self.zero_variance, ZeroVariancePolicy):
            raise TypeError("zero_variance must be a ZeroVariancePolicy")
        if type(self.lookback) is not int or self.lookback < 2:
            raise SleeveCombinerError("config", "lookback must be an integer >= 2")
        if type(self.ddof) is not int or self.ddof not in (0, 1):
            raise SleeveCombinerError("config", "ddof must be 0 or 1")
        shrinkage = _real(self.shrinkage, name="shrinkage")
        if not 0.0 <= shrinkage <= 1.0:
            raise SleeveCombinerError("config", "shrinkage must be within [0, 1]")
        object.__setattr__(self, "shrinkage", shrinkage)
        if self.method is SleeveMethod.FIXED:
            if self.fixed_weights is None:
                raise SleeveCombinerError("config", "FIXED requires fixed_weights")
            fixed = _real_tuple(self.fixed_weights, name="fixed_weights")
            if any(value < 0.0 for value in fixed):
                raise SleeveCombinerError("config", "fixed_weights must be >= 0")
            if abs(math.fsum(fixed) - 1.0) > _WEIGHT_SUM_TOLERANCE:
                raise SleeveCombinerError("config", "fixed_weights must sum to 1")
            object.__setattr__(self, "fixed_weights", fixed)
        elif self.fixed_weights is not None:
            raise SleeveCombinerError("config", "fixed_weights only apply to FIXED")
        if self.risk_budgets is not None:
            if self.method is not SleeveMethod.EQUAL_RISK_CONTRIBUTION:
                raise SleeveCombinerError(
                    "config", "risk_budgets only apply to EQUAL_RISK_CONTRIBUTION"
                )
            budgets = _real_tuple(self.risk_budgets, name="risk_budgets")
            if any(value <= 0.0 for value in budgets):
                raise SleeveCombinerError("config", "risk_budgets must be positive")
            if abs(math.fsum(budgets) - 1.0) > _WEIGHT_SUM_TOLERANCE:
                raise SleeveCombinerError("config", "risk_budgets must sum to 1")
            object.__setattr__(self, "risk_budgets", budgets)
        tolerance = _real(self.erc_tolerance, name="erc_tolerance")
        if not 0.0 < tolerance < 1.0:
            raise SleeveCombinerError("config", "erc_tolerance must be within (0, 1)")
        object.__setattr__(self, "erc_tolerance", tolerance)
        if type(self.erc_max_iterations) is not int or self.erc_max_iterations < 1:
            raise SleeveCombinerError(
                "config", "erc_max_iterations must be a positive integer"
            )
        if self.vol_target is not None and not isinstance(
            self.vol_target, VolatilityTarget
        ):
            raise TypeError("vol_target must be a VolatilityTarget or None")
        if self.max_abs_weight is not None:
            if isinstance(self.max_abs_weight, tuple):
                caps = _real_tuple(self.max_abs_weight, name="max_abs_weight")
                if any(value <= 0.0 for value in caps):
                    raise SleeveCombinerError("config", "max_abs_weight must be > 0")
                object.__setattr__(self, "max_abs_weight", caps)
            else:
                cap = _real(self.max_abs_weight, name="max_abs_weight")
                if cap <= 0.0:
                    raise SleeveCombinerError("config", "max_abs_weight must be > 0")
                object.__setattr__(self, "max_abs_weight", cap)
        if self.max_gross is not None:
            gross = _real(self.max_gross, name="max_gross")
            if gross <= 0.0:
                raise SleeveCombinerError("config", "max_gross must be > 0")
            object.__setattr__(self, "max_gross", gross)


@dataclass(frozen=True, slots=True)
class SleeveCombinationDiagnostics:
    """Audit facts of one combination; ``converged`` is always True on success."""

    method: SleeveMethod
    lookback: int
    ddof: int
    shrinkage: float
    n_observations: int
    iterations: int
    converged: bool
    max_budget_deviation: float | None
    excluded_sleeves: tuple[str, ...]
    predicted_vol: float
    leverage_cap_binding: bool
    symbol_cap_binding: bool
    gross_cap_binding: bool


@dataclass(frozen=True, slots=True)
class SleeveCombination:
    """Immutable combination result.

    ``risk_contributions`` are variance contributions ``w_k (Sigma w)_k`` of the
    sleeve weights before the volatility scale; they sum to ``w' Sigma w``.
    ``netted_weights`` are the symbol weights before scale and caps.
    """

    symbols: tuple[str, ...]
    sleeve_names: tuple[str, ...]
    sleeve_weights: tuple[float, ...]
    scale: float
    netted_weights: tuple[float, ...]
    combined_weights: tuple[float, ...]
    risk_contributions: tuple[float, ...]
    diagnostics: SleeveCombinationDiagnostics

    @property
    def risk_contribution_fractions(self) -> tuple[float, ...] | None:
        """Contributions divided by the total variance; ``None`` if that is zero."""

        peak = max((abs(value) for value in self.risk_contributions), default=0.0)
        if peak == 0.0:
            return None
        unit = [value / peak for value in self.risk_contributions]
        total = math.fsum(unit)
        if total <= 0.0:
            return None
        return tuple(value / total for value in unit)


def combine_sleeves(
    sleeves: Sequence[SleeveTarget],
    *,
    symbols: tuple[str, ...],
    history: object,
    config: SleeveCombinerConfig,
) -> SleeveCombination:
    """Combine ``sleeves`` into one symbol-weight vector (see module docstring).

    ``history`` is a ``(T, K)`` array of realized per-period sleeve returns, one
    column per sleeve in the order of ``sleeves``, oldest row first, containing
    only rows strictly before the decision time.
    """

    if not isinstance(config, SleeveCombinerConfig):
        raise TypeError("config must be a SleeveCombinerConfig")
    if type(symbols) is not tuple or not symbols:
        raise TypeError("symbols must be a non-empty tuple")
    if len(set(symbols)) != len(symbols):
        raise SleeveCombinerError("symbols", "symbols must be unique")
    sleeve_list = tuple(sleeves)
    if not sleeve_list:
        raise SleeveCombinerError("no_sleeves", "at least one sleeve is required")
    if any(type(sleeve) is not SleeveTarget for sleeve in sleeve_list):
        raise TypeError("sleeves must be SleeveTarget instances")
    names = tuple(sleeve.name for sleeve in sleeve_list)
    if len(set(names)) != len(names):
        raise SleeveCombinerError("sleeves", "sleeve names must be unique")
    for sleeve in sleeve_list:
        if sleeve.symbols != symbols:
            raise SleeveCombinerError(
                "symbol_mismatch",
                f"sleeve {sleeve.name!r} symbols do not match the combiner symbols",
            )
    k = len(sleeve_list)
    for name, values in (
        ("fixed_weights", config.fixed_weights),
        ("risk_budgets", config.risk_budgets),
    ):
        if values is not None and len(values) != k:
            raise SleeveCombinerError(
                "config", f"{name} must have one entry per sleeve"
            )
    if isinstance(config.max_abs_weight, tuple) and len(config.max_abs_weight) != len(
        symbols
    ):
        raise SleeveCombinerError("config", "max_abs_weight must match the symbols")

    window = _history_window(history, n_sleeves=k, lookback=config.lookback)
    scaled = _scaled_covariance(window, ddof=config.ddof, shrinkage=config.shrinkage)
    zero_variance = scaled.constant
    excluded = tuple(
        name for name, flag in zip(names, zero_variance, strict=True) if flag
    )
    if excluded and config.zero_variance is ZeroVariancePolicy.REFUSE:
        raise SleeveCombinerError(
            "zero_variance", f"sleeves with zero variance: {', '.join(excluded)}"
        )
    active = ~zero_variance
    if not np.any(active):
        raise SleeveCombinerError("zero_variance", "every sleeve has zero variance")
    relative = scaled.relative_scale
    correlation = scaled.correlation

    iterations = 0
    max_deviation: float | None = None
    weights = np.zeros(k)
    with np.errstate(all="ignore"):
        if config.method is SleeveMethod.FIXED:
            assert config.fixed_weights is not None
            fixed = np.asarray(config.fixed_weights, dtype=np.float64) * active
            total = float(fixed.sum())
            if total <= 0.0:
                raise SleeveCombinerError(
                    "zero_variance", "no fixed weight remains after excluding sleeves"
                )
            weights = fixed / total
        elif config.method is SleeveMethod.INVERSE_VOL:
            inverse = np.zeros(k)
            inverse[active] = 1.0 / (
                relative[active] * np.sqrt(np.diag(correlation)[active])
            )
            weights = _normalized_active(inverse, active)
        else:
            block = correlation[np.ix_(active, active)]
            _require_positive_definite(block)
            if config.risk_budgets is None:
                budgets = np.full(k, 1.0 / k)
            else:
                budgets = np.asarray(config.risk_budgets, dtype=np.float64)
            active_budgets = budgets[active]
            active_budgets = active_budgets / active_budgets.sum()
            solved, iterations, max_deviation = _solve_erc(
                block,
                active_budgets,
                tolerance=config.erc_tolerance,
                max_iterations=config.erc_max_iterations,
            )
            unscaled = np.zeros(k)
            unscaled[active] = solved / relative[active]
            weights = _normalized_active(unscaled, active)

    # Risk in scale-free form: with ``v = relative_scale * w`` (the sleeve weights
    # in units of each sleeve's own volatility), ``z = v / max v`` and the
    # correlation-like matrix ``R``, the true variance is
    # ``(G * dmax * max v)^2 * z' R z``. The exact scale factors are combined with
    # an exponent-tracking product so no intermediate can silently overflow.
    v = relative * weights
    if np.any((relative > 0.0) & (weights > 0.0) & (v == 0.0)):
        raise SleeveCombinerError(
            "non_representable", "sleeve risk weights underflow to zero"
        )
    v_max = float(np.max(v))
    if not v_max > 0.0:
        raise SleeveCombinerError(
            "non_representable", "sleeve risk weights underflow to zero"
        )
    z = v / v_max
    with np.errstate(all="ignore"):
        rz = correlation @ z
        quadratic = max(float(z @ rz), 0.0)
        reference = float(np.sum(z * z * np.diag(correlation)))
    if not (math.isfinite(quadratic) and math.isfinite(reference)):
        raise SleeveCombinerError("non_finite", "predicted variance is not finite")
    vol_zero = quadratic <= _ZERO_VARIANCE_RELATIVE * reference
    periods = 1.0 if config.vol_target is None else config.vol_target.periods_per_year
    if vol_zero:
        predicted_vol = 0.0
    else:
        predicted_vol = _scaled_product(
            scaled.global_scale,
            scaled.max_relative_scale,
            v_max,
            math.sqrt(quadratic),
            math.sqrt(periods),
        )
    contributions = [
        _scaled_product(
            scaled.global_scale,
            scaled.max_relative_scale,
            scaled.global_scale,
            scaled.max_relative_scale,
            v_max,
            v_max,
            float(z[i]),
            float(rz[i]),
        )
        for i in range(k)
    ]

    scale = 1.0
    leverage_binding = False
    if config.vol_target is not None:
        if vol_zero:
            raise SleeveCombinerError(
                "zero_predicted_vol",
                "predicted portfolio volatility is zero; cannot target volatility",
            )
        raw_scale = _scaled_ratio(config.vol_target.target_vol, predicted_vol)
        leverage_binding = raw_scale > config.vol_target.max_leverage
        scale = min(config.vol_target.max_leverage, raw_scale)

    matrix = np.asarray([sleeve.weights for sleeve in sleeve_list], dtype=np.float64)
    with np.errstate(all="ignore"):
        netted = weights @ matrix
        _require_finite(netted, "netted weights")
        combined = scale * netted
        _require_finite(combined, "scaled weights")
        symbol_binding = False
        if config.max_abs_weight is not None:
            caps = np.asarray(config.max_abs_weight, dtype=np.float64)
            clipped = np.clip(combined, -caps, caps)
            symbol_binding = bool(np.any(clipped != combined))
            combined = clipped
        gross_binding = False
        if config.max_gross is not None:
            # Scale-stable gross: sum |c| = peak * sum(|c| / peak), never overflowing.
            peak = float(np.max(np.abs(combined)))
            if peak > 0.0:
                unit = combined / peak
                unit_gross = float(np.abs(unit).sum())
                factor = config.max_gross / unit_gross
                if peak > factor:
                    combined = unit * factor
                    gross_binding = True
        _require_finite(combined, "combined weights")

    return SleeveCombination(
        symbols=symbols,
        sleeve_names=names,
        sleeve_weights=tuple(float(value) for value in weights),
        scale=float(scale),
        netted_weights=tuple(float(value) for value in netted),
        combined_weights=tuple(float(value) for value in combined),
        risk_contributions=tuple(contributions),
        diagnostics=SleeveCombinationDiagnostics(
            method=config.method,
            lookback=config.lookback,
            ddof=config.ddof,
            shrinkage=config.shrinkage,
            n_observations=config.lookback,
            iterations=iterations,
            converged=True,
            max_budget_deviation=max_deviation,
            excluded_sleeves=excluded,
            predicted_vol=float(predicted_vol),
            leverage_cap_binding=leverage_binding,
            symbol_cap_binding=symbol_binding,
            gross_cap_binding=gross_binding,
        ),
    )


def _real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise SleeveCombinerError("non_finite", f"{name} must be finite")
    return resolved


def _real_tuple(values: object, *, name: str) -> tuple[float, ...]:
    if type(values) is not tuple or not values:
        raise TypeError(f"{name} must be a non-empty tuple")
    return tuple(_real(value, name=name) for value in values)


def _history_window(history: object, *, n_sleeves: int, lookback: int) -> np.ndarray:
    try:
        raw = np.asarray(history)
    except (TypeError, ValueError) as error:
        raise SleeveCombinerError(
            "history", "history must be a real (T, K) array"
        ) from error
    if raw.dtype.kind not in "iuf":
        raise SleeveCombinerError(
            "history", f"history must have a real numeric dtype, got {raw.dtype}"
        )
    array = raw.astype(np.float64)
    if array.ndim != 2 or array.shape[1] != n_sleeves:
        raise SleeveCombinerError(
            "shape", f"history must have shape (T, {n_sleeves}), got {array.shape}"
        )
    if array.shape[0] < lookback:
        raise SleeveCombinerError(
            "short_history",
            f"history has {array.shape[0]} rows; lookback {lookback} is required",
        )
    window = array[array.shape[0] - lookback :]
    if not np.all(np.isfinite(window)):
        raise SleeveCombinerError("non_finite", "history window must be finite")
    return window


@dataclass(frozen=True, slots=True)
class _ScaledCovariance:
    """``Sigma = global_scale * diag(rel) R diag(rel) * global_scale`` (shrunk).

    ``relative_scale`` are per-sleeve standard-deviation scales divided by
    ``max_relative_scale`` (so they lie in (0, 1], 0 for constant sleeves);
    ``correlation`` is the shrunk, unit-max-scaled second-moment matrix ``R``.
    """

    constant: np.ndarray
    relative_scale: np.ndarray
    max_relative_scale: float
    global_scale: float
    correlation: np.ndarray


def _scaled_covariance(
    window: np.ndarray, *, ddof: int, shrinkage: float
) -> _ScaledCovariance:
    """Sample covariance factored into scale-free parts that cannot overflow.

    Each column is divided by its max-abs, demeaned and divided by its largest
    deviation, so ``R = X'X / (n - ddof)`` has entries of order one however small or
    large the returns are. The scales are recombined only by exponent-tracking
    products at the end, which refuse genuinely non-representable results.
    """

    n, k = window.shape
    column_peak = np.max(np.abs(window), axis=0)
    constant = (np.ptp(window, axis=0) == 0.0) | (column_peak == 0.0)
    safe_peak = np.where(constant, 1.0, column_peak)
    normalized = window / safe_peak
    centered = normalized - normalized.mean(axis=0)
    deviation = np.max(np.abs(centered), axis=0)
    constant = constant | (deviation == 0.0)
    safe_deviation = np.where(constant, 1.0, deviation)
    unit = np.where(constant, 0.0, centered / safe_deviation)
    sample = unit.T @ unit / (n - ddof)
    correlation = (1.0 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
    correlation = 0.5 * (correlation + correlation.T)
    if not np.all(np.isfinite(correlation)):
        raise SleeveCombinerError("non_finite", "covariance is not finite")
    active = ~constant
    if not np.any(active):
        return _ScaledCovariance(constant, np.zeros(k), 1.0, 1.0, correlation)
    global_scale = float(np.max(column_peak[active]))
    relative = np.zeros(k)
    relative[active] = (column_peak[active] / global_scale) * deviation[active]
    if np.any(relative[active] <= 0.0):
        raise SleeveCombinerError(
            "non_representable", "sleeve volatility scales span too many magnitudes"
        )
    max_relative = float(np.max(relative))
    relative = relative / max_relative
    if np.any(relative[active] <= 0.0):
        raise SleeveCombinerError(
            "non_representable", "sleeve volatility scales span too many magnitudes"
        )
    return _ScaledCovariance(
        constant, relative, max_relative, global_scale, correlation
    )


def _normalized_active(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(values[active])) or np.any(values[active] <= 0.0):
        raise SleeveCombinerError(
            "non_representable", "sleeve weights are not representable"
        )
    total = float(values.sum())
    weights = values / total
    if not np.all(np.isfinite(weights)) or np.any(weights[active] <= 0.0):
        raise SleeveCombinerError(
            "non_representable", "sleeve weights are not representable"
        )
    return weights


def _require_finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise SleeveCombinerError("non_finite", f"{name} are not finite")


def _scaled_product(*factors: float) -> float:
    """Exact-exponent product; refuses overflow and underflow to zero."""

    sign = 1.0
    mantissa = 1.0
    exponent = 0
    for factor in factors:
        if not math.isfinite(factor):
            raise SleeveCombinerError("non_finite", "scale factor is not finite")
        if factor == 0.0:
            return 0.0
        if factor < 0.0:
            sign = -sign
            factor = -factor
        part, power = math.frexp(factor)
        mantissa *= part
        exponent += power
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise SleeveCombinerError(
            "non_representable", "result overflows the float range"
        ) from error
    if result == 0.0:
        raise SleeveCombinerError(
            "non_representable", "result underflows the float range"
        )
    return sign * result


def _scaled_ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` for positive finite floats; ``inf`` on overflow."""

    part_n, power_n = math.frexp(numerator)
    part_d, power_d = math.frexp(denominator)
    try:
        result = math.ldexp(part_n / part_d, power_n - power_d)
    except OverflowError:
        return math.inf
    if result == 0.0:
        raise SleeveCombinerError(
            "non_representable", "volatility scale underflows the float range"
        )
    return result


def _require_positive_definite(block: np.ndarray) -> None:
    eigenvalues = np.linalg.eigvalsh(block)
    if eigenvalues[0] <= _SINGULAR_RELATIVE_EIGENVALUE * eigenvalues[-1]:
        raise SleeveCombinerError(
            "singular_covariance",
            "sleeve covariance is singular; add shrinkage or history",
        )


def _budget_deviation(
    weights: np.ndarray, covariance: np.ndarray, budgets: np.ndarray
) -> float:
    contributions = weights * (covariance @ weights)
    return float(np.max(np.abs(contributions / contributions.sum() - budgets)))


def _solve_erc(
    covariance: np.ndarray,
    budgets: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float]:
    """Cyclic coordinate descent for ``x_i (Sigma x)_i = b_i``, ``w = x / sum x``.

    Each coordinate update is the exact positive root of
    ``Sigma_ii x_i^2 + c_i x_i - b_i = 0`` with ``c_i = sum_{j != i} Sigma_ij x_j``.
    For positive-definite ``Sigma`` and positive budgets the long-only solution is
    unique. Starts at inverse volatility, uses no randomness and refuses to return
    a point whose budget deviation exceeds ``tolerance``.
    """

    n = budgets.shape[0]
    if n == 1:
        return np.ones(1), 0, 0.0
    diagonal = np.diag(covariance)
    x = 1.0 / np.sqrt(diagonal)
    deviation = math.inf
    for iteration in range(max_iterations + 1):
        weights = x / x.sum()
        deviation = _budget_deviation(weights, covariance, budgets)
        if not math.isfinite(deviation):
            raise SleeveCombinerError("non_finite", "ERC iterate is not finite")
        if deviation <= tolerance:
            return weights, iteration, deviation
        if iteration == max_iterations:
            break
        for i in range(n):
            c = float(covariance[i] @ x - diagonal[i] * x[i])
            root = math.sqrt(c * c + 4.0 * diagonal[i] * budgets[i])
            if c >= 0.0:
                x[i] = 2.0 * budgets[i] / (c + root)
            else:
                x[i] = (root - c) / (2.0 * diagonal[i])
    raise SleeveCombinerError(
        "non_convergence",
        f"equal-risk-contribution solve did not converge in {max_iterations} "
        f"iterations (max budget deviation {deviation:.3e} > {tolerance:.3e})",
    )


__all__ = [
    "SleeveCombination",
    "SleeveCombinationDiagnostics",
    "SleeveCombinerConfig",
    "SleeveCombinerError",
    "SleeveMethod",
    "SleeveTarget",
    "VolatilityTarget",
    "ZeroVariancePolicy",
    "combine_sleeves",
]
