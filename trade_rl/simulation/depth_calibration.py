"""Distribution-free execution-cost bounds from cumulative depth bands.

The input is a sequence of order-book *band* snapshots, such as the roughly
30-second Binance Vision USD-M ``bookDepth`` records: for each side, a set of
percentages ``p_1 < ... < p_m`` of the mid price (1% .. 5%) and the cumulative
resting quote notional ``N_1 <= ... <= N_m`` within that percentage. The caller
converts its own store rows into :class:`DepthBandSnapshot`; this module reads
no store, network or file, imports no integration, fits nothing, and makes no
profitability claim. It only aggregates the snapshots it is given.

What the data can and cannot say
--------------------------------
A band snapshot constrains only the cumulative notional at a few distances. It
does not resolve the bid/ask spread, the queue within the first band, or the
price impact of any order that fits inside the first band (``Q <= N_1``). For
such an order the data bounds the cost from above by ``p_1`` (typically 1%,
i.e. 100 bps) and from below only by zero. It therefore cannot validate a
small-order cost of a few bps; such assumptions are reported ``UNRESOLVABLE``,
never "validated". Beyond the outermost band the data says nothing, and
nothing is extrapolated: the result is explicit
:class:`BeyondObservedDepth`.

Units and numeric domain
------------------------
``DepthBand.percentage`` is in percent of the mid price (``1`` is 1%), as in the
source column. Every cost output is in basis points of the mid price
(``percentage * 100``) and exact (:class:`fractions.Fraction`). "Cost" here is
the notional-weighted mean fractional distance of the consumed liquidity from
the mid price. It excludes fees. It is not the exact price-VWAP slippage, which
differs from this arithmetic mean at second order in the distance; the
adversarial placements below are identical for any increasing cost function of
the distance because they depend only on the order of the liquidity.

All arithmetic is exact rational arithmetic: no float is produced, so nothing
can overflow, underflow or round, and metamorphic scaling of the notionals
scales the notional-valued results exactly. Band values, positive notionals,
quantiles and capacity thresholds have a bounded input domain: native
``int``, ``Fraction``, finite ``Decimal`` (and finite ``float`` for thresholds
only), zero or absolute value in ``[1e-308, 1e308]``, at most 700 significant
Decimal digits and at most 4096 bits in a Fraction numerator or denominator.
The classifier and existing-cell lookups intentionally compare exact values
outside the magnitude and Fraction bit limits, including derived bounds;
the conversion/type/finite and Decimal/integer guards still apply. Finite
float assumed costs are accepted. These comparisons do not admit inputs to
the bounded solver or grid. Numeric and band tuple subclasses are refused before conversion or indexing.
Calculation entry points require native DepthBandSnapshot instances.
Numeric validation errors name the offending field, never its value.

Derivation of the bounds
------------------------
Model: the book is a measure ``mu`` of resting quote notional over the distance
``d >= 0`` from the mid price with ``mu([0, p_k]) = N_k`` for every band. A
market order of quote notional ``Q <= N_m`` consumes the nearest ``Q`` units of
``mu``. Let ``d(u)`` be the distance of the ``u``-th unit of notional (the
quantile function of ``mu``), and ``N_0 = 0``, ``p_0 = 0``. For ``u`` in
``(N_{j-1}, N_j]`` the unit lies within ``p_j`` (``mu([0, p_j]) = N_j >= u``)
and strictly beyond ``p_{j-1}`` (``mu([0, p_{j-1}]) = N_{j-1} < u``), so

    p_{j-1} < d(u) <= p_j.

Let ``k`` be the smallest band with ``N_k >= Q``.

* Marginal (worst) fill distance: ``d(Q) <= p_k`` (upper bound) and
  ``d(Q) > p_{k-1}`` (lower bound; an infimum that is approached, not
  attained). For ``Q <= N_1`` the lower bound is ``0``.
* Average cost ``A(Q) = (1/Q) * integral_0^Q d(u) du``. Integrating the
  pointwise bounds gives

      A_upper(Q) = (1/Q) * ( sum_{j<k} (N_j - N_{j-1}) p_j + (Q - N_{k-1}) p_k )
      A_lower(Q) = (1/Q) * ( sum_{j<k} (N_j - N_{j-1}) p_{j-1} + (Q - N_{k-1}) p_{k-1} )

  ``A_upper`` is attained by the adversarial placement that puts all notional
  of band ``j`` exactly at its outer edge ``p_j`` (consistent with every
  ``N_j``), so no smaller valid upper bound exists. ``A_lower`` is the
  infimum approached by placing each band's notional just beyond its inner
  edge. The tests enumerate every discretised placement on small examples and
  confirm both statements.
* All four bounds are non-decreasing in ``Q`` (an average of a non-decreasing
  step function is non-decreasing).

Aggregation, quantiles and capacity
-----------------------------------
For a grid of notionals the module reports empirical quantiles of each bound
across snapshots using the *type-1* (inverse empirical CDF) definition on exact
values: the ``q``-quantile of ``n`` values is the ``ceil(q * n)``-th smallest
(at least the first) for ``0 < q <= 1``. A snapshot whose observed depth is
smaller than ``Q`` carries the trivial interval ``[0, +inf)``: it contributes
``+inf`` (reported as ``None``) to upper-bound quantiles and ``0`` to
lower-bound quantiles, and is also counted in ``beyond_depth_fraction``.
Quantiles are descriptive statistics of the bounds across snapshots, not
confidence intervals about future fills.

``depth_capacity`` returns the exact largest ``Q`` whose ``q``-quantile of
``A_upper`` does not exceed a caller threshold. Per snapshot ``A_upper`` is
continuous and non-decreasing on ``(0, N_m]`` and infinite beyond, so a
snapshot's feasible set is ``(0, Q_s]`` with ``Q_s`` solved band by band from
``C(Q) <= theta * Q``; the type-1 quantile is at most ``theta`` exactly when at
least ``ceil(q * n)`` snapshots are feasible, i.e. for ``Q`` up to the
``ceil(q * n)``-th largest ``Q_s``. A positive selected boundary must also fit
the bounded public notional domain; otherwise the solver refuses it with
``ValueError`` instead of rounding or returning an unqueryable capacity.
This is a bound on assumed cost, not a fill or profitability claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from typing import Literal, NamedTuple

from trade_rl.artifacts.hashing import content_digest

Side = Literal["buy", "sell"]
Exact = Fraction | Decimal | int

DEPTH_CALIBRATION_SCHEMA = "depth_band_cost_calibration_v1"
_SIDES: tuple[Side, ...] = ("buy", "sell")
_BPS_PER_PERCENT = Fraction(100)
_ZERO = Fraction(0)
_MIN_MAGNITUDE = Fraction(1, 10**308)
_MAX_MAGNITUDE = Fraction(10**308)
_MAX_DECIMAL_DIGITS = 700
_MAX_DECIMAL_ADJUSTED = 309
_MAX_BITS = 4096

# Normalised side: ((distance_bps, cumulative_notional), ...), exact.
_Bands = tuple[tuple[Fraction, Fraction], ...]


class DepthBand(NamedTuple):
    """One band: cumulative resting quote notional within ``percentage``% of mid."""

    percentage: Exact
    cumulative_notional: Exact


def _decimal_to_fraction(value: Decimal, field: str) -> Fraction:
    if not value.is_finite():
        raise ValueError(f"{field} must be finite")
    if value == 0:
        # Fraction(Decimal) would materialise 10**exponent even for zero.
        return _ZERO
    if (
        len(value.as_tuple().digits) > _MAX_DECIMAL_DIGITS
        or abs(value.adjusted()) > _MAX_DECIMAL_ADJUSTED
    ):
        raise ValueError(f"{field} is outside the representable range")
    return Fraction(value)


def _exact(
    value: object, field: str, *, allow_float: bool = False, bounded: bool = True
) -> Fraction:
    """Convert to an exact Fraction inside the documented numeric domain.

    ``bounded=False`` skips the magnitude and size limits for values that are
    outputs of this module (bounds can legitimately fall outside the input
    range, for example a tiny average cost after extreme notional ratios).
    """
    if type(value) is bool:
        raise ValueError(f"{field} must be an exact number")
    if type(value) is Fraction:
        result = value
    elif type(value) is int:
        if value.bit_length() > _MAX_BITS:
            raise ValueError(f"{field} is outside the representable range")
        result = Fraction(value)
    elif type(value) is Decimal:
        result = _decimal_to_fraction(value, field)
    elif allow_float and type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        result = Fraction(value)
    else:
        raise ValueError(f"{field} must be an exact Fraction, Decimal or int")
    if not bounded:
        return result
    if (
        result.numerator.bit_length() > _MAX_BITS
        or result.denominator.bit_length() > _MAX_BITS
    ):
        raise ValueError(f"{field} is outside the representable range")
    if result != 0 and not _MIN_MAGNITUDE <= abs(result) <= _MAX_MAGNITUDE:
        raise ValueError(f"{field} is outside the representable range")
    return result


def _coerce_side(bands: object, field: str) -> tuple[DepthBand, ...]:
    if type(bands) is not tuple or not bands:
        raise ValueError(f"{field} must be a non-empty tuple of bands")
    coerced: list[DepthBand] = []
    previous_percentage: Fraction | None = None
    previous_notional: Fraction | None = None
    for band in bands:
        if type(band) not in (tuple, DepthBand) or len(band) != 2:
            raise ValueError(f"{field} bands must be (percentage, notional) pairs")
        percentage = _exact(band[0], f"{field} percentage")
        notional = _exact(band[1], f"{field} cumulative notional")
        if percentage <= 0:
            raise ValueError(f"{field} percentages must be positive")
        if notional < 0:
            raise ValueError(f"{field} cumulative notional must be non-negative")
        if previous_percentage is not None and percentage <= previous_percentage:
            raise ValueError(f"{field} percentages must be strictly increasing")
        if previous_notional is not None and notional < previous_notional:
            raise ValueError(f"{field} cumulative notional must be non-decreasing")
        previous_percentage, previous_notional = percentage, notional
        coerced.append(DepthBand(band[0], band[1]))
    return tuple(coerced)


def _normalise(bands: tuple[DepthBand, ...]) -> _Bands:
    return tuple(
        (
            _exact(band.percentage, "percentage") * _BPS_PER_PERCENT,
            _exact(band.cumulative_notional, "notional"),
        )
        for band in bands
    )


@dataclass(frozen=True, slots=True)
class DepthBandSnapshot:
    """Immutable cumulative-depth-band snapshot of one instrument.

    ``bid_bands`` (used by sells) and ``ask_bands`` (used by buys) are tuples of
    ``DepthBand(percentage, cumulative_notional)`` with percentages in percent
    of mid. Native tuples of native ``DepthBand`` or plain tuple pairs and
    native exact numbers are required. Construction accepts native aware
    datetimes with zero UTC offset,
    samples the offset once and stores a native UTC datetime. It refuses
    non-UTC or naive times, non-finite or out-of-range values, non-positive
    or non-increasing percentages, and negative or decreasing notionals.
    """

    time: datetime
    bid_bands: tuple[DepthBand, ...]
    ask_bands: tuple[DepthBand, ...]

    def __post_init__(self) -> None:
        if type(self.time) is not datetime:
            raise ValueError("snapshot time must be a native timezone-aware datetime")
        offset = self.time.utcoffset()
        if offset is None:
            raise ValueError("snapshot time must be timezone-aware")
        if offset != timedelta(0):
            raise ValueError("snapshot time must be UTC")
        # Preserve the admitted instant without keeping caller-owned tzinfo.
        object.__setattr__(self, "time", self.time.replace(tzinfo=UTC, fold=0))
        object.__setattr__(self, "bid_bands", _coerce_side(self.bid_bands, "bid"))
        object.__setattr__(self, "ask_bands", _coerce_side(self.ask_bands, "ask"))

    def bands_for(self, side: Side) -> tuple[DepthBand, ...]:
        """Return the book side a market order of ``side`` consumes."""
        _require_side(side)
        return self.ask_bands if side == "buy" else self.bid_bands


def _require_side(side: object) -> None:
    if side not in _SIDES:
        raise ValueError("side must be 'buy' or 'sell'")


@dataclass(frozen=True, slots=True)
class OrderCostBounds:
    """Bounds in bps of mid for one order inside the observed depth.

    ``*_lower_bps`` are infima that are approached but not attained. The bounds
    exclude fees. ``within_first_band`` flags ``Q <= N_1``, where the data
    cannot resolve spread or impact (lower bounds are zero).
    """

    side: Side
    notional: Fraction
    observed_depth: Fraction
    marginal_lower_bps: Fraction
    marginal_upper_bps: Fraction
    average_lower_bps: Fraction
    average_upper_bps: Fraction
    within_first_band: bool


@dataclass(frozen=True, slots=True)
class BeyondObservedDepth:
    """The order exceeds the outermost band; the data bounds nothing."""

    side: Side
    notional: Fraction
    observed_depth: Fraction


OrderCostResult = OrderCostBounds | BeyondObservedDepth


def _bounds(
    bands: _Bands, notional: Fraction
) -> tuple[Fraction, Fraction, Fraction, Fraction, bool] | None:
    if notional > bands[-1][1]:
        return None
    previous_distance = _ZERO
    previous_notional = _ZERO
    upper_sum = _ZERO
    lower_sum = _ZERO
    for distance, cumulative in bands:
        if cumulative >= notional:
            take = notional - previous_notional
            return (
                previous_distance,
                distance,
                (lower_sum + take * previous_distance) / notional,
                (upper_sum + take * distance) / notional,
                notional <= bands[0][1],
            )
        width = cumulative - previous_notional
        upper_sum += width * distance
        lower_sum += width * previous_distance
        previous_distance, previous_notional = distance, cumulative
    raise AssertionError("unreachable: notional is within observed depth")


def _require_notional(value: object, field: str) -> Fraction:
    exact = _exact(value, field)
    if exact <= 0:
        raise ValueError(f"{field} must be positive")
    return exact


def order_cost_bounds(
    snapshot: DepthBandSnapshot, *, side: Side, notional: Exact
) -> OrderCostResult:
    """Bound the cost of a market order of quote ``notional`` on one snapshot.

    Buys consume ``ask_bands``; sells consume ``bid_bands``. Returns
    :class:`BeyondObservedDepth` when ``notional`` exceeds the outermost band.
    """
    if type(snapshot) is not DepthBandSnapshot:
        raise ValueError("snapshot must be a DepthBandSnapshot")
    quantity = _require_notional(notional, "notional")
    bands = _normalise(snapshot.bands_for(side))
    observed = bands[-1][1]
    found = _bounds(bands, quantity)
    if found is None:
        return BeyondObservedDepth(side, quantity, observed)
    marginal_lower, marginal_upper, average_lower, average_upper, first = found
    return OrderCostBounds(
        side=side,
        notional=quantity,
        observed_depth=observed,
        marginal_lower_bps=marginal_lower,
        marginal_upper_bps=marginal_upper,
        average_lower_bps=average_lower,
        average_upper_bps=average_upper,
        within_first_band=first,
    )


class CostAssumptionCheck(StrEnum):
    BELOW_OBSERVED_LOWER_BOUND = "below_observed_lower_bound"
    WITHIN_BOUNDS = "within_bounds"
    ABOVE_UPPER_BOUND = "above_upper_bound"
    UNRESOLVABLE = "unresolvable"
    BEYOND_OBSERVED_DEPTH = "beyond_observed_depth"


def classify_assumed_cost(
    assumed_cost_bps: Exact | float,
    *,
    lower_bound_bps: Exact,
    upper_bound_bps: Exact | None,
) -> CostAssumptionCheck:
    """Compare an assumed average cost (bps, excluding fees) with data bounds.

    ``upper_bound_bps=None`` means unbounded (beyond observed depth). Lower
    bounds are infima, so an assumption equal to a positive lower bound is
    already below the attainable costs. A zero lower bound (``Q <= N_1``) means
    the data cannot exclude any cost under the upper bound: an assumption below
    it is ``UNRESOLVABLE`` rather than validated.

    * ``BELOW_OBSERVED_LOWER_BOUND``: optimistic; flag it.
    * ``ABOVE_UPPER_BOUND``: conservative.
    * ``WITHIN_BOUNDS``: consistent with the bounds (which are wide).
    """
    assumed = _exact(
        assumed_cost_bps, "assumed_cost_bps", allow_float=True, bounded=False
    )
    lower = _exact(lower_bound_bps, "lower_bound_bps", bounded=False)
    upper = (
        None
        if upper_bound_bps is None
        else _exact(upper_bound_bps, "upper_bound_bps", bounded=False)
    )
    if assumed < 0 or lower < 0:
        raise ValueError("costs must be non-negative")
    if upper is not None and upper < lower:
        raise ValueError("upper bound must not be below the lower bound")
    if lower > 0 and assumed <= lower:
        return CostAssumptionCheck.BELOW_OBSERVED_LOWER_BOUND
    if upper is None:
        return CostAssumptionCheck.BEYOND_OBSERVED_DEPTH
    if assumed > upper:
        return CostAssumptionCheck.ABOVE_UPPER_BOUND
    if lower == 0 and assumed < upper:
        return CostAssumptionCheck.UNRESOLVABLE
    return CostAssumptionCheck.WITHIN_BOUNDS


def check_snapshot_assumption(
    result: OrderCostResult, assumed_cost_bps: Exact | float
) -> CostAssumptionCheck:
    """Classify an assumed average cost against one snapshot's bounds."""
    if isinstance(result, BeyondObservedDepth):
        return classify_assumed_cost(
            assumed_cost_bps, lower_bound_bps=0, upper_bound_bps=None
        )
    return classify_assumed_cost(
        assumed_cost_bps,
        lower_bound_bps=result.average_lower_bps,
        upper_bound_bps=result.average_upper_bps,
    )


@dataclass(frozen=True, slots=True)
class QuantileBounds:
    """Type-1 quantiles of each bound across snapshots (bps of mid).

    ``None`` upper bounds are unbounded: the quantile rank falls on snapshots
    whose observed depth is smaller than the order.
    """

    quantile: Fraction
    marginal_lower_bps: Fraction
    marginal_upper_bps: Fraction | None
    average_lower_bps: Fraction
    average_upper_bps: Fraction | None


@dataclass(frozen=True, slots=True)
class CalibrationCell:
    side: Side
    hour_utc: int | None
    notional: Fraction
    snapshot_count: int
    beyond_depth_count: int
    first_band_count: int
    bounds: tuple[QuantileBounds, ...]

    @property
    def beyond_depth_fraction(self) -> Fraction:
        return Fraction(self.beyond_depth_count, max(1, self.snapshot_count))

    @property
    def first_band_fraction(self) -> Fraction:
        """Share of snapshots where ``Q <= N_1`` (spread/impact unresolvable)."""
        return Fraction(self.first_band_count, max(1, self.snapshot_count))

    def bounds_at(self, quantile: Exact) -> QuantileBounds:
        wanted = _exact(quantile, "quantile", bounded=False)
        for entry in self.bounds:
            if entry.quantile == wanted:
                return entry
        raise ValueError("quantile was not calibrated")

    def check_assumption(
        self, quantile: Exact, assumed_cost_bps: Exact | float
    ) -> CostAssumptionCheck:
        """Classify an assumed average cost against this cell's quantile bounds."""
        entry = self.bounds_at(quantile)
        return classify_assumed_cost(
            assumed_cost_bps,
            lower_bound_bps=entry.average_lower_bps,
            upper_bound_bps=entry.average_upper_bps,
        )


@dataclass(frozen=True, slots=True)
class DepthCostCalibration:
    """Aggregated bounds with the identity of the snapshots they came from."""

    schema: str
    time_start: datetime
    time_end: datetime
    snapshot_count: int
    input_digest: str
    notionals: tuple[Fraction, ...]
    quantiles: tuple[Fraction, ...]
    sides: tuple[Side, ...]
    by_utc_hour: bool
    cells: tuple[CalibrationCell, ...]

    def cell(
        self, *, side: Side, notional: Exact, hour_utc: int | None = None
    ) -> CalibrationCell:
        wanted = _exact(notional, "notional", bounded=False)
        for entry in self.cells:
            if (
                entry.side == side
                and entry.hour_utc == hour_utc
                and entry.notional == wanted
            ):
                return entry
        raise ValueError("no calibration cell for that side, hour and notional")


@dataclass(frozen=True, slots=True)
class DepthCapacity:
    """Exact largest notional whose quantile average-cost upper bound fits.

    ``capacity_notional == 0`` means no positive order is feasible at that
    quantile. A positive returned capacity is within the public notional input
    domain; an exact boundary outside that domain is refused, not rounded.
    This bounds an assumed cost; it does not claim fills, profit or live capacity.
    """

    side: Side
    hour_utc: int | None
    quantile: Fraction
    max_average_cost_bps: Fraction
    capacity_notional: Fraction
    selected_snapshot_count: int
    snapshot_count: int
    time_start: datetime
    time_end: datetime
    input_digest: str


def _validated_snapshots(
    snapshots: Sequence[DepthBandSnapshot],
) -> tuple[DepthBandSnapshot, ...]:
    candidate: object = snapshots
    if isinstance(candidate, (str, bytes)) or not isinstance(candidate, Sequence):
        raise ValueError("snapshots must be a sequence")
    resolved = tuple(candidate)
    if not resolved:
        raise ValueError("snapshots must not be empty")
    previous: datetime | None = None
    for snapshot in resolved:
        if type(snapshot) is not DepthBandSnapshot:
            raise ValueError("snapshots must be DepthBandSnapshot values")
        if previous is not None and snapshot.time <= previous:
            raise ValueError("snapshot times must be strictly increasing")
        previous = snapshot.time
    return resolved


def _side_payload(bands: tuple[DepthBand, ...]) -> list[list[str]]:
    return [
        [
            str(_exact(band.percentage, "percentage")),
            str(_exact(band.cumulative_notional, "notional")),
        ]
        for band in bands
    ]


def _input_digest(snapshots: tuple[DepthBandSnapshot, ...]) -> str:
    return content_digest(
        {
            "schema": DEPTH_CALIBRATION_SCHEMA,
            "snapshots": [
                {
                    "time": snapshot.time.astimezone(UTC).isoformat(),
                    "bid": _side_payload(snapshot.bid_bands),
                    "ask": _side_payload(snapshot.ask_bands),
                }
                for snapshot in snapshots
            ],
        }
    )


def _strictly_increasing(
    values: Iterable[Exact], field: str, *, upper_one: bool
) -> tuple[Fraction, ...]:
    resolved = tuple(_exact(value, field) for value in values)
    if not resolved:
        raise ValueError(f"{field} must not be empty")
    if any(value <= 0 for value in resolved):
        raise ValueError(f"{field} must be positive")
    if upper_one and any(value > 1 for value in resolved):
        raise ValueError(f"{field} must be within (0, 1]")
    if any(b <= a for a, b in zip(resolved, resolved[1:], strict=False)):
        raise ValueError(f"{field} must be strictly increasing")
    return resolved


def _validated_sides(sides: Iterable[Side]) -> tuple[Side, ...]:
    resolved = tuple(sides)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("sides must be non-empty and unique")
    for side in resolved:
        _require_side(side)
    return resolved


def _rank(count: int, quantile: Fraction) -> int:
    """Type-1 quantile rank: ``ceil(q * n)``, at least 1 (1-indexed)."""
    return max(1, math.ceil(quantile * count))


def _pick_upper(finite_sorted: list[Fraction], rank: int) -> Fraction | None:
    # Beyond-depth snapshots are +inf, the largest values.
    return finite_sorted[rank - 1] if rank <= len(finite_sorted) else None


def _pick_lower(finite_sorted: list[Fraction], rank: int, beyond: int) -> Fraction:
    # Beyond-depth snapshots carry the trivial lower bound 0, the smallest values.
    return _ZERO if rank <= beyond else finite_sorted[rank - beyond - 1]


def _cell(
    side: Side,
    hour: int | None,
    notional: Fraction,
    books: Sequence[_Bands],
    quantiles: tuple[Fraction, ...],
) -> CalibrationCell:
    count = len(books)
    marginal_lower: list[Fraction] = []
    marginal_upper: list[Fraction] = []
    average_lower: list[Fraction] = []
    average_upper: list[Fraction] = []
    beyond = 0
    first = 0
    for bands in books:
        found = _bounds(bands, notional)
        if found is None:
            beyond += 1
            continue
        marginal_lower.append(found[0])
        marginal_upper.append(found[1])
        average_lower.append(found[2])
        average_upper.append(found[3])
        first += int(found[4])
    for values in (marginal_lower, marginal_upper, average_lower, average_upper):
        values.sort()
    entries = []
    for quantile in quantiles:
        rank = _rank(count, quantile)
        entries.append(
            QuantileBounds(
                quantile=quantile,
                marginal_lower_bps=_pick_lower(marginal_lower, rank, beyond),
                marginal_upper_bps=_pick_upper(marginal_upper, rank),
                average_lower_bps=_pick_lower(average_lower, rank, beyond),
                average_upper_bps=_pick_upper(average_upper, rank),
            )
        )
    return CalibrationCell(
        side=side,
        hour_utc=hour,
        notional=notional,
        snapshot_count=count,
        beyond_depth_count=beyond,
        first_band_count=first,
        bounds=tuple(entries),
    )


def calibrate_depth_costs(
    snapshots: Sequence[DepthBandSnapshot],
    *,
    notionals: Iterable[Exact],
    quantiles: Iterable[Exact],
    sides: Iterable[Side] = _SIDES,
    by_utc_hour: bool = False,
) -> DepthCostCalibration:
    """Aggregate per-snapshot cost bounds over a notional grid.

    ``snapshots`` must be strictly time-ordered. ``notionals`` must be positive
    and ``quantiles`` in ``(0, 1]``; both grids must be strictly increasing. With
    ``by_utc_hour`` the result adds one cell per UTC hour that has snapshots,
    besides the overall cell (``hour_utc=None``). Nothing is fitted. The
    ``input_digest`` identifies only the snapshots; grids, sides and grouping
    are separate result fields.
    """
    resolved = _validated_snapshots(snapshots)
    grid = _strictly_increasing(notionals, "notionals", upper_one=False)
    levels = _strictly_increasing(quantiles, "quantiles", upper_one=True)
    chosen = _validated_sides(sides)
    if not isinstance(by_utc_hour, bool):
        raise ValueError("by_utc_hour must be a boolean")
    cells: list[CalibrationCell] = []
    for side in chosen:
        normalised = [_normalise(s.bands_for(side)) for s in resolved]
        groups: list[tuple[int | None, list[_Bands]]] = [(None, normalised)]
        if by_utc_hour:
            for utc_hour in range(24):
                members = [
                    bands
                    for snapshot, bands in zip(resolved, normalised, strict=True)
                    if snapshot.time.hour == utc_hour
                ]
                if members:
                    groups.append((utc_hour, members))
        for group_hour, group_books in groups:
            for notional in grid:
                cells.append(_cell(side, group_hour, notional, group_books, levels))
    return DepthCostCalibration(
        schema=DEPTH_CALIBRATION_SCHEMA,
        time_start=resolved[0].time,
        time_end=resolved[-1].time,
        snapshot_count=len(resolved),
        input_digest=_input_digest(resolved),
        notionals=grid,
        quantiles=levels,
        sides=chosen,
        by_utc_hour=by_utc_hour,
        cells=tuple(cells),
    )


def _feasible_notional(bands: _Bands, threshold_bps: Fraction) -> Fraction:
    """Largest ``Q`` with ``A_upper(Q) <= threshold`` on one snapshot (0 if none)."""
    previous_notional = _ZERO
    upper_sum = _ZERO
    for distance, cumulative in bands:
        if distance > threshold_bps:
            # Within this band C(Q) = upper_sum + distance * (Q - previous_notional)
            # and C(Q) <= threshold * Q  <=>  Q <= D / (distance - threshold),
            # where the divisor is positive by the branch condition.
            candidate = (distance * previous_notional - upper_sum) / (
                distance - threshold_bps
            )
            if candidate < cumulative:
                return candidate
        upper_sum += (cumulative - previous_notional) * distance
        previous_notional = cumulative
    return previous_notional


def depth_capacity(
    snapshots: Sequence[DepthBandSnapshot],
    *,
    side: Side,
    quantile: Exact,
    max_average_cost_bps: Exact | float,
    hour_utc: int | None = None,
) -> DepthCapacity:
    """Exact largest ``Q`` whose ``quantile`` of ``A_upper`` is ``<=`` the threshold.

    Uses the type-1 quantile and counts snapshots too shallow for ``Q`` as
    infinite cost. ``hour_utc`` restricts to snapshots of one UTC hour. A
    positive exact capacity outside the public notional input domain raises
    ``ValueError``; a returned positive capacity can be queried without rounding.
    """
    resolved = _validated_snapshots(snapshots)
    _require_side(side)
    level = _exact(quantile, "quantile")
    if not 0 < level <= 1:
        raise ValueError("quantile must be within (0, 1]")
    threshold = _exact(max_average_cost_bps, "max_average_cost_bps", allow_float=True)
    if threshold < 0:
        raise ValueError("max_average_cost_bps must be non-negative")
    if hour_utc is not None and (
        isinstance(hour_utc, bool)
        or not isinstance(hour_utc, int)
        or not 0 <= hour_utc <= 23
    ):
        raise ValueError("hour_utc must be an integer within 0..23")
    selected = [
        _normalise(s.bands_for(side))
        for s in resolved
        if hour_utc is None or s.time.hour == hour_utc
    ]
    if not selected:
        raise ValueError("no snapshots in the requested UTC hour")
    limits = sorted((_feasible_notional(b, threshold) for b in selected), reverse=True)
    capacity = limits[_rank(len(selected), level) - 1]
    # Validate only the selected order statistic, not discarded candidates.
    if capacity != 0:
        _require_notional(capacity, "capacity_notional")
    return DepthCapacity(
        side=side,
        hour_utc=hour_utc,
        quantile=level,
        max_average_cost_bps=threshold,
        capacity_notional=capacity,
        selected_snapshot_count=len(selected),
        snapshot_count=len(resolved),
        time_start=resolved[0].time,
        time_end=resolved[-1].time,
        input_digest=_input_digest(resolved),
    )


__all__ = [
    "DEPTH_CALIBRATION_SCHEMA",
    "BeyondObservedDepth",
    "CalibrationCell",
    "CostAssumptionCheck",
    "DepthBand",
    "DepthBandSnapshot",
    "DepthCapacity",
    "DepthCostCalibration",
    "OrderCostBounds",
    "OrderCostResult",
    "QuantileBounds",
    "Side",
    "calibrate_depth_costs",
    "check_snapshot_assumption",
    "classify_assumed_cost",
    "depth_capacity",
    "order_cost_bounds",
]
