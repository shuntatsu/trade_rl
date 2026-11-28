from __future__ import annotations

import dataclasses
import itertools
import random
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

import pytest

from trade_rl.simulation.depth_calibration import (
    BeyondObservedDepth,
    CostAssumptionCheck,
    DepthBand,
    DepthBandSnapshot,
    OrderCostBounds,
    calibrate_depth_costs,
    check_snapshot_assumption,
    classify_assumed_cost,
    depth_capacity,
    order_cost_bounds,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
F = Fraction


def bands(*pairs: tuple[object, object]) -> tuple[DepthBand, ...]:
    return tuple(DepthBand(Decimal(str(p)), Decimal(str(n))) for p, n in pairs)


def snap(
    index: int = 0,
    *,
    asks: tuple[DepthBand, ...],
    bids: tuple[DepthBand, ...] | None = None,
    time: datetime | None = None,
) -> DepthBandSnapshot:
    return DepthBandSnapshot(
        time=time or T0 + timedelta(seconds=30 * index),
        bid_bands=asks if bids is None else bids,
        ask_bands=asks,
    )


# Percent bands: 1% -> 100 bps etc. Notional in quote units.
HAND_ASKS = bands((1, 100), (2, 300), (3, 300), (4, 600), (5, 1000))
HAND = snap(asks=HAND_ASKS)


def _bounds(
    snapshot: DepthBandSnapshot, q: object, side: str = "buy"
) -> OrderCostBounds:
    result = order_cost_bounds(snapshot, side=side, notional=q)  # type: ignore[arg-type]
    assert isinstance(result, OrderCostBounds)
    return result


# --- literal hand-computed examples ----------------------------------------


@pytest.mark.parametrize(
    ("q", "m_lo", "m_up", "a_lo", "a_up", "first"),
    [
        (50, 0, 100, 0, 100, True),
        (100, 0, 100, 0, 100, True),
        (200, 100, 200, 50, 150, False),
        (300, 100, 200, F(200, 3), F(500, 3), False),
        # band 3 has zero width, so k = 4 for Q = 500
        (500, 300, 400, 160, 260, False),
        (1000, 400, 500, 270, 370, False),
    ],
)
def test_hand_computed_bounds(q, m_lo, m_up, a_lo, a_up, first) -> None:
    result = _bounds(HAND, q)
    assert (
        result.marginal_lower_bps,
        result.marginal_upper_bps,
        result.average_lower_bps,
        result.average_upper_bps,
        result.within_first_band,
    ) == (F(m_lo), F(m_up), F(a_lo), F(a_up), first)
    assert result.observed_depth == 1000
    assert result.notional == q


def test_beyond_observed_depth_is_explicit_and_not_extrapolated() -> None:
    result = order_cost_bounds(HAND, side="buy", notional=Decimal("1000.0001"))
    assert isinstance(result, BeyondObservedDepth)
    assert result.observed_depth == 1000
    assert not hasattr(result, "average_upper_bps")


def test_sell_uses_bids_and_buy_uses_asks() -> None:
    snapshot = snap(asks=HAND_ASKS, bids=bands((1, 10), (2, 20)))
    assert _bounds(snapshot, 15, "buy").average_upper_bps == 100
    sell = _bounds(snapshot, 15, "sell")
    assert sell.marginal_lower_bps == 100 and sell.marginal_upper_bps == 200
    assert sell.average_upper_bps == F(100 * 10 + 200 * 5, 15)


def test_leading_empty_band_raises_lower_bound() -> None:
    result = _bounds(snap(asks=bands((1, 0), (2, 50))), 10)
    assert result.marginal_lower_bps == 100 and result.marginal_upper_bps == 200
    assert result.average_lower_bps == 100 and result.within_first_band is False


# --- brute-force adversarial tightness -------------------------------------


def _brute(grid_bands: list[tuple[int, int]], q: int):
    """Enumerate every integer-grid liquidity placement consistent with bands."""
    per_band: list[list[tuple[int, ...]]] = []
    previous_p, previous_n = 0, 0
    for index, (p, n) in enumerate(grid_bands):
        low = 0 if index == 0 else previous_p + 1
        choices = list(
            itertools.combinations_with_replacement(range(low, p + 1), n - previous_n)
        )
        per_band.append(choices)
        previous_p, previous_n = p, n
    marginals: list[int] = []
    averages: list[F] = []
    for combo in itertools.product(*per_band):
        atoms = sorted(d for part in combo for d in part)
        marginals.append(atoms[q - 1])
        averages.append(F(sum(atoms[:q]), q))
    return min(marginals), max(marginals), min(averages), max(averages)


def _check_against_brute(grid_bands: list[tuple[int, int]]) -> None:
    snapshot = snap(asks=bands(*grid_bands))
    for q in range(1, grid_bands[-1][1] + 1):
        m_min, m_max, a_min, a_max = _brute(grid_bands, q)
        got = _bounds(snapshot, q)
        # Upper bounds are attained by an enumerated placement: tight and valid.
        assert got.marginal_upper_bps == 100 * m_max
        assert got.average_upper_bps == 100 * a_max
        # Lower bounds are infima: valid, and tight to one grid step (1% = 100 bps).
        assert got.marginal_lower_bps <= 100 * m_min
        assert 100 * m_min - got.marginal_lower_bps <= 100
        assert got.average_lower_bps <= 100 * a_min
        # Exactly one grid step per atom outside the first band.
        outside = max(0, q - grid_bands[0][1])
        assert 100 * a_min == got.average_lower_bps + F(100 * outside, q)


@pytest.mark.parametrize(
    "grid_bands",
    [
        [(1, 2)],
        [(2, 2), (4, 3), (6, 5)],
        [(1, 1), (2, 3), (3, 3), (5, 6)],
        [(2, 0), (3, 2), (5, 4)],
    ],
)
def test_brute_force_adversarial_examples(grid_bands) -> None:
    _check_against_brute(grid_bands)


def test_brute_force_seeded_random_examples() -> None:
    rng = random.Random(20261127)
    for _ in range(25):
        count = rng.randint(1, 4)
        percentages = sorted(rng.sample(range(1, 8), count))
        increments = [rng.randint(0, 2) for _ in range(count)]
        if sum(increments) == 0:
            increments[-1] = 1
        cumulative = list(itertools.accumulate(increments))
        _check_against_brute(list(zip(percentages, cumulative, strict=True)))


# --- monotonicity and symmetry ----------------------------------------------


def test_all_bounds_are_monotone_in_notional() -> None:
    rng = random.Random(7)
    for _ in range(20):
        count = rng.randint(1, 6)
        percentages = sorted(rng.sample(range(1, 40), count))
        cumulative = list(
            itertools.accumulate(rng.randint(0, 50) for _ in range(count))
        )
        cumulative[-1] += 1
        snapshot = snap(asks=bands(*zip(percentages, cumulative, strict=True)))
        previous = None
        for step in range(1, 41):
            current = _bounds(snapshot, F(cumulative[-1] * step, 40))
            row = (
                current.marginal_lower_bps,
                current.marginal_upper_bps,
                current.average_lower_bps,
                current.average_upper_bps,
            )
            assert row[0] <= row[1] and row[2] <= row[3] and row[2] <= row[1]
            if previous is not None:
                assert all(a <= b for a, b in zip(previous, row, strict=True))
            previous = row


def test_buy_sell_symmetry_under_mirrored_inputs() -> None:
    left = bands((1, 7), (3, 20), (4, 21))
    right = bands((2, 5), (3, 6))
    one = snap(asks=left, bids=right)
    mirrored = snap(asks=right, bids=left)
    for q in (1, 5, 6, 7, 20):
        a = order_cost_bounds(one, side="buy", notional=q)
        b = order_cost_bounds(mirrored, side="sell", notional=q)
        assert dataclasses.replace(a, side="sell") == b  # type: ignore[type-var]


# --- aggregation, quantile definition, beyond depth -------------------------


def test_quantile_is_type_one_with_beyond_depth_as_infinite_upper_and_zero_lower() -> (
    None
):
    rows = [snap(i, asks=bands((p, 1000))) for i, p in enumerate((1, 2, 3, 4))]
    rows.append(snap(4, asks=bands((9, 10))))  # shallow: beyond depth for Q=500
    cal = calibrate_depth_costs(
        rows,
        notionals=[500],
        quantiles=[F(1, 5), F(2, 5), F(4, 5), 1],
        sides=["buy"],
    )
    cell = cal.cell(side="buy", notional=500)
    assert cell.snapshot_count == 5 and cell.beyond_depth_count == 1
    assert cell.beyond_depth_fraction == F(1, 5)
    # Resolved upper bounds 100, 200, 300, 400 bps; the shallow snapshot is +inf.
    # rank = ceil(q * 5): 1, 2, 4, 5.
    assert [cell.bounds_at(q).average_upper_bps for q in cal.quantiles] == [
        100,
        200,
        400,
        None,
    ]
    assert all(b.average_lower_bps == 0 for b in cell.bounds)
    assert cell.first_band_fraction == F(4, 5)


def test_quantile_rank_boundaries_use_ceiling() -> None:
    rows = [snap(i, asks=bands((p, 1000))) for i, p in enumerate((1, 2, 3, 4))]
    cal = calibrate_depth_costs(
        rows,
        notionals=[1],
        quantiles=[F(1, 100), F(1, 2), F(51, 100), 1],
        sides=["buy"],
    )
    cell = cal.cell(side="buy", notional=1)
    assert [cell.bounds_at(q).average_upper_bps for q in cal.quantiles] == [
        100,
        200,
        300,
        400,
    ]


def test_aggregation_identity_and_time_range() -> None:
    rows = [snap(i, asks=bands((p, 1000))) for i, p in enumerate((1, 2, 3, 4))]
    cal = calibrate_depth_costs(rows, notionals=[1], quantiles=[F(1, 2)])
    assert (cal.time_start, cal.time_end) == (rows[0].time, rows[-1].time)
    assert cal.snapshot_count == 4 and len(cal.input_digest) == 64
    assert cal == calibrate_depth_costs(rows, notionals=[1], quantiles=[F(1, 2)])
    assert cal.sides == ("buy", "sell") and len(cal.cells) == 2
    other = calibrate_depth_costs(rows[:-1], notionals=[1], quantiles=[F(1, 2)])
    assert other.input_digest != cal.input_digest
    with pytest.raises(dataclasses.FrozenInstanceError):
        cal.snapshot_count = 1  # type: ignore[misc]


def test_digest_depends_on_values_not_representation() -> None:
    def one(percentage: object, notional: object) -> DepthBandSnapshot:
        side = (DepthBand(percentage, notional),)  # type: ignore[arg-type]
        return DepthBandSnapshot(T0, side, side)

    def digest(snapshot: DepthBandSnapshot) -> str:
        return calibrate_depth_costs(
            [snapshot], notionals=[1], quantiles=[1]
        ).input_digest

    a = one(Decimal("1.0"), Decimal("5.00"))
    b = one(F(1), Decimal(5))
    c = one(F(1), Decimal(6))
    assert digest(a) == digest(b) != digest(c)


def test_utc_hour_buckets() -> None:
    rows = [
        snap(asks=bands((1, 50)), time=T0),
        snap(asks=bands((2, 50)), time=T0 + timedelta(minutes=1)),
        snap(asks=bands((3, 50)), time=T0 + timedelta(hours=1)),
    ]
    cal = calibrate_depth_costs(
        rows, notionals=[10], quantiles=[1], sides=["buy"], by_utc_hour=True
    )
    assert [(c.hour_utc, c.snapshot_count) for c in cal.cells] == [
        (None, 3),
        (0, 2),
        (1, 1),
    ]
    zero = cal.cell(side="buy", notional=10, hour_utc=0)
    one = cal.cell(side="buy", notional=10, hour_utc=1)
    assert zero.bounds[0].average_upper_bps == 200
    assert one.bounds[0].average_upper_bps == 300
    with pytest.raises(ValueError, match="no calibration cell"):
        cal.cell(side="buy", notional=10, hour_utc=5)


# --- capacity ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(50, 0), (99, 0), (100, 100), (150, 200), (260, 500), (370, 1000), (10**6, 1000)],
)
def test_capacity_single_snapshot_hand_computed(threshold, expected) -> None:
    result = depth_capacity(
        [HAND], side="buy", quantile=1, max_average_cost_bps=threshold
    )
    assert result.capacity_notional == expected
    assert result.snapshot_count == 1
    assert result.time_start == result.time_end == HAND.time
    assert len(result.input_digest) == 64


def test_capacity_quantile_selects_order_statistic() -> None:
    rows = [snap(i, asks=bands((1, n))) for i, n in enumerate((100, 200, 300))]
    caps = [
        depth_capacity(
            rows, side="buy", quantile=q, max_average_cost_bps=150
        ).capacity_notional
        for q in (F(1, 3), F(2, 3), 1)
    ]
    assert caps == [300, 200, 100]


def test_capacity_hour_filter_and_refusal() -> None:
    rows = [
        snap(asks=bands((1, 100)), time=T0),
        snap(asks=bands((1, 900)), time=T0 + timedelta(hours=2)),
    ]
    result = depth_capacity(
        rows, side="buy", quantile=1, max_average_cost_bps=100, hour_utc=2
    )
    assert result.capacity_notional == 900 and result.selected_snapshot_count == 1
    assert result.snapshot_count == 2 and result.hour_utc == 2
    with pytest.raises(ValueError, match="no snapshots"):
        depth_capacity(
            rows, side="buy", quantile=1, max_average_cost_bps=100, hour_utc=7
        )


def test_capacity_is_exact_boundary_of_calibrated_quantile() -> None:
    rng = random.Random(99)
    for _ in range(15):
        rows = []
        for i in range(rng.randint(1, 6)):
            count = rng.randint(1, 5)
            percentages = sorted(rng.sample(range(1, 30), count))
            cumulative = list(
                itertools.accumulate(rng.randint(0, 40) for _ in range(count))
            )
            cumulative[-1] += 5
            rows.append(snap(i, asks=bands(*zip(percentages, cumulative, strict=True))))
        quantile = F(rng.randint(1, 10), 10)
        threshold = F(rng.randint(50, 2500), rng.randint(1, 5))
        capacity = depth_capacity(
            rows, side="buy", quantile=quantile, max_average_cost_bps=threshold
        ).capacity_notional

        def upper(q: Fraction, rows=rows, quantile=quantile) -> Fraction | None:
            cal = calibrate_depth_costs(
                rows, notionals=[q], quantiles=[quantile], sides=["buy"]
            )
            return cal.cell(side="buy", notional=q).bounds[0].average_upper_bps

        if capacity > 0:
            value = upper(capacity)
            assert value is not None and value <= threshold
        bumped = upper(capacity + F(1, 10**6))
        assert bumped is None or bumped > threshold


# --- profile check -------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "assumed", "expected"),
    [
        (500, 10, CostAssumptionCheck.BELOW_OBSERVED_LOWER_BOUND),
        (500, 160, CostAssumptionCheck.BELOW_OBSERVED_LOWER_BOUND),
        (500, 200, CostAssumptionCheck.WITHIN_BOUNDS),
        (500, 260, CostAssumptionCheck.WITHIN_BOUNDS),
        (500, 261, CostAssumptionCheck.ABOVE_UPPER_BOUND),
        (50, 5, CostAssumptionCheck.UNRESOLVABLE),
        (50, 99.5, CostAssumptionCheck.UNRESOLVABLE),
        (50, 0, CostAssumptionCheck.UNRESOLVABLE),
        (50, 100, CostAssumptionCheck.WITHIN_BOUNDS),
        (50, 101, CostAssumptionCheck.ABOVE_UPPER_BOUND),
        (1001, 5, CostAssumptionCheck.BEYOND_OBSERVED_DEPTH),
    ],
)
def test_profile_check_every_classification(q, assumed, expected) -> None:
    result = order_cost_bounds(HAND, side="buy", notional=q)
    assert check_snapshot_assumption(result, assumed) is expected


def test_profile_check_on_aggregated_cell() -> None:
    cal = calibrate_depth_costs(
        [HAND], notionals=[50, 500, 1001], quantiles=[F(1, 2)], sides=["buy"]
    )

    def check(q: int, assumed: int) -> CostAssumptionCheck:
        cell = cal.cell(side="buy", notional=q)
        return cell.check_assumption(F(1, 2), assumed)

    assert check(50, 3) is CostAssumptionCheck.UNRESOLVABLE
    assert check(500, 100) is CostAssumptionCheck.BELOW_OBSERVED_LOWER_BOUND
    assert check(500, 200) is CostAssumptionCheck.WITHIN_BOUNDS
    assert check(500, 300) is CostAssumptionCheck.ABOVE_UPPER_BOUND
    assert check(1001, 300) is CostAssumptionCheck.BEYOND_OBSERVED_DEPTH
    with pytest.raises(ValueError, match="quantile was not calibrated"):
        cal.cell(side="buy", notional=50).check_assumption(F(1, 3), 3)


# --- refusals (and no echo of raw values) -----------------------------------------

SECRET = Decimal("123456.789012345")
OK = bands((1, 10), (2, 20))


def _snapshot_with(**changes: object) -> DepthBandSnapshot:
    values: dict[str, object] = {"time": T0, "bid_bands": OK, "ask_bands": OK}
    values.update(changes)
    return DepthBandSnapshot(**values)  # type: ignore[arg-type]


BAD_SNAPSHOTS = [
    {"time": datetime(2026, 1, 1)},
    {"time": datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=9)))},
    {"time": "2026-01-01"},
    {"ask_bands": ()},
    {"bid_bands": [(Decimal(1), Decimal(1))]},
    {"ask_bands": ((Decimal(1), Decimal(1), Decimal(1)),)},
    {"ask_bands": ((Decimal(0), Decimal(1)),)},
    {"ask_bands": ((-SECRET, Decimal(1)),)},
    {"ask_bands": ((Decimal(1), Decimal(1)), (Decimal(1), Decimal(2)))},
    {"ask_bands": ((Decimal(2), Decimal(1)), (Decimal(1), Decimal(2)))},
    {"ask_bands": ((Decimal(1), -SECRET),)},
    {"ask_bands": ((Decimal(1), Decimal(5)), (Decimal(2), Decimal(4)))},
    {"ask_bands": ((Decimal("NaN"), Decimal(1)),)},
    {"ask_bands": ((Decimal(1), Decimal("Infinity")),)},
    {"ask_bands": ((Decimal(1), Decimal("sNaN")),)},
    {"ask_bands": ((1.5, Decimal(1)),)},
    {"ask_bands": ((Decimal(1), 2.5),)},
    {"ask_bands": ((True, Decimal(1)),)},
    {"ask_bands": ((Decimal(1), "5"),)},
]


@pytest.mark.parametrize("changes", BAD_SNAPSHOTS)
def test_snapshot_refusals(changes) -> None:
    with pytest.raises(ValueError, match=r"\w") as caught:
        _snapshot_with(**changes)
    assert "123456" not in str(caught.value)


def test_exact_integers_and_fractions_are_accepted() -> None:
    snapshot = DepthBandSnapshot(
        T0,
        ((F(1, 2), 5), (1, 5)),  # type: ignore[arg-type]
        ((F(1, 2), 5),),  # type: ignore[arg-type]
    )
    assert snapshot.ask_bands[0].percentage == F(1, 2)
    assert _bounds(snapshot, 5, "sell").average_upper_bps == 50


def test_calibration_and_query_refusals() -> None:
    good = [HAND]
    grid = {"notionals": [1], "quantiles": [F(1, 2)]}
    with pytest.raises(ValueError, match="not be empty"):
        calibrate_depth_costs([], **grid)
    with pytest.raises(ValueError, match="sequence"):
        calibrate_depth_costs("abc", **grid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="DepthBandSnapshot"):
        calibrate_depth_costs([object()], **grid)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="strictly increasing"):
        calibrate_depth_costs([HAND, HAND], **grid)
    with pytest.raises(ValueError, match="strictly increasing"):
        calibrate_depth_costs([snap(1, asks=OK), snap(0, asks=OK)], **grid)
    for bad_notionals in ([], [0], [-SECRET], [1, 1], [2, 1], [float("nan")], [True]):
        with pytest.raises(ValueError, match=r"\w") as caught:
            calibrate_depth_costs(good, notionals=bad_notionals, quantiles=[1])
        assert "123456" not in str(caught.value)
    for bad_quantiles in (
        [],
        [0],
        [F(3, 2)],
        [F(1, 2), F(1, 2)],
        [0.5],
        [Decimal("NaN")],
    ):
        with pytest.raises(ValueError, match=r"\w"):
            calibrate_depth_costs(good, notionals=[1], quantiles=bad_quantiles)
    for bad_sides in ([], ["buy", "buy"], ["hold"]):
        with pytest.raises(ValueError, match="side"):
            calibrate_depth_costs(
                good,
                notionals=[1],
                quantiles=[1],
                sides=bad_sides,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="boolean"):
        calibrate_depth_costs(
            good,
            notionals=[1],
            quantiles=[1],
            by_utc_hour=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="DepthBandSnapshot"):
        order_cost_bounds(object(), side="buy", notional=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="side"):
        order_cost_bounds(HAND, side="hold", notional=1)  # type: ignore[arg-type]
    for bad in (0, -1, Decimal("NaN"), 1.0, True, "1"):
        with pytest.raises(ValueError, match=r"\w"):
            order_cost_bounds(HAND, side="buy", notional=bad)  # type: ignore[arg-type]


def test_capacity_and_classifier_refusals() -> None:
    for quantile in (0, F(3, 2), 0.5):
        with pytest.raises(ValueError, match=r"\w"):
            depth_capacity(
                [HAND],
                side="buy",
                quantile=quantile,  # type: ignore[arg-type]
                max_average_cost_bps=1,
            )
    for threshold in (-1, float("inf"), float("nan"), Decimal("NaN"), "5"):
        with pytest.raises(ValueError, match=r"\w"):
            depth_capacity(
                [HAND],
                side="buy",
                quantile=1,
                max_average_cost_bps=threshold,  # type: ignore[arg-type]
            )
    for hour in (24, -1, True, 1.0):
        with pytest.raises(ValueError, match="hour_utc"):
            depth_capacity(
                [HAND],
                side="buy",
                quantile=1,
                max_average_cost_bps=1,
                hour_utc=hour,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="side"):
        depth_capacity(
            [HAND],
            side="x",  # type: ignore[arg-type]
            quantile=1,
            max_average_cost_bps=1,
        )
    with pytest.raises(ValueError, match="empty"):
        depth_capacity([], side="buy", quantile=1, max_average_cost_bps=1)
    with pytest.raises(ValueError, match="non-negative"):
        classify_assumed_cost(-1, lower_bound_bps=0, upper_bound_bps=1)
    with pytest.raises(ValueError, match="not be below"):
        classify_assumed_cost(1, lower_bound_bps=5, upper_bound_bps=1)


# --- extreme magnitudes: refusals and exactness ---------------------------------


@pytest.mark.parametrize(
    "value",
    [
        Decimal("1E999999999"),
        Decimal("1E-999999999"),
        Decimal("1E309"),
        Decimal("1E-309"),
        Decimal("9" * 701),
        10**5000,
        F(1, 10**5000),
        F(10**5000 + 1, 10**5000),
    ],
    ids=[
        "huge-exponent",
        "tiny-exponent",
        "just-above-range",
        "just-below-range",
        "too-many-digits",
        "huge-int",
        "huge-denominator",
        "huge-fraction-near-one",
    ],
)
def test_non_representable_inputs_are_refused_without_overflow(value) -> None:
    with pytest.raises(ValueError, match="representable"):
        order_cost_bounds(HAND, side="buy", notional=value)
    with pytest.raises(ValueError, match="representable"):
        DepthBandSnapshot(T0, ((value, Decimal(1)),), OK)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="representable"):
        DepthBandSnapshot(T0, OK, ((Decimal(1), value),))  # type: ignore[arg-type]


def test_tiny_float_threshold_is_refused_but_zero_is_not() -> None:
    with pytest.raises(ValueError, match="representable"):
        depth_capacity([HAND], side="buy", quantile=1, max_average_cost_bps=5e-324)
    zero = depth_capacity([HAND], side="buy", quantile=1, max_average_cost_bps=0)
    assert zero.capacity_notional == 0


def test_zero_with_huge_decimal_exponent_is_cheap_and_valid() -> None:
    empty_first = ((Decimal(1), Decimal("0E+999999999")), (Decimal(2), Decimal(5)))
    snapshot = DepthBandSnapshot(T0, OK, empty_first)
    result = _bounds(snapshot, 1, "buy")
    assert result.marginal_lower_bps == 100 and result.marginal_upper_bps == 200
    assert result.within_first_band is False


@pytest.mark.parametrize("edge", [Decimal("1E308"), Decimal("1E-308")])
def test_range_edges_are_accepted(edge) -> None:
    snapshot = DepthBandSnapshot(T0, ((Decimal(1), edge),), ((Decimal(1), edge),))
    assert _bounds(snapshot, edge).average_upper_bps == 100


@pytest.mark.parametrize("exponent", [-300, -150, 0, 150, 300])
def test_notional_scale_metamorphic(exponent: int) -> None:
    scale = Decimal(f"1E{exponent}")
    scaled = snap(
        asks=tuple(
            DepthBand(b.percentage, b.cumulative_notional * scale) for b in HAND_ASKS
        )
    )
    for q in (50, 100, 200, 300, 500, 1000, 1001):
        base = order_cost_bounds(HAND, side="buy", notional=q)
        other = order_cost_bounds(scaled, side="buy", notional=Decimal(q) * scale)
        assert type(base) is type(other)
        if isinstance(base, BeyondObservedDepth):
            assert other.observed_depth == base.observed_depth * Fraction(scale)
            continue
        assert isinstance(other, OrderCostBounds)
        for field in (
            "marginal_lower_bps",
            "marginal_upper_bps",
            "average_lower_bps",
            "average_upper_bps",
            "within_first_band",
        ):
            assert getattr(other, field) == getattr(base, field)
        assert other.notional == base.notional * Fraction(scale)
    for threshold in (99, 100, 150, 260, 370):
        base_cap = depth_capacity(
            [HAND], side="buy", quantile=1, max_average_cost_bps=threshold
        )
        cap = depth_capacity(
            [scaled], side="buy", quantile=1, max_average_cost_bps=threshold
        )
        assert cap.capacity_notional == base_cap.capacity_notional * Fraction(scale)
    grid = [Decimal(q) * scale for q in (50, 500, 1000)]
    cal = calibrate_depth_costs([scaled], notionals=grid, quantiles=[1], sides=["buy"])
    assert cal.cell(side="buy", notional=grid[1]).bounds[0].average_upper_bps == 260
    first = cal.cell(side="buy", notional=grid[0])
    assert first.check_assumption(1, 5) is CostAssumptionCheck.UNRESOLVABLE


@pytest.mark.parametrize("exponent", [-300, -100, 100, 300])
def test_percentage_scale_metamorphic(exponent: int) -> None:
    scale = Decimal(f"1E{exponent}")
    scaled = snap(
        asks=tuple(
            DepthBand(b.percentage * scale, b.cumulative_notional) for b in HAND_ASKS
        )
    )
    for q in (50, 200, 500, 1000):
        base = _bounds(HAND, q)
        other = _bounds(scaled, q)
        for field in (
            "marginal_lower_bps",
            "marginal_upper_bps",
            "average_lower_bps",
            "average_upper_bps",
        ):
            assert getattr(other, field) == getattr(base, field) * Fraction(scale)
    base_cap = depth_capacity([HAND], side="buy", quantile=1, max_average_cost_bps=260)
    cap = depth_capacity(
        [scaled],
        side="buy",
        quantile=1,
        max_average_cost_bps=Fraction(260) * Fraction(scale),
    )
    assert cap.capacity_notional == base_cap.capacity_notional


def test_extreme_literal_values_are_exact() -> None:
    tiny = snap(asks=bands(("1E-300", "1E-300"), ("2E-300", "3E-300")))
    result = _bounds(tiny, Decimal("2E-300"))
    unit = F(1, 10**298)  # 1e-300 percent is 1e-298 bps
    assert result.marginal_lower_bps == unit and result.marginal_upper_bps == 2 * unit
    assert result.average_upper_bps == F(3, 2) * unit
    assert result.average_lower_bps == F(1, 2) * unit

    huge = snap(asks=bands(("1E300", "1E300"), ("2E300", "3E300")))
    result = _bounds(huge, Decimal("2E300"))
    assert result.marginal_upper_bps == 2 * 10**302
    assert result.average_upper_bps == F(3, 2) * 10**302
    assert result.average_lower_bps == F(1, 2) * 10**302

    cal = calibrate_depth_costs(
        [tiny], notionals=[Decimal("2E-300")], quantiles=[1], sides=["buy"]
    )
    cell = cal.cell(side="buy", notional=Decimal("2E-300"))
    # Own tiny outputs may be fed back to the classifier without range errors.
    assert cell.check_assumption(1, 1) is CostAssumptionCheck.ABOVE_UPPER_BOUND
    assert cell.check_assumption(1, 0) is CostAssumptionCheck.BELOW_OBSERVED_LOWER_BOUND
    capacity = depth_capacity([tiny], side="buy", quantile=1, max_average_cost_bps=unit)
    assert capacity.capacity_notional == F(1, 10**300)


def test_mixed_extreme_ratios_never_raise_arithmetic_errors() -> None:
    snapshot = snap(asks=bands(("1E-300", "1E-300"), ("1E300", "1E300")))
    for q in ("1E-300", "1E-299", "5E299", "1E300"):
        result = order_cost_bounds(snapshot, side="buy", notional=Decimal(q))
        assert isinstance(result, OrderCostBounds)
        assert result.average_lower_bps <= result.average_upper_bps
    cap = depth_capacity([snapshot], side="buy", quantile=1, max_average_cost_bps=1)
    assert 0 <= cap.capacity_notional <= Fraction(10) ** 300
    classify_assumed_cost(
        F(1, 10**320), lower_bound_bps=F(1, 10**400), upper_bound_bps=F(10**400)
    )
