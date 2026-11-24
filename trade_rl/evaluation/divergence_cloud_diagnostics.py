"""Read-only single-symbol divergence/cloud portfolio replay diagnostics.

This module never estimates trades, invents fills, replays its own accounting,
or labels a known-used historical study as an independent economic outcome.
Net directional contributions are buckets of *native book equity changes*
over executed intervals, assigned to their observed position direction.
They are not causally identified alpha or a hypothetical trade-by-trade P&L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from trade_rl.simulation.quantities import parse_quantity


class LedgerInterval(Protocol):
    """Minimal native shared replay interval evidence needed by diagnostics."""

    @property
    def start_index(self) -> int: ...

    @property
    def next_index(self) -> int: ...

    @property
    def exact_quantities_before(self) -> tuple[str, ...]: ...

    @property
    def exact_quantities_after(self) -> tuple[str, ...]: ...

    @property
    def portfolio_value_before(self) -> float: ...

    @property
    def portfolio_value_after(self) -> float: ...

    @property
    def interval_cost(self) -> float: ...

    @property
    def interval_funding(self) -> float: ...

    @property
    def interval_borrow_cost(self) -> float: ...


class PortfolioDecision(Protocol):
    index: int
    proposal_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DirectionalContribution:
    """Signed USDT equity delta, costs, and elapsed intervals by observed side."""

    net_long_usdt: float
    net_short_usdt: float
    net_flat_usdt: float
    net_reversal_usdt: float
    net_total_usdt: float
    long_cost_usdt: float
    short_cost_usdt: float
    flat_cost_usdt: float
    reversal_cost_usdt: float
    long_funding_usdt: float
    short_funding_usdt: float
    long_intervals: int
    short_intervals: int
    flat_intervals: int
    reversal_intervals: int


@dataclass(frozen=True, slots=True)
class OpposingExitObservation:
    """Ex-post mark-price movement after an observed C3 opposite-signal exit."""

    index: int
    direction: str
    exit_filled_flat: bool
    c2_same_side_at_event: bool
    future_signed_returns_pct: dict[int, float]
    c2_continued_same_side: dict[int, bool]


def _sign_quantity(raw: str) -> int:
    if not isinstance(raw, str):
        raise ValueError("ledger quantity must be an exact numeric string")
    try:
        qty = parse_quantity(raw)
    except (ValueError, OverflowError) as error:
        raise ValueError("invalid native ledger exact quantity") from error
    return int((qty > 0) - (qty < 0))


def _single_sign(before: tuple[str, ...], after: tuple[str, ...]) -> int:
    if len(before) != 1 or len(after) != 1:
        raise ValueError("diagnostics require one symbol in one shared account")
    left = _sign_quantity(before[0])
    right = _sign_quantity(after[0])
    if left and right and left != right:
        return 2  # reversal, not automatically allocable to long or short
    return left or right


def _validate_contiguous(intervals: Sequence[LedgerInterval]) -> None:
    if not intervals:
        raise ValueError("native replay ledger intervals must be nonempty")
    previous: LedgerInterval | None = None
    for interval in intervals:
        if interval.next_index != interval.start_index + 1:
            raise ValueError("ledger interval must cover exactly one base bar")
        if (
            len(interval.exact_quantities_before) != 1
            or len(interval.exact_quantities_after) != 1
        ):
            raise ValueError("native single-symbol ledger required")
        _single_sign(interval.exact_quantities_before, interval.exact_quantities_after)
        for value in (
            interval.portfolio_value_before,
            interval.portfolio_value_after,
            interval.interval_cost,
            interval.interval_funding,
            interval.interval_borrow_cost,
        ):
            if not math.isfinite(value):
                raise ValueError("nonfinite native interval evidence")
        if interval.interval_cost < 0 or interval.interval_borrow_cost < 0:
            raise ValueError("native cost fields must be non-negative")
        if previous is not None:
            if previous.next_index != interval.start_index:
                raise ValueError("gapped native interval index")
            if previous.exact_quantities_after != interval.exact_quantities_before:
                raise ValueError("inconsistent native quantity chain")
            if not math.isclose(
                previous.portfolio_value_after,
                interval.portfolio_value_before,
                abs_tol=1e-8,
                rel_tol=1e-11,
            ):
                raise ValueError("inconsistent native portfolio equity chain")
        previous = interval


def attribute_single_symbol_ledger(
    intervals: Sequence[LedgerInterval],
) -> DirectionalContribution:
    """Partition all native equity deltas by position sign without re-accounting."""
    _validate_contiguous(intervals)
    net: dict[int, list[float]] = {-1: [], 0: [], 1: [], 2: []}
    costs: dict[int, list[float]] = {-1: [], 0: [], 1: [], 2: []}
    funding: dict[int, list[float]] = {-1: [], 0: [], 1: [], 2: []}
    for entry in intervals:
        side = _single_sign(entry.exact_quantities_before, entry.exact_quantities_after)
        net[side].append(entry.portfolio_value_after - entry.portfolio_value_before)
        costs[side].append(entry.interval_cost)
        funding[side].append(entry.interval_funding)
    long = math.fsum(net[1])
    short = math.fsum(net[-1])
    flat = math.fsum(net[0])
    reversal = math.fsum(net[2])
    total = math.fsum((long, short, flat, reversal))
    observed_total = (
        intervals[-1].portfolio_value_after - intervals[0].portfolio_value_before
    )
    if not math.isclose(total, observed_total, abs_tol=1e-7, rel_tol=1e-11):
        raise ValueError("directional attribution did not conserve native equity")
    return DirectionalContribution(
        net_long_usdt=long,
        net_short_usdt=short,
        net_flat_usdt=flat,
        net_reversal_usdt=reversal,
        net_total_usdt=total,
        long_cost_usdt=math.fsum(costs[1]),
        short_cost_usdt=math.fsum(costs[-1]),
        flat_cost_usdt=math.fsum(costs[0]),
        reversal_cost_usdt=math.fsum(costs[2]),
        long_funding_usdt=math.fsum(funding[1]),
        short_funding_usdt=math.fsum(funding[-1]),
        long_intervals=len(net[1]),
        short_intervals=len(net[-1]),
        flat_intervals=len(net[0]),
        reversal_intervals=len(net[2]),
    )


def inspect_opposing_divergence_exits(
    *,
    c3_intervals: Sequence[LedgerInterval],
    c3_decisions: Sequence[PortfolioDecision],
    c2_intervals: Sequence[LedgerInterval],
    close: np.ndarray,
    cloud_state: np.ndarray,
    bull_4h: np.ndarray,
    bear_4h: np.ndarray,
    horizons: tuple[int, ...] = (24, 72),
    start_index: int,
    stop_index: int,
) -> tuple[OpposingExitObservation, ...]:
    """Look *after* C3 exits to describe price moves, never to trade or fit.

    Reported future signed price returns exclude any financing, execution,
    differences in C2 exposure and costs. They MUST NOT be summed as missed
    realized profit or treated as independent events.
    """
    prices = np.asarray(close, dtype=np.float64)
    state = np.asarray(cloud_state, dtype=np.int8)
    bullish = np.asarray(bull_4h)
    bearish = np.asarray(bear_4h)
    if (
        prices.ndim != 1
        or prices.shape != state.shape
        or prices.shape != bullish.shape
        or prices.shape != bearish.shape
        or not np.isfinite(prices).all()
        or np.any(prices <= 0)
    ):
        raise ValueError("price and signal arrays require finite aligned values")
    if not (0 <= start_index < stop_index < len(prices)):
        raise ValueError("study bounds must be within the supplied market")
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(type(h) is not int or h < 1 for h in horizons)
    ):
        raise ValueError("future horizons must be distinct positive integers")

    _validate_contiguous(c3_intervals)
    _validate_contiguous(c2_intervals)
    if (
        c3_intervals[0].start_index != start_index
        or c2_intervals[0].start_index != start_index
        or c3_intervals[-1].next_index != stop_index
        or c2_intervals[-1].next_index != stop_index
    ):
        raise ValueError("two native ledgers must cover the same exact study bounds")
    by_index = {decision.index: decision for decision in c3_decisions}
    if len(by_index) != len(c3_decisions):
        raise ValueError("duplicate C3 decision index")
    c2_by_index = {entry.start_index: entry for entry in c2_intervals}
    observations: list[OpposingExitObservation] = []
    for entry in c3_intervals:
        i = entry.start_index
        if i >= len(prices) or i not in by_index:
            continue  # includes terminal settlement interval with no decision
        direction = _sign_quantity(entry.exact_quantities_before[0])
        if direction not in (-1, 1):
            continue
        if state[i] != direction or not (
            bearish[i] > 0 if direction > 0 else bullish[i] > 0
        ):
            continue
        decision = by_index[i]
        if len(decision.proposal_weights) != 1 or decision.proposal_weights[0] != 0:
            continue
        other = c2_by_index[i]
        same = _sign_quantity(other.exact_quantities_before[0]) == direction
        observed: dict[int, float] = {}
        continued: dict[int, bool] = {}
        for hours in horizons:
            if i + hours > stop_index:
                continue
            observed[hours] = float(
                100 * direction * (prices[i + hours] / prices[i] - 1)
            )
            continued[hours] = same and all(
                _sign_quantity(c2_by_index[j].exact_quantities_before[0]) == direction
                for j in range(i, i + hours)
            )
        observations.append(
            OpposingExitObservation(
                index=i,
                direction="long" if direction > 0 else "short",
                exit_filled_flat=(_sign_quantity(entry.exact_quantities_after[0]) == 0),
                c2_same_side_at_event=same,
                future_signed_returns_pct=observed,
                c2_continued_same_side=continued,
            )
        )
    return tuple(observations)


__all__ = [
    "DirectionalContribution",
    "OpposingExitObservation",
    "attribute_single_symbol_ledger",
    "inspect_opposing_divergence_exits",
]
