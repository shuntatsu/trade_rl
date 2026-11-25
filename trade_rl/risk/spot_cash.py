"""Conservative decision-mark spot cash budget, not an execution-time guarantee."""

from __future__ import annotations

import math

from trade_rl.risk._cash_scalars import finite_cash_scalar


def spot_cash_target_factor(
    current: tuple[float, ...],
    target: tuple[float, ...],
    *,
    cash_fraction: float,
    gross_limit: float,
    reserve_rate: float,
) -> float:
    """Uniform scale of effective long-only spot target weights at one mark.

    All weights and cash are fractions of the same positive account equity.
    All scalar inputs must be native binary64-compatible real values, checked
    before narrowing (the same domain as the mixed cash projection).
    Reserve worst-case current plus target turnover; separately reserve the
    new-buy principal and its fees against existing cash, never unexecuted
    sales. Zero can be returned for an already over-budget account: reducing
    exposure is not prohibited by an infeasible reserve on current holdings.
    Caller owns causal mark and fee-envelope selection. Price gaps, pending
    orders, stochastic or unmodelled costs need actual execution-time controls.
    """
    if (
        type(current) is not tuple
        or type(target) is not tuple
        or not current
        or len(current) != len(target)
    ):
        raise ValueError("aligned nonempty spot weight tuples required")
    c = tuple(finite_cash_scalar(v, "current weight") for v in current)
    t = tuple(finite_cash_scalar(v, "target weight") for v in target)
    if min(*c, *t) < 0:
        raise ValueError("spot weights must be nonnegative")
    cash = finite_cash_scalar(cash_fraction, "cash fraction")
    limit = finite_cash_scalar(gross_limit, "gross limit")
    rate = finite_cash_scalar(reserve_rate, "reserve rate")
    if not 0 < limit <= 1 or not 0 <= rate < 1:
        raise ValueError("gross limit must be in (0, 1] and reserve rate in [0, 1)")
    try:
        g0, g1 = math.fsum(c), math.fsum(t)
    except OverflowError as exc:
        raise ValueError("spot gross must be finite") from exc
    if g1 == 0:
        return 1.0
    gross_budget = max(0.0, (limit - rate * g0) / (1 + rate))
    buy_budget = max(0.0, min(cash, 1 - g0)) / (1 + rate)
    upper = min(1.0, gross_budget / g1)

    def feasible(factor: float) -> bool:
        scaled = tuple(factor * v for v in t)
        return (
            math.fsum(scaled) <= gross_budget
            and math.fsum(max(v - old, 0.0) for v, old in zip(scaled, c, strict=True))
            <= buy_budget
        )

    if feasible(upper):
        # Leave a rounding unit of headroom on a binding limit; no gratuitous
        # change to an already admissible proposal.
        return math.nextafter(upper, 0.0) if upper < 1.0 else 1.0
    lower = 0.0
    for _ in range(64):
        middle = (lower + upper) / 2
        if feasible(middle):
            lower = middle
        else:
            upper = middle
    return math.nextafter(lower, 0.0)
