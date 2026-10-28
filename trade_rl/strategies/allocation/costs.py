"""Derive the allocator's per-unit-weight transaction cost from an execution profile."""

from __future__ import annotations

from trade_rl.simulation.execution import ExecutionCostConfig


def one_way_cost_rate(cost: ExecutionCostConfig) -> float:
    """Return the deterministic one-way proportional cost of a full-notional trade.

    This mirrors the executor's unit cost for the configured order type:
    ``multiplier * (fee + venue fee + spread factor * spread)``. The result is a
    **lower bound** on the executor's realized cost for the same profile: the
    participation dependent impact and random slippage are not included, and
    per-row dataset fee/spread overrides are ignored (they can move the realized
    cost either way). The allocator therefore trades slightly more than a fully
    cost-aware one would. Pass explicit, larger ``cost_rates`` to model this, and
    state the bound whenever results are reported.
    """

    if not isinstance(cost, ExecutionCostConfig):
        raise TypeError("cost must be an ExecutionCostConfig")
    if cost.order_type == "limit":
        venue_fee = cost.maker_fee_rate
        spread_factor = 0.5
    else:
        venue_fee = cost.taker_fee_rate
        spread_factor = 1.0
    return float(
        cost.multiplier * (cost.fee_rate + venue_fee + spread_factor * cost.spread_rate)
    )


def one_way_cost_rates(
    cost: ExecutionCostConfig, *, n_symbols: int
) -> tuple[float, ...]:
    """Return the same one-way cost rate for every symbol of the roster."""

    if type(n_symbols) is not int or n_symbols < 1:
        raise ValueError("n_symbols must be a positive integer")
    return (one_way_cost_rate(cost),) * n_symbols


__all__ = ["one_way_cost_rate", "one_way_cost_rates"]
