"""Fixed Task18 reference-size cost beliefs, not realized native charges.

The archived declaration uses quote notional, symmetric fee/full-spread/
square-root-impact, and rates-only funding over (decision-24h, decision].
The four tables are indexed by (H24/H72, BASEL0/COST2L0, symbol, actor row,
one-way execution rate/funding return). BASEL2 uses BASEL0 beliefs.
No venue or historical data is opened here.
"""

from __future__ import annotations

import math

import numpy as np

from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.execution import ExecutionCostConfig

_PROTOCOL = build_task18_protocol()
_PLAN = _PROTOCOL.index_plan
_ECONOMICS = _PROTOCOL.economics
HORIZONS = tuple(sorted({c.horizon_hours for c in _PROTOCOL.configurations}))
COST_SCENARIOS = ("BASEL0", "COST2L0")
ACTOR_ROWS = _PLAN.actor_stop_exclusive - _PLAN.actor_start_inclusive
HOUR_NS = 3_600_000_000_000
COST_SHAPE = (
    len(HORIZONS),
    len(COST_SCENARIOS),
    len(_PROTOCOL.dataset.symbols),
    ACTOR_ROWS,
    2,
)


def build_cost_rates(dataset: MarketDataset) -> np.ndarray:
    """Prepare the fixed reference-cost proxy tables exactly once.

    The caller first binds the Dataset identity and full Task18 clock/roster.
    The returned dense table has no estimator, mutable state or native ledger.
    """
    if (
        dataset.symbols != _PROTOCOL.dataset.symbols
        or dataset.n_bars != _PROTOCOL.dataset.row_count
        or dataset.calendar_kind != MarketCalendarKind.CONTINUOUS
        or not dataset.regular_cadence
        or dataset.bar_hours != 1.0
        or any(unit != VolumeUnit.QUOTE_NOTIONAL for unit in dataset.volume_units)
    ):
        raise ValueError("Task18 costs require the frozen hourly quote-notional roster")
    econ = _ECONOMICS
    first, stop = _PLAN.actor_start_inclusive, _PLAN.actor_stop_exclusive
    for field, expected in (
        ("fee_rate", econ.base_fee_rate),
        ("maker_fee_rate", 0.0),
        ("taker_fee_rate", 0.0),
        ("spread_rate", econ.base_spread_rate),
        ("max_participation_rate", econ.maximum_participation_rate),
        ("borrow_available", econ.borrow_available),
        ("borrow_rate", econ.borrow_rate),
        ("cash_rate", econ.cash_rate),
    ):
        if not np.all(dataset.resolved_array(field) == expected):
            raise ValueError(f"Task18 cost declaration differs: {field}")
    # Identity v7 binds a price-weighted funding array, even for the old
    # rates-only witness. Its bytes must match rate times bar-close mark.
    # Do not silently substitute actual venue settlement marks or a v6 array.
    if dataset.funding_price_rate is None or not np.array_equal(
        dataset.funding_price_rate,
        dataset.funding_rate * dataset.resolved_array("mark_price"),
    ):
        raise ValueError("Task18 costs require the exact v7 bar-close funding proxy")
    if not np.array_equal(
        dataset.close[first:stop], dataset.resolved_array("mark_price")[first:stop]
    ):
        raise ValueError("Task18 cost/forecast valuation requires same-close marks")
    available = dataset.resolved_array("information_available")
    publication = dataset.resolved_array("available_at")
    counts = dataset.resolved_array("funding_event_count")
    due = dataset.resolved_array("funding_due")
    rates = dataset.funding_rate
    # Once over the available prefix, rather than quadratic prefix scans for
    # every decision, as in the historical implementation.
    prefix = available[:stop] & (publication[:stop] <= dataset.timestamps[:stop, None])
    if np.any(prefix & (due[:stop] != (counts[:stop] > 0))) or np.any(
        prefix & (counts[:stop] == 0) & (rates[:stop] != 0)
    ):
        raise ValueError("Task18 funding due/count/rates disagree in available prefix")
    lookback = econ.funding_observation_window_hours
    table = np.empty(COST_SHAPE, dtype=np.float64)
    for symbol in range(dataset.n_symbols):
        for offset, row in enumerate(range(first, stop)):
            left = row - lookback
            window = slice(left + 1, row + 1)
            if left < 0 or not prefix[window, symbol].all():
                raise ValueError("Task18 funding window is incomplete or unavailable")
            event_counts = counts[window, symbol]
            count = sum(int(c) for c in event_counts)
            if not (
                econ.funding_observed_event_count_minimum
                <= count
                <= econ.funding_observed_event_count_maximum
            ):
                raise ValueError("Task18 funding window has inadmissible event count")
            event_offsets = np.flatnonzero(event_counts > 0) + 1
            if (
                np.any(
                    np.diff(np.concatenate(([0], event_offsets)))
                    > econ.maximum_funding_close_gap_hours
                )
                or lookback - int(event_offsets[-1]) > econ.maximum_funding_age_hours
            ):
                raise ValueError(
                    "Task18 funding counted-close gap or age exceeds declaration"
                )
            volume = float(dataset.volume[row, symbol])
            if (
                not math.isfinite(volume)
                or volume <= 0
                or (
                    econ.impact_reference_notional_usdt
                    > volume * econ.maximum_participation_rate
                )
            ):
                raise ValueError(
                    "Task18 reference notional exceeds OPEN participation capacity"
                )
            one_way = (
                econ.base_fee_rate
                + econ.base_spread_rate
                + econ.impact_coefficient
                * math.sqrt(econ.impact_reference_notional_usdt / volume)
            )
            past_funding = math.fsum(float(rate) for rate in rates[window, symbol])
            for h, horizon in enumerate(HORIZONS):
                funding = past_funding * horizon / lookback
                if not math.isfinite(funding):
                    raise ValueError("Task18 funding proxy is not finite")
                table[h, 0, symbol, offset] = (one_way, funding)
                table[h, 1, symbol, offset] = (2.0 * one_way, funding)
    return table


def native_execution_cost(scenario: str) -> ExecutionCostConfig:
    """The frozen actual execution overlay, distinct from estimated beliefs."""
    declaration = next((s for s in _PROTOCOL.scenarios if s.name == scenario), None)
    if declaration is None:
        raise ValueError("unknown Task18 native scenario")
    if (
        not declaration.fee_multiplier
        == declaration.spread_multiplier
        == declaration.impact_multiplier
    ):
        raise ValueError("Task18 requires one common native execution multiplier")
    return ExecutionCostConfig(
        fee_rate=0.0,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=_ECONOMICS.impact_coefficient,
        multiplier=declaration.fee_multiplier,
        max_participation_rate=_ECONOMICS.maximum_participation_rate,
        processing_bar_volume_capacity=False,
        borrow_rate_multiplier=1.0,
        order_latency_bars=declaration.execution_latency_bars,
        random_seed=0,
    )
