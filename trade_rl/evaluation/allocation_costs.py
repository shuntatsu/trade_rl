"""Declared reference-size costs on available, complete synthetic/development inputs.

Current-condition symmetric execution and past-funding extrapolation are proxies,
not calibrated future fills. Declarations/content pins supply no artifact read,
research or execution authority. Realized costs remain with the native ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.forecasts.stream import _after
from trade_rl.strategies.forecasts.training_trace import _timestamp

_NS_PER_HOUR = 3_600_000_000_000


def _positive_integer(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True, slots=True)
class DeclaredAllocationCostRecipe:
    """Owned immutable declarations; funding bounds cover complete (start, stop] intervals.

    Gap means left-window-boundary to first counted bar close, then successive
    counted bar closes. Age means decision minus last counted bar close. These
    proxies lose exact settlement times and may understate age by one native bar.
    """

    execution_cost: ExecutionCostConfig
    expected_execution_policy_digest: str
    reference_notional: float
    assumptions_identity: str
    assumptions_available_at: np.datetime64
    fee_assumption: str
    spread_assumption: str
    impact_assumption: str
    slippage_assumption: str
    borrow_assumption: str
    funding_source_identity: str
    funding_coverage_start: np.datetime64
    funding_coverage_stop: np.datetime64
    funding_lookback_hours: int
    funding_minimum_event_count: int
    funding_maximum_counted_bar_close_gap_hours: float
    funding_maximum_counted_bar_close_age_hours: float
    cash_zero_rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.execution_cost, ExecutionCostConfig):
            raise ValueError("execution_cost must be an ExecutionCostConfig")
        # Native frozen config accepts lists: reject the alias BEFORE policy hashing.
        if not isinstance(self.execution_cost.trigger_volume_fractions, tuple):
            raise ValueError("trigger_volume_fractions must be an immutable tuple")
        require_sha256(
            self.expected_execution_policy_digest,
            field="expected_execution_policy_digest",
        )
        if (
            self.expected_execution_policy_digest
            != self.execution_cost.execution_policy_digest
        ):
            raise ValueError("ExecutionCostConfig execution policy digest differs")
        if (
            self.execution_cost.order_type != "market"
            or self.execution_cost.order_latency_bars != 0
            or self.execution_cost.processing_bar_volume_capacity
        ):
            raise ValueError(
                "declared producer requires MARKET, zero extra latency and previous-bar capacity"
            )
        for name in (
            "assumptions_identity",
            "fee_assumption",
            "spread_assumption",
            "impact_assumption",
            "slippage_assumption",
            "borrow_assumption",
            "funding_source_identity",
            "cash_zero_rationale",
        ):
            object.__setattr__(
                self, name, require_non_empty(getattr(self, name), field=name)
            )
        for name in (
            "assumptions_available_at",
            "funding_coverage_start",
            "funding_coverage_stop",
        ):
            object.__setattr__(self, name, _timestamp(getattr(self, name), field=name))
        if self.funding_coverage_start >= self.funding_coverage_stop:
            raise ValueError("funding coverage bounds must be ordered")
        for name in ("funding_lookback_hours", "funding_minimum_event_count"):
            _positive_integer(getattr(self, name), field=name)
        for name in (
            "reference_notional",
            "funding_maximum_counted_bar_close_gap_hours",
            "funding_maximum_counted_bar_close_age_hours",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (
                name != "funding_maximum_counted_bar_close_age_hours" and value == 0
            ):
                raise ValueError(f"{name} must be positive (age may be zero)")
            object.__setattr__(self, name, float(value))

    def require_available(self, decision: np.datetime64) -> None:
        """Check declaration clocks before a loader and again at emitted decisions."""
        if (
            self.assumptions_available_at > decision
            or self.funding_coverage_start >= decision
        ):
            raise ValueError("cost source declarations are unavailable at decision")


def _recipe_declarations(recipe: DeclaredAllocationCostRecipe) -> dict[str, object]:
    declarations = {field.name: getattr(recipe, field.name) for field in fields(recipe)}
    declarations["execution_cost"] = recipe.execution_cost.execution_policy_payload()
    for name in (
        "assumptions_available_at",
        "funding_coverage_start",
        "funding_coverage_stop",
    ):
        declarations[name] = int(getattr(recipe, name).astype(np.int64))
    return declarations


def estimate_declared_horizon_costs(
    dataset: MarketDataset,
    *,
    decision_indices: tuple[int, ...],
    horizon_hours: int,
    recipe: DeclaredAllocationCostRecipe,
) -> tuple[HorizonCostEstimates, ...]:
    """Emit complete existing six-rate rows in Dataset-symbol/decision order.

    Restricts volume to quote notional and the financial clock to regular 24/7
    bars. Available current cap/size and OPEN fraction describe one reference
    order, without pending competition or promises about the next fill. Buy,
    sell and future exit reuse a symmetric nonlinear reference-size proxy.
    """
    if not isinstance(dataset, MarketDataset) or not isinstance(
        recipe, DeclaredAllocationCostRecipe
    ):
        raise ValueError("declared costs require typed Dataset and recipe")
    _positive_integer(horizon_hours, field="horizon_hours")
    if (
        not isinstance(decision_indices, tuple)
        or not decision_indices
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < dataset.n_bars - 1
            for index in decision_indices
        )
        or len(set(decision_indices)) != len(decision_indices)
    ):
        raise ValueError("decision_indices must be unique valid decision rows")
    decisions = tuple(sorted(decision_indices))
    if (
        dataset.calendar_kind is not MarketCalendarKind.CONTINUOUS
        or not dataset.regular_cadence
    ):
        raise ValueError("declared producer requires a regular continuous clock")
    if any(unit is not VolumeUnit.QUOTE_NOTIONAL for unit in dataset.volume_units):
        raise ValueError("declared producer requires quote-notional volume")
    bar_ns = int(
        (dataset.timestamps[1] - dataset.timestamps[0])
        .astype("timedelta64[ns]")
        .astype(np.int64)
    )
    lookback_ns = recipe.funding_lookback_hours * _NS_PER_HOUR
    if lookback_ns % bar_ns:
        raise ValueError("funding lookback must align exactly with native bars")
    lookback_bars = lookback_ns // bar_ns
    if dataset.identity_payload_json is None:
        raise ValueError(
            "declared costs require an explicit execution economics profile"
        )
    identity = parse_identity_json(dataset.identity_payload_json)
    profile = ExecutionEconomicsProfile.from_payload(
        identity.get("execution_economics")
    )
    cost = recipe.execution_cost
    arrays = {
        name: dataset.resolved_array(name)
        for name in (
            "fee_rate",
            "maker_fee_rate",
            "taker_fee_rate",
            "spread_rate",
            "max_participation_rate",
            "borrow_available",
            "borrow_rate",
            "cash_rate",
            "mark_price",
            "available_at",
            "information_available",
            "funding_due",
            "funding_event_count",
        )
    }
    expected_slippage = (
        cost.slippage_std
        * math.sqrt(2 / math.pi)
        * (1 + cost.tail_slippage_probability * (cost.tail_slippage_multiplier - 1))
    )
    declarations = _recipe_declarations(recipe)
    estimates = []
    for symbol_index, symbol in enumerate(dataset.symbols):
        for row in decisions:
            decision = dataset.timestamps[row]
            recipe.require_available(decision)
            left = row - lookback_bars
            if left < 0:
                raise ValueError("funding history lacks complete lookback intervals")
            boundary = dataset.timestamps[left]
            if (
                int((decision - boundary).astype("timedelta64[ns]").astype(np.int64))
                != lookback_ns
            ):
                raise ValueError(
                    "funding window boundary does not align with the clock"
                )
            if (
                recipe.funding_coverage_start > boundary
                or recipe.funding_coverage_stop < decision
            ):
                raise ValueError(
                    "declared funding coverage does not cover the complete window"
                )
            window = slice(left + 1, row + 1)
            if (
                not np.all(arrays["information_available"][window, symbol_index])
                or np.any(arrays["available_at"][window, symbol_index] > decision)
                or dataset.close[row, symbol_index]
                != arrays["mark_price"][row, symbol_index]
            ):
                raise ValueError(
                    "cost inputs require complete available intervals and same-close marks"
                )
            for name in (
                "fee_rate",
                "maker_fee_rate",
                "taker_fee_rate",
                "spread_rate",
                "max_participation_rate",
                "borrow_available",
                "borrow_rate",
            ):
                if arrays[name][row, symbol_index] != getattr(profile, name):
                    raise ValueError(
                        "execution economics profile disagrees with decision rows"
                    )
            if arrays["cash_rate"][row] != 0:
                raise ValueError(
                    "declared producer requires explicit zero cash interest"
                )
            prefix_available = arrays["information_available"][
                : row + 1, symbol_index
            ] & (arrays["available_at"][: row + 1, symbol_index] <= decision)
            prefix_counts = arrays["funding_event_count"][: row + 1, symbol_index]
            prefix_due = arrays["funding_due"][: row + 1, symbol_index]
            prefix_rates = dataset.funding_rate[: row + 1, symbol_index]
            if np.any(prefix_available & (prefix_due != (prefix_counts > 0))) or np.any(
                prefix_available & (prefix_counts == 0) & (prefix_rates != 0)
            ):
                raise ValueError(
                    "funding due/count/rates disagree in the available prefix"
                )
            counts = arrays["funding_event_count"][window, symbol_index]
            if sum(int(count) for count in counts) < recipe.funding_minimum_event_count:
                raise ValueError("funding history has insufficient counted events")
            counted_closes = dataset.timestamps[window][counts > 0]
            gap_ns = (
                np.diff(np.concatenate((np.array([boundary]), counted_closes)))
                .astype("timedelta64[ns]")
                .astype(np.int64)
            )
            age_ns = int(
                (decision - counted_closes[-1])
                .astype("timedelta64[ns]")
                .astype(np.int64)
            )
            if np.any(
                gap_ns / _NS_PER_HOUR
                > recipe.funding_maximum_counted_bar_close_gap_hours
            ):
                raise ValueError("funding counted-bar-close gap exceeds declaration")
            if (
                age_ns / _NS_PER_HOUR
                > recipe.funding_maximum_counted_bar_close_age_hours
            ):
                raise ValueError("funding counted-bar-close age exceeds declaration")
            turnover = float(
                dataset.market_notional(row, dataset.close[row])[symbol_index]
            )
            capacity = (
                turnover
                * min(cost.max_participation_rate, profile.max_participation_rate)
                * cost.trigger_volume_fractions[0]
            )
            if turnover <= 0 or recipe.reference_notional > capacity:
                raise ValueError("reference notional exceeds declared OPEN capacity")
            participation = recipe.reference_notional / turnover
            one_way = cost.multiplier * (
                cost.fee_rate
                + profile.fee_rate
                + cost.taker_fee_rate
                + profile.taker_fee_rate
                + cost.spread_rate
                + profile.spread_rate
                + cost.impact_rate * math.sqrt(participation)
                + expected_slippage
            )
            funding = (
                math.fsum(
                    float(rate) for rate in dataset.funding_rate[window, symbol_index]
                )
                * horizon_hours
                / recipe.funding_lookback_hours
            )
            borrow = (
                profile.borrow_rate
                * horizon_hours
                / (365 * 24)
                * cost.borrow_rate_multiplier
            )
            source = content_digest(
                {
                    "dataset_id": dataset.dataset_id,
                    "execution_economics": profile.to_payload(),
                    "recipe": declarations,
                    "symbol": symbol,
                    "decision_time": int(decision.astype(np.int64)),
                    "horizon_hours": horizon_hours,
                }
            )
            estimates.append(
                HorizonCostEstimates(
                    symbol=symbol,
                    decision_time=decision,
                    available_at=decision,
                    horizon_end=_after(decision, horizon_hours * 3600),
                    source_identity=source,
                    buy_cost=one_way,
                    sell_cost=one_way,
                    exit_cost=one_way,
                    funding_return=funding,
                    borrow_return=borrow,
                    cash_return=0.0,
                )
            )
    return tuple(estimates)


__all__ = ["DeclaredAllocationCostRecipe", "estimate_declared_horizon_costs"]
