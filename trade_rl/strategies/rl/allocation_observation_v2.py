"""Immutable v2 layout declaration; numeric generation is not implemented.

Projection/sorting identifiers specify requirements for a subsequent encoder.
This contract has no account reader, policy consumer or training integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from trade_rl.artifacts.hashing import content_digest

ALLOCATION_OBSERVATION_SCHEMA = "allocation_account_observation_v2"
FORECAST_FIELDS = tuple(
    "expected_simple_return return_variance buy_cost sell_cost exit_cost "
    "funding_return borrow_return cash_return baseline_target_weight".split()
)
ACCOUNT_FIELDS = tuple(
    "position_notional_over_initial_capital current_weight cash_over_initial_capital "
    "equity_over_initial_capital peak_over_initial_capital current_drawdown maximum_drawdown "
    "margin_used_over_initial_capital maintenance_margin maintenance_requirement_over_initial_capital "
    "margin_deficit_over_initial_capital observable_tradable remaining_horizon_fraction".split()
)
ORDER_FIELDS = tuple(
    "presence requested_notional_over_initial_capital cumulative_quantity_notional_over_initial_capital "
    "remaining_notional_over_initial_capital cumulative_filled_notional_over_initial_capital "
    "submission_reference_over_current_mark decision_equity_over_initial_capital submit_offset eligible_offset "
    "expiry_presence expiry_offset trigger_presence trigger_offset last_transition_presence "
    "last_transition_offset reduce_only tif_ioc tif_day tif_gtc status_submitted "
    "status_latency_wait status_eligible status_triggered status_partially_filled".split()
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _native_capital(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("initial_capital must be a finite non-boolean number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(
            "initial_capital is outside its finite reporting range"
        ) from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError("initial_capital must normalize to a positive finite float")
    return result


@dataclass(frozen=True, slots=True)
class AllocationObservationSchema:
    feature_names: tuple[str, ...]
    max_active_orders: int
    initial_capital: float
    episode_steps: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_names, tuple)
            or not self.feature_names
            or any(
                not isinstance(name, str) or not name.strip()
                for name in self.feature_names
            )
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError(
                "features must be unique nonempty names in an immutable tuple"
            )
        if _positive_integer(self.max_active_orders, "max_active_orders") > 64:
            raise ValueError("max_active_orders must be within [1, 64]")
        object.__setattr__(
            self, "initial_capital", _native_capital(self.initial_capital)
        )
        _positive_integer(self.episode_steps, "episode_steps")

    @property
    def fields(self) -> tuple[str, ...]:
        return (
            tuple(f"feature:{name}" for name in self.feature_names)
            + FORECAST_FIELDS
            + ACCOUNT_FIELDS
            + tuple(
                f"order[{slot}].{name}"
                for slot in range(self.max_active_orders)
                for name in ORDER_FIELDS
            )
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": ALLOCATION_OBSERVATION_SCHEMA,
            "feature_names": list(self.feature_names),
            "max_active_orders": self.max_active_orders,
            "initial_capital": self.initial_capital,
            "episode_steps": self.episode_steps,
            "fields": list(self.fields),
            "projection": "exact_reporting_decimal_toward_zero_float32_v1",
            "raw_feature_projection": "ordinary_guarded_float32_v1",
            "order_sort": "lexicographic_exact_economic_slot_facts_v1",
            "order_overflow": "reject",
            "unused_slots": "all_zero",
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


__all__ = ["AllocationObservationSchema"]
