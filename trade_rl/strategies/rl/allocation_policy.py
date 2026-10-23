"""Versioned allocation observations; no forecast lookup or account mutation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
    AllocationDecision,
)

ALLOCATION_OBSERVATION_SCHEMA = "allocation_account_observation_v1"
ALLOCATION_OBSERVATION_FIELDS = (
    "expected_simple_return",
    "return_variance",
    "buy_cost",
    "sell_cost",
    "exit_cost",
    "funding_return",
    "borrow_return",
    "cash_return",
    "baseline_target_weight",
    "current_weight",
    "cash_over_initial_capital",
    "equity_over_initial_capital",
    "maximum_drawdown",
    "pending_gross_over_initial_capital",
    "pending_order_count",
    "remaining_horizon_fraction",
)


def _capital(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("initial capital must be positive and finite")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class AllocationRuntimeProfile:
    economics_digest: str
    risk_digest: str
    initial_capital: float
    currency: str
    decision_interval_seconds: int
    economic_horizon_seconds: int
    insolvency_valuation: Literal["retain_debt"] = "retain_debt"
    calendar_kind: str = "continuous_24_7"
    execution_bar_hours: float | None = None

    def __post_init__(self) -> None:
        require_sha256(self.economics_digest, field="economics_digest")
        require_sha256(self.risk_digest, field="risk_digest")
        object.__setattr__(self, "initial_capital", _capital(self.initial_capital))
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("currency must be nonempty")
        for name in ("decision_interval_seconds", "economic_horizon_seconds"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if self.economic_horizon_seconds % self.decision_interval_seconds:
            raise ValueError("finite horizon must align with decisions")
        if self.insolvency_valuation != "retain_debt":
            raise ValueError("allocation profit profile must retain debt")
        if self.calendar_kind not in ("continuous_24_7", "session_calendar"):
            raise ValueError("runtime calendar kind is unsupported")
        hours: object = (
            self.decision_interval_seconds / 3600
            if self.execution_bar_hours is None
            else self.execution_bar_hours
        )
        if (
            isinstance(hours, bool)
            or not isinstance(hours, Real)
            or not math.isfinite(hours)
            or hours <= 0
        ):
            raise ValueError("execution bar hours must be positive and finite")
        object.__setattr__(self, "execution_bar_hours", float(hours))

    def payload(self) -> dict[str, object]:
        return {"schema": "allocation_runtime_profile_v1", **asdict(self)}


def allocation_recipe_payload(
    contract: AllocationActionContract,
    feature_names: tuple[str, ...],
    *,
    allocator: AfterCostTargetAllocator,
    expected_horizon_seconds: int,
    runtime_profile: AllocationRuntimeProfile,
) -> dict[str, object]:
    if (
        not feature_names
        or len(set(feature_names)) != len(feature_names)
        or any(not isinstance(n, str) or not n.strip() for n in feature_names)
    ):
        raise ValueError("recipe features must be unique nonempty names")
    horizon = _positive_integer(expected_horizon_seconds, "forecast horizon")
    return {
        "schema": "allocation_ppo_recipe_v1",
        "observation": {
            "schema": ALLOCATION_OBSERVATION_SCHEMA,
            "fields": list(ALLOCATION_OBSERVATION_FIELDS),
        },
        "action": contract.payload(),
        "action_mapping": "bounded_direct_or_residual_v1",
        "feature_names": list(feature_names),
        "allocator": asdict(allocator),
        "expected_horizon_seconds": horizon,
        "feature_preprocessing": "raw_available_values_v1",
        "terminal_valuation": "marked_continuation",
        "runtime_profile": runtime_profile.payload(),
    }


def allocation_recipe_digest(
    contract: AllocationActionContract,
    feature_names: tuple[str, ...],
    *,
    allocator: AfterCostTargetAllocator,
    expected_horizon_seconds: int,
    runtime_profile: AllocationRuntimeProfile,
) -> str:
    return content_digest(
        allocation_recipe_payload(
            contract,
            feature_names,
            allocator=allocator,
            expected_horizon_seconds=expected_horizon_seconds,
            runtime_profile=runtime_profile,
        )
    )


def encode_allocation_observation(
    decision: AllocationDecision, *, episode_steps: int
) -> np.ndarray:
    if (
        isinstance(episode_steps, bool)
        or not isinstance(episode_steps, int)
        or episode_steps < decision.remaining_steps
    ):
        raise ValueError("episode steps must contain the remaining horizon")
    inputs = decision.baseline.inputs
    context = decision.baseline.context
    values = (
        decision.feature_values
        + tuple(
            float(getattr(inputs, name))
            for name in (
                "expected_simple_return",
                "return_variance",
                "buy_cost",
                "sell_cost",
                "exit_cost",
                "funding_return",
                "borrow_return",
                "cash_return",
            )
        )
        + (
            decision.baseline.target_weight,
            context.current_weight,
            context.cash / decision.initial_capital,
            context.equity / decision.initial_capital,
            decision.max_drawdown,
            decision.pending_gross,
            float(decision.pending_count),
            decision.remaining_steps / episode_steps,
        )
    )
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.asarray(values, dtype=np.float32)
        except FloatingPointError as error:
            raise ValueError(
                "allocation observation must fit finite float32 values"
            ) from error
    if not np.isfinite(result).all():
        raise ValueError("allocation observation must be finite")
    return result
