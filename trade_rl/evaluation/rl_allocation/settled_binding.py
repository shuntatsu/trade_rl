"""Bind one settled account to separate actor and native reserve clocks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, timedelta

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.objectives import ObjectiveContract
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_settled_clock import (
    SettledAllocationClockContract,
)


@dataclass(frozen=True, slots=True)
class BoundSettledAllocationClock:
    objective: ObjectiveContract
    clock: SettledAllocationClockContract

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveContract) or not isinstance(
            self.clock, SettledAllocationClockContract
        ):
            raise ValueError(
                "settled binding requires its objective and distinct clock"
            )
        objective, clock = self.objective, self.clock
        if (
            objective.terminal_valuation != "settled"
            or objective.capital.account_mode != "independent_symbol"
            or len(objective.capital.initial_equities) != 1
            or objective.evaluation_stop_exclusive - objective.evaluation_start
            != timedelta(
                seconds=clock.physical_horizon_steps * clock.execution_interval_seconds
            )
        ):
            raise ValueError(
                "settled binding requires one account and the full physical horizon"
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "bound_allocation_settled_clock_v1",
            "objective_digest": self.objective.digest,
            "settled_clock_digest": self.clock.digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


def validate_settled_runtime_binding(
    dataset: MarketDataset,
    bound: BoundSettledAllocationClock,
    executor: MarketExecutor,
    risk: PreTradeRiskConfig,
    recipe_digest: str,
    *,
    start_index: int,
    actor_stop_index: int,
    settlement_stop_index: int,
) -> None:
    if not isinstance(bound, BoundSettledAllocationClock):
        raise ValueError("a distinct settled allocation binding is required")
    if (
        any(
            type(i) is not int
            for i in (start_index, actor_stop_index, settlement_stop_index)
        )
        or not 0
        <= start_index
        < actor_stop_index
        < settlement_stop_index
        < dataset.n_bars
        or actor_stop_index - start_index != bound.clock.actor_steps
        or settlement_stop_index - start_index != bound.clock.physical_horizon_steps
    ):
        raise ValueError(
            "settled actor/reserve interval differs from the Dataset declaration"
        )
    if (
        executor.dataset is not dataset
        or executor.insolvency_valuation != "retain_debt"
        or executor.cost.order_type != "market"
        or executor.cost.order_latency_bars != 0
    ):
        raise ValueError(
            "settled runtime requires its actual Dataset, MARKET, zero latency and retained debt"
        )
    objective = bound.objective
    if not 0 <= risk.drawdown_start < risk.drawdown_stop <= 0.20 or (
        objective.maximum_drawdown != risk.drawdown_stop
    ):
        raise ValueError("settled hard risk differs from the declared drawdown limit")
    if (
        objective.economics_digest != executor.execution_policy_digest
        or objective.risk_digest != content_digest(risk)
        or objective.deployment_recipe_digest != recipe_digest
    ):
        raise ValueError("settled profile or recipe differs from the actual runtime")
    times = dataset.timestamps[start_index : settlement_stop_index + 1].astype(
        "datetime64[ns]"
    )
    if np.any(np.isnat(times)) or np.any(
        np.diff(times.astype(np.int64))
        != bound.clock.execution_interval_seconds * 1_000_000_000
    ):
        raise ValueError("Dataset intervals differ from the settled physical clock")
    for actual, declared in (
        (times[0], objective.evaluation_start),
        (times[-1], objective.evaluation_stop_exclusive),
    ):
        expected = np.datetime64(declared.astimezone(UTC).replace(tzinfo=None), "ns")
        if actual != expected:
            raise ValueError(
                "Dataset endpoints differ from the declared settled UTC horizon"
            )


__all__ = ["BoundSettledAllocationClock", "validate_settled_runtime_binding"]
