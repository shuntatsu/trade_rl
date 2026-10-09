"""Opt-in native NONRL/cash terminal closing; no Study or policy admission.

The marked prefix stays unchanged. Reserved processing bars belong to a distinct
full-horizon evaluation contract, and use its actual account and executor. These
detached runtime observations are not a serialized independent-evidence reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import fsum
from typing import Any

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation import _execute_allocation_target
from trade_rl.evaluation.allocation_global_execution import (
    GlobalAllocationExecutionPlan,
    ObservedGlobalAllocationExecution,
    declare_global_allocation_execution,
    run_declared_global_allocation_execution,
)
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    FinancialClockContract,
    net_equity_increment,
)
from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.global_execution_context import (
    allocation_array_pin,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    native_allocation_execution_facts,
)
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold
from trade_rl.evaluation.runs import build_candidate_run_provenance


def _close_source(env: AllocationTradingEnv, stop: int) -> str:
    dataset = env.dataset
    arrays = {}
    for name, values in dataset.identity_arrays().items():
        if name.startswith(("feature", "global_feature")):
            continue
        selected = values
        if (
            name != "contract_multipliers"
            and values.ndim
            and values.shape[0] == dataset.n_bars
        ):
            selected = values[env.stop_index : stop + 1]
        if not np.isfinite(selected).all():
            raise ValueError("terminal financial source must be finite")
        arrays[name] = allocation_array_pin(selected)
    return content_digest(
        {
            "schema": "allocation_terminal_financial_source_v1",
            "dataset_id": dataset.dataset_id,
            "dataset_contract": dataset.identity_contract_payload(),
            "range": [env.stop_index, stop],
            "arrays": arrays,
        }
    )


def _terminal_recipe(
    native_plan: GlobalAllocationExecutionPlan,
    stop: int,
    clocks: tuple[int, ...],
    source: str,
    clock: FinancialClockContract,
) -> dict[str, Any]:
    return {
        "schema": "allocation_terminal_recipe_v1",
        "native_plan_digest": content_digest(native_plan.payload),
        "policy_range": native_plan.payload["common"]["range"],
        "closing_range": [native_plan.payload["common"]["range"][1], stop],
        "closing_clocks_ns": list(clocks),
        "closing_source_digest": source,
        "clock": clock.payload(),
        "terminal_valuation": "settled",
        "target": "zero_through_unchanged_hard_risk",
        "coverage": "all_reserved_processing_bars_even_when_flat",
    }


@dataclass(frozen=True, slots=True)
class TerminalAllocationExecutionPlan:
    """Frozen evaluation-only terminal window; no learned-policy authorization."""

    native_plan: GlobalAllocationExecutionPlan
    settlement_stop_index: int
    closing_clocks_ns: tuple[int, ...]
    closing_source_digest: str
    bound: BoundObjectiveClock

    def __post_init__(self) -> None:
        if type(self.native_plan) is not GlobalAllocationExecutionPlan:
            raise ValueError("terminal closing requires an exact native prefix plan")
        self.native_plan.__post_init__()
        raw = self.native_plan.payload
        common, cell = raw["common"], raw["cell"]
        if cell["kind"] not in {"nonrl", "cash"} or cell["fee"] is not None:
            raise ValueError("terminal closing supports only standalone NONRL/cash")
        if (
            type(self.settlement_stop_index) is not int
            or self.settlement_stop_index <= common["range"][1]
            or type(self.closing_clocks_ns) is not tuple
            or len(self.closing_clocks_ns)
            != self.settlement_stop_index - common["range"][1] + 1
            or any(type(t) is not int for t in self.closing_clocks_ns)
            or self.closing_clocks_ns[0] != common["source_clocks_ns"][-1]
        ):
            raise ValueError("terminal closing requires a nonempty exact clock window")
        require_sha256(self.closing_source_digest, field="closing_source_digest")
        if type(self.bound) is not BoundObjectiveClock:
            raise ValueError("terminal closing requires a distinct bound full clock")
        self.bound.__post_init__()
        objective, clock = self.bound.objective, self.bound.clock
        interval = clock.execution_interval_seconds
        if (
            not clock.terminal_profit_aligned
            or clock.decision_interval_seconds != interval
            or objective.terminal_valuation != "settled"
            or objective.capital.account_mode != "independent_symbol"
            or objective.capital.initial_equities != (common["initial_capital"],)
            or objective.economics_digest != common["economics_digest"]
            or objective.risk_digest != common["risk_digest"]
            or clock.economic_horizon_seconds
            != (self.settlement_stop_index - common["range"][0]) * interval
            or any(
                b - a != interval * 10**9
                for a, b in zip(self.closing_clocks_ns, self.closing_clocks_ns[1:])
            )
            or objective.deployment_recipe_digest != content_digest(self.recipe)
        ):
            raise ValueError("terminal objective/clock differs from its native window")

    @property
    def recipe(self) -> dict[str, Any]:
        return _terminal_recipe(
            self.native_plan,
            self.settlement_stop_index,
            self.closing_clocks_ns,
            self.closing_source_digest,
            self.bound.clock,
        )

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": "allocation_terminal_execution_plan_v1",
            "native_plan": self.native_plan.payload,
            "recipe": self.recipe,
            "objective": self.bound.objective.payload(),
            "clock": self.bound.clock.payload(),
            "bound_digest": self.bound.digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload)


def declare_terminal_allocation_execution(
    folds: tuple[WalkForwardFold, ...],
    env: AllocationTradingEnv,
    native_plan: GlobalAllocationExecutionPlan,
    *,
    settlement_stop_index: int,
) -> TerminalAllocationExecutionPlan:
    if (
        type(env) is not AllocationTradingEnv
        or type(native_plan) is not GlobalAllocationExecutionPlan
    ):
        raise ValueError("terminal declaration requires native environment and plan")
    native_plan.__post_init__()
    cell, common = native_plan.payload["cell"], native_plan.payload["common"]
    if cell["kind"] not in {"nonrl", "cash"} or cell["fee"] is not None:
        raise ValueError("terminal closing supports only standalone NONRL/cash")
    if (
        type(settlement_stop_index) is not int
        or not env.stop_index < settlement_stop_index < env.dataset.n_bars
    ):
        raise ValueError("terminal reserve must be nonempty and inside the Dataset")
    actual = declare_global_allocation_execution(
        folds,
        env,
        kind=cell["kind"],
        scenario=common["scenario"],
        expected_implementation_digest=common["implementation_digest"],
        expected_runtime_digest=common["runtime_environment_digest"],
    )
    if actual != native_plan:
        raise ValueError("terminal current native prefix differs from frozen plan")
    clocks = tuple(
        int(t)
        for t in env.dataset.timestamps[env.stop_index : settlement_stop_index + 1]
        .astype("datetime64[ns]")
        .astype("int64")
    )
    clock = replace(
        env.bound.clock,
        economic_horizon_seconds=(settlement_stop_index - env.start_index)
        * env.bound.clock.execution_interval_seconds,
    )
    source = _close_source(env, settlement_stop_index)
    recipe = _terminal_recipe(native_plan, settlement_stop_index, clocks, source, clock)
    objective = replace(
        env.bound.objective,
        evaluation_stop_exclusive=env.dataset.timestamps[settlement_stop_index]
        .astype("datetime64[us]")
        .astype(datetime)
        .replace(tzinfo=UTC),
        terminal_valuation="settled",
        deployment_recipe_digest=content_digest(recipe),
    )
    return TerminalAllocationExecutionPlan(
        native_plan,
        settlement_stop_index,
        clocks,
        source,
        BoundObjectiveClock(objective, clock),
    )


def _closing_binding(
    env: AllocationTradingEnv, plan: TerminalAllocationExecutionPlan
) -> None:
    env.validate_binding()
    runtime = plan.native_plan.payload["cell"]["runtime"]
    if (
        env.executor.cost.execution_policy_payload() != runtime["cost_payload"]
        or env.recipe != runtime["recipe"]
        or _close_source(env, plan.settlement_stop_index) != plan.closing_source_digest
    ):
        raise ValueError("terminal native runtime or frozen financial source changed")


def _provenance(plan: TerminalAllocationExecutionPlan) -> None:
    actual, common = (
        build_candidate_run_provenance(),
        plan.native_plan.payload["common"],
    )
    if any(
        actual[name] != common[name]
        for name in ("implementation_digest", "runtime_environment_digest")
    ):
        raise ValueError("terminal implementation/runtime provenance changed")


@dataclass(frozen=True, slots=True)
class ObservedTerminalAllocationExecution:
    """Runtime observation only; no from-payload Study or approval adapter."""

    native_execution: ObservedGlobalAllocationExecution
    _bytes: bytes

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @property
    def digest(self) -> str:
        return content_digest(self.payload)


def _observation(
    plan: TerminalAllocationExecutionPlan,
    native: dict[str, Any] | None,
    closing: list[dict[str, Any]],
    opening: str | None,
    *,
    error: Exception | None = None,
    attempted: int | None = None,
) -> dict[str, Any]:
    rows = (
        []
        if native is None
        else [
            r["native"]
            for r in native["rows"]
            if r["native"] is not None and r["native_validation_failure"] is None
        ]
    )
    prefix_final = None if not rows else rows[-1]["book"]
    final = prefix_final if not closing else closing[-1]["native"]["book"]
    active = (
        ([] if not rows else rows[-1]["active_orders"])
        if not closing
        else closing[-1]["native"]["active_orders"]
    )
    capital = plan.bound.objective.capital.initial_equities[0]
    rewards = [
        net_equity_increment(
            r["proposal"]["decision"]["baseline"]["context"]["equity"],
            r["book"]["equity"],
            initial_capital=capital,
        )
        for r in rows
    ] + [r["fixed_capital_reward"] for r in closing]
    coverage = (
        native is not None
        and native["status"] == "completed"
        and len(closing) == len(plan.closing_clocks_ns) - 1
        and all(row["clock_valid"] for row in closing)
    )
    stopped = final is not None and final["termination_reason"] is not None
    complete = (
        error is None
        and coverage
        and not stopped
        and final is not None
        and final["equity"] > 0
        and all(q == "0" for q in final["exact_quantities"])
        and not active
    )
    profit = (
        None
        if final is None
        else plan.bound.objective.net_profit_rate((final["equity"],))
    )
    return {
        "schema": "allocation_terminal_runtime_observation_v1",
        "plan": plan.payload,
        "plan_digest": plan.digest,
        "status": "economic_stop"
        if stopped or (native is not None and native["status"] == "economic_stop")
        else "integrity_failure"
        if error is not None
        else "completed",
        "native_receipt": native,
        "closing_opening_state_digest": opening,
        "closing_rows": closing,
        "final_book": final,
        "final_active_orders": active,
        "full_close_coverage": coverage,
        "settlement_complete": complete,
        "risk_eligible": complete
        and final is not None
        and final["max_drawdown"] <= plan.bound.objective.maximum_drawdown,
        "marked_prefix_profit_rate": None
        if prefix_final is None
        else (prefix_final["equity"] - capital) / capital,
        "observed_profit_rate": profit,
        "settled_profit_rate": profit if complete else None,
        "fixed_capital_rewards": rewards,
        "fixed_capital_reward_sum": fsum(rewards),
        "attempted_start_index": attempted,
        "failure": None
        if error is None
        else {"error_type": type(error).__name__, "message": str(error)},
        "claim_scope": "bounded_native_software_observation_no_economic_qualification",
    }


def run_terminal_allocation_execution(
    folds: tuple[WalkForwardFold, ...],
    env: AllocationTradingEnv,
    plan: TerminalAllocationExecutionPlan,
) -> ObservedTerminalAllocationExecution:
    if type(plan) is not TerminalAllocationExecutionPlan:
        raise ValueError("terminal execution requires its exact frozen plan")
    plan.__post_init__()
    current = declare_terminal_allocation_execution(
        folds, env, plan.native_plan, settlement_stop_index=plan.settlement_stop_index
    )
    if current != plan:
        raise ValueError("terminal current declaration differs from frozen plan")
    closing: list[dict[str, Any]] = []
    native: ObservedGlobalAllocationExecution | None = None
    opening = None
    attempted = None
    try:
        native = run_declared_global_allocation_execution(folds, env, plan.native_plan)
        opening = allocation_state_digest(env)
        if opening != native.receipt.payload["closing_state_digest"]:
            raise ValueError("terminal opening state differs from native prefix close")
        _provenance(plan)
        executor, risk = env.executor, env.risk
        book, orders, index = env.book, env.order_book, env.index
        while index < plan.settlement_stop_index:
            if env.executor is not executor or env.risk is not risk:
                raise ValueError("terminal executor/risk instance changed")
            _closing_binding(env, plan)
            attempted, before = index, book.portfolio_value
            target, execution = _execute_allocation_target(
                executor,
                book,
                orders,
                target_weight=0.0,
                decision_digest=content_digest(
                    {"plan": plan.digest, "closing_index": index}
                ),
                pretrade_risk=risk,
                symbol_index=env.symbol_index,
                start_index=index,
            )
            facts = native_allocation_execution_facts(execution)
            clock_valid = (
                type(execution.next_index) is int
                and type(execution.bars_advanced) is int
                and execution.bars_advanced == 1
                and execution.next_index == index + 1
            )
            processing_time = (
                int(
                    env.dataset.timestamps[execution.next_index]
                    .astype("datetime64[ns]")
                    .astype("int64")
                )
                if type(execution.next_index) is int
                and 0 <= execution.next_index < env.dataset.n_bars
                else None
            )
            closing.append(
                {
                    "start_index": index,
                    "processing_index": facts["execution"]["next_index"],
                    "processing_time_ns": processing_time,
                    "clock_valid": clock_valid,
                    "returned_clock_types": {
                        "next_index": type(execution.next_index).__name__,
                        "bars_advanced": type(execution.bars_advanced).__name__,
                    },
                    "opening_equity": before,
                    "risk_target": target.weights.tolist(),
                    "risk_reasons": list(target.reasons),
                    "native": facts,
                    "fixed_capital_reward": net_equity_increment(
                        before,
                        execution.book.portfolio_value,
                        initial_capital=env.initial_capital,
                    ),
                }
            )
            book, orders, index = (
                execution.book,
                execution.order_book,
                execution.next_index,
            )
            if not clock_valid:
                raise ValueError("terminal native execution changed its one-bar clock")
            if env.executor is not executor or env.risk is not risk:
                raise ValueError("terminal executor/risk instance changed")
            _closing_binding(env, plan)
            if book.termination_reason is not None:
                break
        _provenance(plan)
        payload = _observation(plan, native.receipt.payload, closing, opening)
        return ObservedTerminalAllocationExecution(
            native, canonical_json_bytes(payload)
        )
    except Exception as error:
        receipt = (
            native.receipt.payload
            if native is not None
            else getattr(error, "global_execution_receipt", None)
        )
        if receipt is not None and not isinstance(receipt, dict):
            receipt = receipt.payload
        try:
            setattr(
                error,
                "terminal_execution_receipt",
                _observation(
                    plan, receipt, closing, opening, error=error, attempted=attempted
                ),
            )
        except Exception as recording_error:
            setattr(
                error,
                "terminal_execution_receipt_error",
                {
                    "error_type": type(recording_error).__name__,
                    "message": str(recording_error),
                },
            )
        raise


__all__ = [
    "TerminalAllocationExecutionPlan",
    "ObservedTerminalAllocationExecution",
    "declare_terminal_allocation_execution",
    "run_terminal_allocation_execution",
]
