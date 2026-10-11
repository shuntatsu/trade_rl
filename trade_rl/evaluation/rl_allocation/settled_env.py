"""Separate Gym composition over one native account and its closing reserve."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.allocation_decision import (
    execute_allocation_action,
    prepare_allocation_decision,
)
from trade_rl.evaluation.allocation_reserved_closing import (
    _iter_reserved_allocation_closing,
)
from trade_rl.evaluation.allocation_snapshot import snapshot_allocation_account
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import net_equity_increment
from trade_rl.evaluation.rl_allocation.settled_binding import (
    BoundSettledAllocationClock,
    validate_settled_runtime_binding,
)
from trade_rl.evaluation.rl_allocation.settled_training_source import (
    settled_training_source_receipt,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _json,
    _pending,
    native_allocation_account_facts,
    native_allocation_execution_facts,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.simulation.stateful.execution import StatefulExecutionResult
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream
from trade_rl.strategies.rl.allocation_observation_encoder_v2 import (
    encode_allocation_observation_v2,
)
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_v4 import allocation_recipe_payload_v4


class SettledAllocationTradingEnv(gym.Env):
    """P actor transitions, H=P+R native bars; no marked-lane inheritance.

    The final actor reward includes every actually processed reserve bar. A
    terminal diagnostic is not a publication admission or a profitable result.
    """

    book: BookState
    order_book: OrderBookState
    index: int
    actor_step: int
    _episode_executor: MarketExecutor
    _episode_risk: PreTradeRisk
    _native_book: BookState
    _native_orders: OrderBookState

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        stream: FrozenSimpleReturnStream,
        estimates: tuple[HorizonCostEstimates, ...],
        bound: BoundSettledAllocationClock,
        action_contract: AllocationActionContract,
        allocator: AfterCostTargetAllocator,
        execution_cost: ExecutionCostConfig,
        risk_config: PreTradeRiskConfig,
        feature_indices: tuple[int, ...],
        symbol_index: int,
        start_index: int,
        account_id: str,
        observation_schema: AllocationObservationSchema,
        expected_horizon_seconds: int = 3600,
    ) -> None:
        if not isinstance(bound, BoundSettledAllocationClock):
            raise ValueError("settled environment requires a distinct settled binding")
        if type(start_index) is not int or not 0 <= start_index < dataset.n_bars:
            raise ValueError("settled start index is outside the Dataset")
        if (
            not feature_indices
            or len(set(feature_indices)) != len(feature_indices)
            or any(
                type(i) is not int or not 0 <= i < dataset.n_features
                for i in feature_indices
            )
            or type(symbol_index) is not int
            or not 0 <= symbol_index < dataset.n_symbols
        ):
            raise ValueError(
                "settled observation requires unique valid features and symbol"
            )
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("settled account identity must be nonempty")
        self.dataset, self.stream, self.bound = dataset, stream, bound
        self.action_contract, self.allocator = action_contract, allocator
        self.execution_cost, self.risk_config = execution_cost, risk_config
        self.feature_indices, self.symbol_index = tuple(feature_indices), symbol_index
        self.start_index = start_index
        self.actor_stop_index = start_index + bound.clock.actor_steps
        self.settlement_stop_index = start_index + bound.clock.physical_horizon_steps
        self.account_id, self.expected_horizon_seconds = (
            account_id,
            expected_horizon_seconds,
        )
        self.observation_schema = observation_schema
        self.initial_capital = bound.objective.capital.initial_equities[0]
        self.executor = MarketExecutor(
            dataset, execution_cost, insolvency_valuation="retain_debt"
        )
        self.risk = PreTradeRisk(risk_config)
        self._bound_digest = bound.digest
        self._recipe_digest = self.recipe_digest
        self._validate_runtime()
        self._estimates = {
            int(c.decision_time.astype("datetime64[ns]").astype(np.int64)): c
            for c in estimates
        }
        times = (
            dataset.timestamps[start_index : self.actor_stop_index]
            .astype("datetime64[ns]")
            .astype(np.int64)
        )
        if len(self._estimates) != len(estimates) or set(self._estimates) != set(
            map(int, times)
        ):
            raise ValueError(
                "settled cost declarations must cover each actor clock exactly once"
            )
        self._source_envelope = settled_training_source_receipt(self)
        self._source_digest = content_digest(self._source_envelope)
        self._observation_shape = (len(observation_schema.fields),)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self._observation_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self._terminated = True
        self._episode_sequence = 0
        self.publication_invalid = False

    @property
    def recipe(self) -> dict[str, object]:
        clock, capital = self.bound.clock, self.bound.objective.capital
        return allocation_recipe_payload_v4(
            self.action_contract,
            tuple(self.dataset.feature_names[i] for i in self.feature_indices),
            allocator=self.allocator,
            expected_horizon_seconds=self.expected_horizon_seconds,
            runtime_profile=AllocationRuntimeProfile(
                self.executor.execution_policy_digest,
                content_digest(self.risk.config),
                capital.initial_equities[0],
                capital.currency,
                clock.execution_interval_seconds,
                clock.physical_horizon_steps * clock.execution_interval_seconds,
                calendar_kind=MarketCalendarKind(self.dataset.calendar_kind).value,
                execution_bar_hours=self.dataset.bar_hours,
            ),
            settled_clock=clock,
            observation_schema=self.observation_schema,
        )

    @property
    def recipe_digest(self) -> str:
        return content_digest(self.recipe)

    def _validate_runtime(self) -> None:
        if (
            self.bound.digest != self._bound_digest
            or self.executor.cost != self.execution_cost
            or self.risk.config != self.risk_config
            or self.initial_capital != self.bound.objective.capital.initial_equities[0]
            or self.recipe_digest != self._recipe_digest
        ):
            raise ValueError("settled objective, capital or runtime recipe changed")
        validate_settled_runtime_binding(
            self.dataset,
            self.bound,
            self.executor,
            self.risk.config,
            self.recipe_digest,
            start_index=self.start_index,
            actor_stop_index=self.actor_stop_index,
            settlement_stop_index=self.settlement_stop_index,
        )

    def validate_binding(self) -> None:
        if self._episode_sequence and (
            self.executor is not self._episode_executor
            or self.risk is not self._episode_risk
            or self.book is not self._native_book
            or self.order_book is not self._native_orders
        ):
            raise ValueError("settled native account/executor/risk instance changed")
        self._validate_runtime()
        if settled_training_source_receipt(self) != self._source_envelope:
            raise ValueError("settled declared actor/native source envelope changed")

    def _arguments(self) -> dict[str, Any]:
        if not self.start_index <= self.index < self.actor_stop_index:
            raise ValueError("reserve bars cannot prepare an actor decision")
        time = int(
            self.dataset.timestamps[self.index]
            .astype("datetime64[ns]")
            .astype(np.int64)
        )
        return dict(
            account_id=self.account_id,
            stream=self.stream,
            estimates=self._estimates[time],
            allocator=self.allocator,
            pretrade_risk=self.risk,
            symbol_index=self.symbol_index,
            start_index=self.index,
            expected_horizon_seconds=self.expected_horizon_seconds,
            action_contract=self.action_contract,
            initial_capital=self.initial_capital,
            remaining_steps=self.settlement_stop_index - self.index,
            feature_indices=self.feature_indices,
        )

    def _observation(self) -> np.ndarray:
        self.decision = prepare_allocation_decision(
            self.executor, self.book, self.order_book, **self._arguments()
        )
        snapshot = snapshot_allocation_account(
            self.executor,
            self.book,
            self.order_book,
            account_id=self.account_id,
            pretrade_risk=self.risk,
            symbol_index=self.symbol_index,
            start_index=self.index,
        )
        # Preserve the actual encoded input for read-only fit observers.
        self._encoded_snapshot = snapshot
        return encode_allocation_observation_v2(
            snapshot,
            self.decision,
            schema=self.observation_schema,
            episode_steps=self.bound.clock.physical_horizon_steps,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if options:
            raise ValueError(
                "settled reset does not accept alternate account or clock options"
            )
        if not self._terminated and self.actor_step:
            self.publication_invalid = True
            raise ValueError("cannot reset an unfinished settled account")
        self.validate_binding()
        super().reset(seed=seed)
        self.executor = MarketExecutor(
            self.dataset, self.execution_cost, insolvency_valuation="retain_debt"
        )
        self.executor.reset_random_state(self.execution_cost.random_seed)
        self.risk = PreTradeRisk(self.risk_config)
        self._validate_runtime()
        self.book = BookState(
            np.zeros(self.dataset.n_symbols),
            self.initial_capital,
            self.dataset.resolved_array("mark_price")[self.start_index],
            self.initial_capital,
            self.dataset.resolved_array("contract_multipliers"),
            as_of_index=self.start_index,
            as_of_dataset_id=self.dataset.dataset_id,
        )
        self.executor._update_margin(self.book)
        self.order_book, self.index, self.actor_step = (
            OrderBookState.empty(),
            self.start_index,
            0,
        )
        self._episode_sequence += 1
        self._episode_executor, self._episode_risk = self.executor, self.risk
        self._native_book, self._native_orders = self.book, self.order_book
        self._episode_rows: list[dict[str, Any]] = []
        self._episode_failed = False
        self._terminated = False
        try:
            observation = self._observation()
        except BaseException:
            self._episode_failed = self._terminated = self.publication_invalid = True
            raise
        return observation, {
            "symbol": self.dataset.symbols[self.symbol_index],
            "episode_id": self._episode_sequence,
        }

    def _state_digest(self, index: int, book: BookState, orders: OrderBookState) -> str:
        return content_digest(
            {
                "schema": "allocation_settled_visible_state_v1",
                "bound_digest": self._bound_digest,
                "recipe_digest": self._recipe_digest,
                "source_envelope_digest": self._source_digest,
                "episode_id": self._episode_sequence,
                "index": index,
                **native_allocation_account_facts(book, orders),
            }
        )

    def _note_stage(self, index: int, stage: str) -> None:
        self._attempted_start_index, self._failure_stage = index, stage

    def _adopt(self, execution: StatefulExecutionResult) -> None:
        self.book, self.order_book, self.index = (
            execution.book,
            execution.order_book,
            execution.next_index,
        )
        self._native_book, self._native_orders = self.book, self.order_book

    def _row(
        self,
        kind: str,
        start: int,
        before: float,
        before_digest: str,
        execution: StatefulExecutionResult,
        native: dict[str, Any],
        risk: Any,
    ) -> dict[str, Any]:
        valid = (
            type(execution.next_index) is int
            and type(execution.bars_advanced) is int
            and execution.bars_advanced == 1
            and execution.next_index == start + 1
        )
        observed_ids = {event.order_id for event in execution.order_events}
        accepted_orders = [
            _pending(order)
            for order in (
                *execution.order_book.active_orders,
                *execution.order_book.terminal_orders,
            )
            if order.order_id in observed_ids
        ]
        return {
            "kind": kind,
            "start_index": start,
            "processing_index": execution.next_index,
            "clock_valid": valid,
            "returned_clock_types": {
                "next_index": type(execution.next_index).__name__,
                "bars_advanced": type(execution.bars_advanced).__name__,
            },
            "opening_equity": before,
            "before_state_digest": before_digest,
            "after_state_digest": self._state_digest(
                execution.next_index, execution.book, execution.order_book
            ),
            "risk": _json(risk),
            "native": native,
            "accepted_orders": accepted_orders,
        }

    def _witness(self) -> dict[str, Any]:
        reserve = sum(row["kind"] == "reserve" for row in self._episode_rows)
        complete = (
            self.actor_step == self.bound.clock.actor_steps
            and len(self._episode_rows) == self.bound.clock.physical_horizon_steps
            and reserve == self.bound.clock.reserve_steps
            and self.index == self.settlement_stop_index
            and all(row["clock_valid"] for row in self._episode_rows)
        )
        settled = (
            complete
            and not self._episode_failed
            and self.book.termination_reason is None
            and all(q == 0 for q in self.book.exact_quantities)
            and not self.order_book.active_orders
        )
        return {
            "actor_steps_processed": self.actor_step,
            "physical_steps_processed": len(self._episode_rows),
            "reserve_steps_processed": reserve,
            "full_coverage": complete,
            "settled": settled,
            "eligible": settled
            and self.book.max_drawdown <= self.bound.objective.maximum_drawdown,
            "maximum_drawdown": self.book.max_drawdown,
            "frontier_index": self.index,
            "native_frontier": native_allocation_account_facts(
                self.book, self.order_book
            ),
        }

    def _payload(
        self,
        actor_step: int,
        actor_index: int,
        rows: list[dict[str, Any]],
        reward: float | None,
    ) -> dict[str, Any]:
        return {
            "schema": "allocation_settled_transition_v1",
            "episode_id": self._episode_sequence,
            "account_id": self.account_id,
            "actor_step": actor_step,
            "actor_index": actor_index,
            "bound_digest": self._bound_digest,
            "recipe_digest": self._recipe_digest,
            "source_envelope_digest": self._source_digest,
            "initial_capital": self.initial_capital,
            "rows": rows,
            "reward": reward,
            "terminated": self._terminated,
            "truncated": False,
            "settlement_witness": self._witness(),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError(
                "settled allocation episode is terminal; reset before step"
            )
        actor_step, actor_index, before = (
            self.actor_step,
            self.index,
            self.book.portfolio_value,
        )
        rows: list[dict[str, Any]] = []
        self._note_stage(actor_index, "actor_preflight")
        try:
            self.validate_binding()
            before_digest = self._state_digest(self.index, self.book, self.order_book)
            self._note_stage(actor_index, "actor_execution")
            result = execute_allocation_action(
                self.executor,
                self.book,
                self.order_book,
                self.decision,
                action,
                **self._arguments(),
            )
            self._adopt(result.execution)
            self._note_stage(actor_index, "actor_returned")
            native = native_allocation_execution_facts(result.execution)
            row = self._row(
                "actor",
                actor_index,
                before,
                before_digest,
                result.execution,
                native,
                result.risk_target,
            )
            row["proposal"] = _json(result.proposal)
            rows.append(row)
            self._episode_rows.append(row)
            self.actor_step += 1
            self._note_stage(actor_index, "actor_validation")
            if not row["clock_valid"]:
                raise ValueError("settled actor execution changed its one-bar clock")
            self.validate_binding()
            if (
                self.actor_step == self.bound.clock.actor_steps
                and self.book.termination_reason is None
            ):
                for closing in _iter_reserved_allocation_closing(
                    self.executor,
                    self.book,
                    self.order_book,
                    risk=self.risk,
                    symbol_index=self.symbol_index,
                    start_index=self.index,
                    stop_index=self.settlement_stop_index,
                    decision_digest=lambda index: content_digest(
                        {
                            "schema": "allocation_settled_reserve_decision_v1",
                            "bound_digest": self._bound_digest,
                            "episode_id": self._episode_sequence,
                            "index": index,
                        }
                    ),
                    before_state=self._state_digest,
                    validate_binding=self.validate_binding,
                    note_stage=self._note_stage,
                    retain_returned=self._adopt,
                ):
                    row = self._row(
                        "reserve",
                        closing.start_index,
                        closing.opening_equity,
                        closing.before_state_digest,
                        closing.execution,
                        closing.native,
                        closing.risk_target,
                    )
                    rows.append(row)
                    self._episode_rows.append(row)
            reward = net_equity_increment(
                before, self.book.portfolio_value, initial_capital=self.initial_capital
            )
            self._terminated = (
                self.actor_step == self.bound.clock.actor_steps
                or self.book.termination_reason is not None
            )
            if self._terminated and not self._witness()["settled"]:
                self.publication_invalid = True
            self._note_stage(
                self.index,
                "terminal_observation" if self._terminated else "actor_observation",
            )
            observation = (
                np.zeros(self._observation_shape, dtype=np.float32)
                if self._terminated
                else self._observation()
            )
            payload = canonical_json_bytes(
                self._payload(actor_step, actor_index, rows, reward)
            )
            return (
                observation,
                reward,
                self._terminated,
                False,
                {
                    "settled_transition": payload,
                    "actor_execution": result.execution,
                    "action_proposal": result.proposal,
                    "risk_target": result.risk_target,
                    "terminal_valuation": "settled",
                },
            )
        except BaseException as error:
            # A native return may precede an interruption before actor_step
            # advances. Retain that frontier as a failed, never eligible episode.
            self._episode_failed = self._terminated = self.publication_invalid = True
            # Keep the original failure even if its returned diagnostics cannot
            # themselves be encoded as finite native JSON.
            try:
                diagnostic = self._payload(actor_step, actor_index, rows, None) | {
                    "failure_stage": self._failure_stage,
                    "attempted_start_index": self._attempted_start_index,
                }
                error.settled_execution_receipt = canonical_json_bytes(diagnostic)  # type: ignore[attr-defined]
            except BaseException:
                pass
            raise


__all__ = ["SettledAllocationTradingEnv"]
