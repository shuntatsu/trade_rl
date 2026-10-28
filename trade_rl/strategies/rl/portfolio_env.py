"""Opt-in joint portfolio Gym adapter over the native shared-account session."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from numbers import Real
from typing import TYPE_CHECKING, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.risk.pretrade import (
    PreTradeRisk,
    PreTradeRiskConfig,
    RiskConstrainedTarget,
)
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.contract import net_equity_increment
from trade_rl.simulation.orders.model import (
    PendingOrder,
)
from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPendingOrderSnapshot,
)
from trade_rl.strategies.portfolio_observation import (
    build_portfolio_decision_snapshot,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_CODES,
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    ACTIVE_ORDER_FIELDS,
    PORTFOLIO_GYM_OBSERVATION_KEYS,
    PORTFOLIO_GYM_OBSERVATION_SCHEMA,
    PORTFOLIO_STATE_FIELDS,
    RISK_STATE_FIELDS,
    decode_portfolio_action_codes,
    encode_portfolio_active_orders,
    encode_portfolio_decision_snapshot,
    portfolio_action_to_proposal,
)

if TYPE_CHECKING:
    from trade_rl.simulation.stateful.execution import StatefulExecutionObservation

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _finite_real(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real non-boolean number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _utc_datetime_ns(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    elapsed = normalized - _EPOCH
    seconds = elapsed.days * 86_400 + elapsed.seconds
    return seconds * 1_000_000_000 + elapsed.microseconds * 1_000


def _market_timestamp_ns(dataset: MarketDataset, index: int) -> int:
    timestamp = np.datetime64(dataset.timestamps[index], "ns")
    if np.isnat(timestamp):
        raise ValueError("dataset timestamp is NaT")
    return int(timestamp.astype(np.int64))


def _integer_index(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return int(value)


def _float_box(shape: tuple[int, ...]) -> spaces.Box:
    return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float64)


class SharedPortfolioGymEnv(gym.Env):
    """One joint action per native interval over a single shared account.

    This is an opt-in execution adapter, not a trainer or a second accounting
    engine. Gym's seed controls only Gym's own RNG; each reset creates a fresh
    native session with the fixed execution seed in ``execution_cost``.
    """

    metadata = {"render_modes": []}
    observation_schema = PORTFOLIO_GYM_OBSERVATION_SCHEMA
    observation_keys = PORTFOLIO_GYM_OBSERVATION_KEYS

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        start_index: int,
        stop_index: int,
        bound: BoundObjectiveClock,
        target_abs_weights: tuple[float, ...],
        execution_cost: ExecutionCostConfig,
        risk_config: PreTradeRiskConfig,
        market_order_profile: MarketOrderProfile | None = None,
        execution_observer: Callable[[StatefulExecutionObservation], None]
        | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dataset, MarketDataset):
            raise TypeError("dataset must be a MarketDataset")
        if not isinstance(bound, BoundObjectiveClock):
            raise TypeError("bound must be a BoundObjectiveClock")
        if not isinstance(execution_cost, ExecutionCostConfig):
            raise TypeError("execution_cost must be an ExecutionCostConfig")
        if not isinstance(risk_config, PreTradeRiskConfig):
            raise TypeError("risk_config must be a PreTradeRiskConfig")
        if market_order_profile is not None and not isinstance(
            market_order_profile, MarketOrderProfile
        ):
            raise TypeError("market_order_profile must be verified or None")
        if execution_observer is not None and not callable(execution_observer):
            raise TypeError("execution_observer must be callable")

        resolved_start = _integer_index(start_index, field="start_index")
        resolved_stop = _integer_index(stop_index, field="stop_index")
        if not 0 <= resolved_start < resolved_stop < dataset.n_bars:
            raise ValueError("range must satisfy 0 <= start < stop < n_bars")
        if type(target_abs_weights) is not tuple or len(target_abs_weights) != (
            dataset.n_symbols
        ):
            raise TypeError("target_abs_weights must be a symbol-aligned tuple")
        resolved_weights = tuple(
            _finite_real(value, field="target_abs_weights")
            for value in target_abs_weights
        )
        if any(value < 0.0 for value in resolved_weights):
            raise ValueError("target_abs_weights must be non-negative")

        objective = bound.objective
        clock = bound.clock
        if objective.capital.account_mode != "shared_portfolio":
            raise ValueError("the joint environment requires shared_portfolio capital")
        if len(objective.capital.initial_equities) != 1:
            raise ValueError("the joint environment requires exactly one capital book")
        if objective.terminal_valuation != "settled":
            raise ValueError("the joint environment requires a settled endpoint")
        if not clock.terminal_profit_aligned:
            raise ValueError("the joint environment requires gamma-one equity deltas")
        if risk_config.drawdown_stop > objective.maximum_drawdown:
            raise ValueError("risk drawdown stop exceeds the objective maximum")

        timestamps = tuple(
            _market_timestamp_ns(dataset, index)
            for index in range(resolved_start, resolved_stop + 1)
        )
        intervals_ns = tuple(
            current - previous
            for previous, current in zip(timestamps[:-1], timestamps[1:], strict=True)
        )
        if not intervals_ns or any(interval <= 0 for interval in intervals_ns):
            raise ValueError("dataset timestamps must increase across the window")
        if len(set(intervals_ns)) != 1 or intervals_ns[0] % 1_000_000_000:
            raise ValueError("dataset window must use one regular whole-second cadence")
        native_interval = intervals_ns[0] // 1_000_000_000
        if (
            clock.decision_interval_seconds != native_interval
            or clock.execution_interval_seconds != native_interval
            or clock.reward_interval_seconds != native_interval
        ):
            raise ValueError("financial clock must match one native interval per step")
        if timestamps[0] != _utc_datetime_ns(objective.evaluation_start):
            raise ValueError("objective start must match the dataset start row")
        if timestamps[-1] != _utc_datetime_ns(objective.evaluation_stop_exclusive):
            raise ValueError("objective stop must match the dataset stop row")
        if clock.economic_horizon_seconds != (
            (resolved_stop - resolved_start) * native_interval
        ):
            raise ValueError("financial horizon must match the native index window")
        agent_stop = resolved_stop - execution_cost.order_latency_bars - 1
        if agent_stop <= resolved_start:
            raise ValueError(
                "window must leave at least one actor decision before close"
            )
        if market_order_profile is not None:
            market_order_profile.validate_dataset(dataset)

        self.dataset = dataset
        self.start_index = resolved_start
        self.stop_index = resolved_stop
        self.bound = bound
        self.target_abs_weights = resolved_weights
        self.execution_cost = execution_cost
        self.risk_config = risk_config
        self.market_order_profile = market_order_profile
        self.execution_observer = execution_observer
        self.initial_capital = objective.capital.total_initial_capital
        n_symbols = dataset.n_symbols
        n_features = len(dataset.feature_names)
        n_global_features = len(dataset.global_feature_names)

        self.action_space = spaces.MultiDiscrete(
            np.full(n_symbols, len(ACTION_CODES), dtype=np.int64)
        )
        int64_min = np.iinfo(np.int64).min
        int64_max = np.iinfo(np.int64).max
        self.observation_space = spaces.Dict(
            OrderedDict(
                (
                    ("features", _float_box((n_symbols, n_features))),
                    (
                        "feature_available",
                        spaces.MultiBinary((n_symbols, n_features)),
                    ),
                    ("feature_staleness", _float_box((n_symbols, n_features))),
                    ("global_features", _float_box((n_global_features,))),
                    (
                        "global_feature_available",
                        spaces.MultiBinary((n_global_features,)),
                    ),
                    ("account", _float_box((n_symbols, 4))),
                    ("portfolio", _float_box((len(PORTFOLIO_STATE_FIELDS),))),
                    ("risk", _float_box((len(RISK_STATE_FIELDS),))),
                    (
                        "active_orders",
                        _float_box((n_symbols, len(ACTIVE_ORDER_FIELDS))),
                    ),
                    (
                        "clock",
                        spaces.Box(
                            low=np.full(4, int64_min, dtype=np.int64),
                            high=np.full(4, int64_max, dtype=np.int64),
                            dtype=np.int64,
                        ),
                    ),
                    ("terminal", spaces.MultiBinary(1)),
                )
            )
        )
        self._session: SharedPortfolioSession | None = None
        self._episode_done = True
        self._episode_poisoned = False
        self._capturing_execution_observations: (
            list[StatefulExecutionObservation] | None
        ) = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        if options:
            raise ValueError("reset options are not supported")
        super().reset(seed=seed)
        self._session = SharedPortfolioSession(
            self.dataset,
            start_index=self.start_index,
            stop_index=self.stop_index,
            initial_capital=self.initial_capital,
            execution_cost=self.execution_cost,
            risk=PreTradeRisk(self.risk_config),
            market_order_profile=self.market_order_profile,
            settle_terminal_position=True,
            execution_observer=self._observe_execution,
        )
        self._episode_done = False
        self._episode_poisoned = False
        observation = self._encode_snapshot(self._snapshot())
        return observation, {
            "current_index": self._session.current_index,
            "observation_schema": self.observation_schema,
            "execution_seed": self.execution_cost.random_seed,
        }

    def step(
        self, action: object
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        session = self._require_live_session()
        codes = self._validated_action(action)
        snapshot = self._snapshot()
        target_weights, hold_symbols = self._proposal(codes, snapshot)
        equity_before = session.book.portfolio_value
        cash_before = float(session.book.cash)
        quantities_before = tuple(str(value) for value in session.book.exact_quantities)
        intervals_advanced = 0
        effective_holds: tuple[int, ...] = ()
        mutated = False
        execution_observations: list[StatefulExecutionObservation] = []
        transitions: list[dict[str, object]] = []
        self._capturing_execution_observations = execution_observations
        try:
            start_index = session.current_index
            risk_target, _, effective_holds = session.advance(
                target_weights,
                hold_current_quantity_symbols=hold_symbols,
            )
            mutated = True
            intervals_advanced = 1
            transitions.append(
                self._decision_trace_transition(
                    kind="actor_action",
                    start_index=start_index,
                    risk_target=risk_target,
                    execution_observations=execution_observations,
                    expected_observation_count=1,
                )
            )
            while session.can_settle_terminal:
                start_index = session.current_index
                previous_observation_count = len(execution_observations)
                risk_target, _, _ = session.settle_next_interval()
                intervals_advanced += 1
                transitions.append(
                    self._decision_trace_transition(
                        kind="terminal_settlement",
                        start_index=start_index,
                        risk_target=risk_target,
                        execution_observations=execution_observations,
                        expected_observation_count=previous_observation_count + 1,
                    )
                )

            book = session.book
            reward = net_equity_increment(
                equity_before,
                book.portfolio_value,
                initial_capital=self.initial_capital,
            )
            terminated = session.finished
            if terminated:
                terminal_state = self._terminal_state(session, book)
                settlement_complete = bool(terminal_state["settlement_complete"])
                endpoint_valid = bool(terminal_state["objective_endpoint_valid"])
                observation = self._terminal_observation()
                self._episode_done = True
            else:
                terminal_state = None
                settlement_complete = False
                endpoint_valid = False
                observation = self._encode_snapshot(self._snapshot())
            equity_after = float(book.portfolio_value)
            decision_trace = {
                "schema": "shared_portfolio_decision_trace_v1",
                "raw_action_codes": list(codes),
                "proposal_weights": list(target_weights),
                "transitions": transitions,
                "account_delta": {
                    "cash_before": cash_before,
                    "cash_after": float(book.cash),
                    "equity_before": float(equity_before),
                    "equity_after": equity_after,
                    "equity_delta": equity_after - equity_before,
                    "quantities_before": list(quantities_before),
                    "quantities_after": [str(value) for value in book.exact_quantities],
                },
            }
            return (
                observation,
                reward,
                terminated,
                False,
                {
                    "current_index": session.current_index,
                    "elapsed_intervals": intervals_advanced,
                    "raw_action": codes,
                    "target_weights": target_weights,
                    "effective_hold_symbols": effective_holds,
                    "decision_trace": decision_trace,
                    "terminal_state": terminal_state,
                    "settlement_complete": settlement_complete,
                    "objective_endpoint_valid": endpoint_valid,
                },
            )
        except BaseException:
            if mutated or session.poisoned:
                self._episode_poisoned = True
                self._episode_done = True
            raise
        finally:
            self._capturing_execution_observations = None

    def _observe_execution(self, observation: StatefulExecutionObservation) -> None:
        """Forward the caller observer once, then retain this native interval."""

        if self.execution_observer is not None:
            self.execution_observer(observation)
        capture = self._capturing_execution_observations
        if capture is None:
            raise RuntimeError("execution trace capture is not active")
        capture.append(observation)

    @staticmethod
    def _decision_trace_transition(
        *,
        kind: str,
        start_index: int,
        risk_target: RiskConstrainedTarget,
        execution_observations: list[StatefulExecutionObservation],
        expected_observation_count: int,
    ) -> dict[str, object]:
        if len(execution_observations) != expected_observation_count:
            raise RuntimeError("native execution trace observation count is invalid")
        observation = execution_observations[-1]
        if observation.next_index != start_index + 1:
            raise RuntimeError("native execution trace index is inconsistent")
        proposal_weights = risk_target.proposal_weights
        pretrade_weights = risk_target.pretrade_weights
        if proposal_weights is None or pretrade_weights is None:
            raise RuntimeError("risk target is missing pipeline-stage weights")
        return {
            "kind": kind,
            "start_index": start_index,
            "next_index": observation.next_index,
            "proposal_weights": [float(value) for value in proposal_weights],
            "pretrade_weights": [float(value) for value in pretrade_weights],
            "approved_weights": [float(value) for value in risk_target.weights],
            "reasons": list(risk_target.reasons),
            "was_constrained": bool(risk_target.was_constrained),
            "risk_scale": float(risk_target.risk_scale),
            "requested_turnover": float(risk_target.requested_turnover),
            "constrained_turnover": float(risk_target.constrained_turnover),
            "execution_observation": {
                "next_index": observation.next_index,
                "order_events": [
                    event.canonical_payload() for event in observation.order_events
                ],
                "active_order_remainders": [
                    {
                        "order_id": order_id,
                        "remaining_quantity": float(remaining_quantity),
                    }
                    for order_id, remaining_quantity in observation.active_order_remainders
                ],
                "terminal_order_reasons": [
                    {"order_id": order_id, "reason": reason}
                    for order_id, reason in observation.terminal_order_reasons
                ],
            },
        }

    def _require_live_session(self) -> SharedPortfolioSession:
        if self._session is None:
            raise RuntimeError("reset must be called before step")
        if self._episode_poisoned:
            raise RuntimeError("the session failed after possible state mutation")
        if self._episode_done:
            raise RuntimeError("the episode has ended; call reset for a new episode")
        return self._session

    def _validated_action(self, action: object) -> tuple[int, ...]:
        return decode_portfolio_action_codes(action, n_symbols=self.dataset.n_symbols)

    def _snapshot(self) -> PortfolioDecisionSnapshot:
        session = self._require_live_session()
        return build_portfolio_decision_snapshot(
            self.dataset,
            index=session.current_index,
            agent_stop_index=session.agent_stop_index,
            book=session.book,
            active_orders=session.active_orders,
            risk_config=self.risk_config,
        )

    def _proposal(
        self,
        codes: tuple[int, ...],
        snapshot: PortfolioDecisionSnapshot,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        proposal = portfolio_action_to_proposal(
            codes, snapshot, self.target_abs_weights
        )
        return proposal.target_weights, proposal.hold_current_quantity_symbols

    def _encode_snapshot(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> dict[str, np.ndarray]:
        return encode_portfolio_decision_snapshot(snapshot)

    def _encode_active_orders(
        self, active_orders: tuple[PortfolioPendingOrderSnapshot, ...]
    ) -> np.ndarray:
        return encode_portfolio_active_orders(self.dataset.n_symbols, active_orders)

    def _terminal_observation(self) -> dict[str, np.ndarray]:
        spaces_by_key = cast(spaces.Dict, self.observation_space).spaces
        observation: dict[str, np.ndarray] = {}
        for key, space in spaces_by_key.items():
            if space.shape is None or space.dtype is None:
                raise RuntimeError(
                    "observation spaces must have a fixed shape and dtype"
                )
            observation[key] = np.zeros(space.shape, dtype=space.dtype)
        observation["terminal"][0] = 1
        return observation

    def _terminal_state(
        self, session: SharedPortfolioSession, book: BookState
    ) -> dict[str, object]:
        reason = book.termination_reason
        reason_text = (
            None
            if reason is None
            else reason.value
            if isinstance(reason, EconomicTerminationReason)
            else str(reason)
        )
        orders = tuple(
            self._pending_order_facts(order) for order in session.active_orders
        )
        flat = all(quantity == 0 for quantity in book.exact_quantities)
        settlement_complete = (
            session.current_index == self.stop_index and flat and not orders
        )
        endpoint_valid = settlement_complete and not book.insolvent and reason is None
        return {
            "current_index": session.current_index,
            "cash": float(book.cash),
            "portfolio_value": float(book.portfolio_value),
            "peak_value": float(book.peak_value),
            "max_drawdown": float(book.max_drawdown),
            "total_cost": float(book.total_cost),
            "funding_pnl": float(book.funding_pnl),
            "borrow_cost": float(book.borrow_cost),
            "quantities": tuple(float(value) for value in book.quantities),
            "exact_quantities": tuple(str(value) for value in book.exact_quantities),
            "active_orders": orders,
            "insolvent": bool(book.insolvent),
            "termination_reason": reason_text,
            "settlement_complete": settlement_complete,
            "objective_endpoint_valid": endpoint_valid,
        }

    @staticmethod
    def _pending_order_facts(order: PendingOrder) -> dict[str, object]:
        intent = order.intent
        return {
            "symbol_index": int(intent.symbol_index),
            "order_type": intent.order_type.value,
            "time_in_force": intent.time_in_force.value,
            "requested_quantity": float(intent.requested_quantity),
            "remaining_quantity": float(order.remaining_quantity),
            "status": order.status.value,
            "submit_index": int(intent.submit_index),
            "eligible_index": int(intent.eligible_index),
            "last_processed_index": order.last_processed_index,
            "trigger_index": order.trigger_index,
            "expiry_index": intent.expiry_index,
            "reduce_only": bool(intent.reduce_only),
            "limit_price": intent.limit_price,
            "stop_price": intent.stop_price,
            "submission_reference_price": intent.submission_reference_price,
        }


__all__ = [
    "ACTION_CODES",
    "ACTION_FLAT",
    "ACTION_HOLD",
    "ACTION_LONG",
    "ACTION_SHORT",
    "ACTIVE_ORDER_FIELDS",
    "PORTFOLIO_GYM_OBSERVATION_KEYS",
    "PORTFOLIO_GYM_OBSERVATION_SCHEMA",
    "PORTFOLIO_STATE_FIELDS",
    "RISK_STATE_FIELDS",
    "SharedPortfolioGymEnv",
]
