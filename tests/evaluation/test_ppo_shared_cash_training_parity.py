from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.orders.model import PendingOrder
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.intent import PPO_OBSERVATION_SCHEMA_V3
from trade_rl.strategies.rl.ppo import PPOTradingEnv

_INTENTS = (
    PositionIntent.LONG,
    PositionIntent.LONG,
    PositionIntent.LONG,
    PositionIntent.FLAT,
)
_ACTIONS = tuple(int(intent) + 1 for intent in _INTENTS)
_INITIAL_CAPITAL = 1_000.0
_GROSS_BUDGET = 0.5
_MINIMUM_HOLD_BARS = 2


@dataclass
class _ScriptedIntent:
    intents: tuple[PositionIntent, ...]
    index: int = 0
    observations: list[StrategyObservation] = field(default_factory=list)
    observation_schema: str = PPO_OBSERVATION_SCHEMA_V3

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        self.observations.append(observation)
        intent = self.intents[self.index]
        self.index += 1
        return intent


@dataclass(frozen=True)
class _TrainingRow:
    age_before: int
    active_orders: tuple[PendingOrder, ...]
    cash_after: float
    equity_after: float
    info: dict[str, object]
    quantity_before: float
    quantity_after: float
    terminated: bool
    truncated: bool


def _hourly_market() -> MarketDataset:
    prices = np.full((6, 1), 100.0, dtype=np.float64)
    participation = np.asarray((1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0)).reshape(-1, 1)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("SYNTHUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(6) * np.timedelta64(1, "h"),
        features=np.arange(6, dtype=np.float32).reshape(6, 1, 1),
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.full((6, 1), 1_000_000.0),
        funding_rate=np.zeros((6, 1)),
        tradable=np.ones((6, 1), dtype=np.bool_),
        feature_available=np.ones((6, 1, 1), dtype=np.bool_),
        feature_names=("synthetic_signal",),
        global_feature_names=("synthetic_regime",),
        periods_per_year=8_760,
        mark_price=prices.copy(),
        max_participation_rate=participation,
    )


def _risk() -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )


def test_ppo_training_env_matches_single_symbol_shared_cash_replay() -> None:
    dataset = _hourly_market()
    execution_cost = ExecutionCostConfig.zero()
    risk_config = _risk().config
    environment = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=5,
        gross_budget=_GROSS_BUDGET,
        initial_capital=_INITIAL_CAPITAL,
        execution_cost=execution_cost,
        risk_config=risk_config,
        settle_terminal_position=True,
        minimum_hold_bars=_MINIMUM_HOLD_BARS,
        observation_schema=PPO_OBSERVATION_SCHEMA_V3,
    )
    environment.reset(seed=23)

    strategy = _ScriptedIntent(_INTENTS)
    replay = run_shared_cash_replay(
        dataset,
        (strategy,),
        start_index=0,
        stop_index=5,
        gross_budget=_GROSS_BUDGET,
        initial_capital=_INITIAL_CAPITAL,
        execution_cost=execution_cost,
        risk=_risk(),
        minimum_hold_bars=_MINIMUM_HOLD_BARS,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    training_rows: list[_TrainingRow] = []
    for action in _ACTIONS:
        age_before = environment.position_age_bars
        quantity_before = float(environment.book.quantities[0])
        _, _, terminated, truncated, info = environment.step(action)
        training_rows.append(
            _TrainingRow(
                age_before=age_before,
                active_orders=environment.executor.compatibility_order_book.active_orders,
                cash_after=float(environment.book.cash),
                equity_after=float(environment.book.portfolio_value),
                info=info,
                quantity_before=quantity_before,
                quantity_after=float(environment.book.quantities[0]),
                terminated=terminated,
                truncated=truncated,
            )
        )

    evidence = replay.ledger_evidence
    assert evidence is not None
    assert evidence.schema_version == "shared_cash_replay_ledger_v2"
    assert len(replay.decisions) == len(_INTENTS)
    assert len(evidence.intervals) == 5
    assert len(replay.returns.values) == 5
    assert len(strategy.observations) == len(_INTENTS)

    for index, (training, decision) in enumerate(
        zip(training_rows, replay.decisions, strict=True)
    ):
        info = training.info
        observation = strategy.observations[index]
        ledger_interval = evidence.intervals[index]

        assert training.age_before == decision.position_age_bars_before[0]
        assert observation.position_age_bars == decision.position_age_bars_before[0]
        assert info["position_age_bars"] == decision.position_age_bars_after[0]
        assert info["requested_intent"] is decision.intents[0]
        assert info["effective_intent"] is decision.effective_intents[0]
        assert info["target_weight"] == pytest.approx(decision.target_weights[0])
        assert training.quantity_before == pytest.approx(
            decision.position_quantity_before[0]
        )
        assert training.quantity_after == pytest.approx(
            decision.position_quantity_after[0]
        )
        assert training.quantity_after - training.quantity_before == pytest.approx(
            float(ledger_interval.exact_quantities_after[0])
            - float(ledger_interval.exact_quantities_before[0])
        )
        assert info["interval_net_return"] == pytest.approx(
            replay.returns.values[index]
        )
        assert info["interval_net_return"] == pytest.approx(
            ledger_interval.interval_net_return
        )
        assert training.cash_after == pytest.approx(ledger_interval.cash_after)
        assert training.equity_after == pytest.approx(
            ledger_interval.portfolio_value_after
        )

    first_info = training_rows[0].info
    locked_info = training_rows[1].info
    unlocked_info = training_rows[2].info

    first_fill = training_rows[0].quantity_after
    assert first_fill > 0.0
    assert first_fill < _INITIAL_CAPITAL * _GROSS_BUDGET / 100.0
    training_entry_orders = training_rows[0].active_orders
    assert len(training_entry_orders) == 1
    first_partial_fill_events = [
        event
        for event in evidence.intervals[0].order_events
        if event.event_type == "partial_fill"
    ]
    assert len(first_partial_fill_events) == 1
    assert first_partial_fill_events[0].remaining_quantity == pytest.approx(
        training_entry_orders[0].remaining_quantity
    )

    assert locked_info["minimum_hold_suppressed"] is True
    assert replay.decisions[1].minimum_hold_suppressed == (True,)
    assert training_rows[1].active_orders == ()
    assert any(
        event.event_type == "cancelled" and event.reason == "superseded"
        for event in evidence.intervals[1].order_events
    )

    assert unlocked_info["requested_intent"] is PositionIntent.LONG
    assert unlocked_info["effective_intent"] is PositionIntent.LONG
    assert unlocked_info["minimum_hold_suppressed"] is False
    assert unlocked_info["minimum_hold_unlocked"] is True
    assert replay.decisions[2].minimum_hold_unlocked == (True,)
    assert training_rows[2].quantity_after > training_rows[2].quantity_before
    assert unlocked_info["target_weight"] == pytest.approx(_GROSS_BUDGET)
    unlocked_fill_events = [
        event
        for event in evidence.intervals[2].order_events
        if event.event_type == "filled"
    ]
    assert len(unlocked_fill_events) == 1
    assert unlocked_fill_events[0].filled_quantity == pytest.approx(
        training_rows[2].quantity_after - training_rows[2].quantity_before
    )
    assert training_rows[2].active_orders == ()

    assert training_rows[3].terminated is True
    assert training_rows[3].truncated is False
    assert first_info["requested_intent"] is PositionIntent.LONG
    assert evidence.terminal_exact_quantities == ("0",)
    assert evidence.active_order_remainders == ()
    assert environment.executor.compatibility_order_book.active_orders == ()
    assert environment.book.quantities[0] == pytest.approx(0.0)
    assert replay.book.quantities[0] == pytest.approx(0.0)
    assert environment.book.cash == pytest.approx(evidence.final_cash)
    assert environment.book.cash == pytest.approx(replay.book.cash)
    assert environment.book.portfolio_value == pytest.approx(
        evidence.final_portfolio_value
    )

    terminal_info = training_rows[3].info
    terminal_interval = evidence.intervals[4]
    assert terminal_info["terminal_settlement_intervals"] == 1
    assert terminal_info["terminal_settlement_net_return"] == pytest.approx(
        terminal_interval.interval_net_return
    )
    assert terminal_interval.exact_quantities_after == ("0",)
