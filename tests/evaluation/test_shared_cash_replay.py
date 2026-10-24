from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import cast

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation import replay as replay_module
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import (
    BookState,
    ExecutionCostConfig,
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.simulation.orders.model import OrderEvent, OrderStatus
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.adaptive import (
    AdaptiveProfitConfig,
    RegimeAdaptiveStrategy,
)


@dataclass
class FixedIntent:
    intent: PositionIntent
    observations: list[object] = field(default_factory=list)

    def decide(self, observation: object) -> PositionIntent:
        self.observations.append(observation)
        return self.intent


@dataclass
class SequenceIntent:
    intents: tuple[PositionIntent, ...]
    index: int = 0
    observations: list[object] = field(default_factory=list)

    def decide(self, observation: object) -> PositionIntent:
        self.observations.append(observation)
        intent = self.intents[min(self.index, len(self.intents) - 1)]
        self.index += 1
        return intent


def _market(
    close: np.ndarray,
    *,
    symbols: tuple[str, ...] | None = None,
    volume: float | np.ndarray = 1_000_000.0,
    signal_values: np.ndarray | None = None,
    tradable_values: np.ndarray | None = None,
    split_factors: np.ndarray | None = None,
    open_values: np.ndarray | None = None,
) -> MarketDataset:
    close_array = np.asarray(close, dtype=np.float64)
    if close_array.ndim != 2:
        raise ValueError("close must be two-dimensional")
    n_bars, n_symbols = close_array.shape
    resolved_symbols = symbols or tuple(f"SYM{index}" for index in range(n_symbols))
    open_price = np.vstack((close_array[0], close_array[:-1]))
    if open_values is not None:
        open_price = np.asarray(open_values, dtype=np.float64)
        if open_price.shape != (n_bars, n_symbols):
            raise ValueError("open_values must match bars and symbols")
    features = np.zeros((n_bars, n_symbols, 1), dtype=np.float32)
    for symbol_index in range(n_symbols):
        features[:, symbol_index, 0] = np.arange(n_bars) + 10 * symbol_index
    if signal_values is not None:
        signals = np.asarray(signal_values, dtype=np.float32)
        if signals.shape != (n_bars, n_symbols):
            raise ValueError("signal_values must match bars and symbols")
        features[:, :, 0] = signals
    tradable = np.ones((n_bars, n_symbols), dtype=np.bool_)
    if tradable_values is not None:
        tradable = np.asarray(tradable_values, dtype=np.bool_)
        if tradable.shape != (n_bars, n_symbols):
            raise ValueError("tradable_values must match bars and symbols")
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=resolved_symbols,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close_array),
        low=np.minimum(open_price, close_array),
        close=close_array,
        volume=np.broadcast_to(volume, (n_bars, n_symbols)).copy(),
        funding_rate=np.zeros((n_bars, n_symbols)),
        tradable=tradable,
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        split_factor=split_factors,
    )


def _two_symbol_market() -> MarketDataset:
    return _market(
        np.asarray(
            [
                [100.0, 200.0],
                [100.0, 200.0],
                [110.0, 190.0],
                [120.0, 180.0],
                [130.0, 170.0],
                [140.0, 160.0],
            ]
        ),
        symbols=("BTCUSDT", "ETHUSDT"),
    )


def test_shared_replay_rejects_late_flat_roster_mark_before_target_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _two_symbol_market()
    assert dataset.available_at is not None
    assert dataset.information_available is not None
    available_at = dataset.available_at.copy()
    information_available = dataset.information_available.copy()
    available_at[2, 1] = dataset.timestamps[3]
    information_available[2, 1] = False
    delayed = replace(
        dataset,
        available_at=available_at,
        information_available=information_available,
    )
    strategies = (
        FixedIntent(PositionIntent.FLAT),
        FixedIntent(PositionIntent.FLAT),
    )
    execution_starts: list[int] = []
    original_execute_interval = MarketExecutor.execute_interval

    def record_execution_start(
        executor: MarketExecutor,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> ExecutionResult:
        execution_starts.append(start_index)
        return original_execute_interval(
            executor, book, target, start_index=start_index, bars=bars
        )

    monkeypatch.setattr(
        MarketExecutor,
        "execute_interval",
        record_execution_start,
    )
    with pytest.raises(ValueError, match="decision-time mark unavailable"):
        replay_module.run_shared_cash_replay(
            delayed,
            strategies,
            start_index=0,
            stop_index=4,
            gross_budget=0.2,
        )

    assert execution_starts == [0]
    assert [item.index for item in strategies[0].observations] == [0, 1]
    assert [item.index for item in strategies[1].observations] == [0, 1]


def _risk(*, max_gross: float, max_turnover: float | None) -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=max_gross,
            max_abs_weight=max_gross,
            max_turnover=max_turnover,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )


def test_shared_cash_replay_api_exists() -> None:
    assert callable(replay_module.run_shared_cash_replay)


def test_scripted_hold_preserves_exact_lot_quantity_across_repeated_bars() -> None:
    price = 459.4451860856874
    dataset = _market(
        np.asarray([[price], [price], [price * 1.1], [price * 0.9], [price * 1.2]]),
        volume=1e18,
    )
    hold = replay_module.ScriptedReplayAction.HOLD_QUANTITY

    result = replay_module.run_scripted_shared_cash_replay(
        dataset,
        actions_by_symbol={
            "SYM0": (
                PositionIntent.LONG,
                hold,
                hold,
                None,
            )
        },
        start_index=0,
        stop_index=4,
        gross_budget=0.28465406229772494,
        initial_capital=12_644_598_901.709024,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
            lot_size=1e-12,
            max_leverage=1.0,
        ),
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    after_entry = ledger.intervals[0].exact_quantities_after
    assert after_entry == ("1958523319267917/250000000",)
    assert ledger.intervals[1].exact_quantities_after == after_entry
    assert ledger.intervals[2].exact_quantities_after == after_entry
    assert ledger.intervals[1].order_events == ()
    assert ledger.intervals[2].order_events == ()
    assert ledger.final_total_cost == 0.0


def test_scripted_hold_cancels_unfilled_order_from_actual_flat_position() -> None:
    volume = np.full((5, 1), 1_000_000.0)
    volume[1, 0] = 0.0
    result = replay_module.run_scripted_shared_cash_replay(
        _market(np.full((5, 1), 100.0), volume=volume),
        actions_by_symbol={
            "SYM0": (
                PositionIntent.LONG,
                replay_module.ScriptedReplayAction.HOLD_QUANTITY,
                None,
            )
        },
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            order_latency_bars=1,
        ),
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    hold = ledger.decisions[1]
    assert hold.intents == (PositionIntent.FLAT,)
    assert hold.effective_intents == (PositionIntent.FLAT,)
    assert hold.position_quantity_before == (0.0,)
    hold_events = ledger.intervals[1].order_events
    assert any(
        event.event_type == "cancelled" and event.reason == "superseded"
        for event in hold_events
    )
    assert ledger.terminal_exact_quantities == ("0",)
    assert ledger.active_order_remainders == ()


def test_scripted_hold_cancels_partial_remainder_and_keeps_exact_fill() -> None:
    volume = np.full((5, 1), 1_000_000.0)
    volume[1, 0] = 1.0
    result = replay_module.run_scripted_shared_cash_replay(
        _market(np.full((5, 1), 100.0), volume=volume),
        actions_by_symbol={
            "SYM0": (
                PositionIntent.LONG,
                replay_module.ScriptedReplayAction.HOLD_QUANTITY,
                None,
            )
        },
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            lot_size=0.1,
        ),
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    partial_quantity = ledger.intervals[0].exact_quantities_after
    assert partial_quantity == ("1",)
    hold_events = ledger.intervals[1].order_events
    assert any(
        event.event_type == "cancelled" and event.reason == "superseded"
        for event in hold_events
    )
    assert ledger.intervals[1].exact_quantities_after == partial_quantity
    assert ledger.intervals[2].exact_quantities_after == partial_quantity
    assert ledger.active_order_remainders == ()


def test_scripted_none_preserves_pending_target_and_native_fill_progression() -> None:
    volume = np.full((5, 1), 1_000_000.0)
    volume[1:3, 0] = 1.0
    result = replay_module.run_scripted_shared_cash_replay(
        _market(np.full((5, 1), 100.0), volume=volume),
        actions_by_symbol={
            "SYM0": (PositionIntent.LONG, None, None),
        },
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            lot_size=0.1,
        ),
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    assert ledger.intervals[0].exact_quantities_after == ("1",)
    assert ledger.intervals[1].exact_quantities_after == ("2",)
    assert not any(
        event.event_type == "cancelled" for event in ledger.intervals[1].order_events
    )
    first_order_id = next(
        event.order_id
        for event in ledger.intervals[0].order_events
        if event.event_type == "submitted"
    )
    assert all(
        event.order_id == first_order_id for event in ledger.intervals[1].order_events
    )


def test_scripted_hold_is_still_constrained_by_drawdown_risk() -> None:
    result = replay_module.run_scripted_shared_cash_replay(
        _market(np.asarray([[100.0], [100.0], [70.0], [70.0], [70.0]])),
        actions_by_symbol={
            "SYM0": (
                PositionIntent.LONG,
                None,
                replay_module.ScriptedReplayAction.HOLD_QUANTITY,
                None,
            )
        },
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=1.0,
                max_turnover=None,
                drawdown_start=0.01,
                drawdown_stop=0.02,
            )
        ),
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    assert "drawdown_deleveraging" in ledger.decisions[2].risk_reasons
    assert ledger.intervals[2].exact_quantities_after == ("0",)
    assert ledger.active_order_remainders == ()


def test_scripted_replay_rejects_invalid_future_action_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_starts: list[int] = []
    original_execute_interval = MarketExecutor.execute_interval

    def record_execution_start(
        executor: MarketExecutor,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> ExecutionResult:
        execution_starts.append(start_index)
        return original_execute_interval(
            executor,
            book,
            target,
            start_index=start_index,
            bars=bars,
        )

    monkeypatch.setattr(MarketExecutor, "execute_interval", record_execution_start)
    with pytest.raises(ValueError, match="unsupported scripted action"):
        replay_module.run_scripted_shared_cash_replay(
            _market(np.full((5, 1), 100.0)),
            actions_by_symbol={
                "SYM0": (PositionIntent.LONG, None, object()),
            },
            start_index=0,
            stop_index=3,
            gross_budget=0.5,
            execution_cost=ExecutionCostConfig.zero(),
        )

    assert execution_starts == []


@pytest.mark.parametrize("invalid_cost", [False, 0])
def test_scripted_replay_rejects_falsey_nonconfig_cost_before_execution(
    invalid_cost: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_starts: list[int] = []
    original_execute_interval = MarketExecutor.execute_interval

    def record_execution_start(
        executor: MarketExecutor,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> ExecutionResult:
        execution_starts.append(start_index)
        return original_execute_interval(
            executor,
            book,
            target,
            start_index=start_index,
            bars=bars,
        )

    monkeypatch.setattr(MarketExecutor, "execute_interval", record_execution_start)
    with pytest.raises(
        TypeError,
        match="execution_cost must be an ExecutionCostConfig or None",
    ):
        replay_module.run_scripted_shared_cash_replay(
            _market(np.full((5, 1), 100.0)),
            actions_by_symbol={"SYM0": (PositionIntent.LONG, None, None)},
            start_index=0,
            stop_index=3,
            gross_budget=0.5,
            execution_cost=cast(ExecutionCostConfig, invalid_cost),
        )

    assert execution_starts == []


def test_scripted_small_position_action_remains_a_real_retarget() -> None:
    result = replay_module.run_scripted_shared_cash_replay(
        _market(np.full((4, 1), 100.0), volume=1_000_000.0),
        actions_by_symbol={
            "SYM0": (PositionIntent.LONG, PositionIntent.FLAT),
        },
        start_index=0,
        stop_index=2,
        gross_budget=1e-8,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    assert ledger.intervals[0].exact_quantities_after != ("0",)
    assert ledger.intervals[1].exact_quantities_after == ("0",)


def test_shared_cash_replay_settles_every_symbol_before_the_exclusive_close() -> None:
    dataset = _market(np.full((6, 2), [100.0, 200.0]))
    result = replay_module.run_shared_cash_replay(
        dataset,
        (
            FixedIntent(PositionIntent.LONG),
            FixedIntent(PositionIntent.SHORT),
        ),
        start_index=0,
        stop_index=5,
        gross_budget=0.2,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    assert len(result.decisions) == 4
    assert len(result.returns.values) == 5
    np.testing.assert_allclose(result.book.quantities, np.zeros(2))
    assert result.ledger_evidence is not None
    assert result.ledger_evidence.schema_version == "shared_cash_replay_ledger_v2"
    assert result.ledger_evidence.terminal_exact_quantities == ("0", "0")
    assert result.ledger_evidence.active_order_remainders == ()
    assert len(result.ledger_evidence.intervals) == len(result.returns.values)


@pytest.mark.parametrize(
    ("intents", "drifted_prices"),
    [
        pytest.param(
            (PositionIntent.LONG, PositionIntent.LONG),
            (120.0, 240.0),
            id="long-long-drift",
        ),
        pytest.param(
            (PositionIntent.SHORT, PositionIntent.SHORT),
            (120.0, 240.0),
            id="short-short-drift",
        ),
        pytest.param(
            (PositionIntent.LONG, PositionIntent.SHORT),
            (90.0, 220.0),
            id="mixed-side-drift",
        ),
    ],
)
def test_shared_cash_terminal_settlement_flattens_past_turnover_cap(
    intents: tuple[PositionIntent, PositionIntent],
    drifted_prices: tuple[float, float],
) -> None:
    close = np.tile(np.asarray([[100.0, 200.0]]), (6, 1))
    close[4:, :] = drifted_prices
    dataset = _market(close)
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.4,
            max_abs_weight=0.2,
            max_turnover=0.4,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )
    execution_cost = ExecutionCostConfig(
        fee_rate=0.001,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=1.0,
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        tuple(FixedIntent(intent) for intent in intents),
        start_index=0,
        stop_index=5,
        gross_budget=0.2,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk=risk,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    terminal_interval = ledger.intervals[-1]
    terminal_weights = (
        np.asarray(
            [
                float(Fraction(quantity)) * price
                for quantity, price in zip(
                    terminal_interval.exact_quantities_before,
                    drifted_prices,
                    strict=True,
                )
            ]
        )
        / terminal_interval.portfolio_value_before
    )
    assert float(np.abs(terminal_weights).sum()) > risk.config.max_turnover
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert ledger.active_order_remainders == ()
    assert result.book.total_cost > 0.0
    assert ledger.final_total_cost == pytest.approx(result.book.total_cost)
    assert ledger.final_turnover_total == pytest.approx(result.book.turnover_total)


def test_shared_cash_without_terminal_settlement_keeps_ordinary_turnover_cap() -> None:
    close = np.tile(np.asarray([[100.0, 200.0]]), (6, 1))
    close[4:, :] = (120.0, 240.0)
    strategies = tuple(
        SequenceIntent(
            (
                PositionIntent.LONG,
                PositionIntent.LONG,
                PositionIntent.LONG,
                PositionIntent.LONG,
                PositionIntent.FLAT,
            )
        )
        for _ in range(2)
    )

    result = replay_module.run_shared_cash_replay(
        _market(close),
        strategies,
        start_index=0,
        stop_index=5,
        gross_budget=0.2,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.5,
                max_turnover=0.01,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=False,
    )

    last_decision = result.decisions[-1]
    assert last_decision.index == 4
    assert "max_turnover" in last_decision.risk_reasons
    assert "emergency_flatten" not in last_decision.risk_reasons


def test_shared_cash_terminal_settlement_preserves_residual_when_capacity_is_zero() -> (
    None
):
    close = np.tile(np.asarray([[100.0, 200.0]]), (6, 1))
    close[4:, :] = (120.0, 240.0)
    volume = np.full((6, 2), 1_000_000.0)
    volume[5, :] = 0.0
    dataset = _market(close, volume=volume)
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.4,
            max_abs_weight=0.2,
            max_turnover=0.4,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=5,
        gross_budget=0.2,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    assert any(Fraction(quantity) != 0 for quantity in ledger.terminal_exact_quantities)
    assert ledger.active_order_remainders


def test_shared_cash_replay_enforces_minimum_hold_from_actual_shared_fills() -> None:
    dataset = _market(np.full((6, 1), 100.0))
    strategy = SequenceIntent(
        (
            PositionIntent.LONG,
            PositionIntent.FLAT,
            PositionIntent.FLAT,
            PositionIntent.FLAT,
        )
    )
    result = replay_module.run_shared_cash_replay(
        dataset,
        (strategy,),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        minimum_hold_bars=3,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    assert [item.position_age_bars for item in strategy.observations] == [0, 1, 2, 3]
    assert [item.intents for item in result.decisions] == [
        (PositionIntent.LONG,),
        (PositionIntent.FLAT,),
        (PositionIntent.FLAT,),
        (PositionIntent.FLAT,),
    ]
    assert [item.effective_intents for item in result.decisions] == [
        (PositionIntent.LONG,),
        (PositionIntent.LONG,),
        (PositionIntent.LONG,),
        (PositionIntent.FLAT,),
    ]
    assert [item.minimum_hold_suppressed for item in result.decisions] == [
        (False,),
        (True,),
        (True,),
        (False,),
    ]
    assert [item.position_age_bars_before for item in result.decisions] == [
        (0,),
        (1,),
        (2,),
        (3,),
    ]
    assert [item.position_age_bars_after for item in result.decisions] == [
        (1,),
        (2,),
        (3,),
        (0,),
    ]
    np.testing.assert_allclose(result.book.quantities, np.zeros(1))
    assert result.ledger_evidence is not None
    assert result.ledger_evidence.decisions[1].minimum_hold_suppressed == (True,)


def test_adaptive_protective_exit_uses_filled_gross_return_and_bypasses_hold() -> None:
    dataset = _market(
        np.asarray([[100.0], [100.0], [103.0], [103.0], [103.0], [103.0]]),
        signal_values=np.asarray([[0.5], [0.5], [0.5], [0.0], [0.0], [0.0]]),
        tradable_values=np.asarray([[True], [True], [True], [False], [True], [True]]),
    )
    strategy = RegimeAdaptiveStrategy(
        AdaptiveProfitConfig(
            signal_index=0,
            volatility_index=0,
            trend_entry_threshold=0.01,
            trend_exit_threshold=0.002,
            volatility_regime_threshold=0.005,
            take_profit_threshold=0.02,
        )
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        (strategy,),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        minimum_hold_bars=5,
    )

    assert [decision.intents for decision in result.decisions[:3]] == [
        (PositionIntent.LONG,),
        (PositionIntent.LONG,),
        (PositionIntent.FLAT,),
    ]
    assert result.decisions[1].minimum_hold_suppressed == (True,)
    assert result.decisions[2].minimum_hold_suppressed == (False,)
    assert result.decisions[2].position_quantity_after[0] != pytest.approx(0.0)
    assert result.decisions[2].effective_intents == (PositionIntent.FLAT,)
    assert result.decisions[3].minimum_hold_suppressed == (False,)
    assert result.decisions[3].intents == (PositionIntent.FLAT,)
    assert result.decisions[3].position_quantity_after == pytest.approx((0.0,))


def test_adaptive_protective_exit_stays_latched_after_missed_fill_and_recovery() -> (
    None
):
    class RecordingAdaptiveStrategy(RegimeAdaptiveStrategy):
        def __init__(self, config: AdaptiveProfitConfig) -> None:
            super().__init__(config)
            self.pending_by_index: dict[int, bool] = {}

        def decide(self, observation: StrategyObservation) -> PositionIntent:
            intent = super().decide(observation)
            self.pending_by_index[observation.index] = self.protective_exit_pending
            return intent

    dataset = _market(
        np.asarray([[100.0], [100.0], [103.0], [101.0], [101.0], [101.0]]),
        signal_values=np.full((6, 1), 0.5),
        tradable_values=np.asarray([[True], [True], [True], [False], [True], [True]]),
    )
    strategy = RecordingAdaptiveStrategy(
        AdaptiveProfitConfig(
            signal_index=0,
            volatility_index=0,
            trend_entry_threshold=0.01,
            trend_exit_threshold=0.002,
            volatility_regime_threshold=0.005,
            take_profit_threshold=0.02,
        )
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        (strategy,),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        minimum_hold_bars=5,
    )

    missed_exit = result.decisions[2]
    recovered_mark = result.decisions[3]
    assert missed_exit.intents == (PositionIntent.FLAT,)
    assert missed_exit.effective_intents == (PositionIntent.FLAT,)
    assert missed_exit.position_quantity_after[0] != pytest.approx(0.0)
    assert recovered_mark.position_quantity_before[0] != pytest.approx(0.0)
    assert recovered_mark.intents == (PositionIntent.FLAT,)
    assert strategy.pending_by_index[3]


def test_execution_gross_position_return_is_invariant_across_split() -> None:
    dataset = _market(
        np.asarray([[100.0], [100.0], [100.0], [1e14], [1e14], [1e14]]),
        open_values=np.asarray([[100.0], [100.0], [100.0], [1e14], [1e14], [1e14]]),
        split_factors=np.asarray([[1.0], [1.0], [1.0], [1e-12], [1.0], [1.0]]),
    )
    strategy = FixedIntent(PositionIntent.LONG)

    replay_module.run_shared_cash_replay(
        dataset,
        (strategy,),
        start_index=0,
        stop_index=4,
        gross_budget=0.02,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    pre_split_return = strategy.observations[2].gross_position_return
    post_split_return = strategy.observations[3].gross_position_return
    assert strategy.observations[0].intent_gross_budget == 0.02
    assert pre_split_return == pytest.approx(0.0)
    assert post_split_return is not None
    assert post_split_return == pytest.approx(pre_split_return)


def test_fill_tracker_handles_tiny_position_reversal_without_product_underflow() -> (
    None
):
    tracker = replay_module._ExecutedEntryPrices(1)

    def event(sequence: int, filled: float, price: float) -> OrderEvent:
        return OrderEvent(
            schema_version="order_event_v1",
            sequence=sequence,
            order_id=f"{sequence + 1:064x}",
            replaced_order_id=None,
            dataset_id="d" * 64,
            execution_policy_digest="e" * 64,
            symbol_index=0,
            event_type="partial_fill",
            processing_index=sequence,
            timestamp_ns=sequence,
            previous_status=OrderStatus.ELIGIBLE,
            new_status=OrderStatus.PARTIALLY_FILLED,
            requested_quantity=1.0,
            remaining_quantity=1.0,
            filled_quantity=filled,
            execution_price=price,
            filled_notional=abs(filled * price),
            capacity_before=1.0,
            capacity_after=1.0,
            participation_rate=0.0,
            trigger_segment=None,
            available_volume_fraction=1.0,
            reason=None,
            path_mode="conservative",
            path_points=(),
        )

    tracker.ingest((event(0, 1e-200, 100.0),), np.asarray([1e-200]))
    tracker.ingest((event(1, -2e-200, 200.0),), np.asarray([-1e-200]))

    assert tracker.mark_gross_return(
        0, quantity=-1e-200, mark_price=200.0
    ) == pytest.approx(0.0)


def test_shared_cash_hold_age_uses_partial_fill_and_cancels_unfilled_remainder() -> (
    None
):
    dataset = _market(
        np.full((6, 1), 100.0),
        volume=np.asarray(
            [
                [1_000_000.0],
                [2.0],
                [1_000_000.0],
                [1_000_000.0],
                [1_000_000.0],
                [1_000_000.0],
            ]
        ),
    )
    strategy = SequenceIntent(
        (
            PositionIntent.LONG,
            PositionIntent.FLAT,
            PositionIntent.FLAT,
            PositionIntent.FLAT,
        )
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        (strategy,),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        minimum_hold_bars=3,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    first_fill = result.decisions[0].position_quantity_after[0]
    assert 0.0 < first_fill < 5.0
    assert result.decisions[0].position_age_bars_after == (1,)
    assert result.decisions[1].position_quantity_before == pytest.approx((first_fill,))
    assert result.decisions[1].minimum_hold_suppressed == (True,)
    assert result.ledger_evidence is not None
    assert any(
        event.event_type == "cancelled" and event.reason == "superseded"
        for event in result.ledger_evidence.intervals[1].order_events
    )
    assert result.ledger_evidence.terminal_exact_quantities == ("0",)
    assert result.ledger_evidence.active_order_remainders == ()


def test_shared_cash_drawdown_stop_overrides_minimum_hold() -> None:
    dataset = _market(np.asarray([[100.0], [100.0], [60.0], [60.0], [60.0], [60.0]]))
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )
    result = replay_module.run_shared_cash_replay(
        dataset,
        (
            SequenceIntent(
                (
                    PositionIntent.LONG,
                    PositionIntent.FLAT,
                    PositionIntent.FLAT,
                    PositionIntent.FLAT,
                )
            ),
        ),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        minimum_hold_bars=3,
    )

    stop_decision = result.decisions[2]
    assert stop_decision.minimum_hold_suppressed == (True,)
    assert stop_decision.target_weights == pytest.approx((0.0,))
    assert "drawdown_deleveraging" in stop_decision.risk_reasons
    assert stop_decision.position_quantity_after == pytest.approx((0.0,))


def test_shared_cash_minimum_hold_requires_age_aware_observation() -> None:
    dataset = _market(np.full((4, 1), 100.0))
    strategy = FixedIntent(PositionIntent.LONG)
    strategy.observation_schema = "ppo_observation_v2"

    with pytest.raises(ValueError, match="age-aware observation"):
        replay_module.run_shared_cash_replay(
            dataset,
            (strategy,),
            start_index=0,
            stop_index=3,
            gross_budget=0.5,
            minimum_hold_bars=1,
        )


def test_shared_cash_minimum_hold_roster_must_match_dataset() -> None:
    dataset = _two_symbol_market()

    with pytest.raises(ValueError, match="match the dataset symbol roster"):
        replay_module.run_shared_cash_replay(
            dataset,
            (FixedIntent(PositionIntent.FLAT), FixedIntent(PositionIntent.FLAT)),
            start_index=0,
            stop_index=5,
            gross_budget=0.2,
            minimum_hold_bars=(1,),
        )


def test_one_symbol_shared_cash_is_single_symbol_equivalent() -> None:
    dataset = _market(
        np.asarray([[100.0], [100.0], [110.0], [120.0], [130.0], [140.0]])
    )
    single_strategy = FixedIntent(PositionIntent.LONG)
    shared_strategy = FixedIntent(PositionIntent.LONG)
    execution_cost = ExecutionCostConfig.zero()

    single = replay_module.run_single_symbol_replay(
        dataset,
        single_strategy,
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
    )
    shared = replay_module.run_shared_cash_replay(
        dataset,
        (shared_strategy,),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
    )

    assert shared.returns.values == pytest.approx(single.returns.values)
    np.testing.assert_allclose(shared.book.quantities, single.book.quantities)
    assert shared.book.cash == pytest.approx(single.book.cash)
    assert shared.book.portfolio_value == pytest.approx(single.book.portfolio_value)
    assert shared.diagnostics == single.diagnostics
    assert [item.intents[0] for item in shared.decisions] == [
        item.intent for item in single.decisions
    ]
    assert [item.changed_intents[0] for item in shared.decisions] == [
        item.changed_intent for item in single.decisions
    ]
    assert [item.target_weights[0] for item in shared.decisions] == pytest.approx(
        [item.target_weight for item in single.decisions]
    )


def test_simultaneous_long_proposals_share_one_gross_budget() -> None:
    dataset = _market(np.full((5, 2), [100.0, 200.0]))
    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=4,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    first = result.decisions[0]
    assert first.proposal_weights == pytest.approx((1.0, 1.0))
    assert first.target_weights == pytest.approx((0.5, 0.5))
    assert "max_gross" in first.risk_reasons
    assert result.book.portfolio_value == pytest.approx(1_000.0)
    assert result.book.cash == pytest.approx(0.0)
    np.testing.assert_allclose(result.book.weights, np.asarray([0.5, 0.5]))
    assert 1_000.0 * np.prod(1.0 + np.asarray(result.returns.values)) == pytest.approx(
        result.book.portfolio_value
    )


def test_risk_and_execution_run_once_per_bar_after_all_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _market(np.full((5, 2), [100.0, 200.0]))
    strategies = (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG))
    execution_calls: list[int] = []
    risk_calls: list[tuple[float, ...]] = []
    original_execute = MarketExecutor.execute_interval
    original_constrain = PreTradeRisk.constrain

    def record_execute(
        executor: MarketExecutor,
        book: object,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> object:
        execution_calls.append(start_index)
        return original_execute(
            executor,
            book,
            target,
            start_index=start_index,
            bars=bars,  # type: ignore[arg-type]
        )

    def record_constrain(
        controller: PreTradeRisk,
        target: np.ndarray,
        *,
        current: np.ndarray,
        drawdown: float,
        emergency_flatten_mask: np.ndarray | None = None,
        reduce_only_mask: np.ndarray | None = None,
    ) -> object:
        risk_calls.append(tuple(float(value) for value in target))
        return original_constrain(
            controller,
            target,
            current=current,
            drawdown=drawdown,
            emergency_flatten_mask=emergency_flatten_mask,
            reduce_only_mask=reduce_only_mask,
        )

    monkeypatch.setattr(MarketExecutor, "execute_interval", record_execute)
    monkeypatch.setattr(PreTradeRisk, "constrain", record_constrain)

    result = replay_module.run_shared_cash_replay(
        dataset,
        strategies,
        start_index=0,
        stop_index=4,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert execution_calls == [0, 1, 2, 3]
    assert len(risk_calls) == 4
    assert risk_calls[0] == pytest.approx((1.0, 1.0))
    assert [
        getattr(strategy.observations[0], "current_weight") for strategy in strategies
    ] == [0.0, 0.0]
    assert len(result.decisions) == 4


def test_turnover_only_projection_keeps_desired_quantities_for_convergence() -> None:
    dataset = _market(np.full((5, 2), [100.0, 200.0]))
    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=_risk(max_gross=1.0, max_turnover=0.5),
    )

    assert result.decisions[0].proposal_weights == pytest.approx((0.5, 0.5))
    assert result.decisions[0].target_weights == pytest.approx((0.25, 0.25))
    assert result.decisions[0].risk_reasons == ("max_turnover",)
    assert result.decisions[1].proposal_weights == pytest.approx((0.5, 0.5))
    assert result.decisions[1].target_weights == pytest.approx((0.5, 0.5))


def test_shared_reversal_preserves_proposal_through_hard_turnover_override() -> None:
    dataset = _market(
        np.asarray(
            [
                [100.0],
                [100.0],
                [100.0],
                [100.0],
                [100.0],
                [200.0],
                [200.0],
                [200.0],
            ]
        )
    )
    result = replay_module.run_shared_cash_replay(
        dataset,
        (
            SequenceIntent(
                (
                    PositionIntent.LONG,
                    PositionIntent.LONG,
                    PositionIntent.LONG,
                    PositionIntent.LONG,
                    PositionIntent.LONG,
                    PositionIntent.SHORT,
                    PositionIntent.SHORT,
                )
            ),
        ),
        start_index=0,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.5,
                max_turnover=0.1,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
    )

    reversal = result.decisions[5]
    follow_up = result.decisions[6]
    assert reversal.proposal_weights == pytest.approx((-0.5,))
    assert reversal.target_weights == pytest.approx((0.5,))
    assert "max_turnover" in reversal.risk_reasons
    assert "max_abs_weight" in reversal.risk_reasons
    assert "hard_risk_turnover_override" in reversal.risk_reasons
    assert follow_up.proposal_weights == pytest.approx((-0.5,))
    assert follow_up.target_weights == pytest.approx((0.4,))


def test_hard_risk_projection_rebinds_desired_quantities() -> None:
    dataset = _market(np.full((5, 2), [100.0, 200.0]))
    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=_risk(max_gross=0.5, max_turnover=None),
    )

    assert result.decisions[0].proposal_weights == pytest.approx((0.5, 0.5))
    assert result.decisions[0].target_weights == pytest.approx((0.25, 0.25))
    assert "max_gross" in result.decisions[0].risk_reasons
    assert result.decisions[1].proposal_weights == pytest.approx((0.25, 0.25))
    assert result.decisions[1].target_weights == pytest.approx((0.25, 0.25))


def test_shared_cash_fixed_drawdown_does_not_compound_risk_scale() -> None:
    dataset = _market(np.asarray([[100.0], [70.0], [70.0], [70.0], [70.0]]))
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG),),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
    )

    expected_proposal = (0.4117647058823529,)
    expected_target = (0.20588235294117646,)
    assert result.book.max_drawdown == pytest.approx(0.15)
    assert result.decisions[0].target_weights == pytest.approx((0.5,))
    assert result.decisions[1].proposal_weights == pytest.approx(expected_proposal)
    assert result.decisions[2].proposal_weights == pytest.approx(expected_proposal)
    assert result.decisions[1].target_weights == pytest.approx(expected_target)
    assert result.decisions[2].target_weights == pytest.approx(expected_target)


def test_symbol_permutation_preserves_shared_portfolio_economics() -> None:
    original = _two_symbol_market()
    swapped = _market(
        np.asarray(original.close)[:, [1, 0]],
        symbols=("ETHUSDT", "BTCUSDT"),
    )
    first = replay_module.run_shared_cash_replay(
        original,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.SHORT)),
        start_index=0,
        stop_index=5,
        gross_budget=0.6,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    second = replay_module.run_shared_cash_replay(
        swapped,
        (FixedIntent(PositionIntent.SHORT), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=5,
        gross_budget=0.6,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert second.returns.values == pytest.approx(first.returns.values)
    assert second.book.portfolio_value == pytest.approx(first.book.portfolio_value)
    assert second.book.turnover_total == pytest.approx(first.book.turnover_total)
    assert second.book.total_cost == pytest.approx(first.book.total_cost)
    np.testing.assert_allclose(second.book.quantities[[1, 0]], first.book.quantities)
    np.testing.assert_allclose(second.book.weights[[1, 0]], first.book.weights)


def test_strategy_roster_fails_closed() -> None:
    dataset = _two_symbol_market()
    shared = FixedIntent(PositionIntent.LONG)
    common = dict(
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        execution_cost=ExecutionCostConfig.zero(),
    )
    with pytest.raises(ValueError, match="one strategy per dataset symbol"):
        replay_module.run_shared_cash_replay(dataset, (shared,), **common)
    with pytest.raises(ValueError, match="distinct strategy instance"):
        replay_module.run_shared_cash_replay(dataset, (shared, shared), **common)


def test_shared_cash_drawdown_stop_uses_combined_multi_symbol_equity() -> None:
    dataset = _market(
        np.asarray(
            [
                [100.0, 100.0],
                [100.0, 100.0],
                [40.0, 40.0],
                [40.0, 40.0],
                [40.0, 40.0],
            ]
        )
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
    )

    assert result.book.max_drawdown > 0.20
    assert result.decisions[2].target_weights == pytest.approx((0.0, 0.0))
    assert "drawdown_deleveraging" in result.decisions[2].risk_reasons


def test_portfolio_termination_stops_all_symbols() -> None:
    dataset = _market(
        np.asarray(
            [
                [100.0, 100.0],
                [100.0, 100.0],
                [300.0, 100.0],
                [300.0, 100.0],
                [300.0, 100.0],
                [300.0, 100.0],
            ]
        )
    )
    result = replay_module.run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.SHORT), FixedIntent(PositionIntent.FLAT)),
        start_index=0,
        stop_index=5,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    assert result.book.termination_reason is not None
    assert len(result.returns.values) < 5
    assert len(result.decisions) == len(result.returns.values)


def test_shared_cash_partial_reversal_locks_actual_short_fill_and_cancels_remainder() -> (
    None
):
    dataset = _market(
        np.full((7, 1), 100.0),
        volume=np.asarray(
            [
                [1_000_000.0],
                [1_000_000.0],
                [1_000_000.0],
                [1_000_000.0],
                [6.0],
                [1_000_000.0],
                [1_000_000.0],
            ]
        ),
    )
    strategy = SequenceIntent(
        (
            PositionIntent.LONG,
            PositionIntent.LONG,
            PositionIntent.LONG,
            PositionIntent.SHORT,
            PositionIntent.FLAT,
        )
    )

    result = replay_module.run_shared_cash_replay(
        dataset,
        (strategy,),
        start_index=0,
        stop_index=6,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        minimum_hold_bars=3,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None

    reversal = result.decisions[3]
    reversed_quantity = reversal.position_quantity_after[0]
    assert -5.0 < reversed_quantity < 0.0
    assert reversal.position_age_bars_after == (1,)
    assert any(
        event.event_type == "partial_fill" and event.remaining_quantity != 0.0
        for event in ledger.intervals[3].order_events
    )

    locked = result.decisions[4]
    assert locked.position_quantity_before == pytest.approx((reversed_quantity,))
    assert locked.intents == (PositionIntent.FLAT,)
    assert locked.effective_intents == (PositionIntent.SHORT,)
    assert locked.minimum_hold_suppressed == (True,)
    assert locked.position_quantity_after == pytest.approx((reversed_quantity,))
    assert any(
        event.event_type == "cancelled"
        and event.reason == "superseded"
        and event.symbol_index == 0
        for event in ledger.intervals[4].order_events
    )
    assert ledger.terminal_exact_quantities == ("0",)
    assert ledger.active_order_remainders == ()


def test_shared_cash_locked_partial_fill_is_reduced_by_cross_symbol_hard_risk() -> None:
    dataset = _market(
        np.full((6, 2), 100.0),
        volume=np.asarray(
            [
                [1_000_000.0, 1_000_000.0],
                [2.0, 1_000_000.0],
                [1_000_000.0, 1_000_000.0],
                [1_000_000.0, 1_000_000.0],
                [1_000_000.0, 1_000_000.0],
                [1_000_000.0, 1_000_000.0],
            ]
        ),
    )
    strategies = (
        SequenceIntent(
            (
                PositionIntent.LONG,
                PositionIntent.FLAT,
                PositionIntent.FLAT,
                PositionIntent.FLAT,
            )
        ),
        SequenceIntent(
            (
                PositionIntent.FLAT,
                PositionIntent.LONG,
                PositionIntent.LONG,
                PositionIntent.LONG,
            )
        ),
    )

    risk = _risk(max_gross=0.5, max_turnover=None)
    result = replay_module.run_shared_cash_replay(
        dataset,
        strategies,
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        minimum_hold_bars=3,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None

    partial_quantity = result.decisions[0].position_quantity_after[0]
    assert 0.0 < partial_quantity < 5.0
    locked = result.decisions[1]
    assert locked.minimum_hold_suppressed == (True, False)
    assert locked.position_quantity_before[0] == pytest.approx(partial_quantity)
    assert "max_gross" in locked.risk_reasons
    assert locked.target_weights == pytest.approx((1.0 / 7.0, 2.5 / 7.0))
    assert sum(abs(weight) for weight in locked.target_weights) <= 0.5
    locked_interval = ledger.intervals[1]
    assert locked_interval.start_index == locked.index
    assert locked_interval.next_index == locked.index + 1
    actual_quantities = np.asarray(
        [float(Fraction(value)) for value in locked_interval.exact_quantities_after],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        actual_quantities,
        np.asarray(locked.position_quantity_after, dtype=np.float64),
    )
    mark_prices = dataset.close[locked_interval.next_index]
    contract_multipliers = result.book.contract_multipliers
    assert locked_interval.portfolio_value_after > 0.0
    actual_gross_value = sum(
        (
            abs(Fraction(quantity))
            * Fraction(str(mark_price))
            * Fraction(str(contract_multiplier))
            for quantity, mark_price, contract_multiplier in zip(
                locked_interval.exact_quantities_after,
                mark_prices,
                contract_multipliers,
                strict=True,
            )
        ),
        Fraction(0),
    )
    actual_gross_fraction = actual_gross_value / Fraction(
        str(locked_interval.portfolio_value_after)
    )
    # Match the existing risk contract's configured numerical tolerance.
    risk_gross_limit = Fraction(str(risk.config.max_gross)) + Fraction(
        str(risk.config.fail_closed_tolerance)
    )
    assert actual_gross_fraction <= risk_gross_limit, (
        f"actual post-fill gross exposure {actual_gross_fraction} exceeds "
        f"configured gross limit {risk_gross_limit}"
    )
    assert locked.position_quantity_after[0] < locked.position_quantity_before[0]
    assert locked.position_quantity_after[1] > 0.0
    assert any(
        event.event_type == "cancelled"
        and event.reason == "superseded"
        and event.symbol_index == 0
        for event in ledger.intervals[1].order_events
    )
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert ledger.active_order_remainders == ()
