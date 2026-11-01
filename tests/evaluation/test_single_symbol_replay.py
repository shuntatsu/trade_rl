from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest

import trade_rl.evaluation as evaluation
from trade_rl.data.market import MarketDataset
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.position_intent import PositionIntent


@dataclass
class AlwaysLong:
    observations: list[object] = field(default_factory=list)

    def decide(self, observation: object) -> PositionIntent:
        self.observations.append(observation)
        return PositionIntent.LONG


@dataclass
class AlwaysShort:
    observations: list[object] = field(default_factory=list)

    def decide(self, observation: object) -> PositionIntent:
        self.observations.append(observation)
        return PositionIntent.SHORT


@dataclass
class IntentSequence:
    intents: tuple[PositionIntent, ...]
    index: int = 0

    def decide(self, observation: object) -> PositionIntent:
        del observation
        intent = self.intents[min(self.index, len(self.intents) - 1)]
        self.index += 1
        return intent


def _rising_market() -> MarketDataset:
    close = np.asarray([[100.0], [100.0], [110.0], [120.0], [130.0], [140.0]])
    open_price = np.vstack((close[0], close[:-1]))
    n_bars = close.shape[0]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.arange(n_bars, dtype=np.float32).reshape(n_bars, 1, 1),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _two_symbol_market() -> MarketDataset:
    close = np.asarray(
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [110.0, 190.0],
            [120.0, 180.0],
            [130.0, 170.0],
            [140.0, 160.0],
        ]
    )
    open_price = np.vstack((close[0], close[:-1]))
    n_bars = close.shape[0]
    features = np.zeros((n_bars, 2, 1), dtype=np.float32)
    features[:, 0, 0] = np.arange(n_bars)
    features[:, 1, 0] = -np.arange(n_bars)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _with_late_mark(
    dataset: MarketDataset,
    *,
    row_index: int,
    symbol_index: int = 0,
) -> MarketDataset:
    assert dataset.available_at is not None
    assert dataset.information_available is not None
    available_at = dataset.available_at.copy()
    information_available = dataset.information_available.copy()
    available_at[row_index, symbol_index] = dataset.timestamps[
        row_index
    ] + np.timedelta64(1, "h")
    information_available[row_index, symbol_index] = False
    return replace(
        dataset,
        available_at=available_at,
        information_available=information_available,
    )


def test_repeated_long_intent_holds_quantity_instead_of_rebalancing_weight() -> None:
    run_replay = getattr(evaluation, "run_single_symbol_replay", None)
    assert callable(run_replay), "lean single-symbol replay is not implemented"

    strategy = AlwaysLong()
    result = run_replay(
        _rising_market(),
        strategy,
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )

    assert result.book.fill_count == 1
    assert result.book.rebalance_events == 1
    assert result.book.quantities[0] == 5.0
    assert result.book.portfolio_value == 1_200.0
    assert len(result.returns.values) == 5
    assert len(result.decisions) == 5
    assert result.decisions[0].intent is PositionIntent.LONG
    assert result.decisions[0].changed_intent is True
    assert all(not item.changed_intent for item in result.decisions[1:])

    first_observation = strategy.observations[0]
    assert getattr(first_observation, "symbol") == "BTCUSDT"
    assert getattr(first_observation, "index") == 0
    assert getattr(first_observation, "intent_gross_budget") == 0.5
    features = getattr(first_observation, "features")
    assert features.flags.writeable is False


def test_cost_aware_replay_cancels_unfilled_stale_long_before_latency_fill() -> None:
    weak_forecast = float(np.log1p(0.005))
    forecast_values = np.asarray(
        [0.11, weak_forecast, weak_forecast, -0.02, -0.02, -0.02],
        dtype=np.float32,
    )
    n_bars = forecast_values.size
    price = np.full((n_bars, 1), 100.0, dtype=np.float64)
    dataset = MarketDataset(
        dataset_id="c" * 64,
        symbols=("SYNTH",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=forecast_values.reshape(n_bars, 1, 1),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=price.copy(),
        high=price.copy(),
        low=price.copy(),
        close=price.copy(),
        volume=np.full((n_bars, 1), 10.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("literal_forecast",),
        global_feature_names=("constant",),
        periods_per_year=8_760,
    )
    model = RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.asarray([0.0]),
        feature_scale=np.asarray([1.0]),
        coefficients=np.asarray([1.0]),
        intercept=0.0,
        horizon_hours=1,
        alpha=1.0,
        n_samples=1,
        fit_cutoff=np.datetime64("2025-12-31T00:00:00", "ns"),
    )
    strategy = RidgeForecastStrategy(
        model,
        entry_threshold=0.10,
        exit_threshold=0.02,
        one_way_switch_cost=0.01,
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
            emergency_turnover_override=True,
            fail_closed_tolerance=1e-10,
        )
    )

    result = evaluation.run_single_symbol_replay(
        dataset,
        strategy,
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            fee_rate=0.01,
            max_participation_rate=0.01,
            order_latency_bars=2,
        ),
        risk=risk,
        minimum_hold_bars=0,
        settle_terminal_position=False,
    )

    assert result.decisions[0].intent is PositionIntent.LONG
    assert result.decisions[1].intent is PositionIntent.FLAT
    assert result.decisions[1].changed_intent
    assert result.book.fill_count == 0
    assert result.book.total_cost == pytest.approx(0.0)
    assert result.book.turnover_total == pytest.approx(0.0)
    np.testing.assert_array_equal(result.book.quantities, np.zeros(1))
    assert result.active_order_remainders == ()


def test_replay_rejects_late_start_mark_before_strategy_observation() -> None:
    strategy = AlwaysLong()

    with pytest.raises(ValueError, match="decision-time mark unavailable"):
        evaluation.run_single_symbol_replay(
            _with_late_mark(_rising_market(), row_index=0),
            strategy,
            start_index=0,
            stop_index=4,
            gross_budget=0.5,
        )

    assert strategy.observations == []


def test_replay_allows_on_time_row_with_false_information_flag() -> None:
    dataset = _rising_market()
    assert dataset.available_at is not None
    assert dataset.information_available is not None
    information_available = dataset.information_available.copy()
    information_available[2, 0] = False
    on_time_unavailable = replace(
        dataset,
        information_available=information_available,
    )

    result = evaluation.run_single_symbol_replay(
        on_time_unavailable,
        AlwaysLong(),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
    )

    assert len(result.decisions) == 4


def test_replay_ignores_late_unused_tail_and_unselected_flat_symbol() -> None:
    late_tail = _with_late_mark(_rising_market(), row_index=4)
    tail_result = evaluation.run_single_symbol_replay(
        late_tail,
        AlwaysLong(),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
    )
    assert len(tail_result.returns.values) == 3

    late_unselected = _with_late_mark(_two_symbol_market(), row_index=2, symbol_index=1)
    selected_result = evaluation.run_single_symbol_replay(
        late_unselected,
        AlwaysLong(),
        symbol_index=0,
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
    )
    assert selected_result.book.quantities[1] == 0.0
    assert len(selected_result.decisions) == 4


def test_terminal_settlement_rejects_late_target_before_processing_it() -> None:
    strategy = AlwaysLong()

    with pytest.raises(ValueError, match="decision-time mark unavailable"):
        evaluation.run_single_symbol_replay(
            _with_late_mark(_rising_market(), row_index=4),
            strategy,
            start_index=0,
            stop_index=4,
            gross_budget=0.5,
            settle_terminal_position=True,
        )

    assert [item.index for item in strategy.observations] == [0, 1, 2]


def test_replay_exposes_terminal_active_order_remainders() -> None:
    result = evaluation.run_single_symbol_replay(
        _rising_market(),
        AlwaysLong(),
        start_index=0,
        stop_index=1,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            order_latency_bars=2,
        ),
    )

    np.testing.assert_array_equal(result.book.quantities, np.zeros(1))
    assert len(result.active_order_remainders) == 1
    order_id, remaining_quantity = result.active_order_remainders[0]
    assert isinstance(order_id, str) and order_id
    assert remaining_quantity > 0.0
    assert isinstance(result.terminal_order_reasons, tuple)


def test_reversal_keeps_short_proposal_after_hard_override() -> None:
    close = np.asarray(
        [[100.0], [100.0], [100.0], [100.0], [100.0], [200.0], [200.0], [200.0]]
    )
    open_price = np.vstack((close[0], close[:-1]))
    dataset = MarketDataset(
        dataset_id="c" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(close.shape[0]) * np.timedelta64(1, "h"),
        features=np.zeros((close.shape[0], 1, 1), dtype=np.float32),
        global_features=np.zeros((close.shape[0], 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((close.shape[0], 1), 1_000_000.0),
        funding_rate=np.zeros((close.shape[0], 1)),
        tradable=np.ones((close.shape[0], 1), dtype=np.bool_),
        feature_available=np.ones((close.shape[0], 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=0.1,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )

    result = evaluation.run_single_symbol_replay(
        dataset,
        IntentSequence(
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
        start_index=0,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
    )

    assert result.decisions[5].target_weight == pytest.approx(0.5)
    assert result.decisions[6].target_weight == pytest.approx(0.4)


def test_adverse_short_drift_is_hard_deleveraged_instead_of_crashing() -> None:
    result = evaluation.run_single_symbol_replay(
        _rising_market(),
        AlwaysShort(),
        start_index=0,
        stop_index=5,
        gross_budget=1.0,
        initial_capital=1_000.0,
    )

    assert len(result.returns.values) == 5
    assert result.book.fill_count > 1
    assert result.book.quantities[0] > -10.0
    assert all(
        abs(decision.target_weight) <= 1.0 + 1e-10 for decision in result.decisions
    )


def test_pooled_dataset_replays_only_selected_symbol() -> None:
    strategy = AlwaysLong()
    result = evaluation.run_single_symbol_replay(
        _two_symbol_market(),
        strategy,
        symbol_index=1,
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )

    np.testing.assert_array_equal(result.book.quantities[[0]], np.zeros(1))
    assert result.book.quantities[1] == pytest.approx(2.5)
    assert result.book.portfolio_value == pytest.approx(900.0)
    first_observation = strategy.observations[0]
    assert getattr(first_observation, "symbol") == "ETHUSDT"
    assert getattr(first_observation, "features")[0] == 0.0


def test_replay_rejects_invalid_symbol_index() -> None:
    with pytest.raises(ValueError, match="symbol_index"):
        evaluation.run_single_symbol_replay(
            _two_symbol_market(),
            AlwaysLong(),
            symbol_index=2,
            start_index=0,
            stop_index=5,
            gross_budget=0.5,
        )


def test_fixed_drawdown_does_not_compound_risk_scale_into_next_proposal() -> None:
    close = np.asarray([[100.0], [70.0], [70.0], [70.0], [70.0]])
    open_price = np.vstack((close[0], close[:-1]))
    dataset = MarketDataset(
        dataset_id="c" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(close.shape[0]) * np.timedelta64(1, "h"),
        features=np.zeros((close.shape[0], 1, 1), dtype=np.float32),
        global_features=np.zeros((close.shape[0], 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((close.shape[0], 1), 1_000_000.0),
        funding_rate=np.zeros((close.shape[0], 1)),
        tradable=np.ones((close.shape[0], 1), dtype=np.bool_),
        feature_available=np.ones((close.shape[0], 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )

    result = evaluation.run_single_symbol_replay(
        dataset,
        AlwaysLong(),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
    )

    assert result.book.max_drawdown == pytest.approx(0.15)
    assert result.decisions[0].target_weight == pytest.approx(0.5)
    assert result.decisions[1].target_weight == pytest.approx(0.20588235294117646)
    assert result.decisions[2].target_weight == pytest.approx(
        result.decisions[1].target_weight
    )
