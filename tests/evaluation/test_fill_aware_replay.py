from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import trade_rl.evaluation as evaluation
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation import replay as replay_module
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


class _RecordingRidgeForecastStrategy:
    def __init__(self) -> None:
        self.observations: list[StrategyObservation] = []
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
        self.strategy = RidgeForecastStrategy(
            model,
            entry_threshold=0.10,
            exit_threshold=0.02,
            one_way_switch_cost=0.01,
        )

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        self.observations.append(observation)
        return self.strategy.decide(observation)


class _SequenceStrategy:
    def __init__(self, intents: tuple[PositionIntent, ...]) -> None:
        self.intents = intents
        self.observations: list[StrategyObservation] = []

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        self.observations.append(observation)
        return self.intents[min(len(self.observations) - 1, len(self.intents) - 1)]


def _drifting_mark_dataset() -> MarketDataset:
    simple_return = -0.0095
    forecasts = np.asarray([0.11, math.log1p(simple_return), 0.0])
    mark_after_fill = 900.0 / 11.0
    close = np.asarray([[100.0], [mark_after_fill], [mark_after_fill]])
    open_price = np.vstack((close[0], close[:-1]))
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=("SYNTH",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(close.shape[0]) * np.timedelta64(1, "h"),
        features=forecasts.astype(np.float32).reshape(-1, 1, 1),
        global_features=np.zeros((close.shape[0], 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((close.shape[0], 1), 1_000_000.0),
        funding_rate=np.zeros((close.shape[0], 1)),
        tradable=np.ones((close.shape[0], 1), dtype=np.bool_),
        feature_available=np.ones((close.shape[0], 1, 1), dtype=np.bool_),
        feature_names=("literal_forecast",),
        global_feature_names=("constant",),
        periods_per_year=8_760,
    )


def _flat_market_dataset(
    forecast_values: tuple[float, ...] | None = None,
) -> MarketDataset:
    close = np.full((6, 1), 100.0)
    values = np.zeros(6, dtype=np.float32)
    if forecast_values is not None:
        values = np.asarray(forecast_values, dtype=np.float32)
        if values.shape != (6,):
            raise ValueError("forecast_values must contain one value per bar")
    forecasts = values.reshape(6, 1, 1)
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("SYNTH",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(close.shape[0]) * np.timedelta64(1, "h"),
        features=forecasts,
        global_features=np.zeros((close.shape[0], 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((close.shape[0], 1), 1_000_000.0),
        funding_rate=np.zeros((close.shape[0], 1)),
        tradable=np.ones((close.shape[0], 1), dtype=np.bool_),
        feature_available=np.ones((close.shape[0], 1, 1), dtype=np.bool_),
        feature_names=("unused",),
        global_feature_names=("constant",),
        periods_per_year=8_760,
    )


def _partial_fill_market_dataset() -> MarketDataset:
    close = np.full((3, 1), 100.0)
    forecasts = np.asarray([0.11, math.log1p(-0.005), math.log1p(-0.005)])
    return MarketDataset(
        dataset_id="7" * 64,
        symbols=("SYNTH",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(close.shape[0]) * np.timedelta64(1, "h"),
        features=forecasts.astype(np.float32).reshape(-1, 1, 1),
        global_features=np.zeros((close.shape[0], 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((close.shape[0], 1), 100.0),
        funding_rate=np.zeros((close.shape[0], 1)),
        tradable=np.ones((close.shape[0], 1), dtype=np.bool_),
        feature_available=np.ones((close.shape[0], 1, 1), dtype=np.bool_),
        feature_names=("literal_forecast",),
        global_feature_names=("constant",),
        periods_per_year=8_760,
    )


def _run_replay(
    replay_kind: str,
    dataset: MarketDataset,
    strategy: object,
    *,
    stop_index: int,
    minimum_hold_bars: int = 0,
    max_abs_weight: float = 1.0,
    execution_cost: ExecutionCostConfig | None = None,
    capture_ledger_evidence: bool = False,
) -> object:
    kwargs = dict(
        start_index=0,
        stop_index=stop_index,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost or ExecutionCostConfig.zero(),
        minimum_hold_bars=minimum_hold_bars,
        settle_terminal_position=False,
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=max_abs_weight,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
    )
    if replay_kind == "single":
        return evaluation.run_single_symbol_replay(dataset, strategy, **kwargs)
    return replay_module.run_shared_cash_replay(
        dataset,
        (strategy,),
        capture_ledger_evidence=capture_ledger_evidence,
        **kwargs,
    )


@pytest.mark.parametrize("replay_kind", ("single", "shared"))
def test_cost_aware_replay_preserves_fully_filled_cached_quantity_hold(
    replay_kind: str,
) -> None:
    strategy = _RecordingRidgeForecastStrategy()
    dataset = _drifting_mark_dataset()
    result = _run_replay(
        replay_kind,
        dataset,
        strategy,
        stop_index=2,
    )

    held_observation = strategy.observations[1]
    assert held_observation.current_weight == pytest.approx(0.45)
    assert held_observation.current_position_quantity == pytest.approx(5.0)
    assert held_observation.same_intent_target_weight == pytest.approx(0.45)
    if replay_kind == "single":
        assert result.decisions[1].intent is PositionIntent.LONG
        assert result.decisions[1].risk_reasons == ()
    else:
        assert result.decisions[1].intents == (PositionIntent.LONG,)
        assert result.decisions[1].risk_reasons == ()
    assert result.book.quantities == pytest.approx((5.0,))
    assert result.book.fill_count == 1
    assert result.book.turnover_total == pytest.approx(0.5)


@pytest.mark.parametrize("replay_kind", ("single", "shared"))
def test_cost_aware_replay_prices_persistent_risk_cap_target(
    replay_kind: str,
) -> None:
    strategy = _RecordingRidgeForecastStrategy()
    forecasts = (
        0.11,
        math.log1p(-0.0095),
        -0.02,
        -0.02,
        -0.02,
        -0.02,
    )
    result = _run_replay(
        replay_kind,
        _flat_market_dataset(forecasts),
        strategy,
        stop_index=3,
        max_abs_weight=0.4,
        capture_ledger_evidence=True,
    )

    held_observation = strategy.observations[1]
    assert result.decisions[0].risk_reasons == ("max_abs_weight",)
    assert held_observation.current_weight == pytest.approx(0.4)
    assert held_observation.current_position_quantity == pytest.approx(4.0)
    assert held_observation.same_intent_target_weight == pytest.approx(0.4)
    if replay_kind == "single":
        assert result.decisions[0].target_weight == pytest.approx(0.4)
        assert result.decisions[1].intent is PositionIntent.LONG
        assert result.decisions[1].target_weight == pytest.approx(0.4)
        assert result.decisions[2].intent is PositionIntent.FLAT
        assert result.book.quantities == pytest.approx((0.0,))
        assert result.active_order_remainders == ()
    else:
        assert result.decisions[0].target_weights == pytest.approx((0.4,))
        assert result.decisions[1].intents == (PositionIntent.LONG,)
        assert result.decisions[1].target_weights == pytest.approx((0.4,))
        assert result.decisions[2].intents == (PositionIntent.FLAT,)
        assert result.book.quantities == pytest.approx((0.0,))
        assert result.ledger_evidence is not None
        assert result.ledger_evidence.active_order_remainders == ()


@pytest.mark.parametrize("replay_kind", ("single", "shared"))
def test_partial_fill_residual_is_cancelled_when_cached_target_loses_to_flat(
    replay_kind: str,
) -> None:
    strategy = _RecordingRidgeForecastStrategy()
    dataset = _partial_fill_market_dataset()
    execution_cost = replace(
        ExecutionCostConfig.zero(),
        max_participation_rate=0.01,
    )
    result = _run_replay(
        replay_kind,
        dataset,
        strategy,
        stop_index=2,
        execution_cost=execution_cost,
        capture_ledger_evidence=True,
    )

    partial_observation = strategy.observations[1]
    assert partial_observation.current_weight == pytest.approx(0.1)
    assert partial_observation.current_position_quantity == pytest.approx(1.0)
    assert partial_observation.same_intent_target_weight == pytest.approx(0.5)
    if replay_kind == "single":
        assert result.decisions[0].target_weight == pytest.approx(0.5)
        assert result.decisions[0].position_quantity_after == pytest.approx(1.0)
        assert result.decisions[1].intent is PositionIntent.FLAT
        assert result.decisions[1].position_quantity_before == pytest.approx(1.0)
        assert result.decisions[1].position_quantity_after == pytest.approx(0.0)
        assert result.terminal_order_reasons
        assert "superseded" in {reason for _, reason in result.terminal_order_reasons}
    else:
        assert result.decisions[1].intents == (PositionIntent.FLAT,)
        assert result.ledger_evidence is not None
        first_fill = next(
            event
            for event in result.ledger_evidence.intervals[0].order_events
            if event.event_type == "partial_fill"
        )
        assert first_fill.requested_quantity == pytest.approx(5.0)
        assert first_fill.filled_quantity == pytest.approx(1.0)
        assert first_fill.remaining_quantity == pytest.approx(4.0)
        cancelled = (
            event
            for event in result.ledger_evidence.intervals[1].order_events
            if event.event_type == "cancelled"
        )
        assert any(event.reason == "superseded" for event in cancelled)
        sell = next(
            event
            for event in result.ledger_evidence.intervals[1].order_events
            if event.event_type == "filled"
        )
        assert sell.filled_quantity == pytest.approx(-1.0)
        assert result.ledger_evidence.active_order_remainders == ()

    assert result.book.fill_count == 2
    assert result.book.turnover_total == pytest.approx(0.2)
    assert result.book.quantities == pytest.approx((0.0,))
    if replay_kind == "single":
        assert result.active_order_remainders == ()


@pytest.mark.parametrize("replay_kind", ("single", "shared"))
def test_minimum_hold_lock_omits_cached_target_until_unlock_reset(
    replay_kind: str,
) -> None:
    strategy = _SequenceStrategy(
        (
            PositionIntent.LONG,
            PositionIntent.FLAT,
            PositionIntent.FLAT,
            PositionIntent.FLAT,
        )
    )
    result = _run_replay(
        replay_kind,
        _flat_market_dataset(),
        strategy,
        stop_index=4,
        minimum_hold_bars=3,
    )

    assert strategy.observations[1].same_intent_target_weight is None
    assert strategy.observations[2].same_intent_target_weight is None
    assert strategy.observations[3].same_intent_target_weight is None
    if replay_kind == "single":
        assert result.decisions[1].minimum_hold_suppressed
        assert result.decisions[2].minimum_hold_suppressed
        assert result.decisions[3].minimum_hold_unlocked
        assert result.decisions[3].target_weight == pytest.approx(0.0)
    else:
        assert result.decisions[1].minimum_hold_suppressed == (True,)
        assert result.decisions[2].minimum_hold_suppressed == (True,)
        assert result.decisions[3].minimum_hold_unlocked == (True,)
        assert result.decisions[3].target_weights == pytest.approx((0.0,))
