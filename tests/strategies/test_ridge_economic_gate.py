from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.strategies.forecasts.ridge import RidgeForecastModel
from trade_rl.strategies.forecasts.ridge_economic_gate import (
    RidgeEconomicGateConfig,
    RidgeEconomicGateStrategy,
    evaluate_ridge_economic_transition,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


def _model(*, coefficient: float = 1.0, intercept: float = 0.0) -> RidgeForecastModel:
    return RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.asarray([0.0]),
        feature_scale=np.asarray([1.0]),
        coefficients=np.asarray([coefficient]),
        intercept=intercept,
        horizon_hours=24,
        alpha=1.0,
        n_samples=10,
        fit_cutoff=np.datetime64(datetime(2023, 1, 1, tzinfo=UTC)),
    )


def _observation(
    feature: float,
    *,
    current: PositionIntent = PositionIntent.FLAT,
    available: bool = True,
) -> StrategyObservation:
    return StrategyObservation(
        index=0,
        timestamp=np.datetime64("2023-01-01T00:00:00", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([feature], dtype=np.float64),
        feature_available=np.asarray([available], dtype=np.bool_),
        feature_staleness=np.asarray([0.0 if available else 1.0], dtype=np.float32),
        global_features=np.asarray([0.0]),
        global_feature_available=np.asarray([True]),
        current_intent=current,
        current_weight=float(current),
    )


def _strategy(*, cost: float = 0.0007) -> RidgeEconomicGateStrategy:
    return RidgeEconomicGateStrategy(
        RidgeEconomicGateConfig(
            model=_model(),
            entry_threshold=0.0025,
            exit_threshold=0.0005,
            one_way_explicit_cost=cost,
        )
    )


def test_config_preserves_exact_ridge_proposal_model_and_thresholds() -> None:
    model = _model()
    config = RidgeEconomicGateConfig(
        model=model,
        entry_threshold=0.0025,
        exit_threshold=0.0005,
        one_way_explicit_cost=0.0007,
    )
    strategy = RidgeEconomicGateStrategy(config)

    assert strategy.config.model is model
    assert strategy.proposal.model is model
    assert strategy.proposal.controller.config.entry_threshold == 0.0025
    assert strategy.proposal.controller.config.exit_threshold == 0.0005


def test_flat_to_long_is_allowed_only_when_forecast_edge_strictly_exceeds_cost() -> (
    None
):
    strategy = _strategy()

    assert strategy.decide(_observation(0.01)) is PositionIntent.LONG
    assert strategy.decide(_observation(0.0007)) is PositionIntent.FLAT


def test_flat_to_short_uses_signed_delta_times_forecast() -> None:
    strategy = _strategy()

    assert strategy.decide(_observation(-0.01)) is PositionIntent.SHORT


def test_long_to_short_reversal_uses_two_transition_legs() -> None:
    decision = evaluate_ridge_economic_transition(
        current=PositionIntent.LONG,
        proposed=PositionIntent.SHORT,
        ridge_forecast=-0.001,
        one_way_explicit_cost=0.0007,
    )

    assert decision.delta_position == -2
    assert decision.expected_incremental_edge == pytest.approx(0.002)
    assert decision.transition_cost == pytest.approx(0.0014)
    assert decision.allowed is True


def test_equal_edge_and_transition_cost_holds_current_intent() -> None:
    decision = evaluate_ridge_economic_transition(
        current=PositionIntent.FLAT,
        proposed=PositionIntent.LONG,
        ridge_forecast=0.0007,
        one_way_explicit_cost=0.0007,
    )

    assert decision.expected_incremental_edge == pytest.approx(decision.transition_cost)
    assert decision.allowed is False


def test_missing_feature_bypasses_cost_gate_and_preserves_fail_safe_flat() -> None:
    strategy = _strategy(cost=100.0)

    assert (
        strategy.decide(
            _observation(0.01, current=PositionIntent.LONG, available=False)
        )
        is PositionIntent.FLAT
    )


def test_nonfinite_feature_bypasses_cost_gate_and_preserves_fail_safe_flat() -> None:
    strategy = _strategy(cost=100.0)

    assert (
        strategy.decide(_observation(float("nan"), current=PositionIntent.LONG))
        is PositionIntent.FLAT
    )


def test_no_proposed_state_change_does_not_apply_economic_gate() -> None:
    strategy = _strategy(cost=100.0)

    assert (
        strategy.decide(_observation(0.01, current=PositionIntent.LONG))
        is PositionIntent.LONG
    )


def test_transition_helper_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="ridge_forecast"):
        evaluate_ridge_economic_transition(
            current=PositionIntent.FLAT,
            proposed=PositionIntent.LONG,
            ridge_forecast=float("nan"),
            one_way_explicit_cost=0.0007,
        )
    with pytest.raises(ValueError, match="one_way_explicit_cost"):
        evaluate_ridge_economic_transition(
            current=PositionIntent.FLAT,
            proposed=PositionIntent.LONG,
            ridge_forecast=0.01,
            one_way_explicit_cost=-0.1,
        )


def test_config_rejects_threshold_or_cost_drift() -> None:
    model = _model()
    with pytest.raises(ValueError, match="entry_threshold"):
        RidgeEconomicGateConfig(
            model=model,
            entry_threshold=0.0,
            exit_threshold=0.0,
            one_way_explicit_cost=0.0007,
        )
    with pytest.raises(ValueError, match="exit_threshold"):
        RidgeEconomicGateConfig(
            model=model,
            entry_threshold=0.0025,
            exit_threshold=0.0025,
            one_way_explicit_cost=0.0007,
        )
    with pytest.raises(ValueError, match="one_way_explicit_cost"):
        RidgeEconomicGateConfig(
            model=model,
            entry_threshold=0.0025,
            exit_threshold=0.0005,
            one_way_explicit_cost=float("nan"),
        )
