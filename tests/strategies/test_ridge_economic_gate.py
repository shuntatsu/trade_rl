from __future__ import annotations

import inspect
import math

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


def _model() -> RidgeForecastModel:
    return RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.asarray([0.0], dtype=np.float64),
        feature_scale=np.asarray([1.0], dtype=np.float64),
        coefficients=np.asarray([1.0], dtype=np.float64),
        intercept=0.0,
        horizon_hours=24,
        alpha=1.0,
        n_samples=100,
        fit_cutoff=np.datetime64("2023-01-01T00:00:00", "ns"),
    )


def _observation(
    forecast_feature: float,
    *,
    current: PositionIntent = PositionIntent.FLAT,
    available: bool = True,
) -> StrategyObservation:
    return StrategyObservation(
        index=100,
        timestamp=np.datetime64("2024-01-01T00:00:00", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([forecast_feature], dtype=np.float32),
        feature_available=np.asarray([available], dtype=np.bool_),
        global_features=np.asarray([0.0], dtype=np.float32),
        global_feature_available=np.asarray([True], dtype=np.bool_),
        current_intent=current,
        current_weight=0.0,
    )


def _strategy(
    *,
    model: RidgeForecastModel | None = None,
    cost: float = 0.0007,
) -> RidgeEconomicGateStrategy:
    resolved = _model() if model is None else model
    return RidgeEconomicGateStrategy(
        RidgeEconomicGateConfig(
            model=resolved,
            entry_threshold=0.0025,
            exit_threshold=0.0005,
            one_way_explicit_cost=cost,
        )
    )


def test_wrapper_reuses_exact_ridge_model_and_proposal_thresholds() -> None:
    model = _model()
    strategy = _strategy(model=model)

    assert strategy.model is model
    assert strategy.proposal.model is model
    assert strategy.proposal.controller.config.entry_threshold == 0.0025
    assert strategy.proposal.controller.config.exit_threshold == 0.0005


def test_gate_allows_flat_entries_with_positive_net_edge() -> None:
    strategy = _strategy()

    assert strategy.decide(_observation(0.003)) is PositionIntent.LONG
    assert strategy.decide(_observation(-0.003)) is PositionIntent.SHORT


def test_gate_exit_sign_arithmetic_is_directionally_correct() -> None:
    strategy = _strategy()

    assert (
        strategy.decide(_observation(-0.001, current=PositionIntent.LONG))
        is PositionIntent.FLAT
    )
    assert (
        strategy.decide(_observation(0.001, current=PositionIntent.SHORT))
        is PositionIntent.FLAT
    )


def test_gate_holds_below_cost_exit_and_is_inert_for_same_proposal() -> None:
    strategy = _strategy()

    assert (
        strategy.decide(_observation(0.0005, current=PositionIntent.LONG))
        is PositionIntent.LONG
    )
    assert (
        strategy.decide(_observation(0.003, current=PositionIntent.LONG))
        is PositionIntent.LONG
    )


def test_transition_diagnostics_charge_reversal_as_two_legs() -> None:
    decision = evaluate_ridge_economic_transition(
        current=PositionIntent.LONG,
        proposed=PositionIntent.SHORT,
        forecast=-0.003,
        one_way_explicit_cost=0.0007,
    )

    assert decision.delta_position == -2
    assert decision.expected_incremental_edge == pytest.approx(0.006)
    assert decision.transition_cost == pytest.approx(0.0014)
    assert decision.allowed is True


def test_transition_equality_is_rejected() -> None:
    decision = evaluate_ridge_economic_transition(
        current=PositionIntent.FLAT,
        proposed=PositionIntent.LONG,
        forecast=0.0007,
        one_way_explicit_cost=0.0007,
    )

    assert decision.expected_incremental_edge == decision.transition_cost
    assert decision.allowed is False


def test_unavailable_or_nonfinite_features_preserve_proposal_fail_safe_flat() -> None:
    strategy = _strategy()

    assert (
        strategy.decide(
            _observation(float("nan"), current=PositionIntent.LONG, available=False)
        )
        is PositionIntent.FLAT
    )
    assert (
        strategy.decide(
            _observation(float("nan"), current=PositionIntent.SHORT, available=True)
        )
        is PositionIntent.FLAT
    )


def test_gate_config_rejects_wrong_model_invalid_thresholds_and_invalid_cost() -> None:
    with pytest.raises(TypeError, match="model"):
        RidgeEconomicGateConfig(  # type: ignore[arg-type]
            model=object(),
            entry_threshold=0.0025,
            exit_threshold=0.0005,
            one_way_explicit_cost=0.0007,
        )

    for entry, exit_threshold in ((0.0, 0.0005), (0.0025, 0.0025)):
        with pytest.raises(ValueError):
            RidgeEconomicGateConfig(
                model=_model(),
                entry_threshold=entry,
                exit_threshold=exit_threshold,
                one_way_explicit_cost=0.0007,
            )

    for cost in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="one_way_explicit_cost"):
            RidgeEconomicGateConfig(
                model=_model(),
                entry_threshold=0.0025,
                exit_threshold=0.0005,
                one_way_explicit_cost=cost,
            )


def test_strategy_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="config"):
        RidgeEconomicGateStrategy(object())  # type: ignore[arg-type]


def test_transition_rejects_nonfinite_forecast_and_invalid_cost() -> None:
    with pytest.raises(ValueError, match="forecast"):
        evaluate_ridge_economic_transition(
            current=PositionIntent.FLAT,
            proposed=PositionIntent.LONG,
            forecast=math.nan,
            one_way_explicit_cost=0.0007,
        )
    with pytest.raises(ValueError, match="one_way_explicit_cost"):
        evaluate_ridge_economic_transition(
            current=PositionIntent.FLAT,
            proposed=PositionIntent.LONG,
            forecast=0.01,
            one_way_explicit_cost=-0.001,
        )


def test_wrapper_api_has_no_dataset_economics_or_hard_risk_inputs() -> None:
    config_fields = set(RidgeEconomicGateConfig.__dataclass_fields__)
    assert config_fields == {
        "model",
        "entry_threshold",
        "exit_threshold",
        "one_way_explicit_cost",
    }
    decide_parameters = set(
        inspect.signature(RidgeEconomicGateStrategy.decide).parameters
    )
    assert decide_parameters == {"self", "observation"}
