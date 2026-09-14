from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.mean_reversion import MeanReversionIntentConfig
from trade_rl.strategies.rules.mean_reversion_economic_gate import (
    MeanReversionEconomicGateConfig,
    MeanReversionEconomicGateStrategy,
    evaluate_economic_transition,
)


def _observation(
    signal: float,
    *,
    current: PositionIntent = PositionIntent.FLAT,
    available: bool = True,
) -> StrategyObservation:
    return StrategyObservation(
        index=100,
        timestamp=np.datetime64("2024-01-01T00:00:00", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([0.0, 0.0, signal], dtype=np.float32),
        feature_available=np.asarray([True, True, available], dtype=np.bool_),
        global_features=np.asarray([1.0], dtype=np.float32),
        global_feature_available=np.asarray([True], dtype=np.bool_),
        current_intent=current,
        current_weight=0.0,
    )


def _strategy(
    *,
    beta_gate: float = -0.1,
    one_way_cost: float = 0.001,
) -> MeanReversionEconomicGateStrategy:
    return MeanReversionEconomicGateStrategy(
        MeanReversionEconomicGateConfig(
            proposal=MeanReversionIntentConfig(
                signal_index=2,
                entry_threshold=0.01,
                exit_threshold=0.0025,
            ),
            beta_gate=beta_gate,
            one_way_explicit_cost=one_way_cost,
        )
    )


def test_gate_allows_only_positive_net_incremental_edge() -> None:
    strategy = _strategy(beta_gate=-0.1, one_way_cost=0.001)

    assert strategy.decide(_observation(-0.02)) is PositionIntent.LONG
    assert strategy.decide(_observation(0.02)) is PositionIntent.SHORT

    # The proposal enters at exactly -1%; expected edge is only 0.1%, equal to
    # the one-way cost, so the preregistered strict '>' gate must hold FLAT.
    equality = _strategy(beta_gate=-1.0, one_way_cost=0.01)
    assert equality.decide(_observation(-0.01)) is PositionIntent.FLAT


def test_gate_holds_current_when_proposal_has_no_incremental_economic_value() -> None:
    strategy = _strategy(beta_gate=-0.1, one_way_cost=0.001)

    # Baseline hysteresis proposes FLAT here, but moving SHORT -> FLAT has zero
    # expected incremental edge at signal zero and therefore is not voluntary.
    assert (
        strategy.decide(_observation(0.0, current=PositionIntent.SHORT))
        is PositionIntent.SHORT
    )

    # When the proposal already equals current intent, the wrapper is inert.
    assert (
        strategy.decide(_observation(-0.02, current=PositionIntent.LONG))
        is PositionIntent.LONG
    )


def test_unavailable_or_nonfinite_signal_preserves_proposal_fail_safe_flat() -> None:
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


def test_transition_diagnostics_charge_reversal_as_two_legs() -> None:
    decision = evaluate_economic_transition(
        current=PositionIntent.LONG,
        proposed=PositionIntent.SHORT,
        signal=0.02,
        beta_gate=-0.1,
        one_way_explicit_cost=0.001,
    )

    assert decision.delta_position == -2
    assert decision.expected_incremental_edge == pytest.approx(0.004)
    assert decision.transition_cost == pytest.approx(0.002)
    assert decision.allowed is True


def test_transition_equality_is_rejected() -> None:
    decision = evaluate_economic_transition(
        current=PositionIntent.FLAT,
        proposed=PositionIntent.LONG,
        signal=-0.01,
        beta_gate=-1.0,
        one_way_explicit_cost=0.01,
    )

    assert decision.expected_incremental_edge == decision.transition_cost
    assert decision.allowed is False


def test_gate_config_rejects_nonnegative_beta_and_invalid_cost() -> None:
    proposal = MeanReversionIntentConfig(
        signal_index=2,
        entry_threshold=0.01,
        exit_threshold=0.0025,
    )
    for beta in (0.0, 0.1, float("nan")):
        with pytest.raises(ValueError, match="beta_gate"):
            MeanReversionEconomicGateConfig(
                proposal=proposal,
                beta_gate=beta,
                one_way_explicit_cost=0.0007,
            )
    for cost in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="one_way_explicit_cost"):
            MeanReversionEconomicGateConfig(
                proposal=proposal,
                beta_gate=-0.1,
                one_way_explicit_cost=cost,
            )


def test_transition_rejects_nonfinite_signal() -> None:
    with pytest.raises(ValueError, match="signal"):
        evaluate_economic_transition(
            current=PositionIntent.FLAT,
            proposed=PositionIntent.LONG,
            signal=math.nan,
            beta_gate=-0.1,
            one_way_explicit_cost=0.0007,
        )
