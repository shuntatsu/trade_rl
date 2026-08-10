from __future__ import annotations

import numpy as np

from trade_rl.rl.transition import classify_economic_transition
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason


def _book() -> BookState:
    return BookState.zero(
        n_symbols=1,
        initial_capital=100_000.0,
        initial_prices=np.array([100.0], dtype=np.float64),
    )


def test_shadow_failure_remains_diagnostic_for_policy_mdp() -> None:
    hybrid = _book()
    shadow = _book()
    shadow.terminate(EconomicTerminationReason.MINIMUM_EQUITY)

    transition = classify_economic_transition(
        hybrid=hybrid,
        shadow=shadow,
        time_limit_reached=False,
        liquidation_terminal=False,
        liquidation_complete=True,
    )

    assert transition.terminated is False
    assert transition.truncated is False
    assert transition.reason is None


def test_hybrid_failure_remains_a_true_policy_termination() -> None:
    hybrid = _book()
    shadow = _book()
    hybrid.terminate(EconomicTerminationReason.DRAWDOWN_STOP)

    transition = classify_economic_transition(
        hybrid=hybrid,
        shadow=shadow,
        time_limit_reached=False,
        liquidation_terminal=False,
        liquidation_complete=True,
    )

    assert transition.terminated is True
    assert transition.truncated is False
    assert transition.reason == EconomicTerminationReason.DRAWDOWN_STOP.value


def test_hybrid_failure_takes_precedence_when_both_books_fail() -> None:
    hybrid = _book()
    shadow = _book()
    hybrid.terminate(EconomicTerminationReason.MARGIN_CALL)
    shadow.terminate(EconomicTerminationReason.MINIMUM_EQUITY)

    transition = classify_economic_transition(
        hybrid=hybrid,
        shadow=shadow,
        time_limit_reached=False,
        liquidation_terminal=False,
        liquidation_complete=True,
    )

    assert transition.terminated is True
    assert transition.truncated is False
    assert transition.reason == EconomicTerminationReason.MARGIN_CALL.value


def test_external_time_limit_remains_a_truncation() -> None:
    transition = classify_economic_transition(
        hybrid=_book(),
        shadow=_book(),
        time_limit_reached=True,
        liquidation_terminal=False,
        liquidation_complete=True,
        time_limit_terminates=False,
    )

    assert transition.terminated is False
    assert transition.truncated is True
    assert transition.reason is None


def test_intrinsic_finite_horizon_is_a_true_termination() -> None:
    transition = classify_economic_transition(
        hybrid=_book(),
        shadow=_book(),
        time_limit_reached=True,
        liquidation_terminal=False,
        liquidation_complete=True,
        time_limit_terminates=True,
    )

    assert transition.terminated is True
    assert transition.truncated is False
    assert transition.reason == "finite_horizon"
