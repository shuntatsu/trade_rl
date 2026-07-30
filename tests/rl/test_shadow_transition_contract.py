from __future__ import annotations

from trade_rl.rl.transition import classify_economic_transition
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason


def test_shadow_insolvency_is_diagnostic_not_an_episode_boundary() -> None:
    hybrid = BookState.zero(1, 1_000.0)
    shadow = BookState.zero(1, 1_000.0)
    shadow.terminate(EconomicTerminationReason.INSOLVENCY)

    outcome = classify_economic_transition(
        hybrid=hybrid,
        shadow=shadow,
        time_limit_reached=False,
        liquidation_terminal=False,
        liquidation_complete=True,
    )

    assert outcome.terminated is False
    assert outcome.truncated is False
    assert outcome.reason is None
