from __future__ import annotations

import numpy as np

from trade_rl.risk.guardrails import OperationalGuardrailConfig, OperationalGuardrails
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.actions import ResidualAction
from trade_rl.rl.decision import DecisionContext, ResidualDecisionEngine
from trade_rl.strategies.trend import TrendTargets


def trends() -> TrendTargets:
    return TrendTargets(
        fast=np.array([0.5, -0.5]),
        base=np.array([0.5, -0.5]),
        slow=np.array([0.5, -0.5]),
    )


def context(
    *,
    data_age_hours: float = 0.0,
    features_available: bool = True,
    next_tradable: np.ndarray | None = None,
    portfolio_value: float = 1.0,
    day_start_value: float = 1.0,
) -> DecisionContext:
    return DecisionContext(
        current_weights=np.zeros(2),
        portfolio_value=portfolio_value,
        peak_value=max(1.0, portfolio_value),
        day_start_value=day_start_value,
        next_tradable=(
            np.ones(2, dtype=np.bool_)
            if next_tradable is None
            else next_tradable
        ),
        data_age_hours=data_age_hours,
        features_available=features_available,
    )


def engine() -> ResidualDecisionEngine:
    return ResidualDecisionEngine(
        guardrails=OperationalGuardrails(
            OperationalGuardrailConfig(max_data_age_hours=2.0, max_daily_loss=0.05)
        ),
        pre_trade_risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.4,
                max_turnover=2.0,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
    )


def test_zero_action_is_constrained_baseline_identity() -> None:
    decision = engine().decide(
        action=ResidualAction(0.0, 0.0),
        trends=trends(),
        alpha=np.zeros(2),
        alpha_enabled=False,
        context=context(),
    )

    np.testing.assert_allclose(decision.target_weights, np.array([0.4, -0.4]))
    assert decision.guardrail.triggered == ()
    assert decision.risk.reasons == ("max_abs_weight",)


def test_stale_market_flattens_through_shared_guardrail_path() -> None:
    decision = engine().decide(
        action=ResidualAction(1.0, 1.0),
        trends=trends(),
        alpha=np.array([0.5, -0.5]),
        alpha_enabled=True,
        context=context(data_age_hours=3.0),
    )

    np.testing.assert_allclose(decision.target_weights, np.zeros(2))
    assert decision.guardrail.triggered == ("stale_data",)


def test_unavailable_features_and_daily_loss_are_fail_closed() -> None:
    decision = engine().decide(
        action=ResidualAction(0.0, 0.0),
        trends=trends(),
        alpha=np.zeros(2),
        alpha_enabled=False,
        context=context(
            features_available=False,
            portfolio_value=0.94,
            day_start_value=1.0,
        ),
    )

    np.testing.assert_allclose(decision.target_weights, np.zeros(2))
    assert decision.guardrail.triggered == ("features_unavailable", "daily_loss")


def test_non_tradable_symbol_keeps_existing_position() -> None:
    existing = DecisionContext(
        current_weights=np.array([0.2, -0.2]),
        portfolio_value=1.0,
        peak_value=1.0,
        day_start_value=1.0,
        next_tradable=np.array([False, True]),
        data_age_hours=0.0,
        features_available=True,
    )

    decision = engine().decide(
        action=ResidualAction(0.0, 0.0),
        trends=trends(),
        alpha=np.zeros(2),
        alpha_enabled=False,
        context=existing,
    )

    assert decision.target_weights[0] == 0.2
    assert decision.target_weights[1] == -0.4
