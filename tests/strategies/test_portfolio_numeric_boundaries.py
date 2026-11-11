"""Fail-closed numeric boundaries for portfolio policy contracts (#925)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.strategies.portfolio import PortfolioRiskLimits, PortfolioTargetProposal


@pytest.mark.parametrize("huge", (10**500, -(10**500)))
def test_target_proposal_refuses_unrepresentable_integer_with_typed_error(
    huge: int,
) -> None:
    with pytest.raises(ValueError, match="target_weights must be finite"):
        PortfolioTargetProposal(target_weights=(huge,))


@pytest.mark.parametrize("huge", (10**500, -(10**500)))
@pytest.mark.parametrize(
    "field",
    (
        "max_gross",
        "max_abs_weight",
        "max_turnover",
        "drawdown_start",
        "drawdown_stop",
        "fail_closed_tolerance",
    ),
)
def test_risk_limits_refuse_unrepresentable_integer_with_field_error(
    huge: int, field: str
) -> None:
    baseline = PortfolioRiskLimits(
        max_gross=1.0,
        max_abs_weight=0.5,
        max_turnover=0.25,
        drawdown_start=0.2,
        drawdown_stop=0.4,
        emergency_turnover_override=False,
        fail_closed_tolerance=1e-8,
    )
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        replace(baseline, **{field: huge})


def test_valid_portfolio_contracts_keep_their_original_values() -> None:
    proposal = PortfolioTargetProposal(target_weights=(0.4, -0.25))
    assert proposal.target_weights == (0.4, -0.25)
    limits = PortfolioRiskLimits(1.0, 0.5, None, 0.2, 0.4, False, 1e-8)
    assert limits.max_gross == 1.0
    assert limits.max_turnover is None
