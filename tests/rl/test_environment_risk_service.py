from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.risk.inputs import PortfolioRiskInputs
from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.environment_risk import (
    EnvironmentRiskProjector,
    EnvironmentRiskRequest,
)
from trade_rl.simulation.accounting import BookState


class _Dataset:
    n_symbols = 2
    close = np.array([[10.0, 20.0], [11.0, 25.0]])
    volume = np.array([[100.0, 5.0], [120.0, 4.0]])
    volume_units = ("quote_notional", "base_units")


class _EmergencyMonitor:
    def __init__(self, *, flatten: tuple[bool, bool], reasons: tuple[str, ...]) -> None:
        self.flatten = np.asarray(flatten, dtype=np.bool_)
        self.reasons = reasons

    def assess(
        self,
        dataset: object,
        *,
        index: int,
        weights: np.ndarray,
    ) -> SimpleNamespace:
        assert dataset is not None
        assert index == 1
        assert weights.shape == (2,)
        return SimpleNamespace(flatten_mask=self.flatten, reasons=self.reasons)


class _AdvancedProvider:
    identity_digest = "0" * 64
    minimum_index = 1

    def __init__(self) -> None:
        self.calls: list[int] = []

    def inputs(self, dataset: object, *, index: int) -> PortfolioRiskInputs:
        assert dataset is not None
        self.calls.append(index)
        return PortfolioRiskInputs(
            covariance=np.eye(2) * 0.04,
            beta=np.array([1.0, -0.5]),
            stress_losses=np.array([-0.2, -0.1]),
            as_of_index=index,
            provider_digest=self.identity_digest,
            observation_count=10,
        )


def _book() -> BookState:
    return BookState.from_weights(
        weights=np.array([0.2, -0.2]),
        capital=100.0,
        prices=np.array([11.0, 25.0]),
        max_gross=1.0,
    )


def test_market_notional_respects_quote_and_base_volume_units() -> None:
    projector = EnvironmentRiskProjector(
        _Dataset(),
        emergency_risk_monitor=_EmergencyMonitor(flatten=(False, False), reasons=()),
        pre_trade_risk=PreTradeRisk(),
        portfolio_risk=PortfolioRiskModel(),
        portfolio_risk_inputs_provider=None,
    )

    np.testing.assert_allclose(projector.market_notional(1), np.array([120.0, 100.0]))


def test_projection_preserves_reason_order_and_projection_distance() -> None:
    projector = EnvironmentRiskProjector(
        _Dataset(),
        emergency_risk_monitor=_EmergencyMonitor(
            flatten=(False, True),
            reasons=("market_halt",),
        ),
        pre_trade_risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.4,
                max_turnover=2.0,
            )
        ),
        portfolio_risk=PortfolioRiskModel(
            PortfolioRiskConfig(max_net_exposure=0.2)
        ),
        portfolio_risk_inputs_provider=None,
    )

    result = projector.project(
        EnvironmentRiskRequest(
            proposal=np.array([0.8, 0.3]),
            book=_book(),
            current_index=1,
        )
    )

    np.testing.assert_allclose(result.weights, np.array([0.2, 0.0]))
    assert result.reasons == (
        "emergency_flatten",
        "max_abs_weight",
        "emergency_turnover_override",
        "market_halt",
        "portfolio:max_net_exposure",
    )
    assert result.projection_l1 == pytest.approx(0.9)
    assert result.was_constrained is True


@pytest.mark.parametrize(
    "proposal",
    [np.array([0.1]), np.array([0.1, np.nan])],
)
def test_projection_rejects_invalid_proposals(proposal: np.ndarray) -> None:
    projector = EnvironmentRiskProjector(
        _Dataset(),
        emergency_risk_monitor=_EmergencyMonitor(flatten=(False, False), reasons=()),
        pre_trade_risk=PreTradeRisk(),
        portfolio_risk=PortfolioRiskModel(),
        portfolio_risk_inputs_provider=None,
    )

    with pytest.raises(ValueError, match="proposal does not match dataset symbols"):
        projector.project(
            EnvironmentRiskRequest(
                proposal=proposal,
                book=_book(),
                current_index=1,
            )
        )


def test_advanced_risk_fails_closed_without_causal_provider() -> None:
    projector = EnvironmentRiskProjector(
        _Dataset(),
        emergency_risk_monitor=_EmergencyMonitor(flatten=(False, False), reasons=()),
        pre_trade_risk=PreTradeRisk(),
        portfolio_risk=PortfolioRiskModel(
            PortfolioRiskConfig(volatility_target=0.1)
        ),
        portfolio_risk_inputs_provider=None,
    )

    with pytest.raises(
        RuntimeError,
        match="advanced portfolio risk requires a causal input provider",
    ):
        projector.project(
            EnvironmentRiskRequest(
                proposal=np.array([0.2, -0.1]),
                book=_book(),
                current_index=1,
            )
        )


def test_advanced_risk_uses_inputs_from_current_index() -> None:
    provider = _AdvancedProvider()
    projector = EnvironmentRiskProjector(
        _Dataset(),
        emergency_risk_monitor=_EmergencyMonitor(flatten=(False, False), reasons=()),
        pre_trade_risk=PreTradeRisk(
            PreTradeRiskConfig(max_abs_weight=1.0, max_turnover=2.0)
        ),
        portfolio_risk=PortfolioRiskModel(
            PortfolioRiskConfig(
                volatility_target=0.1,
                max_abs_beta=0.2,
                max_stress_loss=0.03,
            )
        ),
        portfolio_risk_inputs_provider=provider,
    )

    result = projector.project(
        EnvironmentRiskRequest(
            proposal=np.array([0.5, -0.25]),
            book=_book(),
            current_index=1,
        )
    )

    assert provider.calls == [1]
    assert any(reason.startswith("portfolio:") for reason in result.reasons)
