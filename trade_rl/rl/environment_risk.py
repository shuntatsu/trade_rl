"""Causal emergency, pre-trade, and portfolio risk projection services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trade_rl.risk.inputs import PortfolioRiskInputsProvider
from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.simulation.accounting import BookState


class RiskDataset(Protocol):
    @property
    def n_bars(self) -> int: ...

    @property
    def n_symbols(self) -> int: ...

    @property
    def periods_per_year(self) -> int: ...

    @property
    def close(self) -> np.ndarray: ...

    @property
    def volume(self) -> np.ndarray: ...

    @property
    def volume_units(self) -> tuple[object, ...]: ...


class EmergencyRiskAssessmentLike(Protocol):
    @property
    def flatten_mask(self) -> np.ndarray: ...

    @property
    def reasons(self) -> tuple[str, ...]: ...


class EmergencyRiskMonitor(Protocol):
    def assess(
        self,
        dataset: RiskDataset,
        *,
        index: int,
        weights: np.ndarray,
    ) -> EmergencyRiskAssessmentLike: ...


@dataclass(frozen=True, slots=True)
class EnvironmentRiskRequest:
    proposal: np.ndarray
    book: BookState
    current_index: int


class EnvironmentRiskProjector:
    """Apply risk controls in the maintained emergency-to-portfolio order."""

    def __init__(
        self,
        dataset: RiskDataset,
        *,
        emergency_risk_monitor: EmergencyRiskMonitor,
        pre_trade_risk: PreTradeRisk,
        portfolio_risk: PortfolioRiskModel,
        portfolio_risk_inputs_provider: PortfolioRiskInputsProvider | None,
    ) -> None:
        self.dataset = dataset
        self.emergency_risk_monitor = emergency_risk_monitor
        self.pre_trade_risk = pre_trade_risk
        self.portfolio_risk = portfolio_risk
        self.portfolio_risk_inputs_provider = portfolio_risk_inputs_provider

    @staticmethod
    def drawdown(book: BookState) -> float:
        value = max(book.portfolio_value, 0.0)
        return min(
            1.0,
            max(0.0, 1.0 - value / max(book.peak_value, value, 1e-12)),
        )

    def market_notional(self, index: int) -> np.ndarray:
        prices = self.dataset.close[index]
        volume = self.dataset.volume[index]
        return np.asarray(
            [
                float(value)
                if str(getattr(unit, "value", unit)) == "quote_notional"
                else float(price * value)
                for price, value, unit in zip(
                    prices,
                    volume,
                    self.dataset.volume_units,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )

    def project(self, request: EnvironmentRiskRequest) -> RiskConstrainedTarget:
        target = np.asarray(request.proposal, dtype=np.float64).reshape(-1).copy()
        if target.shape != (self.dataset.n_symbols,) or not np.isfinite(target).all():
            raise ValueError("proposal does not match dataset symbols")
        assessment = self.emergency_risk_monitor.assess(
            self.dataset,
            index=request.current_index,
            weights=request.book.weights,
        )
        pretrade = self.pre_trade_risk.constrain(
            target,
            current=request.book.weights,
            drawdown=self.drawdown(request.book),
            emergency_flatten_mask=assessment.flatten_mask,
        )
        risk_inputs = None
        if self.portfolio_risk.requires_advanced_inputs:
            provider = self.portfolio_risk_inputs_provider
            if provider is None:
                raise RuntimeError(
                    "advanced portfolio risk requires a causal input provider"
                )
            risk_inputs = provider.inputs(
                self.dataset,
                index=request.current_index,
            )
        portfolio = self.portfolio_risk.constrain(
            pretrade.weights,
            portfolio_value=max(request.book.portfolio_value, 1e-12),
            market_notional=self.market_notional(request.current_index),
            covariance=None if risk_inputs is None else risk_inputs.covariance,
            beta=None if risk_inputs is None else risk_inputs.beta,
            stress_losses=None if risk_inputs is None else risk_inputs.stress_losses,
        )
        final_weights = np.asarray(portfolio.weights, dtype=np.float64)
        constrained_turnover = float(np.abs(final_weights - request.book.weights).sum())
        reasons = tuple(
            dict.fromkeys(
                (
                    *pretrade.reasons,
                    *assessment.reasons,
                    *(f"portfolio:{item}" for item in portfolio.reasons),
                )
            )
        )
        return RiskConstrainedTarget(
            weights=final_weights,
            requested_turnover=pretrade.requested_turnover,
            constrained_turnover=constrained_turnover,
            was_constrained=bool(reasons),
            reasons=reasons,
            risk_scale=pretrade.risk_scale,
            projection_l1=float(np.abs(target - final_weights).sum()),
            turnover_overridden=pretrade.turnover_overridden,
        )


__all__ = [
    "EnvironmentRiskProjector",
    "EnvironmentRiskRequest",
]
