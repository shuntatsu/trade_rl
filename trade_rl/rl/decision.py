"""Shared residual decision path used by training, evaluation, and serving."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.risk.guardrails import GuardrailTarget, OperationalGuardrails
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.rl.actions import (
    BaselineResidualComposer,
    ResidualAction,
    ResidualComposition,
)
from trade_rl.strategies.trend import TrendTargets


@dataclass(frozen=True, slots=True)
class DecisionContext:
    current_weights: np.ndarray
    portfolio_value: float
    peak_value: float
    day_start_value: float
    next_tradable: np.ndarray
    data_age_hours: float
    features_available: bool

    def __post_init__(self) -> None:
        current = np.asarray(self.current_weights, dtype=np.float64).reshape(-1).copy()
        tradable = np.asarray(self.next_tradable, dtype=np.bool_).reshape(-1).copy()
        if current.size == 0 or current.shape != tradable.shape:
            raise ValueError("decision state vectors must have identical non-empty shapes")
        if not np.isfinite(current).all():
            raise ValueError("current_weights must be finite")
        for field_name, value in (
            ("portfolio_value", self.portfolio_value),
            ("peak_value", self.peak_value),
            ("day_start_value", self.day_start_value),
            ("data_age_hours", self.data_age_hours),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if min(self.portfolio_value, self.peak_value, self.day_start_value) <= 0.0:
            raise ValueError("portfolio values must be positive")
        if self.peak_value < self.portfolio_value:
            raise ValueError("peak_value must be at least portfolio_value")
        if self.data_age_hours < 0.0:
            raise ValueError("data_age_hours must be non-negative")
        object.__setattr__(self, "current_weights", current)
        object.__setattr__(self, "next_tradable", tradable)

    @property
    def drawdown(self) -> float:
        return 1.0 - self.portfolio_value / self.peak_value


@dataclass(frozen=True, slots=True)
class DecisionResult:
    target_weights: np.ndarray
    composition: ResidualComposition
    guardrail: GuardrailTarget
    risk: RiskConstrainedTarget


class ResidualDecisionEngine:
    """Compose, fail closed, mask tradability, and apply portfolio risk once."""

    def __init__(
        self,
        *,
        composer: BaselineResidualComposer | None = None,
        guardrails: OperationalGuardrails | None = None,
        pre_trade_risk: PreTradeRisk | None = None,
    ) -> None:
        self.composer = composer or BaselineResidualComposer()
        self.guardrails = guardrails or OperationalGuardrails()
        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()

    def decide(
        self,
        *,
        action: ResidualAction,
        trends: TrendTargets,
        alpha: np.ndarray,
        alpha_enabled: bool,
        context: DecisionContext,
    ) -> DecisionResult:
        composition = self.composer.compose(
            action,
            trends,
            alpha,
            alpha_enabled=alpha_enabled,
        )
        guardrail = self.guardrails.apply(
            composition.proposal,
            portfolio_value=context.portfolio_value,
            day_start_value=context.day_start_value,
            data_age_hours=context.data_age_hours,
            features_available=context.features_available,
        )
        proposal = guardrail.weights.copy()
        proposal[~context.next_tradable] = context.current_weights[
            ~context.next_tradable
        ]
        risk = self.pre_trade_risk.constrain(
            proposal,
            current=context.current_weights,
            drawdown=context.drawdown,
        )
        return DecisionResult(
            target_weights=risk.weights,
            composition=composition,
            guardrail=guardrail,
            risk=risk,
        )
