"""Optional portfolio-level risk constraints applied before market execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest

_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PortfolioRiskConfig:
    max_abs_weight: float | None = None
    max_net_exposure: float | None = None
    max_position_to_market_notional: float | None = None
    volatility_target: float | None = None
    max_abs_beta: float | None = None
    max_stress_loss: float | None = None
    schema_version: str = "portfolio_risk_v1"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_abs_weight", self.max_abs_weight),
            ("max_net_exposure", self.max_net_exposure),
            ("max_position_to_market_notional", self.max_position_to_market_notional),
            ("volatility_target", self.volatility_target),
            ("max_abs_beta", self.max_abs_beta),
            ("max_stress_loss", self.max_stress_loss),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.schema_version != "portfolio_risk_v1":
            raise ValueError("unsupported portfolio risk schema")


@dataclass(frozen=True, slots=True)
class PortfolioRiskResult:
    weights: np.ndarray
    was_constrained: bool
    reasons: tuple[str, ...]
    portfolio_volatility: float
    portfolio_beta: float
    stress_loss: float


class PortfolioRiskModel:
    """Apply deterministic concentration, liquidity and scenario constraints."""

    implementation_digest = content_digest({"schema_version": "portfolio-risk-model-v1"})

    def __init__(self, config: PortfolioRiskConfig | None = None) -> None:
        self.config = config or PortfolioRiskConfig()

    @staticmethod
    def _scale_to_limit(
        weights: np.ndarray,
        value: float,
        limit: float | None,
        reason: str,
        reasons: list[str],
    ) -> np.ndarray:
        if limit is None or value <= limit + _TOLERANCE:
            return weights
        if value <= 0.0:
            return weights
        reasons.append(reason)
        return weights * (limit / value)

    def constrain(
        self,
        target: np.ndarray,
        *,
        portfolio_value: float,
        market_notional: np.ndarray,
        covariance: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        stress_losses: np.ndarray | None = None,
    ) -> PortfolioRiskResult:
        weights = np.asarray(target, dtype=np.float64).reshape(-1).copy()
        liquidity = np.asarray(market_notional, dtype=np.float64).reshape(-1)
        if weights.size == 0 or weights.shape != liquidity.shape:
            raise ValueError("target and market_notional must have the same non-empty shape")
        if not np.isfinite(weights).all() or not np.isfinite(liquidity).all():
            raise ValueError("portfolio risk inputs must be finite")
        if not math.isfinite(portfolio_value) or portfolio_value <= 0.0:
            raise ValueError("portfolio_value must be finite and positive")
        if np.any(liquidity < 0.0):
            raise ValueError("market_notional must be non-negative")

        reasons: list[str] = []
        if self.config.max_abs_weight is not None:
            clipped = np.clip(
                weights,
                -self.config.max_abs_weight,
                self.config.max_abs_weight,
            )
            if not np.array_equal(clipped, weights):
                weights = clipped
                reasons.append("max_abs_weight")

        if self.config.max_position_to_market_notional is not None:
            caps = (
                liquidity
                * self.config.max_position_to_market_notional
                / portfolio_value
            )
            clipped = np.clip(weights, -caps, caps)
            if not np.allclose(clipped, weights, atol=0.0, rtol=0.0):
                weights = clipped
                reasons.append("liquidity_cap")

        net = abs(float(weights.sum()))
        weights = self._scale_to_limit(
            weights,
            net,
            self.config.max_net_exposure,
            "max_net_exposure",
            reasons,
        )

        volatility = 0.0
        if covariance is not None:
            matrix = np.asarray(covariance, dtype=np.float64)
            if matrix.shape != (weights.size, weights.size) or not np.isfinite(matrix).all():
                raise ValueError("covariance must be a finite square matrix")
            variance = max(float(weights @ matrix @ weights), 0.0)
            volatility = math.sqrt(variance)
            weights = self._scale_to_limit(
                weights,
                volatility,
                self.config.volatility_target,
                "volatility_target",
                reasons,
            )
            volatility = math.sqrt(max(float(weights @ matrix @ weights), 0.0))
        elif self.config.volatility_target is not None:
            raise ValueError("volatility_target requires covariance")

        portfolio_beta = 0.0
        if beta is not None:
            beta_vector = np.asarray(beta, dtype=np.float64).reshape(-1)
            if beta_vector.shape != weights.shape or not np.isfinite(beta_vector).all():
                raise ValueError("beta must be a finite vector matching target")
            portfolio_beta = float(weights @ beta_vector)
            weights = self._scale_to_limit(
                weights,
                abs(portfolio_beta),
                self.config.max_abs_beta,
                "max_abs_beta",
                reasons,
            )
            portfolio_beta = float(weights @ beta_vector)
        elif self.config.max_abs_beta is not None:
            raise ValueError("max_abs_beta requires beta")

        stress_loss = 0.0
        if stress_losses is not None:
            stress_vector = np.asarray(stress_losses, dtype=np.float64).reshape(-1)
            if stress_vector.shape != weights.shape or not np.isfinite(stress_vector).all():
                raise ValueError("stress_losses must be a finite vector matching target")
            stress_loss = abs(float(weights @ stress_vector))
            weights = self._scale_to_limit(
                weights,
                stress_loss,
                self.config.max_stress_loss,
                "max_stress_loss",
                reasons,
            )
            stress_loss = abs(float(weights @ stress_vector))
        elif self.config.max_stress_loss is not None:
            raise ValueError("max_stress_loss requires stress_losses")

        if not np.isfinite(weights).all():
            raise RuntimeError("portfolio risk projection produced non-finite weights")
        weights.setflags(write=False)
        return PortfolioRiskResult(
            weights=weights,
            was_constrained=bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
            portfolio_volatility=volatility,
            portfolio_beta=portfolio_beta,
            stress_loss=stress_loss,
        )


__all__ = ["PortfolioRiskConfig", "PortfolioRiskModel", "PortfolioRiskResult"]
