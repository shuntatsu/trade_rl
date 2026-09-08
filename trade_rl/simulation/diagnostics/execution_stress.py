"""Deterministic execution-environment overlays for sealed sensitivity runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionRuleStress


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be a finite number")
    return resolved


@dataclass(frozen=True, slots=True)
class ExecutionEnvironmentStress(ExecutionRuleStress):
    """Rule, cost, liquidity, latency, and tail-risk evaluation assumptions."""

    fee_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    impact_multiplier: float = 1.0
    slippage_std_multiplier: float = 1.0
    slippage_std_floor: float = 0.0
    participation_fraction: float = 1.0
    minimum_order_latency_bars: int = 0
    tail_slippage_probability_floor: float = 0.0
    tail_slippage_multiplier_floor: float = 0.0
    borrow_rate_multiplier: float = 1.0

    def __post_init__(self) -> None:
        ExecutionRuleStress.__post_init__(self)
        for field_name in (
            "fee_multiplier",
            "spread_multiplier",
            "impact_multiplier",
            "slippage_std_multiplier",
            "borrow_rate_multiplier",
        ):
            value = _finite_number(getattr(self, field_name), field=field_name)
            if value < 1.0:
                raise ValueError(f"{field_name} must be at least one")
            object.__setattr__(self, field_name, value)
        slippage_floor = _finite_number(
            self.slippage_std_floor,
            field="slippage_std_floor",
        )
        if slippage_floor < 0.0:
            raise ValueError("slippage_std_floor must be non-negative")
        object.__setattr__(self, "slippage_std_floor", slippage_floor)
        participation = _finite_number(
            self.participation_fraction,
            field="participation_fraction",
        )
        if not 0.0 < participation <= 1.0:
            raise ValueError("participation_fraction must be within (0, 1]")
        object.__setattr__(self, "participation_fraction", participation)
        latency = self.minimum_order_latency_bars
        if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
            raise ValueError(
                "minimum_order_latency_bars must be a non-negative integer"
            )
        probability = _finite_number(
            self.tail_slippage_probability_floor,
            field="tail_slippage_probability_floor",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("tail_slippage_probability_floor must be within [0, 1]")
        object.__setattr__(self, "tail_slippage_probability_floor", probability)
        multiplier_floor = _finite_number(
            self.tail_slippage_multiplier_floor,
            field="tail_slippage_multiplier_floor",
        )
        if multiplier_floor != 0.0 and multiplier_floor < 1.0:
            raise ValueError(
                "tail_slippage_multiplier_floor must be zero or at least one"
            )
        if probability > 0.0 and multiplier_floor < 1.0:
            raise ValueError(
                "tail_slippage_multiplier_floor must be at least one when the "
                "tail probability floor is positive"
            )
        object.__setattr__(
            self,
            "tail_slippage_multiplier_floor",
            multiplier_floor,
        )

    @property
    def environment_enabled(self) -> bool:
        return (
            self.fee_multiplier > 1.0
            or self.spread_multiplier > 1.0
            or self.impact_multiplier > 1.0
            or self.slippage_std_multiplier > 1.0
            or self.slippage_std_floor > 0.0
            or self.participation_fraction < 1.0
            or self.minimum_order_latency_bars > 0
            or self.tail_slippage_probability_floor > 0.0
            or self.tail_slippage_multiplier_floor > 0.0
            or self.borrow_rate_multiplier > 1.0
        )

    @property
    def enabled(self) -> bool:
        return (
            self.tick_size_factor != 1.0
            or self.lot_size_factor != 1.0
            or self.minimum_notional_factor != 1.0
            or self.adverse_tick_rounding
            or self.environment_enabled
        )

    def apply(self, base: ExecutionCostConfig) -> ExecutionCostConfig:
        """Return a stressed immutable cost configuration without mutating base."""

        if not isinstance(base, ExecutionCostConfig):
            raise TypeError("base must be an ExecutionCostConfig")
        if not self.environment_enabled:
            return base
        tail_multiplier = base.tail_slippage_multiplier
        if self.tail_slippage_multiplier_floor > 0.0:
            tail_multiplier = max(
                tail_multiplier,
                self.tail_slippage_multiplier_floor,
            )
        return replace(
            base,
            fee_rate=base.fee_rate * self.fee_multiplier,
            maker_fee_rate=base.maker_fee_rate * self.fee_multiplier,
            taker_fee_rate=base.taker_fee_rate * self.fee_multiplier,
            spread_rate=base.spread_rate * self.spread_multiplier,
            impact_rate=base.impact_rate * self.impact_multiplier,
            slippage_std=max(
                base.slippage_std * self.slippage_std_multiplier,
                self.slippage_std_floor,
            ),
            max_participation_rate=(
                base.max_participation_rate * self.participation_fraction
            ),
            order_latency_bars=max(
                base.order_latency_bars,
                self.minimum_order_latency_bars,
            ),
            tail_slippage_probability=max(
                base.tail_slippage_probability,
                self.tail_slippage_probability_floor,
            ),
            tail_slippage_multiplier=tail_multiplier,
            borrow_rate_multiplier=(
                base.borrow_rate_multiplier * self.borrow_rate_multiplier
            ),
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "adverse_tick_rounding": self.adverse_tick_rounding,
            "borrow_rate_multiplier": self.borrow_rate_multiplier,
            "fee_multiplier": self.fee_multiplier,
            "impact_multiplier": self.impact_multiplier,
            "lot_size_factor": self.lot_size_factor,
            "minimum_notional_factor": self.minimum_notional_factor,
            "minimum_order_latency_bars": self.minimum_order_latency_bars,
            "name": self.name,
            "participation_fraction": self.participation_fraction,
            "schema_version": "execution_environment_stress_v1",
            "slippage_std_floor": self.slippage_std_floor,
            "slippage_std_multiplier": self.slippage_std_multiplier,
            "spread_multiplier": self.spread_multiplier,
            "tail_slippage_multiplier_floor": (self.tail_slippage_multiplier_floor),
            "tail_slippage_probability_floor": (self.tail_slippage_probability_floor),
            "tick_size_factor": self.tick_size_factor,
        }


def apply_execution_environment_stress(
    base: ExecutionCostConfig,
    stress: ExecutionRuleStress | None,
) -> ExecutionCostConfig:
    """Apply extended stress while preserving every legacy rule-stress subtype."""

    if not isinstance(base, ExecutionCostConfig):
        raise TypeError("base must be an ExecutionCostConfig")
    if isinstance(stress, ExecutionEnvironmentStress):
        return stress.apply(base)
    if stress is None or isinstance(stress, ExecutionRuleStress):
        return base
    raise TypeError("stress must be an ExecutionRuleStress or null")


__all__ = [
    "ExecutionEnvironmentStress",
    "apply_execution_environment_stress",
]
