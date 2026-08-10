"""Immutable evaluation-only overlays for adverse execution environments."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionRuleStress


@dataclass(frozen=True, slots=True)
class ExecutionEnvironmentStress(ExecutionRuleStress):
    """Rule stress plus deterministic adverse execution-cost assumptions."""

    fee_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    impact_multiplier: float = 1.0
    slippage_std_multiplier: float = 1.0
    participation_fraction: float = 1.0
    minimum_order_latency_bars: int = 0
    tail_slippage_probability_floor: float = 0.0
    tail_slippage_multiplier_floor: float = 0.0
    borrow_rate_multiplier: float = 1.0

    def __post_init__(self) -> None:
        ExecutionRuleStress.__post_init__(self)
        for field_name, value in (
            ("fee_multiplier", self.fee_multiplier),
            ("spread_multiplier", self.spread_multiplier),
            ("impact_multiplier", self.impact_multiplier),
            ("slippage_std_multiplier", self.slippage_std_multiplier),
            ("borrow_rate_multiplier", self.borrow_rate_multiplier),
        ):
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{field_name} must be finite and at least 1.0")
        if (
            not math.isfinite(self.participation_fraction)
            or not 0.0 < self.participation_fraction <= 1.0
        ):
            raise ValueError("participation_fraction must be finite and within (0, 1]")
        if (
            isinstance(self.minimum_order_latency_bars, bool)
            or not isinstance(self.minimum_order_latency_bars, int)
            or self.minimum_order_latency_bars < 0
        ):
            raise ValueError(
                "minimum_order_latency_bars must be a non-negative integer"
            )
        if (
            not math.isfinite(self.tail_slippage_probability_floor)
            or not 0.0 <= self.tail_slippage_probability_floor <= 1.0
        ):
            raise ValueError("tail_slippage_probability_floor must be within [0, 1]")
        if (
            not math.isfinite(self.tail_slippage_multiplier_floor)
            or self.tail_slippage_multiplier_floor < 0.0
        ):
            raise ValueError(
                "tail_slippage_multiplier_floor must be finite and non-negative"
            )

    @property
    def environment_enabled(self) -> bool:
        return (
            self.fee_multiplier > 1.0
            or self.spread_multiplier > 1.0
            or self.impact_multiplier > 1.0
            or self.slippage_std_multiplier > 1.0
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
        """Return one validated stressed cost while leaving ``base`` immutable."""

        if not self.environment_enabled:
            return base
        tail_probability = max(
            base.tail_slippage_probability,
            self.tail_slippage_probability_floor,
        )
        tail_multiplier = max(
            base.tail_slippage_multiplier,
            self.tail_slippage_multiplier_floor,
            1.0 if tail_probability > 0.0 else 0.0,
        )
        return replace(
            base,
            fee_rate=base.fee_rate * self.fee_multiplier,
            maker_fee_rate=base.maker_fee_rate * self.fee_multiplier,
            taker_fee_rate=base.taker_fee_rate * self.fee_multiplier,
            spread_rate=base.spread_rate * self.spread_multiplier,
            impact_rate=base.impact_rate * self.impact_multiplier,
            slippage_std=(base.slippage_std * self.slippage_std_multiplier),
            max_participation_rate=(
                base.max_participation_rate * self.participation_fraction
            ),
            order_latency_bars=max(
                base.order_latency_bars,
                self.minimum_order_latency_bars,
            ),
            tail_slippage_probability=tail_probability,
            tail_slippage_multiplier=tail_multiplier,
            borrow_rate_multiplier=(
                base.borrow_rate_multiplier * self.borrow_rate_multiplier
            ),
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            **ExecutionRuleStress.digest_payload(self),
            "borrow_rate_multiplier": self.borrow_rate_multiplier,
            "fee_multiplier": self.fee_multiplier,
            "impact_multiplier": self.impact_multiplier,
            "minimum_order_latency_bars": self.minimum_order_latency_bars,
            "participation_fraction": self.participation_fraction,
            "schema_version": "execution_environment_stress_v1",
            "slippage_std_multiplier": self.slippage_std_multiplier,
            "spread_multiplier": self.spread_multiplier,
            "tail_slippage_multiplier_floor": (self.tail_slippage_multiplier_floor),
            "tail_slippage_probability_floor": (self.tail_slippage_probability_floor),
        }


__all__ = ["ExecutionEnvironmentStress"]
