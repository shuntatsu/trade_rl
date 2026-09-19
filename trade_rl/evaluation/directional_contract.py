"""Shared execution economics for directional fitting and evaluation."""

from __future__ import annotations

from dataclasses import replace

from trade_rl.simulation import ExecutionCostConfig

DIRECTIONAL_BASE_EXECUTION_COST = replace(
    ExecutionCostConfig.zero(),
    max_leverage=1.0,
    processing_bar_volume_capacity=False,
    borrow_rate_multiplier=1.0,
)

__all__ = ["DIRECTIONAL_BASE_EXECUTION_COST"]
