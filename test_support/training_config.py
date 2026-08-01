from __future__ import annotations

from typing import Any

_EXECUTION_CONFIG_V4: dict[str, object] = {
    "fee_rate": 0.0005,
    "maker_fee_rate": 0.0,
    "taker_fee_rate": 0.0,
    "spread_rate": 0.0002,
    "impact_rate": 0.0001,
    "multiplier": 1.0,
    "max_participation_rate": 0.05,
    "slippage_std": 0.0,
    "tail_slippage_probability": 0.0,
    "tail_slippage_multiplier": 5.0,
    "random_seed": 0,
    "minimum_notional": 0.0,
    "lot_size": 0.0,
    "tick_size": 0.0,
    "allow_short": True,
    "borrow_rate_multiplier": 1.0,
    "max_leverage": 1.0,
    "maintenance_margin_rate": 0.25,
    "collateral_haircut": 1.0,
    "margin_mode": "cross",
    "order_latency_bars": 0,
    "order_type": "market",
    "limit_offset_rate": 0.0005,
    "path_mode": "conservative",
    "processing_bar_volume_capacity": True,
    "partial_fill_carry": True,
    "trigger_volume_fractions": [1.0, 0.5, 0.25, 0.0],
}


def complete_execution_config(**overrides: Any) -> dict[str, object]:
    config = dict(_EXECUTION_CONFIG_V4)
    config.update(overrides)
    return config


__all__ = ["complete_execution_config"]
