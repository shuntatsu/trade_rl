"""Typed, ML-free identity boundary for existing portfolio training configs.

These helpers retain the complete dataclass payload used by the existing
portfolio transform. They do not replace execution-policy identity or admit
a runtime configuration for execution or training.
"""

from __future__ import annotations

from dataclasses import asdict

from trade_rl.artifacts.hashing import content_digest
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig


def training_execution_config_payload(config: ExecutionCostConfig) -> dict[str, object]:
    """Return a detached full config, matching the transform's legacy payload."""
    if type(config) is not ExecutionCostConfig:
        raise TypeError("expected an ExecutionCostConfig")
    return asdict(config)


def training_execution_config_digest(config: ExecutionCostConfig) -> str:
    """Identify the full training configuration, not the runtime execution policy."""
    return content_digest(training_execution_config_payload(config))


def training_risk_config_payload(config: PreTradeRiskConfig) -> dict[str, object]:
    """Return the complete detached risk config used in the training binding."""
    if type(config) is not PreTradeRiskConfig:
        raise TypeError("expected a PreTradeRiskConfig")
    return asdict(config)


def training_risk_config_digest(config: PreTradeRiskConfig) -> str:
    """Identify the complete training risk configuration."""
    return content_digest(training_risk_config_payload(config))


__all__ = [
    "training_execution_config_digest",
    "training_execution_config_payload",
    "training_risk_config_digest",
    "training_risk_config_payload",
]
