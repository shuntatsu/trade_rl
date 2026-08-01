"""Compatibility exports for configuration field validation."""

from trade_rl.domain.config_fields import (
    require_dataclass_fields,
    require_exact_fields,
)

__all__ = ["require_dataclass_fields", "require_exact_fields"]
