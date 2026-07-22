"""Local-only user interface runtime for Trade RL research artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trade_rl.studio import telemetry as _telemetry
from trade_rl.studio.strict_telemetry import StrictStudioTelemetryReader

setattr(_telemetry, "StudioTelemetryReader", StrictStudioTelemetryReader)

if TYPE_CHECKING:
    from trade_rl.studio.catalog import StudioCatalog
    from trade_rl.studio.settings import StudioSettings


def __getattr__(name: str) -> Any:
    if name == "StudioCatalog":
        from trade_rl.studio.catalog import StudioCatalog

        return StudioCatalog
    if name == "StudioSettings":
        from trade_rl.studio.settings import StudioSettings

        return StudioSettings
    raise AttributeError(name)


__all__ = ["StudioCatalog", "StudioSettings"]
