"""Compatibility import for the canonical Studio telemetry reader."""

from trade_rl.studio.telemetry import StudioTelemetryReader

StrictStudioTelemetryReader = StudioTelemetryReader

__all__ = ["StrictStudioTelemetryReader"]
