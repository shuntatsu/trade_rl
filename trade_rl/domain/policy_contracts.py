"""Maintained policy and structured-observation identifiers."""

from __future__ import annotations

SB3_POLICY_IDENTITY_SCHEMA = "sb3_policy_identity_v4"
HIERARCHICAL_SEQUENCE_ENCODER = "hierarchical_sequence_v2"
STRUCTURED_TIMEFRAMES = ("15m", "1h", "4h", "1d")

__all__ = [
    "HIERARCHICAL_SEQUENCE_ENCODER",
    "SB3_POLICY_IDENTITY_SCHEMA",
    "STRUCTURED_TIMEFRAMES",
]
