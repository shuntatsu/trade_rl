"""Canonical serialized policy identity vocabulary."""

from __future__ import annotations

from typing import Final

SB3_POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v4"
HIERARCHICAL_SEQUENCE_ENCODER: Final = "hierarchical_sequence_v2"
STRUCTURED_TIMEFRAMES: Final = ("15m", "1h", "4h", "1d")

__all__ = [
    "HIERARCHICAL_SEQUENCE_ENCODER",
    "SB3_POLICY_IDENTITY_SCHEMA",
    "STRUCTURED_TIMEFRAMES",
]
