"""Fail-closed promotion gates for execution runtime authority."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class RuntimeMode(str, Enum):
    """Authority level for maintained execution evaluation."""

    LEGACY_AUTHORITATIVE = "legacy_authoritative"
    DUAL_SHADOW = "dual_shadow"
    NAUTILUS_AUTHORITATIVE = "nautilus_authoritative"


@dataclass(frozen=True, slots=True)
class ExecutionPromotionEvidence:
    """Independent evidence required before execution authority may advance."""

    capability_passed: bool
    causal_bridge_passed: bool
    funding_passed: bool
    terminal_flat_passed: bool
    exact_parity_passed: bool
    determinism_passed: bool
    performance_approved: bool


@dataclass(frozen=True, slots=True)
class RuntimePromotionDecision:
    requested: RuntimeMode
    allowed: bool
    missing: tuple[str, ...]


_DUAL_SHADOW_REQUIRED = (
    "capability_passed",
    "causal_bridge_passed",
    "funding_passed",
    "terminal_flat_passed",
)
_NAUTILUS_AUTHORITATIVE_REQUIRED = tuple(
    field.name for field in fields(ExecutionPromotionEvidence)
)


def assess_runtime_promotion(
    *,
    requested: RuntimeMode,
    evidence: ExecutionPromotionEvidence,
) -> RuntimePromotionDecision:
    """Return a deterministic promotion decision without changing runtime state."""

    if requested is RuntimeMode.LEGACY_AUTHORITATIVE:
        required: tuple[str, ...] = ()
    elif requested is RuntimeMode.DUAL_SHADOW:
        required = _DUAL_SHADOW_REQUIRED
    elif requested is RuntimeMode.NAUTILUS_AUTHORITATIVE:
        required = _NAUTILUS_AUTHORITATIVE_REQUIRED
    else:  # pragma: no cover - Enum exhaustiveness guard
        raise ValueError(f"unsupported runtime mode: {requested!r}")

    missing = tuple(name for name in required if not getattr(evidence, name))
    return RuntimePromotionDecision(
        requested=requested,
        allowed=not missing,
        missing=missing,
    )


__all__ = [
    "ExecutionPromotionEvidence",
    "RuntimeMode",
    "RuntimePromotionDecision",
    "assess_runtime_promotion",
]
