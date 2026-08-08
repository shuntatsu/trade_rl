"""Fail-closed promotion gates for execution runtime authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum

from trade_rl.artifacts.hashing import content_digest

EXECUTION_PROMOTION_REPORT_SCHEMA = "execution_promotion_report_v1"


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


@dataclass(frozen=True, slots=True)
class ExecutionPromotionReport:
    """Immutable report which records a promotion decision without applying it."""

    digest: str
    requested_mode: RuntimeMode
    evidence: ExecutionPromotionEvidence
    decision: RuntimePromotionDecision
    schema_version: str = EXECUTION_PROMOTION_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_PROMOTION_REPORT_SCHEMA:
            raise ValueError("unsupported execution promotion report schema")
        if self.decision.requested is not self.requested_mode:
            raise ValueError("execution promotion report requested mode mismatch")
        expected_decision = assess_runtime_promotion(
            requested=self.requested_mode,
            evidence=self.evidence,
        )
        if self.decision != expected_decision:
            raise ValueError(
                "execution promotion report decision does not match evidence"
            )
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("execution promotion report digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        """Return canonical decision evidence excluding the self-referential digest."""

        return _promotion_report_payload(
            requested=self.requested_mode,
            evidence=self.evidence,
            decision=self.decision,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the immutable report payload suitable for persisted evidence."""

        return {"digest": self.digest, **self.digest_payload()}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ExecutionPromotionReport:
        """Restore persisted promotion evidence and validate it fail-closed."""

        try:
            requested = RuntimeMode(_require_string(raw, "requested_mode"))
            evidence_raw = raw["evidence"]
            if not isinstance(evidence_raw, Mapping):
                raise ValueError("promotion evidence must be an object")
            evidence = ExecutionPromotionEvidence(
                **{
                    field.name: _require_bool(evidence_raw, field.name)
                    for field in fields(ExecutionPromotionEvidence)
                }
            )
            missing_raw = raw["missing"]
            if not isinstance(missing_raw, (list, tuple)) or any(
                not isinstance(item, str) for item in missing_raw
            ):
                raise ValueError("promotion report missing fields must be strings")
            decision = RuntimePromotionDecision(
                requested=requested,
                allowed=_require_bool(raw, "allowed"),
                missing=tuple(missing_raw),
            )
            return cls(
                digest=_require_string(raw, "digest"),
                requested_mode=requested,
                evidence=evidence,
                decision=decision,
                schema_version=_require_string(raw, "schema_version"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("promotion report is invalid") from error


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


def _promotion_report_payload(
    *,
    requested: RuntimeMode,
    evidence: ExecutionPromotionEvidence,
    decision: RuntimePromotionDecision,
) -> dict[str, object]:
    return {
        "allowed": decision.allowed,
        "evidence": {
            field.name: getattr(evidence, field.name)
            for field in fields(ExecutionPromotionEvidence)
        },
        "missing": decision.missing,
        "requested_mode": requested.value,
        "schema_version": EXECUTION_PROMOTION_REPORT_SCHEMA,
    }


def _require_string(raw: Mapping[str, object], name: str) -> str:
    value = raw[name]
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_bool(raw: Mapping[str, object], name: str) -> bool:
    value = raw[name]
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def build_execution_promotion_report(
    *,
    requested: RuntimeMode,
    evidence: ExecutionPromotionEvidence,
) -> ExecutionPromotionReport:
    """Build signed-selection-ready evidence without changing runtime authority."""

    decision = assess_runtime_promotion(requested=requested, evidence=evidence)
    payload = _promotion_report_payload(
        requested=requested,
        evidence=evidence,
        decision=decision,
    )
    return ExecutionPromotionReport(
        digest=content_digest(payload),
        requested_mode=requested,
        evidence=evidence,
        decision=decision,
    )


__all__ = [
    "EXECUTION_PROMOTION_REPORT_SCHEMA",
    "ExecutionPromotionEvidence",
    "ExecutionPromotionReport",
    "RuntimeMode",
    "RuntimePromotionDecision",
    "assess_runtime_promotion",
    "build_execution_promotion_report",
]
