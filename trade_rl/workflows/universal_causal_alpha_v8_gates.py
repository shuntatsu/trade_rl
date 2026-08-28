"""V8-owned gate evidence using the unchanged V7 numerical gate rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.workflows.universal_causal_alpha_v7_selection import (
    CausalAlphaV7SelectionEvidence,
    evaluate_causal_alpha_v7_selection,
)
from trade_rl.workflows.universal_causal_alpha_v7_signal import (
    CausalAlphaV7SignalEvidence,
    CausalAlphaV7SignalScopeMetric,
    evaluate_causal_alpha_v7_signal_gate,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)

_SIGNAL_SCHEMA: Final = "causal_alpha_v8_signal_evidence_v1"
_SELECTION_SCHEMA: Final = "causal_alpha_v8_selection_evidence_v1"
_V8_BY_V7: Final = {
    CausalAlphaV7Candidate.V6_CONTROL: CausalAlphaV8Candidate.V7_CONTROL,
    CausalAlphaV7Candidate.SYMMETRIC_CONTRARIAN: (
        CausalAlphaV8Candidate.ROBUST_CONTRARIAN
    ),
    CausalAlphaV7Candidate.CAUSAL_CALIBRATED: (
        CausalAlphaV8Candidate.ROBUST_CALIBRATED
    ),
}


def _candidate_payloads(source: object) -> tuple[dict[str, object], ...]:
    candidates = getattr(source, "candidates")
    result: list[dict[str, object]] = []
    for candidate in candidates:
        payload = candidate.to_payload()
        payload["source_v7_candidate"] = payload["candidate"]
        payload["candidate"] = _V8_BY_V7[candidate.candidate].value
        result.append(payload)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CausalAlphaV8SignalEvidence:
    source_v7: CausalAlphaV7SignalEvidence
    schema_version: str = _SIGNAL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_v7, CausalAlphaV7SignalEvidence):
            raise TypeError("V8 Signal source evidence is invalid")
        if self.schema_version != _SIGNAL_SCHEMA:
            raise ValueError("unsupported V8 Signal schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V8 Signal digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return self.source_v7.passed

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.source_v7.rejection_reasons

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidates": _candidate_payloads(self.source_v7),
            "independent_episode_count": self.source_v7.independent_episode_count,
            "passed": self.passed,
            "promotion_eligible": False,
            "raw_scope_count_per_candidate": (
                self.source_v7.raw_scope_count_per_candidate
            ),
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "source_v7_gate_digest": self.source_v7.digest,
            "symbol_count": self.source_v7.symbol_count,
            "v4_fast_lane_digest": self.source_v7.v4_fast_lane_digest,
            "v4_fast_lane_passed": self.source_v7.v4_fast_lane_passed,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV8SelectionEvidence:
    source_v7: CausalAlphaV7SelectionEvidence
    schema_version: str = _SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_v7, CausalAlphaV7SelectionEvidence):
            raise TypeError("V8 Selection source evidence is invalid")
        if self.schema_version != _SELECTION_SCHEMA:
            raise ValueError("unsupported V8 Selection schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V8 Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return self.source_v7.passed

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.source_v7.rejection_reasons

    @property
    def selected_candidate(self) -> CausalAlphaV8Candidate | None:
        selected = self.source_v7.selected_candidate
        return None if selected is None else _V8_BY_V7[selected]

    @property
    def selected_config_digest(self) -> str | None:
        return self.source_v7.selected_config_digest

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidates": _candidate_payloads(self.source_v7),
            "paired_scope_count": self.source_v7.paired_scope_count,
            "passed": self.passed,
            "promotion_eligible": False,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "selected_candidate": (
                None
                if self.selected_candidate is None
                else self.selected_candidate.value
            ),
            "selected_config_digest": self.selected_config_digest,
            "source_v7_gate_digest": self.source_v7.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v8_signal_gate(
    metrics: tuple[CausalAlphaV7SignalScopeMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
    v4_fast_lane_digest: str,
    v4_fast_lane_passed: bool,
) -> CausalAlphaV8SignalEvidence:
    """Evaluate mapped V8 liveness with the frozen Signal gate implementation."""

    return CausalAlphaV8SignalEvidence(
        evaluate_causal_alpha_v7_signal_gate(
            metrics,
            expected_symbols=expected_symbols,
            v4_fast_lane_digest=v4_fast_lane_digest,
            v4_fast_lane_passed=v4_fast_lane_passed,
        )
    )


def evaluate_causal_alpha_v8_selection(
    metrics: tuple[CausalAlphaV8ReplayMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV8SelectionEvidence:
    """Evaluate V8 replay economics with every unchanged universal gate."""

    return CausalAlphaV8SelectionEvidence(
        evaluate_causal_alpha_v7_selection(
            tuple(metric.as_v7_metric() for metric in metrics),
            expected_symbols=expected_symbols,
        )
    )


__all__ = [
    "CausalAlphaV8SelectionEvidence",
    "CausalAlphaV8SignalEvidence",
    "evaluate_causal_alpha_v8_selection",
    "evaluate_causal_alpha_v8_signal_gate",
]
