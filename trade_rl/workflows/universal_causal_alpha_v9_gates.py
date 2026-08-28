"""V9-owned evidence over unchanged universal numerical gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Candidate
from trade_rl.workflows.universal_causal_alpha_v8_gates import (
    CausalAlphaV8SelectionEvidence,
    CausalAlphaV8SignalEvidence,
    evaluate_causal_alpha_v8_selection,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)

_SIGNAL_SCHEMA: Final = "causal_alpha_v9_signal_evidence_v1"
_SELECTION_SCHEMA: Final = "causal_alpha_v9_selection_evidence_v1"
V8_CANDIDATE_BY_V9: Final = {
    CausalAlphaV9Candidate.V7_CONTROL: CausalAlphaV8Candidate.V7_CONTROL,
    CausalAlphaV9Candidate.V8_ROBUST_CONTROL: (
        CausalAlphaV8Candidate.ROBUST_CONTRARIAN
    ),
    CausalAlphaV9Candidate.NONLINEAR_WAVE: CausalAlphaV8Candidate.ROBUST_CALIBRATED,
}
V9_CANDIDATE_BY_V8: Final = {value: key for key, value in V8_CANDIDATE_BY_V9.items()}


@dataclass(frozen=True, slots=True)
class CausalAlphaV9SignalEvidence:
    source_v8: CausalAlphaV8SignalEvidence
    wave_scope_count: int
    qualified_wave_scope_count: int
    wave_fit_digests: tuple[str, ...]
    schema_version: str = _SIGNAL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_v8, CausalAlphaV8SignalEvidence):
            raise TypeError("V9 Signal source is invalid")
        if self.wave_scope_count != 72 or self.qualified_wave_scope_count != 72:
            raise ValueError("V9 Signal requires 72 qualified wave scopes")
        if len(self.wave_fit_digests) != 8 or len(set(self.wave_fit_digests)) != 8:
            raise ValueError("V9 Signal wave fit identities are invalid")
        if not self.source_v8.passed:
            raise ValueError("V9 Signal cannot bypass source Signal")
        if self.schema_version != _SIGNAL_SCHEMA:
            raise ValueError("unsupported V9 Signal schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V9 Signal digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return True

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return ()

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "passed": self.passed,
            "promotion_eligible": False,
            "qualified_wave_scope_count": self.qualified_wave_scope_count,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "source_v8_signal_digest": self.source_v8.digest,
            "wave_fit_digests": self.wave_fit_digests,
            "wave_scope_count": self.wave_scope_count,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _candidate_payloads(
    source: CausalAlphaV8SelectionEvidence,
) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for candidate in source.source_v7.candidates:
        source_v8 = CausalAlphaV8Candidate(
            {
                "v6_control": "v7_control",
                "symmetric_contrarian": "robust_contrarian",
                "causal_calibrated": "robust_calibrated",
            }[candidate.candidate.value]
        )
        payload = candidate.to_payload()
        payload["source_gate_candidate"] = payload["candidate"]
        payload["candidate"] = V9_CANDIDATE_BY_V8[source_v8].value
        result.append(payload)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CausalAlphaV9SelectionEvidence:
    source_v8: CausalAlphaV8SelectionEvidence
    schema_version: str = _SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_v8, CausalAlphaV8SelectionEvidence):
            raise TypeError("V9 Selection source is invalid")
        if self.schema_version != _SELECTION_SCHEMA:
            raise ValueError("unsupported V9 Selection schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V9 Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return self.source_v8.passed

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.source_v8.rejection_reasons

    @property
    def selected_candidate(self) -> CausalAlphaV9Candidate | None:
        selected = self.source_v8.selected_candidate
        return None if selected is None else V9_CANDIDATE_BY_V8[selected]

    @property
    def selected_config_digest(self) -> str | None:
        return self.source_v8.selected_config_digest

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidates": _candidate_payloads(self.source_v8),
            "paired_scope_count": self.source_v8.source_v7.paired_scope_count,
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
            "source_v8_selection_digest": self.source_v8.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v9_selection(
    metrics: tuple[CausalAlphaV8ReplayMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV9SelectionEvidence:
    return CausalAlphaV9SelectionEvidence(
        evaluate_causal_alpha_v8_selection(
            metrics,
            expected_symbols=expected_symbols,
        )
    )


__all__ = [
    "CausalAlphaV9SelectionEvidence",
    "CausalAlphaV9SignalEvidence",
    "V8_CANDIDATE_BY_V9",
    "V9_CANDIDATE_BY_V8",
    "evaluate_causal_alpha_v9_selection",
]
