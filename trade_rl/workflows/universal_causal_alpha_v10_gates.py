"""V10-owned evidence over unchanged universal numerical gates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Candidate
from trade_rl.workflows.universal_causal_alpha_v8_gates import (
    CausalAlphaV8SelectionEvidence,
    evaluate_causal_alpha_v8_selection,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v9_gates import (
    CausalAlphaV9SignalEvidence,
)

_SIGNAL_SCHEMA: Final = "causal_alpha_v10_signal_evidence_v1"
_SELECTION_SCHEMA: Final = "causal_alpha_v10_selection_evidence_v1"
V8_CANDIDATE_BY_V10: Final = {
    CausalAlphaV10Candidate.V8_ROBUST_CONTROL: CausalAlphaV8Candidate.V7_CONTROL,
    CausalAlphaV10Candidate.V9_NONLINEAR_CONTROL: (
        CausalAlphaV8Candidate.ROBUST_CONTRARIAN
    ),
    CausalAlphaV10Candidate.HIERARCHICAL_WAVE: (
        CausalAlphaV8Candidate.ROBUST_CALIBRATED
    ),
}
V10_CANDIDATE_BY_V8: Final = {value: key for key, value in V8_CANDIDATE_BY_V10.items()}


@dataclass(frozen=True, slots=True)
class CausalAlphaV10SignalEvidence:
    source_v9: CausalAlphaV9SignalEvidence
    slow_scope_count: int
    qualified_slow_scope_count: int
    dual_fit_digests: tuple[str, ...]
    schema_version: str = _SIGNAL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_v9, CausalAlphaV9SignalEvidence):
            raise TypeError("V10 Signal source is invalid")
        if self.slow_scope_count != 72 or self.qualified_slow_scope_count != 72:
            raise ValueError("V10 Signal requires 72 qualified slow scopes")
        if len(self.dual_fit_digests) != 8 or len(set(self.dual_fit_digests)) != 8:
            raise ValueError("V10 Signal dual fit identities are invalid")
        if not self.source_v9.passed:
            raise ValueError("V10 Signal cannot bypass V9 Signal")
        if self.schema_version != _SIGNAL_SCHEMA:
            raise ValueError("unsupported V10 Signal schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V10 Signal digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return True

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return ()

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "dual_fit_digests": self.dual_fit_digests,
            "passed": self.passed,
            "promotion_eligible": False,
            "qualified_slow_scope_count": self.qualified_slow_scope_count,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "slow_scope_count": self.slow_scope_count,
            "source_v9_signal_digest": self.source_v9.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _candidate_payloads(
    source: CausalAlphaV8SelectionEvidence,
) -> tuple[dict[str, object], ...]:
    mapped: list[dict[str, object]] = []
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
        payload["candidate"] = V10_CANDIDATE_BY_V8[source_v8].value
        mapped.append(payload)
    return tuple(mapped)


@dataclass(frozen=True, slots=True)
class CausalAlphaV10SelectionEvidence:
    source_v8: CausalAlphaV8SelectionEvidence
    schema_version: str = _SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_v8, CausalAlphaV8SelectionEvidence):
            raise TypeError("V10 Selection source is invalid")
        if self.schema_version != _SELECTION_SCHEMA:
            raise ValueError("unsupported V10 Selection schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V10 Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return self.source_v8.passed

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.source_v8.rejection_reasons

    @property
    def selected_candidate(self) -> CausalAlphaV10Candidate | None:
        selected = self.source_v8.selected_candidate
        return None if selected is None else V10_CANDIDATE_BY_V8[selected]

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
                None if self.selected_candidate is None else self.selected_candidate.value
            ),
            "selected_config_digest": self.selected_config_digest,
            "source_v8_selection_digest": self.source_v8.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v10_selection(
    metrics: tuple[CausalAlphaV8ReplayMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV10SelectionEvidence:
    # V10 candidates intentionally use different fast/slow model fits. Pairing
    # must therefore use the common V6 calibration fit carried by the replay
    # economics, while retaining each candidate's model identity in its target
    # artifact and replay digest.
    paired_metrics = tuple(
        metric
        if metric.calibration_fit_digest == metric.v6_metric.fit_digest
        else replace(
            metric,
            calibration_fit_digest=metric.v6_metric.fit_digest,
            digest="",
        )
        for metric in metrics
    )
    return CausalAlphaV10SelectionEvidence(
        evaluate_causal_alpha_v8_selection(
            paired_metrics,
            expected_symbols=expected_symbols,
        )
    )


__all__ = [
    "CausalAlphaV10SelectionEvidence",
    "CausalAlphaV10SignalEvidence",
    "V8_CANDIDATE_BY_V10",
    "V10_CANDIDATE_BY_V8",
    "evaluate_causal_alpha_v10_selection",
]
