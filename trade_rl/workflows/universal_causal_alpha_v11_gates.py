"""Independent V11 study-arm evidence over unchanged universal gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.causal_alpha_v11 import (
    CausalAlphaV11Candidate,
    CausalAlphaV11SizingFeasibility,
    CausalAlphaV11StudyArm,
)
from trade_rl.workflows.universal_causal_alpha_v8_gates import (
    CausalAlphaV8SelectionEvidence,
    evaluate_causal_alpha_v8_selection,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)

CAUSAL_ALPHA_V11_SELECTION_SCHEMA: Final = "causal_alpha_v11_selection_evidence_v1"
_V11_BY_V8: Final = {
    CausalAlphaV8Candidate.V7_CONTROL: CausalAlphaV11Candidate.V8_CASH_SANITY,
    CausalAlphaV8Candidate.ROBUST_CONTRARIAN: CausalAlphaV11Candidate.V9_CONTROL,
    CausalAlphaV8Candidate.ROBUST_CALIBRATED: CausalAlphaV11Candidate.TREATMENT,
}


def _candidate_payloads(
    source: CausalAlphaV8SelectionEvidence | None,
) -> tuple[dict[str, object], ...]:
    if source is None:
        return tuple(
            {"candidate": candidate.value, "status": "not_replayed"}
            for candidate in CausalAlphaV11Candidate
        )
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
        payload["candidate"] = _V11_BY_V8[source_v8].value
        result.append(payload)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CausalAlphaV11SelectionEvidence:
    """Terminal evidence for exactly one pre-registered V11 treatment arm."""

    study_arm: CausalAlphaV11StudyArm
    study_arm_digest: str
    v11_config_digest: str
    diagnostic_digests: tuple[str, ...]
    source_v8: CausalAlphaV8SelectionEvidence | None
    sizing_feasibility: CausalAlphaV11SizingFeasibility | None = None
    schema_version: str = CAUSAL_ALPHA_V11_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        arm = CausalAlphaV11StudyArm(self.study_arm)
        require_sha256(self.study_arm_digest, field="V11 study_arm_digest")
        require_sha256(self.v11_config_digest, field="V11 config digest")
        diagnostics = tuple(self.diagnostic_digests)
        if not diagnostics:
            raise ValueError("V11 Selection requires D1 diagnostic evidence")
        for index, value in enumerate(diagnostics):
            require_sha256(value, field=f"V11 diagnostic_digests[{index}]")
        if self.source_v8 is None:
            if (
                arm is not CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING
                or self.sizing_feasibility is None
                or self.sizing_feasibility.executable
            ):
                raise ValueError("V11 Selection source may be absent only for S1 stop")
        elif not isinstance(self.source_v8, CausalAlphaV8SelectionEvidence):
            raise TypeError("V11 Selection source evidence is invalid")
        if self.sizing_feasibility is not None and (
            arm is not CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING
        ):
            raise ValueError("V11 sizing feasibility belongs only to S1")
        if self.schema_version != CAUSAL_ALPHA_V11_SELECTION_SCHEMA:
            raise ValueError("unsupported V11 Selection schema")
        object.__setattr__(self, "study_arm", arm)
        object.__setattr__(self, "diagnostic_digests", diagnostics)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V11 Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return self.source_v8 is not None and self.source_v8.passed

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        if self.source_v8 is None:
            return ("sizing_non_executable",)
        return self.source_v8.rejection_reasons

    @property
    def selected_candidate(self) -> CausalAlphaV11Candidate | None:
        if self.source_v8 is None or self.source_v8.selected_candidate is None:
            return None
        return _V11_BY_V8[self.source_v8.selected_candidate]

    @property
    def selected_config_digest(self) -> str | None:
        return None if self.source_v8 is None else self.source_v8.selected_config_digest

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidates": _candidate_payloads(self.source_v8),
            "diagnostic_digests": self.diagnostic_digests,
            "paired_scope_count": (
                0
                if self.source_v8 is None
                else self.source_v8.source_v7.paired_scope_count
            ),
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
            "study_arm": self.study_arm.value,
            "study_arm_digest": self.study_arm_digest,
            "v11_config_digest": self.v11_config_digest,
        }
        if self.source_v8 is not None:
            payload["source_v8_selection_digest"] = self.source_v8.digest
        if self.sizing_feasibility is not None:
            payload["sizing_feasibility"] = self.sizing_feasibility.to_payload()
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v11_selection(
    *,
    study_arm: CausalAlphaV11StudyArm,
    cash_metrics: tuple[CausalAlphaV8ReplayMetric, ...],
    control_metrics: tuple[CausalAlphaV8ReplayMetric, ...],
    treatment_metrics: tuple[CausalAlphaV8ReplayMetric, ...],
    expected_symbols: tuple[str, ...],
    v11_config_digest: str,
    diagnostic_digests: tuple[str, ...],
    sizing_feasibility: CausalAlphaV11SizingFeasibility | None,
) -> CausalAlphaV11SelectionEvidence:
    """Evaluate cash, exact control, and one treatment under frozen gates."""

    arm = CausalAlphaV11StudyArm(study_arm)
    expected_groups = (
        (cash_metrics, CausalAlphaV8Candidate.V7_CONTROL),
        (control_metrics, CausalAlphaV8Candidate.ROBUST_CONTRARIAN),
        (treatment_metrics, CausalAlphaV8Candidate.ROBUST_CALIBRATED),
    )
    study_arm_digest = content_digest(
        {
            "schema_version": "causal_alpha_v11_study_arm_v1",
            "study_arm": arm.value,
            "v11_config_digest": v11_config_digest,
        }
    )
    if (
        arm is CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING
        and sizing_feasibility is not None
        and not sizing_feasibility.executable
    ):
        if any(metrics for metrics, _ in expected_groups):
            raise ValueError("V11 S1 stop must occur before replay metrics exist")
        if len(expected_symbols) != 9 or len(set(expected_symbols)) != 9:
            raise ValueError("V11 Selection requires exactly nine expected symbols")
        return CausalAlphaV11SelectionEvidence(
            study_arm=arm,
            study_arm_digest=study_arm_digest,
            v11_config_digest=v11_config_digest,
            diagnostic_digests=diagnostic_digests,
            source_v8=None,
            sizing_feasibility=sizing_feasibility,
        )
    for metrics, expected_candidate in expected_groups:
        if not metrics or any(
            metric.candidate is not expected_candidate for metric in metrics
        ):
            raise ValueError("V11 Selection candidate group identity drifted")
    source = evaluate_causal_alpha_v8_selection(
        cash_metrics + control_metrics + treatment_metrics,
        expected_symbols=expected_symbols,
    )
    return CausalAlphaV11SelectionEvidence(
        study_arm=arm,
        study_arm_digest=study_arm_digest,
        v11_config_digest=v11_config_digest,
        diagnostic_digests=diagnostic_digests,
        source_v8=source,
        sizing_feasibility=sizing_feasibility,
    )


__all__ = [
    "CAUSAL_ALPHA_V11_SELECTION_SCHEMA",
    "CausalAlphaV11SelectionEvidence",
    "evaluate_causal_alpha_v11_selection",
]
