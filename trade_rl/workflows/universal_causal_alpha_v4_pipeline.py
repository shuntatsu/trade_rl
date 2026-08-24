"""Fail-closed stage orchestration for the research-only Causal Alpha V4 lane."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
)

CAUSAL_ALPHA_V4_RESEARCH_PACKAGE_SCHEMA: Final = "causal_alpha_v4_research_package_v1"


class _Evidence(Protocol):
    passed: bool
    digest: str

    def to_payload(self) -> dict[str, object]: ...


def _validate_evidence(value: object, *, stage: str) -> _Evidence:
    if not isinstance(getattr(value, "passed", None), bool):
        raise TypeError(f"V4 {stage} stage returned evidence without boolean passed")
    digest = getattr(value, "digest", None)
    if not isinstance(digest, str):
        raise TypeError(f"V4 {stage} stage returned evidence without digest")
    require_sha256(digest, field=f"V4 {stage} evidence digest")
    to_payload = getattr(value, "to_payload", None)
    if not callable(to_payload):
        raise TypeError(f"V4 {stage} stage returned evidence without to_payload")
    payload = to_payload()
    if not isinstance(payload, dict):
        raise TypeError(f"V4 {stage} evidence payload must be an object")
    return value  # type: ignore[return-value]


class _StageRejected(RuntimeError):
    stage: str
    schema_version: str

    def __init__(self, evidence: _Evidence) -> None:
        resolved = _validate_evidence(evidence, stage=self.stage)
        self.evidence = resolved
        self.evidence_digest = resolved.digest
        self.digest = content_digest(
            {
                "evidence_digest": self.evidence_digest,
                "promotion_eligible": False,
                "schema_version": self.schema_version,
                "stage": self.stage,
            }
        )
        super().__init__(f"causal alpha V4 {self.stage} rejected")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "evidence_digest": self.evidence_digest,
            "promotion_eligible": False,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }


class CausalAlphaV4SignalRejected(_StageRejected):
    stage = "signal"
    schema_version = "causal_alpha_v4_signal_rejection_v1"


class CausalAlphaV4SelectionRejected(_StageRejected):
    stage = "selection"
    schema_version = "causal_alpha_v4_selection_rejection_v1"


class CausalAlphaV4AdmissionRejected(_StageRejected):
    stage = "admission"
    schema_version = "causal_alpha_v4_admission_rejection_v1"


@dataclass(frozen=True, slots=True)
class CausalAlphaV4ResearchPackage:
    signal_evidence_digest: str
    selection_evidence_digest: str
    admission_evidence_digest: str
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V4_RESEARCH_PACKAGE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "signal_evidence_digest",
            "selection_evidence_digest",
            "admission_evidence_digest",
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V4 package {field_name}")
        if not self.research_only or self.promotion_eligible:
            raise ValueError(
                "V4 teacher package must remain research-only and non-promotable"
            )
        if self.schema_version != CAUSAL_ALPHA_V4_RESEARCH_PACKAGE_SCHEMA:
            raise ValueError("unsupported V4 research package schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 research package digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "admission_evidence_digest": self.admission_evidence_digest,
            "config_digest": self.config_digest,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "selection_evidence_digest": self.selection_evidence_digest,
            "signal_evidence_digest": self.signal_evidence_digest,
            "v4_context_manifest_digest": self.v4_context_manifest_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _persist_stage(
    store: CausalAlphaV4ArtifactStore,
    *,
    stage: str,
    evidence: _Evidence,
) -> None:
    payload = evidence.to_payload()
    store.write_leaf(
        Path(stage) / "evidence.json",
        store.envelope(
            schema_version=f"causal_alpha_v4_{stage}_envelope_v1",
            evidence_digest=evidence.digest,
            payload=payload,
        ),
    )


def run_universal_causal_alpha_v4_research_pipeline(
    *,
    store: CausalAlphaV4ArtifactStore,
    prepare_stage: Callable[[], object],
    signal_stage: Callable[[object], _Evidence],
    selection_stage: Callable[[object, _Evidence], _Evidence],
    admission_stage: Callable[[object, _Evidence, _Evidence], _Evidence],
) -> CausalAlphaV4ResearchPackage:
    """Run V4 stages strictly in order; the first rejected gate terminates the run."""

    if not isinstance(store, CausalAlphaV4ArtifactStore):
        raise TypeError("V4 pipeline requires a CausalAlphaV4ArtifactStore")
    prepared = prepare_stage()

    signal = _validate_evidence(signal_stage(prepared), stage="signal")
    _persist_stage(store, stage="signal", evidence=signal)
    if not signal.passed:
        signal_rejection = CausalAlphaV4SignalRejected(signal)
        store.write_leaf(
            "result.json",
            store.envelope(
                schema_version="causal_alpha_v4_terminal_result_v1",
                evidence_digest=signal_rejection.digest,
                payload={
                    "status": "signal_rejected",
                    **signal_rejection.to_payload(),
                },
            ),
        )
        raise signal_rejection

    selection = _validate_evidence(selection_stage(prepared, signal), stage="selection")
    _persist_stage(store, stage="selection", evidence=selection)
    if not selection.passed:
        selection_rejection = CausalAlphaV4SelectionRejected(selection)
        store.write_leaf(
            "result.json",
            store.envelope(
                schema_version="causal_alpha_v4_terminal_result_v1",
                evidence_digest=selection_rejection.digest,
                payload={
                    "status": "selection_rejected",
                    **selection_rejection.to_payload(),
                },
            ),
        )
        raise selection_rejection

    admission = _validate_evidence(
        admission_stage(prepared, signal, selection), stage="admission"
    )
    _persist_stage(store, stage="admission", evidence=admission)
    if not admission.passed:
        admission_rejection = CausalAlphaV4AdmissionRejected(admission)
        store.write_leaf(
            "result.json",
            store.envelope(
                schema_version="causal_alpha_v4_terminal_result_v1",
                evidence_digest=admission_rejection.digest,
                payload={
                    "status": "admission_rejected",
                    **admission_rejection.to_payload(),
                },
            ),
        )
        raise admission_rejection

    package = CausalAlphaV4ResearchPackage(
        signal_evidence_digest=signal.digest,
        selection_evidence_digest=selection.digest,
        admission_evidence_digest=admission.digest,
        run_manifest_digest=store.run_manifest_digest,
        v4_context_manifest_digest=store.v4_context_manifest_digest,
        config_digest=store.config_digest,
    )
    store.write_leaf("result.json", package.to_payload())
    return package


__all__ = [
    "CAUSAL_ALPHA_V4_RESEARCH_PACKAGE_SCHEMA",
    "CausalAlphaV4AdmissionRejected",
    "CausalAlphaV4ResearchPackage",
    "CausalAlphaV4SelectionRejected",
    "CausalAlphaV4SignalRejected",
    "run_universal_causal_alpha_v4_research_pipeline",
]
