"""Fail-closed Signal-to-Admission orchestration for Causal Alpha V6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.workflows.universal_causal_alpha_v6_artifact_store import (
    CausalAlphaV6ArtifactStore,
)

CAUSAL_ALPHA_V6_RESEARCH_PACKAGE_SCHEMA: Final = "causal_alpha_v6_research_package_v1"
_STAGE_EXIT_CODES: Final = {"signal": 2, "selection": 3, "admission": 4}


class _Evidence(Protocol):
    @property
    def passed(self) -> bool: ...

    @property
    def digest(self) -> str: ...

    def to_payload(self) -> dict[str, object]: ...


def _validate(value: object, *, stage: str) -> _Evidence:
    if not isinstance(getattr(value, "passed", None), bool):
        raise TypeError(f"V6 {stage} evidence has no boolean passed state")
    digest = getattr(value, "digest", None)
    if not isinstance(digest, str):
        raise TypeError(f"V6 {stage} evidence has no digest")
    require_sha256(digest, field=f"V6 {stage} digest")
    if not callable(getattr(value, "to_payload", None)):
        raise TypeError(f"V6 {stage} evidence has no payload")
    return value  # type: ignore[return-value]


class CausalAlphaV6StageRejected(RuntimeError):
    def __init__(self, stage: str, evidence: _Evidence) -> None:
        if stage not in _STAGE_EXIT_CODES:
            raise ValueError("V6 rejected stage is invalid")
        self.stage = stage
        self.exit_code = _STAGE_EXIT_CODES[stage]
        self.evidence = _validate(evidence, stage=stage)
        self.evidence_digest = evidence.digest
        self.digest = content_digest(
            {
                "evidence_digest": evidence.digest,
                "exit_code": self.exit_code,
                "promotion_eligible": False,
                "schema_version": f"causal_alpha_v6_{stage}_rejection_v1",
                "stage": stage,
            }
        )
        super().__init__(f"causal alpha V6 {stage} rejected")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "evidence_digest": self.evidence_digest,
            "exit_code": self.exit_code,
            "promotion_eligible": False,
            "schema_version": f"causal_alpha_v6_{self.stage}_rejection_v1",
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV6ResearchPackage:
    signal_evidence_digest: str
    selection_evidence_digest: str
    admission_evidence_digest: str
    selected_candidate: CausalAlphaV6Candidate
    selected_config_digest: str
    run_manifest_digest: str
    v4_context_manifest_digest: str
    generator_code_digest: str
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V6_RESEARCH_PACKAGE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV6Candidate(self.selected_candidate)
        for name in (
            "signal_evidence_digest",
            "selection_evidence_digest",
            "admission_evidence_digest",
            "selected_config_digest",
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "generator_code_digest",
        ):
            require_sha256(getattr(self, name), field=f"V6 package {name}")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V6 package must remain research-only and non-promotable")
        if self.schema_version != CAUSAL_ALPHA_V6_RESEARCH_PACKAGE_SCHEMA:
            raise ValueError("unsupported V6 package schema")
        object.__setattr__(self, "selected_candidate", candidate)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 package digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "admission_evidence_digest": self.admission_evidence_digest,
            "generator_code_digest": self.generator_code_digest,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "selected_candidate": self.selected_candidate.value,
            "selected_config_digest": self.selected_config_digest,
            "selection_evidence_digest": self.selection_evidence_digest,
            "signal_evidence_digest": self.signal_evidence_digest,
            "v4_context_manifest_digest": self.v4_context_manifest_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _persist(
    store: CausalAlphaV6ArtifactStore,
    stage: str,
    evidence: _Evidence,
) -> None:
    store.write_leaf(
        Path(stage) / "evidence.json",
        store.envelope(
            schema_version=f"causal_alpha_v6_{stage}_envelope_v1",
            evidence_digest=evidence.digest,
            payload=evidence.to_payload(),
        ),
    )


def _reject(
    store: CausalAlphaV6ArtifactStore,
    stage: str,
    evidence: _Evidence,
) -> CausalAlphaV6StageRejected:
    rejection = CausalAlphaV6StageRejected(stage, evidence)
    store.write_leaf(
        "result.json",
        store.envelope(
            schema_version="causal_alpha_v6_terminal_result_v1",
            evidence_digest=rejection.digest,
            payload={"status": f"{stage}_rejected", **rejection.to_payload()},
        ),
    )
    return rejection


def _selected(selection: _Evidence) -> tuple[CausalAlphaV6Candidate, str]:
    raw_candidate = getattr(selection, "selected_candidate", None)
    raw_digest = getattr(selection, "selected_config_digest", None)
    if raw_candidate is None or not isinstance(raw_digest, str):
        raise TypeError("V6 passed Selection omitted selected candidate/config")
    candidate = CausalAlphaV6Candidate(raw_candidate)
    require_sha256(raw_digest, field="V6 selected config digest")
    return candidate, raw_digest


def run_universal_causal_alpha_v6_research_pipeline(
    *,
    store: CausalAlphaV6ArtifactStore,
    prepare_stage: Callable[[], object],
    signal_stage: Callable[[object], object],
    selection_stage: Callable[[object, object], object],
    admission_stage: Callable[[object, object, object], object],
) -> CausalAlphaV6ResearchPackage:
    """Stop at the first rejection and publish only after Admission passes."""

    if not isinstance(store, CausalAlphaV6ArtifactStore):
        raise TypeError("V6 pipeline requires a V6 artifact store")
    prepared = prepare_stage()
    signal = _validate(signal_stage(prepared), stage="signal")
    _persist(store, "signal", signal)
    if not signal.passed:
        raise _reject(store, "signal", signal)
    selection = _validate(
        selection_stage(prepared, signal),
        stage="selection",
    )
    _persist(store, "selection", selection)
    if not selection.passed:
        raise _reject(store, "selection", selection)
    admission = _validate(
        admission_stage(prepared, signal, selection),
        stage="admission",
    )
    _persist(store, "admission", admission)
    if not admission.passed:
        raise _reject(store, "admission", admission)
    selected_candidate, selected_config_digest = _selected(selection)
    if selected_config_digest != store.config_digest:
        raise ValueError("V6 selected config and store identity drifted")
    package = CausalAlphaV6ResearchPackage(
        signal_evidence_digest=signal.digest,
        selection_evidence_digest=selection.digest,
        admission_evidence_digest=admission.digest,
        selected_candidate=selected_candidate,
        selected_config_digest=selected_config_digest,
        run_manifest_digest=store.run_manifest_digest,
        v4_context_manifest_digest=store.v4_context_manifest_digest,
        generator_code_digest=store.generator_code_digest,
    )
    store.write_leaf("package.json", package.to_payload())
    store.write_leaf("result.json", package.to_payload())
    return package


__all__ = [
    "CAUSAL_ALPHA_V6_RESEARCH_PACKAGE_SCHEMA",
    "CausalAlphaV6ResearchPackage",
    "CausalAlphaV6StageRejected",
    "run_universal_causal_alpha_v6_research_pipeline",
]
