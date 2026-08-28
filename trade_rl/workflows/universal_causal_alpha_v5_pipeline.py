"""Fail-closed research stage orchestration for Causal Alpha V5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_v5_artifact_store import (
    CausalAlphaV5ArtifactStore,
)

CAUSAL_ALPHA_V5_RESEARCH_PACKAGE_SCHEMA: Final = "causal_alpha_v5_research_package_v1"


class _Evidence(Protocol):
    @property
    def passed(self) -> bool: ...

    @property
    def digest(self) -> str: ...

    def to_payload(self) -> dict[str, object]: ...


def _validate(value: object, *, stage: str) -> _Evidence:
    if not isinstance(getattr(value, "passed", None), bool):
        raise TypeError(f"V5 {stage} evidence has no boolean passed state")
    digest = getattr(value, "digest", None)
    if not isinstance(digest, str):
        raise TypeError(f"V5 {stage} evidence has no digest")
    require_sha256(digest, field=f"V5 {stage} digest")
    if not callable(getattr(value, "to_payload", None)):
        raise TypeError(f"V5 {stage} evidence has no payload")
    return value  # type: ignore[return-value]


class CausalAlphaV5StageRejected(RuntimeError):
    def __init__(self, stage: str, evidence: _Evidence) -> None:
        self.stage = stage
        self.evidence = _validate(evidence, stage=stage)
        self.evidence_digest = evidence.digest
        self.digest = content_digest(
            {
                "evidence_digest": evidence.digest,
                "promotion_eligible": False,
                "schema_version": f"causal_alpha_v5_{stage}_rejection_v1",
                "stage": stage,
            }
        )
        super().__init__(f"causal alpha V5 {stage} rejected")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "evidence_digest": self.evidence_digest,
            "promotion_eligible": False,
            "schema_version": f"causal_alpha_v5_{self.stage}_rejection_v1",
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV5ResearchPackage:
    calibration_evidence_digest: str
    signal_evidence_digest: str
    selection_evidence_digest: str
    admission_evidence_digest: str
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V5_RESEARCH_PACKAGE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "calibration_evidence_digest",
            "signal_evidence_digest",
            "selection_evidence_digest",
            "admission_evidence_digest",
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
        ):
            require_sha256(getattr(self, name), field=f"V5 package {name}")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V5 package must remain research-only and non-promotable")
        if self.schema_version != CAUSAL_ALPHA_V5_RESEARCH_PACKAGE_SCHEMA:
            raise ValueError("unsupported V5 package schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 package digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "digest"
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _persist(
    store: CausalAlphaV5ArtifactStore, stage: str, evidence: _Evidence
) -> None:
    store.write_leaf(
        Path(stage) / "evidence.json",
        store.envelope(
            schema_version=f"causal_alpha_v5_{stage}_envelope_v1",
            evidence_digest=evidence.digest,
            payload=evidence.to_payload(),
        ),
    )


def run_universal_causal_alpha_v5_research_pipeline(
    *,
    store: CausalAlphaV5ArtifactStore,
    prepare_stage: Callable[[], object],
    calibration_stage: Callable[[object], object],
    signal_stage: Callable[[object, object], object],
    selection_stage: Callable[[object, object, object], object],
    admission_stage: Callable[[object, object, object, object], object],
) -> CausalAlphaV5ResearchPackage:
    """Stop on the first rejected stage and publish a package only after Admission."""

    if not isinstance(store, CausalAlphaV5ArtifactStore):
        raise TypeError("V5 pipeline requires a V5 artifact store")
    prepared = prepare_stage()
    calibration = _validate(calibration_stage(prepared), stage="calibration")
    _persist(store, "calibration", calibration)
    stages: tuple[tuple[str, Callable[[], object]], ...] = (
        ("signal", lambda: signal_stage(prepared, calibration)),
        ("selection", lambda: selection_stage(prepared, calibration, signal)),
        (
            "admission",
            lambda: admission_stage(prepared, calibration, signal, selection),
        ),
    )
    if not calibration.passed:
        rejection = CausalAlphaV5StageRejected("calibration", calibration)
        store.write_leaf(
            "result.json",
            store.envelope(
                schema_version="causal_alpha_v5_terminal_result_v1",
                evidence_digest=rejection.digest,
                payload={"status": "calibration_rejected", **rejection.to_payload()},
            ),
        )
        raise rejection
    signal: _Evidence
    selection: _Evidence
    evidence_by_stage: dict[str, _Evidence] = {}
    for stage, run in stages:
        evidence = _validate(run(), stage=stage)
        evidence_by_stage[stage] = evidence
        if stage == "signal":
            signal = evidence
        elif stage == "selection":
            selection = evidence
        _persist(store, stage, evidence)
        if not evidence.passed:
            rejection = CausalAlphaV5StageRejected(stage, evidence)
            store.write_leaf(
                "result.json",
                store.envelope(
                    schema_version="causal_alpha_v5_terminal_result_v1",
                    evidence_digest=rejection.digest,
                    payload={"status": f"{stage}_rejected", **rejection.to_payload()},
                ),
            )
            raise rejection
    package = CausalAlphaV5ResearchPackage(
        calibration_evidence_digest=calibration.digest,
        signal_evidence_digest=evidence_by_stage["signal"].digest,
        selection_evidence_digest=evidence_by_stage["selection"].digest,
        admission_evidence_digest=evidence_by_stage["admission"].digest,
        run_manifest_digest=store.run_manifest_digest,
        v4_context_manifest_digest=store.v4_context_manifest_digest,
        config_digest=store.config_digest,
    )
    store.write_leaf("package.json", package.to_payload())
    store.write_leaf("result.json", package.to_payload())
    return package


__all__ = [
    "CAUSAL_ALPHA_V5_RESEARCH_PACKAGE_SCHEMA",
    "CausalAlphaV5ResearchPackage",
    "CausalAlphaV5StageRejected",
    "run_universal_causal_alpha_v5_research_pipeline",
]
