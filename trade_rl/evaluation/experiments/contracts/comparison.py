"""Immutable comparison contract for one controlled Experiment."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from trade_rl.artifacts.canonical import to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts._common import (
    contract_sha256,
    contract_text,
)
from trade_rl.evaluation.experiments.errors import ContractViolationError

_COMPARISON_SCHEMA = "controlled_experiment_comparison_v1"


def _freeze_json(value: object) -> object:
    normalized = to_json_value(value)
    if isinstance(normalized, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in normalized.items()}
        )
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _thaw_json(value: object) -> object:
    if isinstance(value, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExperimentComparison:
    """Digest-bound baseline-vs-candidate factor-effect evidence."""

    study_digest: str
    experiment_digest: str
    baseline_evidence_digest: str
    candidate_evidence_digest: str
    verification_digest: str
    baseline_analysis_digest: str
    candidate_analysis_digest: str
    factor_effect_digest: str
    factor_effect: object
    schema_version: str = _COMPARISON_SCHEMA

    def __post_init__(self) -> None:
        for value, field in (
            (self.study_digest, "study_digest"),
            (self.experiment_digest, "experiment_digest"),
            (self.baseline_evidence_digest, "baseline_evidence_digest"),
            (self.candidate_evidence_digest, "candidate_evidence_digest"),
            (self.verification_digest, "verification_digest"),
            (self.baseline_analysis_digest, "baseline_analysis_digest"),
            (self.candidate_analysis_digest, "candidate_analysis_digest"),
            (self.factor_effect_digest, "factor_effect_digest"),
        ):
            contract_sha256(value, field=field)
        schema_version = contract_text(self.schema_version, field="schema_version")
        if schema_version != _COMPARISON_SCHEMA:
            raise ContractViolationError("unsupported ExperimentComparison schema")
        frozen = _freeze_json(self.factor_effect)
        payload = _thaw_json(frozen)
        if not isinstance(payload, dict):
            raise ContractViolationError("factor_effect must be a JSON object")
        analysis_digest = payload.get("analysis_digest")
        if analysis_digest != self.factor_effect_digest:
            raise ContractViolationError("factor_effect digest reference mismatch")
        body = dict(payload)
        body.pop("analysis_digest", None)
        if content_digest(body) != self.factor_effect_digest:
            raise ContractViolationError("factor_effect content digest mismatch")
        object.__setattr__(self, "factor_effect", frozen)
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "experiment_digest": self.experiment_digest,
            "baseline_evidence_digest": self.baseline_evidence_digest,
            "candidate_evidence_digest": self.candidate_evidence_digest,
            "verification_digest": self.verification_digest,
            "baseline_analysis_digest": self.baseline_analysis_digest,
            "candidate_analysis_digest": self.candidate_analysis_digest,
            "factor_effect_digest": self.factor_effect_digest,
            "factor_effect": cast(dict[str, object], _thaw_json(self.factor_effect)),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = ["ExperimentComparison"]
