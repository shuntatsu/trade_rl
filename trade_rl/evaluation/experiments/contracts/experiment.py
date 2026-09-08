"""Immutable Experiment definition and terminal failure contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts._common import (
    contract_aware_datetime,
    contract_positive_int,
    contract_sha256,
    contract_text,
)
from trade_rl.evaluation.experiments.contracts.run import ResolvedRunConfig
from trade_rl.evaluation.experiments.errors import ContractViolationError


class ControlledFactor(StrEnum):
    FEATURE_SET = "FEATURE_SET"
    RULE_SIGNAL = "RULE_SIGNAL"
    RULE_THRESHOLDS = "RULE_THRESHOLDS"
    FORECAST_THRESHOLDS = "FORECAST_THRESHOLDS"
    FIT_SYMBOL_SCOPE = "FIT_SYMBOL_SCOPE"
    PPO_TRAINING_BUDGET = "PPO_TRAINING_BUDGET"
    GROSS_BUDGET = "GROSS_BUDGET"


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    """Pre-registered definition for one controlled development attempt."""

    study_digest: str
    sequence: int
    hypothesis: str
    baseline_evidence_digest: str
    factor: ControlledFactor
    candidate_requested_config_digest: str
    candidate_config: ResolvedRunConfig
    schema_version: str = "controlled_experiment_definition_v1"

    def __post_init__(self) -> None:
        study_digest = contract_sha256(self.study_digest, field="study_digest")
        sequence = contract_positive_int(
            self.sequence,
            field="sequence",
            maximum=9_999,
        )
        hypothesis = contract_text(self.hypothesis, field="hypothesis")
        baseline_evidence_digest = contract_sha256(
            self.baseline_evidence_digest,
            field="baseline_evidence_digest",
        )
        if not isinstance(self.factor, ControlledFactor):
            raise ContractViolationError("factor is unsupported")
        candidate_requested_config_digest = contract_sha256(
            self.candidate_requested_config_digest,
            field="candidate_requested_config_digest",
        )
        if not isinstance(self.candidate_config, ResolvedRunConfig):
            raise ContractViolationError("candidate_config must be a ResolvedRunConfig")
        schema_version = contract_text(self.schema_version, field="schema_version")

        object.__setattr__(self, "study_digest", study_digest)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "hypothesis", hypothesis)
        object.__setattr__(
            self,
            "baseline_evidence_digest",
            baseline_evidence_digest,
        )
        object.__setattr__(
            self,
            "candidate_requested_config_digest",
            candidate_requested_config_digest,
        )
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "sequence": self.sequence,
            "hypothesis": self.hypothesis,
            "baseline_evidence_digest": self.baseline_evidence_digest,
            "factor": self.factor.value,
            "candidate_requested_config_digest": self.candidate_requested_config_digest,
            "candidate_config": self.candidate_config.to_payload(),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class ExperimentFailure:
    """Terminal operational failure recorded before complete candidate evidence."""

    study_digest: str
    experiment_digest: str
    reason: str
    recorded_by: str
    recorded_at: datetime
    schema_version: str = "controlled_experiment_failure_v1"

    def __post_init__(self) -> None:
        study_digest = contract_sha256(self.study_digest, field="study_digest")
        experiment_digest = contract_sha256(
            self.experiment_digest,
            field="experiment_digest",
        )
        reason = contract_text(self.reason, field="reason")
        recorded_by = contract_text(self.recorded_by, field="recorded_by")
        recorded_at = contract_aware_datetime(
            self.recorded_at,
            field="recorded_at",
        )
        schema_version = contract_text(self.schema_version, field="schema_version")

        object.__setattr__(self, "study_digest", study_digest)
        object.__setattr__(self, "experiment_digest", experiment_digest)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "recorded_by", recorded_by)
        object.__setattr__(self, "recorded_at", recorded_at)
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "experiment_digest": self.experiment_digest,
            "reason": self.reason,
            "recorded_by": self.recorded_by,
            "recorded_at": self.recorded_at,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = ["ControlledFactor", "ExperimentDefinition", "ExperimentFailure"]
