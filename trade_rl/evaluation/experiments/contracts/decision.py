"""Immutable evidence-bound Experiment decision contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts._common import (
    contract_aware_datetime,
    contract_sha256,
    contract_text,
)
from trade_rl.evaluation.experiments.errors import ContractViolationError


class ExperimentDecisionKind(StrEnum):
    ACCEPT_CANDIDATE = "ACCEPT_CANDIDATE"
    KEEP_BASELINE = "KEEP_BASELINE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class ExperimentDecision:
    """One immutable human/agent research decision bound to comparison evidence."""

    study_digest: str
    experiment_digest: str
    verification_digest: str
    comparison_digest: str
    decision: ExperimentDecisionKind
    rationale: str
    decided_by: str
    decided_at: datetime
    schema_version: str = "controlled_experiment_decision_v1"

    def __post_init__(self) -> None:
        study_digest = contract_sha256(self.study_digest, field="study_digest")
        experiment_digest = contract_sha256(
            self.experiment_digest,
            field="experiment_digest",
        )
        verification_digest = contract_sha256(
            self.verification_digest,
            field="verification_digest",
        )
        comparison_digest = contract_sha256(
            self.comparison_digest,
            field="comparison_digest",
        )
        if not isinstance(self.decision, ExperimentDecisionKind):
            raise ContractViolationError("decision is unsupported")
        rationale = contract_text(self.rationale, field="rationale")
        decided_by = contract_text(self.decided_by, field="decided_by")
        decided_at = contract_aware_datetime(self.decided_at, field="decided_at")
        schema_version = contract_text(self.schema_version, field="schema_version")

        object.__setattr__(self, "study_digest", study_digest)
        object.__setattr__(self, "experiment_digest", experiment_digest)
        object.__setattr__(self, "verification_digest", verification_digest)
        object.__setattr__(self, "comparison_digest", comparison_digest)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "decided_by", decided_by)
        object.__setattr__(self, "decided_at", decided_at)
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "experiment_digest": self.experiment_digest,
            "verification_digest": self.verification_digest,
            "comparison_digest": self.comparison_digest,
            "decision": self.decision.value,
            "rationale": self.rationale,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = ["ExperimentDecision", "ExperimentDecisionKind"]
