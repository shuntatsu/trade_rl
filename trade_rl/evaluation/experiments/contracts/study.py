"""Immutable Study-level contracts for controlled development research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts._common import (
    contract_aware_datetime,
    contract_non_negative_int,
    contract_non_negative_int_tuple,
    contract_positive_int,
    contract_sha256,
    contract_sha256_tuple,
    contract_text,
    contract_unique_enum_tuple,
    contract_unique_texts,
)
from trade_rl.evaluation.experiments.contracts.experiment import ControlledFactor
from trade_rl.evaluation.experiments.contracts.run import ResolvedRunConfig
from trade_rl.evaluation.experiments.errors import ContractViolationError

CANDIDATE_STRATEGY_NAMES = (
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "ppo",
)
CONTROL_STRATEGY_NAMES = (
    "cash",
    "constant_long",
    "constant_short",
)


class StudyOutcome(StrEnum):
    WINNER = "WINNER"
    NO_WINNER = "NO_WINNER"


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """Frozen development Study contract and adaptive-iteration budget."""

    FIXED_RESOLVED_FIELDS: ClassVar[tuple[str, ...]] = (
        "fit_cutoff",
        "evaluation_start",
        "evaluation_stop_exclusive",
        "initial_capital",
        "execution_overlay",
    )

    STRATEGY_NAMES: ClassVar[tuple[str, ...]] = (
        *CONTROL_STRATEGY_NAMES,
        *CANDIDATE_STRATEGY_NAMES,
    )
    PPO_SEED_INVARIANT_STRATEGY_NAMES: ClassVar[tuple[str, ...]] = (
        CONTROL_STRATEGY_NAMES
        + tuple(name for name in CANDIDATE_STRATEGY_NAMES if name != "ppo")
    )

    research_question: str
    dataset_id: str
    dataset_artifact_schema: str
    dataset_artifact_digest: str
    symbols: tuple[str, ...]
    baseline_config: ResolvedRunConfig
    ppo_seeds: tuple[int, ...]
    allowed_factors: tuple[object, ...]
    max_experiments: int
    n_bootstrap: int
    bootstrap_seed: int
    implementation_digest: str
    runtime_environment_digest: str
    schema_version: str = "controlled_study_plan_v1"

    def __post_init__(self) -> None:
        research_question = contract_text(
            self.research_question,
            field="research_question",
        )
        dataset_id = contract_sha256(self.dataset_id, field="dataset_id")
        dataset_artifact_schema = contract_text(
            self.dataset_artifact_schema,
            field="dataset_artifact_schema",
        )
        dataset_artifact_digest = contract_sha256(
            self.dataset_artifact_digest,
            field="dataset_artifact_digest",
        )
        symbols = contract_unique_texts(self.symbols, field="symbols")
        if not isinstance(self.baseline_config, ResolvedRunConfig):
            raise ContractViolationError("baseline_config must be a ResolvedRunConfig")
        ppo_seeds = contract_non_negative_int_tuple(
            self.ppo_seeds,
            field="ppo_seeds",
            minimum_items=2,
        )
        if self.baseline_config.ppo_seed != ppo_seeds[0]:
            raise ContractViolationError(
                "baseline_config ppo_seed must equal the first registered ppo_seeds value"
            )
        allowed = contract_unique_enum_tuple(
            self.allowed_factors,
            field="allowed_factors",
            expected_type=ControlledFactor,
        )
        max_experiments = contract_positive_int(
            self.max_experiments,
            field="max_experiments",
            maximum=9_999,
        )
        n_bootstrap = contract_positive_int(
            self.n_bootstrap,
            field="n_bootstrap",
        )
        bootstrap_seed = contract_non_negative_int(
            self.bootstrap_seed,
            field="bootstrap_seed",
        )
        implementation_digest = contract_sha256(
            self.implementation_digest,
            field="implementation_digest",
        )
        runtime_environment_digest = contract_sha256(
            self.runtime_environment_digest,
            field="runtime_environment_digest",
        )
        schema_version = contract_text(self.schema_version, field="schema_version")

        object.__setattr__(self, "research_question", research_question)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "dataset_artifact_schema", dataset_artifact_schema)
        object.__setattr__(self, "dataset_artifact_digest", dataset_artifact_digest)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "ppo_seeds", ppo_seeds)
        object.__setattr__(
            self,
            "allowed_factors",
            cast(tuple[ControlledFactor, ...], allowed),
        )
        object.__setattr__(self, "max_experiments", max_experiments)
        object.__setattr__(self, "n_bootstrap", n_bootstrap)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        object.__setattr__(self, "implementation_digest", implementation_digest)
        object.__setattr__(
            self, "runtime_environment_digest", runtime_environment_digest
        )
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def candidate_strategy_names(self) -> tuple[str, ...]:
        return CANDIDATE_STRATEGY_NAMES

    @property
    def control_strategy_names(self) -> tuple[str, ...]:
        return CONTROL_STRATEGY_NAMES

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "research_question": self.research_question,
            "dataset_id": self.dataset_id,
            "dataset_artifact_schema": self.dataset_artifact_schema,
            "dataset_artifact_digest": self.dataset_artifact_digest,
            "symbols": list(self.symbols),
            "baseline_config": self.baseline_config.to_payload(),
            "ppo_seeds": list(self.ppo_seeds),
            "allowed_factors": [
                cast(StrEnum, factor).value for factor in self.allowed_factors
            ],
            "max_experiments": self.max_experiments,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
            "implementation_digest": self.implementation_digest,
            "runtime_environment_digest": self.runtime_environment_digest,
            "candidate_strategy_names": list(CANDIDATE_STRATEGY_NAMES),
            "control_strategy_names": list(CONTROL_STRATEGY_NAMES),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class StudyFreeze:
    """Terminal development-only Study outcome."""

    study_digest: str
    experiment_decision_digests: tuple[str, ...]
    outcome: StudyOutcome
    selected_evidence_digest: str | None
    selected_strategy: str | None
    rationale: str
    frozen_by: str
    frozen_at: datetime
    schema_version: str = "controlled_study_freeze_v1"

    def __post_init__(self) -> None:
        study_digest = contract_sha256(self.study_digest, field="study_digest")
        decision_digests = contract_sha256_tuple(
            self.experiment_decision_digests,
            field="experiment_decision_digests",
        )
        if not isinstance(self.outcome, StudyOutcome):
            raise ContractViolationError("outcome is unsupported")
        rationale = contract_text(self.rationale, field="rationale")
        frozen_by = contract_text(self.frozen_by, field="frozen_by")
        frozen_at = contract_aware_datetime(self.frozen_at, field="frozen_at")
        schema_version = contract_text(self.schema_version, field="schema_version")

        selected_evidence_digest = self.selected_evidence_digest
        selected_strategy = self.selected_strategy
        if self.outcome is StudyOutcome.WINNER:
            if selected_evidence_digest is None:
                raise ContractViolationError("WINNER requires selected_evidence_digest")
            selected_evidence_digest = contract_sha256(
                selected_evidence_digest,
                field="selected_evidence_digest",
            )
            if selected_strategy not in CANDIDATE_STRATEGY_NAMES:
                raise ContractViolationError(
                    "WINNER selected_strategy must be a candidate strategy"
                )
        else:
            if selected_evidence_digest is not None or selected_strategy is not None:
                raise ContractViolationError(
                    "NO_WINNER forbids selected evidence and strategy fields"
                )

        object.__setattr__(self, "study_digest", study_digest)
        object.__setattr__(self, "experiment_decision_digests", decision_digests)
        object.__setattr__(self, "selected_evidence_digest", selected_evidence_digest)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "frozen_by", frozen_by)
        object.__setattr__(self, "frozen_at", frozen_at)
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "experiment_decision_digests": list(self.experiment_decision_digests),
            "outcome": self.outcome.value,
            "selected_evidence_digest": self.selected_evidence_digest,
            "selected_strategy": self.selected_strategy,
            "rationale": self.rationale,
            "frozen_by": self.frozen_by,
            "frozen_at": self.frozen_at,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = [
    "CANDIDATE_STRATEGY_NAMES",
    "CONTROL_STRATEGY_NAMES",
    "StudyFreeze",
    "StudyOutcome",
    "StudyPlan",
]
