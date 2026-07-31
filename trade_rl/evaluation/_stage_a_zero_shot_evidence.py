"""Observation and complete-evidence artifacts for Stage A zero-shot evaluation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation._stage_a_zero_shot_contract_values import (
    STAGE_A_EVIDENCE_SCHEMA,
    STAGE_A_OBSERVATION_SCHEMA,
    StageACandidate,
    StageAEvaluationSplit,
    StageAZeroShotEvaluationPlan,
    _SPLITS,
    _finite,
    _non_negative_int,
    _unique_digests,
    _unique_ints,
    _unique_strings,
)

@dataclass(frozen=True, slots=True)
class StageAEvaluationObservation:
    """Paired policy/baseline growth for one exact evaluation cell."""

    candidate_id: str
    split: StageAEvaluationSplit
    triplet_id: str
    fold: int
    seed: int
    checkpoint_digest: str
    dataset_identity: str
    feature_identity: str
    execution_identity: str
    evaluation_identity: str
    policy_execution_evidence_digest: str
    baseline_execution_evidence_digest: str
    policy_log_growth: float
    baseline_log_growth: float
    schema_version: str = STAGE_A_OBSERVATION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_OBSERVATION_SCHEMA:
            raise ValueError("unsupported Stage A observation schema")
        candidate_id = require_non_empty(
            self.candidate_id, field="stage_a_observation.candidate_id"
        )
        if self.split not in _SPLITS:
            raise ValueError("Stage A observation split is invalid")
        for field, value in (
            ("triplet_id", self.triplet_id),
            ("checkpoint_digest", self.checkpoint_digest),
            ("dataset_identity", self.dataset_identity),
            ("feature_identity", self.feature_identity),
            ("execution_identity", self.execution_identity),
            ("evaluation_identity", self.evaluation_identity),
            ("policy_execution_evidence_digest", self.policy_execution_evidence_digest),
            (
                "baseline_execution_evidence_digest",
                self.baseline_execution_evidence_digest,
            ),
        ):
            require_sha256(value, field=f"stage_a_observation.{field}")
        fold = _non_negative_int(self.fold, field="stage_a_observation.fold")
        seed = _non_negative_int(self.seed, field="stage_a_observation.seed")
        policy_growth = _finite(
            self.policy_log_growth, field="stage_a_observation.policy_log_growth"
        )
        baseline_growth = _finite(
            self.baseline_log_growth, field="stage_a_observation.baseline_log_growth"
        )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "policy_log_growth", policy_growth)
        object.__setattr__(self, "baseline_log_growth", baseline_growth)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A observation digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @classmethod
    def create(cls, **kwargs: object) -> StageAEvaluationObservation:
        return cls(**kwargs)  # type: ignore[arg-type]

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.candidate_id, self.triplet_id, self.fold, self.seed)

    @property
    def baseline_key(self) -> tuple[str, int, int]:
        return (self.triplet_id, self.fold, self.seed)

    @property
    def excess_log_growth(self) -> float:
        return self.policy_log_growth - self.baseline_log_growth

    def digest_payload(self) -> dict[str, object]:
        return {
            "baseline_execution_evidence_digest": (
                self.baseline_execution_evidence_digest
            ),
            "baseline_log_growth": self.baseline_log_growth,
            "candidate_id": self.candidate_id,
            "checkpoint_digest": self.checkpoint_digest,
            "dataset_identity": self.dataset_identity,
            "evaluation_identity": self.evaluation_identity,
            "execution_identity": self.execution_identity,
            "feature_identity": self.feature_identity,
            "fold": self.fold,
            "policy_execution_evidence_digest": self.policy_execution_evidence_digest,
            "policy_log_growth": self.policy_log_growth,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "split": self.split,
            "triplet_id": self.triplet_id,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


@dataclass(frozen=True, slots=True)
class StageAEvaluationEvidence:
    """Complete Cartesian evidence for one split and candidate set."""

    plan_digest: str
    split: StageAEvaluationSplit
    candidate_ids: tuple[str, ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    triplet_ids: tuple[str, ...]
    observations: tuple[StageAEvaluationObservation, ...]
    schema_version: str = STAGE_A_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_EVIDENCE_SCHEMA:
            raise ValueError("unsupported Stage A evidence schema")
        require_sha256(self.plan_digest, field="stage_a_evidence.plan_digest")
        if self.split not in _SPLITS:
            raise ValueError("Stage A evidence split is invalid")
        candidate_ids = _unique_strings(
            tuple(self.candidate_ids), field="stage_a_evidence.candidate_ids"
        )
        folds = _unique_ints(
            tuple(self.folds), field="stage_a_evidence.folds", minimum_count=2
        )
        seeds = _unique_ints(
            tuple(self.seeds), field="stage_a_evidence.seeds", minimum_count=2
        )
        triplet_ids = _unique_digests(
            tuple(self.triplet_ids), field="stage_a_evidence.triplet_ids"
        )
        observations = tuple(sorted(self.observations, key=lambda item: item.key))
        keys = tuple(item.key for item in observations)
        if len(set(keys)) != len(keys):
            raise ValueError("Stage A evidence contains a duplicate observation")
        expected_keys = set(itertools.product(candidate_ids, triplet_ids, folds, seeds))
        if set(keys) != expected_keys:
            raise ValueError("Stage A evidence observation closure mismatch")
        if any(item.split != self.split for item in observations):
            raise ValueError("Stage A evidence contains a cross-split observation")
        baseline_by_key: dict[tuple[str, int, int], tuple[str, float]] = {}
        for observation in observations:
            baseline = (
                observation.baseline_execution_evidence_digest,
                observation.baseline_log_growth,
            )
            previous = baseline_by_key.setdefault(observation.baseline_key, baseline)
            if previous != baseline:
                raise ValueError(
                    "Stage A evidence requires one shared baseline per triplet/fold/seed cell"
                )
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "triplet_ids", triplet_ids)
        object.__setattr__(self, "observations", observations)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A evidence digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def validate_plan(self, plan: StageAZeroShotEvaluationPlan) -> None:
        if self.plan_digest != plan.digest:
            raise ValueError("Stage A evidence plan digest mismatch")
        if self.folds != plan.folds or self.seeds != plan.seeds:
            raise ValueError("Stage A evidence fold or seed closure mismatch")
        if self.triplet_ids != plan.triplet_ids_for(self.split):
            raise ValueError("Stage A evidence triplet closure mismatch")
        if not set(self.candidate_ids) <= set(plan.candidate_ids):
            raise ValueError("Stage A evidence contains an undeclared candidate")
        for observation in self.observations:
            candidate = plan.candidate(observation.candidate_id)
            for label, actual, expected in (
                ("dataset", observation.dataset_identity, plan.dataset_identity),
                ("feature", observation.feature_identity, plan.feature_identity),
                ("execution", observation.execution_identity, plan.execution_identity),
                ("evaluation", observation.evaluation_identity, plan.evaluation_identity),
            ):
                if actual != expected:
                    raise ValueError(f"Stage A observation {label} identity mismatch")
            if observation.checkpoint_digest != candidate.checkpoint_digest(
                observation.seed
            ):
                raise ValueError("Stage A observation checkpoint digest mismatch")

    def observations_for(self, candidate_id: str) -> tuple[StageAEvaluationObservation, ...]:
        resolved = require_non_empty(candidate_id, field="stage_a_candidate_id")
        if resolved not in self.candidate_ids:
            raise ValueError("Stage A evidence candidate is not present")
        return tuple(item for item in self.observations if item.candidate_id == resolved)

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_ids": self.candidate_ids,
            "folds": self.folds,
            "observations": tuple(item.to_json_dict() for item in self.observations),
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
            "seeds": self.seeds,
            "split": self.split,
            "triplet_ids": self.triplet_ids,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def build_stage_a_zero_shot_evaluation_plan(**kwargs: object) -> StageAZeroShotEvaluationPlan:
    return StageAZeroShotEvaluationPlan(**kwargs)  # type: ignore[arg-type]


def build_stage_a_evaluation_evidence(
    *,
    plan: StageAZeroShotEvaluationPlan,
    split: str,
    observations: tuple[StageAEvaluationObservation, ...],
    candidate_ids: tuple[str, ...] | None = None,
) -> StageAEvaluationEvidence:
    resolved_split = cast(StageAEvaluationSplit, split)
    selected_candidates = plan.candidate_ids if candidate_ids is None else candidate_ids
    evidence = StageAEvaluationEvidence(
        plan_digest=plan.digest,
        split=resolved_split,
        candidate_ids=selected_candidates,
        folds=plan.folds,
        seeds=plan.seeds,
        triplet_ids=plan.triplet_ids_for(resolved_split),
        observations=observations,
    )
    evidence.validate_plan(plan)
    return evidence


def write_stage_a_zero_shot_evaluation_plan(
    path: str | Path, plan: StageAZeroShotEvaluationPlan
) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(plan.to_json_dict()))


def write_stage_a_evaluation_evidence(
    path: str | Path, evidence: StageAEvaluationEvidence
) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(evidence.to_json_dict()))


