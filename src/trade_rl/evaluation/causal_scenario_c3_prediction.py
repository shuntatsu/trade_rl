"""Frozen C1 prediction evidence consumed by realized C3 comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256, require_unique_non_empty
from trade_rl.evaluation.causal_scenario_c3_contracts import PersistedScenarioDecision
from trade_rl.evaluation.causal_scenario_values import CausalScenarioEvaluationResult

C3_PREDICTION_EVIDENCE_SCHEMA: Final = "causal_scenario_c3_prediction_evidence_v1"


def _readonly_float_vector(name: str, value: object, *, size: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).copy(order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of the expected size")
    array[array == 0.0] = 0.0
    array.setflags(write=False)
    return array


def _readonly_int_vector(name: str, value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector")
    array = np.asarray(raw, dtype=np.int64).copy(order="C")
    if array.ndim != 1 or array.size == 0 or np.any(array < 0):
        raise ValueError(f"{name} must be a non-empty non-negative vector")
    array.setflags(write=False)
    return array


def _array_payload(value: np.ndarray) -> dict[str, object]:
    return {
        "dtype": value.dtype.str,
        "shape": tuple(int(size) for size in value.shape),
        "values": value.tolist(),
    }


@dataclass(frozen=True, slots=True)
class C3PredictionEvidence:
    result_digest: str
    scenario_library_digest: str
    scenario_set_digest: str
    candidate_digests: tuple[str, ...]
    predicted_score: np.ndarray
    predicted_mean_advantage: np.ndarray
    predicted_loss_cvar: np.ndarray
    predicted_expected_turnover: np.ndarray
    scenario_anchor_indices: np.ndarray
    scenario_distances: np.ndarray
    evidence_digest: str
    schema_version: str = C3_PREDICTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "result_digest",
            "scenario_library_digest",
            "scenario_set_digest",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(str(getattr(self, field)), field=field),
            )
        candidates = tuple(
            require_sha256(value, field="candidate_digests")
            for value in require_unique_non_empty(
                tuple(self.candidate_digests), field="candidate_digests"
            )
        )
        candidate_count = len(candidates)
        arrays = {}
        for name in (
            "predicted_score",
            "predicted_mean_advantage",
            "predicted_loss_cvar",
            "predicted_expected_turnover",
        ):
            arrays[name] = _readonly_float_vector(
                name, getattr(self, name), size=candidate_count
            )
        if np.any(arrays["predicted_loss_cvar"] < 0.0):
            raise ValueError("predicted_loss_cvar must be non-negative")
        if np.any(arrays["predicted_expected_turnover"] < 0.0):
            raise ValueError("predicted_expected_turnover must be non-negative")
        anchors = _readonly_int_vector(
            "scenario_anchor_indices", self.scenario_anchor_indices
        )
        distances = _readonly_float_vector(
            "scenario_distances", self.scenario_distances, size=anchors.size
        )
        if np.any(distances < 0.0):
            raise ValueError("scenario_distances must be non-negative")
        if self.schema_version != C3_PREDICTION_EVIDENCE_SCHEMA:
            raise ValueError("unsupported C3 prediction evidence schema")
        object.__setattr__(self, "candidate_digests", candidates)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "scenario_anchor_indices", anchors)
        object.__setattr__(self, "scenario_distances", distances)
        digest = require_sha256(self.evidence_digest, field="evidence_digest")
        if digest != content_digest(self.digest_payload()):
            raise ValueError("evidence_digest does not match C3 prediction evidence")
        object.__setattr__(self, "evidence_digest", digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_digests": self.candidate_digests,
            "predicted_expected_turnover": _array_payload(
                self.predicted_expected_turnover
            ),
            "predicted_loss_cvar": _array_payload(self.predicted_loss_cvar),
            "predicted_mean_advantage": _array_payload(self.predicted_mean_advantage),
            "predicted_score": _array_payload(self.predicted_score),
            "result_digest": self.result_digest,
            "scenario_anchor_indices": _array_payload(self.scenario_anchor_indices),
            "scenario_distances": _array_payload(self.scenario_distances),
            "scenario_library_digest": self.scenario_library_digest,
            "scenario_set_digest": self.scenario_set_digest,
            "schema_version": self.schema_version,
        }

    def validate_for_decision(self, decision: PersistedScenarioDecision) -> None:
        if not isinstance(decision, PersistedScenarioDecision):
            raise TypeError("decision must be PersistedScenarioDecision")
        if self.result_digest != decision.value_result_digest:
            raise ValueError("C3 prediction result does not match persisted decision")
        if self.scenario_library_digest != decision.scenario_library_digest:
            raise ValueError("C3 prediction library does not match persisted decision")
        if self.scenario_set_digest != decision.scenario_set_digest:
            raise ValueError(
                "C3 prediction scenario set does not match persisted decision"
            )
        if self.candidate_digests != decision.candidate_digests:
            raise ValueError("C3 prediction candidates do not match persisted decision")
        if not np.array_equal(self.predicted_score, decision.score):
            raise ValueError("C3 prediction scores do not match persisted decision")


def create_c3_prediction_evidence(
    *,
    result_digest: str,
    scenario_library_digest: str,
    scenario_set_digest: str,
    candidate_digests: tuple[str, ...],
    predicted_score: np.ndarray,
    predicted_mean_advantage: np.ndarray,
    predicted_loss_cvar: np.ndarray,
    predicted_expected_turnover: np.ndarray,
    scenario_anchor_indices: np.ndarray,
    scenario_distances: np.ndarray,
) -> C3PredictionEvidence:
    """Create prediction evidence with a canonical digest."""

    candidate_count = len(candidate_digests)
    score = _readonly_float_vector(
        "predicted_score", predicted_score, size=candidate_count
    )
    mean_advantage = _readonly_float_vector(
        "predicted_mean_advantage", predicted_mean_advantage, size=candidate_count
    )
    loss_cvar = _readonly_float_vector(
        "predicted_loss_cvar", predicted_loss_cvar, size=candidate_count
    )
    expected_turnover = _readonly_float_vector(
        "predicted_expected_turnover",
        predicted_expected_turnover,
        size=candidate_count,
    )
    anchors = _readonly_int_vector("scenario_anchor_indices", scenario_anchor_indices)
    distances = _readonly_float_vector(
        "scenario_distances", scenario_distances, size=anchors.size
    )
    payload = {
        "candidate_digests": candidate_digests,
        "predicted_expected_turnover": _array_payload(expected_turnover),
        "predicted_loss_cvar": _array_payload(loss_cvar),
        "predicted_mean_advantage": _array_payload(mean_advantage),
        "predicted_score": _array_payload(score),
        "result_digest": result_digest,
        "scenario_anchor_indices": _array_payload(anchors),
        "scenario_distances": _array_payload(distances),
        "scenario_library_digest": scenario_library_digest,
        "scenario_set_digest": scenario_set_digest,
        "schema_version": C3_PREDICTION_EVIDENCE_SCHEMA,
    }
    return C3PredictionEvidence(
        result_digest=result_digest,
        scenario_library_digest=scenario_library_digest,
        scenario_set_digest=scenario_set_digest,
        candidate_digests=candidate_digests,
        predicted_score=score,
        predicted_mean_advantage=mean_advantage,
        predicted_loss_cvar=loss_cvar,
        predicted_expected_turnover=expected_turnover,
        scenario_anchor_indices=anchors,
        scenario_distances=distances,
        evidence_digest=content_digest(payload),
    )


def build_c3_prediction_evidence(
    result: CausalScenarioEvaluationResult,
) -> C3PredictionEvidence:
    if not isinstance(result, CausalScenarioEvaluationResult):
        raise TypeError("result must be CausalScenarioEvaluationResult")
    return create_c3_prediction_evidence(
        result_digest=result.result_digest,
        scenario_library_digest=result.scenario_library_digest,
        scenario_set_digest=result.scenario_set_digest,
        candidate_digests=result.candidate_digests,
        predicted_score=result.score,
        predicted_mean_advantage=result.mean_advantage,
        predicted_loss_cvar=result.loss_cvar,
        predicted_expected_turnover=result.expected_filled_turnover,
        scenario_anchor_indices=result.scenario_anchor_indices,
        scenario_distances=result.scenario_distances,
    )


__all__ = [
    "C3_PREDICTION_EVIDENCE_SCHEMA",
    "C3PredictionEvidence",
    "build_c3_prediction_evidence",
    "create_c3_prediction_evidence",
]
