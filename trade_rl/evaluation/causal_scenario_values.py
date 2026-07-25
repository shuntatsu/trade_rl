"""Evaluation-only causal scenario action values."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final, Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_git_sha,
    require_sha256,
    require_unique_non_empty,
)

CAUSAL_SCENARIO_EVALUATOR_SCHEMA: Final = "causal_scenario_action_evaluator_v1"
_PROJECTED_CANDIDATE_SCHEMA: Final = "projected_residual_candidate_v1"
_ROLLOUT_EVIDENCE_SCHEMA: Final = "scenario_rollout_evidence_v1"
_RAW_CANDIDATE_SCHEMA: Final = "raw_residual_candidate_v1"
_CANDIDATE_GENERATOR_SCHEMA: Final = "residual_candidate_generator_v1"


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _match_shape(
    name: str, array: np.ndarray, shape: tuple[int | None, ...] | None
) -> None:
    if shape is None:
        return
    if len(shape) != array.ndim:
        raise ValueError(f"{name} shape contract is invalid")
    if any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has an invalid shape")


def _readonly_float_array(
    name: str,
    value: object,
    *,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).copy(order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    _match_shape(name, array, shape)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array[array == 0.0] = 0.0
    array.setflags(write=False)
    return array


def _readonly_int_array(
    name: str,
    value: object,
    *,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array")
    array = np.asarray(raw, dtype=np.int64).copy(order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    _match_shape(name, array, shape)
    array.setflags(write=False)
    return array


def _readonly_bool_array(
    name: str,
    value: object,
    *,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array")
    array = np.asarray(raw, dtype=np.bool_).copy(order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    _match_shape(name, array, shape)
    array.setflags(write=False)
    return array


def _array_payload(array: np.ndarray) -> dict[str, object]:
    return {
        "dtype": array.dtype.str,
        "shape": tuple(int(size) for size in array.shape),
        "values": array.tolist(),
    }


def _candidate_payload(
    *, projected_target: np.ndarray, execution_intent_digest: str
) -> dict[str, object]:
    return {
        "execution_intent_digest": execution_intent_digest,
        "projected_target": projected_target.tolist(),
        "schema_version": _PROJECTED_CANDIDATE_SCHEMA,
    }


def _rollout_payload(
    *,
    terminal_equity: float,
    reported_log_return: float,
    filled_turnover: float,
    interval_cost: float,
    fill_ratio: float,
    feasible: bool,
    termination_reason: str,
) -> dict[str, object]:
    return {
        "feasible": feasible,
        "fill_ratio": fill_ratio,
        "filled_turnover": filled_turnover,
        "interval_cost": interval_cost,
        "reported_log_return": reported_log_return,
        "schema_version": _ROLLOUT_EVIDENCE_SCHEMA,
        "terminal_equity": terminal_equity,
        "termination_reason": termination_reason,
    }


@dataclass(frozen=True, slots=True)
class CausalScenarioEvaluatorConfig:
    action_dimension: int
    scenario_count: int = 64
    horizon_decisions: int = 96
    cvar_alpha: float = 0.10
    cvar_penalty: float = 0.25
    bootstrap_resamples: int = 256
    confidence_level: float = 0.90
    score_tolerance: float = 1e-8
    max_candidates: int = 32
    replay_tolerance: float = 1e-10
    probability_tolerance: float = 1e-12
    schema_version: str = CAUSAL_SCENARIO_EVALUATOR_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_dimension",
            _positive_int("action_dimension", self.action_dimension),
        )
        object.__setattr__(
            self, "scenario_count", _positive_int("scenario_count", self.scenario_count)
        )
        object.__setattr__(
            self,
            "horizon_decisions",
            _positive_int("horizon_decisions", self.horizon_decisions),
        )
        object.__setattr__(
            self,
            "bootstrap_resamples",
            _positive_int("bootstrap_resamples", self.bootstrap_resamples),
        )
        object.__setattr__(
            self, "max_candidates", _positive_int("max_candidates", self.max_candidates)
        )
        if self.max_candidates > 32:
            raise ValueError("max_candidates must not exceed 32")
        alpha = _finite_float("cvar_alpha", self.cvar_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("cvar_alpha must be in (0, 1]")
        penalty = _finite_float("cvar_penalty", self.cvar_penalty)
        if penalty < 0.0:
            raise ValueError("cvar_penalty must be non-negative")
        confidence = _finite_float("confidence_level", self.confidence_level)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        score_tolerance = _finite_float("score_tolerance", self.score_tolerance)
        if score_tolerance < 0.0:
            raise ValueError("score_tolerance must be non-negative")
        replay_tolerance = _finite_float("replay_tolerance", self.replay_tolerance)
        probability_tolerance = _finite_float(
            "probability_tolerance", self.probability_tolerance
        )
        if replay_tolerance <= 0.0:
            raise ValueError("replay_tolerance must be positive")
        if probability_tolerance <= 0.0:
            raise ValueError("probability_tolerance must be positive")
        if self.schema_version != CAUSAL_SCENARIO_EVALUATOR_SCHEMA:
            raise ValueError("unsupported causal scenario evaluator schema")
        object.__setattr__(self, "cvar_alpha", alpha)
        object.__setattr__(self, "cvar_penalty", penalty)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "score_tolerance", score_tolerance)
        object.__setattr__(self, "replay_tolerance", replay_tolerance)
        object.__setattr__(self, "probability_tolerance", probability_tolerance)

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_dimension": self.action_dimension,
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence_level": self.confidence_level,
            "cvar_alpha": self.cvar_alpha,
            "cvar_penalty": self.cvar_penalty,
            "horizon_decisions": self.horizon_decisions,
            "max_candidates": self.max_candidates,
            "probability_tolerance": self.probability_tolerance,
            "replay_tolerance": self.replay_tolerance,
            "scenario_count": self.scenario_count,
            "schema_version": self.schema_version,
            "score_tolerance": self.score_tolerance,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


@dataclass(frozen=True, slots=True)
class CausalQuerySnapshot:
    dataset_id: str
    fold_digest: str
    train_start: int
    train_stop: int
    query_index: int
    query_timestamp_ns: int
    source_commit: str
    query_digest: str
    state_snapshot_digest: str
    observation_digest: str
    environment_digest: str
    action_spec_digest: str
    execution_policy_digest: str
    risk_digest: str
    trend_digest: str
    starting_equity: float
    baseline_target: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "fold_digest",
            "query_digest",
            "state_snapshot_digest",
            "observation_digest",
            "environment_digest",
            "action_spec_digest",
            "execution_policy_digest",
            "risk_digest",
            "trend_digest",
        ):
            object.__setattr__(
                self, name, require_sha256(str(getattr(self, name)), field=name)
            )
        object.__setattr__(
            self,
            "source_commit",
            require_git_sha(self.source_commit, field="source_commit"),
        )
        start = _non_negative_int("train_start", self.train_start)
        stop = _positive_int("train_stop", self.train_stop)
        if stop <= start:
            raise ValueError("train_stop must be greater than train_start")
        query_index = _non_negative_int("query_index", self.query_index)
        query_timestamp = _positive_int("query_timestamp_ns", self.query_timestamp_ns)
        equity = _finite_float("starting_equity", self.starting_equity)
        if equity <= 0.0:
            raise ValueError("starting_equity must be positive")
        target = _readonly_float_array("baseline_target", self.baseline_target, ndim=1)
        if target.size == 0:
            raise ValueError("baseline_target must not be empty")
        object.__setattr__(self, "train_start", start)
        object.__setattr__(self, "train_stop", stop)
        object.__setattr__(self, "query_index", query_index)
        object.__setattr__(self, "query_timestamp_ns", query_timestamp)
        object.__setattr__(self, "starting_equity", equity)
        object.__setattr__(self, "baseline_target", target)

    @property
    def action_dimension(self) -> int:
        return int(self.baseline_target.shape[0])


@dataclass(frozen=True, slots=True)
class CausalScenarioSet:
    scenario_ids: tuple[str, ...]
    probabilities: np.ndarray
    anchor_indices: np.ndarray
    distances: np.ndarray
    query_condition: np.ndarray
    anchor_conditions: np.ndarray
    library_digest: str

    def __post_init__(self) -> None:
        ids = require_unique_non_empty(tuple(self.scenario_ids), field="scenario_ids")
        count = len(ids)
        probabilities = _readonly_float_array(
            "probabilities", self.probabilities, ndim=1, shape=(count,)
        )
        if np.any(probabilities < 0.0) or not math.isclose(
            float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("probabilities must be non-negative and sum to one")
        expected = 1.0 / count
        if not np.allclose(probabilities, expected, rtol=0.0, atol=1e-12):
            raise ValueError("probabilities must be uniform in version one")
        anchors = _readonly_int_array(
            "anchor_indices", self.anchor_indices, ndim=1, shape=(count,)
        )
        distances = _readonly_float_array(
            "distances", self.distances, ndim=1, shape=(count,)
        )
        if np.any(distances < 0.0):
            raise ValueError("distances must be non-negative")
        query_condition = _readonly_float_array(
            "query_condition", self.query_condition, ndim=1
        )
        anchor_conditions = _readonly_float_array(
            "anchor_conditions",
            self.anchor_conditions,
            ndim=2,
            shape=(count, query_condition.size),
        )
        object.__setattr__(self, "scenario_ids", ids)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "anchor_indices", anchors)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "query_condition", query_condition)
        object.__setattr__(self, "anchor_conditions", anchor_conditions)
        object.__setattr__(
            self,
            "library_digest",
            require_sha256(self.library_digest, field="library_digest"),
        )

    @property
    def scenario_count(self) -> int:
        return len(self.scenario_ids)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "anchor_conditions": _array_payload(self.anchor_conditions),
                "anchor_indices": _array_payload(self.anchor_indices),
                "distances": _array_payload(self.distances),
                "library_digest": self.library_digest,
                "probabilities": _array_payload(self.probabilities),
                "query_condition": _array_payload(self.query_condition),
                "scenario_ids": self.scenario_ids,
                "schema_version": "causal_scenario_set_v1",
            }
        )


@dataclass(frozen=True, slots=True)
class ProjectedResidualCandidate:
    raw_action: np.ndarray
    projected_target: np.ndarray
    execution_intent_digest: str
    candidate_digest: str
    expected_turnover_hint: float
    is_zero: bool

    def __post_init__(self) -> None:
        raw = _readonly_float_array("raw_action", self.raw_action, ndim=1)
        target = _readonly_float_array(
            "projected_target", self.projected_target, ndim=1, shape=(raw.size,)
        )
        execution_digest = require_sha256(
            self.execution_intent_digest, field="execution_intent_digest"
        )
        candidate_digest = require_sha256(
            self.candidate_digest, field="candidate_digest"
        )
        expected = content_digest(
            _candidate_payload(
                projected_target=target, execution_intent_digest=execution_digest
            )
        )
        if candidate_digest != expected:
            raise ValueError("candidate_digest does not match projected candidate")
        turnover = _finite_float("expected_turnover_hint", self.expected_turnover_hint)
        if turnover < 0.0:
            raise ValueError("expected_turnover_hint must be non-negative")
        zero = bool(self.is_zero)
        if zero != bool(np.all(raw == 0.0)):
            raise ValueError("is_zero does not match raw_action")
        object.__setattr__(self, "raw_action", raw)
        object.__setattr__(self, "projected_target", target)
        object.__setattr__(self, "execution_intent_digest", execution_digest)
        object.__setattr__(self, "candidate_digest", candidate_digest)
        object.__setattr__(self, "expected_turnover_hint", turnover)
        object.__setattr__(self, "is_zero", zero)


@dataclass(frozen=True, slots=True)
class ScenarioRolloutEvidence:
    terminal_equity: float
    reported_log_return: float
    filled_turnover: float
    interval_cost: float
    fill_ratio: float
    feasible: bool
    termination_reason: str
    evidence_digest: str

    def __post_init__(self) -> None:
        terminal = _finite_float("terminal_equity", self.terminal_equity)
        log_return = _finite_float("reported_log_return", self.reported_log_return)
        turnover = _finite_float("filled_turnover", self.filled_turnover)
        cost = _finite_float("interval_cost", self.interval_cost)
        ratio = _finite_float("fill_ratio", self.fill_ratio)
        if terminal <= 0.0:
            raise ValueError("terminal_equity must be positive")
        if turnover < 0.0 or cost < 0.0:
            raise ValueError("filled_turnover and interval_cost must be non-negative")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("fill_ratio must be in [0, 1]")
        if not isinstance(self.feasible, bool):
            raise ValueError("feasible must be boolean")
        reason = self.termination_reason.strip()
        if not reason:
            raise ValueError("termination_reason must be non-empty")
        digest = require_sha256(self.evidence_digest, field="evidence_digest")
        expected = content_digest(
            _rollout_payload(
                terminal_equity=terminal,
                reported_log_return=log_return,
                filled_turnover=turnover,
                interval_cost=cost,
                fill_ratio=ratio,
                feasible=self.feasible,
                termination_reason=reason,
            )
        )
        if digest != expected:
            raise ValueError("evidence_digest does not match scenario rollout")
        object.__setattr__(self, "terminal_equity", terminal)
        object.__setattr__(self, "reported_log_return", log_return)
        object.__setattr__(self, "filled_turnover", turnover)
        object.__setattr__(self, "interval_cost", cost)
        object.__setattr__(self, "fill_ratio", ratio)
        object.__setattr__(self, "termination_reason", reason)
        object.__setattr__(self, "evidence_digest", digest)


class ScenarioRollout(Protocol):
    def run(
        self,
        candidate: ProjectedResidualCandidate,
        *,
        horizon_decisions: int,
        zero_residual_after_first: bool,
    ) -> ScenarioRolloutEvidence: ...


class ScenarioRolloutFactory(Protocol):
    def project_candidate(
        self, query: CausalQuerySnapshot, raw_action: np.ndarray
    ) -> ProjectedResidualCandidate: ...

    def create_rollout(
        self, query: CausalQuerySnapshot, scenario_index: int, scenario_id: str
    ) -> ScenarioRollout: ...


def generate_residual_candidates(
    trend_target: np.ndarray,
    *,
    external_actions: Sequence[np.ndarray] = (),
    max_candidates: int = 32,
) -> tuple[np.ndarray, ...]:
    maximum = _positive_int("max_candidates", max_candidates)
    if maximum > 32:
        raise ValueError("max_candidates must not exceed 32")
    target = _readonly_float_array("trend_target", trend_target, ndim=1)
    if target.size == 0:
        raise ValueError("trend_target must not be empty")
    actions: list[np.ndarray] = [np.zeros_like(target)]
    for asset_index in range(target.size):
        for magnitude in (-1.0, -0.5, 0.5, 1.0):
            action = np.zeros_like(target)
            action[asset_index] = magnitude
            actions.append(action)
    for magnitude in (0.5, 1.0):
        actions.append(-np.sign(target) * magnitude)
    for external in external_actions:
        if not isinstance(external, np.ndarray):
            raise ValueError("external actions must be numpy arrays")
        actions.append(external)
    normalized: list[np.ndarray] = []
    seen: set[str] = set()
    for action in actions:
        value = _readonly_float_array(
            "raw_action", action, ndim=1, shape=(target.size,)
        )
        if np.any(np.abs(value) > 1.0):
            raise ValueError("raw_action must be within [-1, 1]")
        digest = content_digest(
            {"raw_action": value.tolist(), "schema_version": _RAW_CANDIDATE_SCHEMA}
        )
        if digest in seen:
            continue
        seen.add(digest)
        normalized.append(value)
    if len(normalized) > maximum:
        raise ValueError("generated candidate count exceeds max_candidates")
    return tuple(normalized)


def _loss_cvar(advantages: np.ndarray, *, alpha: float) -> np.ndarray:
    tail_count = int(math.ceil(alpha * advantages.shape[0]))
    downside_losses = np.maximum(-advantages, 0.0)
    return np.sort(downside_losses, axis=0)[-tail_count:].mean(axis=0)


def _bootstrap_mean_intervals(
    advantages: np.ndarray,
    *,
    query_digest: str,
    config_digest: str,
    resamples: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    seed_digest = content_digest(
        {
            "config_digest": config_digest,
            "query_digest": query_digest,
            "schema_version": "causal_scenario_bootstrap_seed_v1",
        }
    )
    generator = np.random.Generator(np.random.Philox(int(seed_digest[:16], 16)))
    indices = generator.integers(
        0, advantages.shape[0], size=(resamples, advantages.shape[0]), endpoint=False
    )
    means = advantages[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return (
        np.quantile(means, tail, axis=0, method="linear"),
        np.quantile(means, 1.0 - tail, axis=0, method="linear"),
    )


def _candidate_generator_digest(
    query: CausalQuerySnapshot,
    raw_actions: tuple[np.ndarray, ...],
    external_actions: Sequence[np.ndarray],
    max_candidates: int,
) -> str:
    return content_digest(
        {
            "action_dimension": query.action_dimension,
            "baseline_target": query.baseline_target.tolist(),
            "external_actions": [
                np.asarray(action, dtype=np.float64).tolist()
                for action in external_actions
            ],
            "generated_actions": [action.tolist() for action in raw_actions],
            "isolated_magnitudes": (-1.0, -0.5, 0.5, 1.0),
            "max_candidates": max_candidates,
            "reduction_magnitudes": (0.5, 1.0),
            "schema_version": _CANDIDATE_GENERATOR_SCHEMA,
        }
    )


def _project_candidates(
    query: CausalQuerySnapshot,
    raw_actions: tuple[np.ndarray, ...],
    rollout_factory: ScenarioRolloutFactory,
    *,
    max_candidates: int,
) -> tuple[ProjectedResidualCandidate, ...]:
    selected: list[ProjectedResidualCandidate] = []
    positions: dict[str, int] = {}
    for raw in raw_actions:
        candidate = rollout_factory.project_candidate(query, raw)
        if not isinstance(candidate, ProjectedResidualCandidate):
            raise ValueError("project_candidate must return ProjectedResidualCandidate")
        if candidate.raw_action.shape != raw.shape or not np.array_equal(
            candidate.raw_action, raw
        ):
            raise ValueError("projected candidate raw_action mismatch")
        if candidate.projected_target.shape != query.baseline_target.shape:
            raise ValueError("projected candidate target dimension mismatch")
        existing_index = positions.get(candidate.candidate_digest)
        if existing_index is None:
            positions[candidate.candidate_digest] = len(selected)
            selected.append(candidate)
        else:
            existing = selected[existing_index]
            if (
                existing.execution_intent_digest != candidate.execution_intent_digest
                or not np.array_equal(
                    existing.projected_target, candidate.projected_target
                )
            ):
                raise ValueError("candidate digest collision")
    if len(selected) > max_candidates:
        raise ValueError("projected candidate count exceeds max_candidates")
    if sum(candidate.is_zero for candidate in selected) != 1:
        raise ValueError("exactly one zero residual candidate is required")
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class CausalScenarioEvaluationResult:
    config: CausalScenarioEvaluatorConfig
    dataset_id: str
    fold_digest: str
    train_start: int
    train_stop: int
    query_index: int
    query_timestamp_ns: int
    source_commit: str
    query_digest: str
    state_snapshot_digest: str
    observation_digest: str
    environment_digest: str
    action_spec_digest: str
    execution_policy_digest: str
    risk_digest: str
    trend_digest: str
    starting_equity: float
    candidate_generator_digest: str
    scenario_set_digest: str
    scenario_library_digest: str
    scenario_ids: tuple[str, ...]
    candidate_digests: tuple[str, ...]
    execution_intent_digests: tuple[str, ...]
    termination_reasons: tuple[str, ...]
    raw_candidate_actions: np.ndarray
    projected_targets: np.ndarray
    scenario_probabilities: np.ndarray
    scenario_anchor_indices: np.ndarray
    scenario_distances: np.ndarray
    query_condition: np.ndarray
    anchor_conditions: np.ndarray
    terminal_equity: np.ndarray
    gross_log_returns: np.ndarray
    baseline_relative_advantages: np.ndarray
    filled_turnover: np.ndarray
    interval_cost: np.ndarray
    fill_ratio: np.ndarray
    feasible_mask: np.ndarray
    termination_codes: np.ndarray
    mean_advantage: np.ndarray
    loss_cvar: np.ndarray
    score: np.ndarray
    regret: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    expected_filled_turnover: np.ndarray
    selected_candidate_index: int
    zero_candidate_index: int
    tie_candidate_indices: tuple[int, ...]
    result_digest: str
    schema_version: str = CAUSAL_SCENARIO_EVALUATOR_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.config, CausalScenarioEvaluatorConfig):
            raise ValueError("config must be CausalScenarioEvaluatorConfig")
        if self.schema_version != CAUSAL_SCENARIO_EVALUATOR_SCHEMA:
            raise ValueError("unsupported causal scenario result schema")

        for name in (
            "dataset_id",
            "fold_digest",
            "query_digest",
            "state_snapshot_digest",
            "observation_digest",
            "environment_digest",
            "action_spec_digest",
            "execution_policy_digest",
            "risk_digest",
            "trend_digest",
            "candidate_generator_digest",
            "scenario_set_digest",
            "scenario_library_digest",
            "result_digest",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(str(getattr(self, name)), field=name),
            )
        object.__setattr__(
            self,
            "source_commit",
            require_git_sha(self.source_commit, field="source_commit"),
        )

        start = _non_negative_int("train_start", self.train_start)
        stop = _positive_int("train_stop", self.train_stop)
        if stop <= start:
            raise ValueError("train_stop must be greater than train_start")
        query_index = _non_negative_int("query_index", self.query_index)
        query_timestamp = _positive_int("query_timestamp_ns", self.query_timestamp_ns)
        starting_equity = _finite_float("starting_equity", self.starting_equity)
        if starting_equity <= 0.0:
            raise ValueError("starting_equity must be positive")
        object.__setattr__(self, "train_start", start)
        object.__setattr__(self, "train_stop", stop)
        object.__setattr__(self, "query_index", query_index)
        object.__setattr__(self, "query_timestamp_ns", query_timestamp)
        object.__setattr__(self, "starting_equity", starting_equity)

        scenario_count = self.config.scenario_count
        action_dimension = self.config.action_dimension
        scenario_ids = require_unique_non_empty(
            tuple(self.scenario_ids), field="scenario_ids"
        )
        if len(scenario_ids) != scenario_count:
            raise ValueError("scenario metadata count mismatch")
        object.__setattr__(self, "scenario_ids", scenario_ids)

        candidate_digests = tuple(
            require_sha256(value, field="candidate_digests")
            for value in self.candidate_digests
        )
        execution_digests = tuple(
            require_sha256(value, field="execution_intent_digests")
            for value in self.execution_intent_digests
        )
        if len(set(candidate_digests)) != len(candidate_digests):
            raise ValueError("candidate_digests must be unique")
        candidate_count = len(candidate_digests)
        if candidate_count <= 0 or len(execution_digests) != candidate_count:
            raise ValueError("candidate metadata count mismatch")
        object.__setattr__(self, "candidate_digests", candidate_digests)
        object.__setattr__(self, "execution_intent_digests", execution_digests)

        termination_reasons = require_unique_non_empty(
            tuple(self.termination_reasons), field="termination_reasons"
        )
        if tuple(sorted(termination_reasons)) != termination_reasons:
            raise ValueError("termination_reasons must be sorted")
        object.__setattr__(self, "termination_reasons", termination_reasons)

        float_matrices = {
            "raw_candidate_actions": (
                self.raw_candidate_actions,
                (candidate_count, action_dimension),
            ),
            "projected_targets": (
                self.projected_targets,
                (candidate_count, action_dimension),
            ),
            "terminal_equity": (
                self.terminal_equity,
                (scenario_count, candidate_count),
            ),
            "gross_log_returns": (
                self.gross_log_returns,
                (scenario_count, candidate_count),
            ),
            "baseline_relative_advantages": (
                self.baseline_relative_advantages,
                (scenario_count, candidate_count),
            ),
            "filled_turnover": (
                self.filled_turnover,
                (scenario_count, candidate_count),
            ),
            "interval_cost": (
                self.interval_cost,
                (scenario_count, candidate_count),
            ),
            "fill_ratio": (
                self.fill_ratio,
                (scenario_count, candidate_count),
            ),
        }
        for name, (value, shape) in float_matrices.items():
            object.__setattr__(
                self,
                name,
                _readonly_float_array(name, value, ndim=2, shape=shape),
            )

        for name in (
            "scenario_probabilities",
            "scenario_distances",
            "mean_advantage",
            "loss_cvar",
            "score",
            "regret",
            "confidence_lower",
            "confidence_upper",
            "expected_filled_turnover",
        ):
            expected_size = (
                scenario_count if name.startswith("scenario_") else candidate_count
            )
            object.__setattr__(
                self,
                name,
                _readonly_float_array(
                    name,
                    getattr(self, name),
                    ndim=1,
                    shape=(expected_size,),
                ),
            )
        object.__setattr__(
            self,
            "scenario_anchor_indices",
            _readonly_int_array(
                "scenario_anchor_indices",
                self.scenario_anchor_indices,
                ndim=1,
                shape=(scenario_count,),
            ),
        )
        object.__setattr__(
            self,
            "termination_codes",
            _readonly_int_array(
                "termination_codes",
                self.termination_codes,
                ndim=2,
                shape=(scenario_count, candidate_count),
            ),
        )
        object.__setattr__(
            self,
            "feasible_mask",
            _readonly_bool_array(
                "feasible_mask",
                self.feasible_mask,
                ndim=2,
                shape=(scenario_count, candidate_count),
            ),
        )
        query_condition = _readonly_float_array(
            "query_condition", self.query_condition, ndim=1
        )
        object.__setattr__(self, "query_condition", query_condition)
        object.__setattr__(
            self,
            "anchor_conditions",
            _readonly_float_array(
                "anchor_conditions",
                self.anchor_conditions,
                ndim=2,
                shape=(scenario_count, query_condition.size),
            ),
        )

        zero_index = _non_negative_int(
            "zero_candidate_index", self.zero_candidate_index
        )
        selected_index = _non_negative_int(
            "selected_candidate_index", self.selected_candidate_index
        )
        if zero_index >= candidate_count or selected_index >= candidate_count:
            raise ValueError("selected or zero candidate index is invalid")
        object.__setattr__(self, "zero_candidate_index", zero_index)
        object.__setattr__(self, "selected_candidate_index", selected_index)

        tie_indices = tuple(
            _non_negative_int("tie_candidate_indices", value)
            for value in self.tie_candidate_indices
        )
        if tuple(sorted(set(tie_indices))) != tie_indices:
            raise ValueError("tie_candidate_indices must be sorted and unique")
        if any(value >= candidate_count for value in tie_indices):
            raise ValueError("tie_candidate_indices contain an invalid index")
        object.__setattr__(self, "tie_candidate_indices", tie_indices)

        if np.any(np.abs(self.raw_candidate_actions) > 1.0):
            raise ValueError("raw_candidate_actions must be within [-1, 1]")
        zero_rows = np.all(self.raw_candidate_actions == 0.0, axis=1)
        if int(zero_rows.sum()) != 1 or not bool(zero_rows[zero_index]):
            raise ValueError(
                "zero_candidate_index does not identify the unique zero action"
            )
        for index, (target, execution_digest, candidate_digest) in enumerate(
            zip(
                self.projected_targets,
                execution_digests,
                candidate_digests,
                strict=True,
            )
        ):
            expected_digest = content_digest(
                _candidate_payload(
                    projected_target=target,
                    execution_intent_digest=execution_digest,
                )
            )
            if candidate_digest != expected_digest:
                raise ValueError(
                    f"candidate_digests[{index}] does not match projected target"
                )

        probabilities = self.scenario_probabilities
        if np.any(probabilities < 0.0) or not math.isclose(
            float(probabilities.sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=self.config.probability_tolerance,
        ):
            raise ValueError(
                "scenario_probabilities must be non-negative and sum to one"
            )
        expected_probability = 1.0 / scenario_count
        if not np.allclose(
            probabilities,
            expected_probability,
            rtol=0.0,
            atol=self.config.probability_tolerance,
        ):
            raise ValueError("scenario_probabilities must be uniform")
        if np.any(self.scenario_distances < 0.0):
            raise ValueError("scenario_distances must be non-negative")

        expected_scenario_digest = content_digest(
            {
                "anchor_conditions": _array_payload(self.anchor_conditions),
                "anchor_indices": _array_payload(self.scenario_anchor_indices),
                "distances": _array_payload(self.scenario_distances),
                "library_digest": self.scenario_library_digest,
                "probabilities": _array_payload(self.scenario_probabilities),
                "query_condition": _array_payload(self.query_condition),
                "scenario_ids": self.scenario_ids,
                "schema_version": "causal_scenario_set_v1",
            }
        )
        if self.scenario_set_digest != expected_scenario_digest:
            raise ValueError("scenario_set_digest is inconsistent")

        if not np.all(self.feasible_mask):
            raise ValueError("C1 requires every rollout to be feasible")
        if not np.all(self.terminal_equity > 0.0):
            raise ValueError("terminal_equity must be positive")
        if np.any(self.filled_turnover < 0.0) or np.any(self.interval_cost < 0.0):
            raise ValueError("turnover and cost evidence must be non-negative")
        if np.any(self.fill_ratio < 0.0) or np.any(self.fill_ratio > 1.0):
            raise ValueError("fill_ratio evidence must be in [0, 1]")
        if np.any(self.termination_codes < 0) or np.any(
            self.termination_codes >= len(termination_reasons)
        ):
            raise ValueError("termination_codes reference an unknown reason")

        reconstructed_log = np.log(self.terminal_equity / starting_equity)
        if not np.allclose(
            reconstructed_log,
            self.gross_log_returns,
            rtol=0.0,
            atol=self.config.replay_tolerance,
        ):
            raise ValueError("gross_log_returns do not match terminal_equity")
        advantages = (
            self.gross_log_returns - self.gross_log_returns[:, zero_index][:, None]
        )
        if not np.allclose(
            advantages,
            self.baseline_relative_advantages,
            rtol=0.0,
            atol=self.config.replay_tolerance,
        ):
            raise ValueError("baseline_relative_advantages are inconsistent")

        means = advantages.mean(axis=0)
        cvar = _loss_cvar(advantages, alpha=self.config.cvar_alpha)
        scores = means - self.config.cvar_penalty * cvar
        regrets = float(scores.max()) - scores
        expected_turnover = self.filled_turnover.mean(axis=0)
        lower, upper = _bootstrap_mean_intervals(
            advantages,
            query_digest=self.query_digest,
            config_digest=self.config.digest,
            resamples=self.config.bootstrap_resamples,
            confidence_level=self.config.confidence_level,
        )
        expected_statistics = (
            ("mean_advantage", means),
            ("loss_cvar", cvar),
            ("score", scores),
            ("regret", regrets),
            ("expected_filled_turnover", expected_turnover),
            ("confidence_lower", lower),
            ("confidence_upper", upper),
        )
        for statistic_name, statistic_expected in expected_statistics:
            if not np.allclose(
                getattr(self, statistic_name),
                statistic_expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"{statistic_name} is inconsistent")
        max_score = float(scores.max())
        tie = tuple(
            int(index)
            for index in np.flatnonzero(
                max_score - scores <= self.config.score_tolerance
            )
        )
        if tie_indices != tie:
            raise ValueError("tie_candidate_indices are inconsistent")
        selection = min(
            tie,
            key=lambda index: (
                float(expected_turnover[index]),
                float(np.abs(self.raw_candidate_actions[index]).sum()),
                0 if index == zero_index else 1,
                candidate_digests[index],
            ),
        )
        if selected_index != selection:
            raise ValueError("selected_candidate_index is inconsistent")

        expected_result = content_digest(
            self.digest_payload(include_result_digest=False)
        )
        if self.result_digest != expected_result:
            raise ValueError("result_digest is inconsistent")

    def digest_payload(
        self, *, include_result_digest: bool = True
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "action_spec_digest": self.action_spec_digest,
            "anchor_conditions": _array_payload(self.anchor_conditions),
            "baseline_relative_advantages": _array_payload(
                self.baseline_relative_advantages
            ),
            "candidate_digests": self.candidate_digests,
            "candidate_generator_digest": self.candidate_generator_digest,
            "confidence_lower": _array_payload(self.confidence_lower),
            "confidence_upper": _array_payload(self.confidence_upper),
            "config": self.config.digest_payload(),
            "dataset_id": self.dataset_id,
            "environment_digest": self.environment_digest,
            "execution_intent_digests": self.execution_intent_digests,
            "execution_policy_digest": self.execution_policy_digest,
            "feasible_mask": _array_payload(self.feasible_mask),
            "fill_ratio": _array_payload(self.fill_ratio),
            "filled_turnover": _array_payload(self.filled_turnover),
            "fold_digest": self.fold_digest,
            "gross_log_returns": _array_payload(self.gross_log_returns),
            "interval_cost": _array_payload(self.interval_cost),
            "loss_cvar": _array_payload(self.loss_cvar),
            "mean_advantage": _array_payload(self.mean_advantage),
            "observation_digest": self.observation_digest,
            "projected_targets": _array_payload(self.projected_targets),
            "query_condition": _array_payload(self.query_condition),
            "query_digest": self.query_digest,
            "query_index": self.query_index,
            "query_timestamp_ns": self.query_timestamp_ns,
            "raw_candidate_actions": _array_payload(self.raw_candidate_actions),
            "regret": _array_payload(self.regret),
            "risk_digest": self.risk_digest,
            "scenario_anchor_indices": _array_payload(self.scenario_anchor_indices),
            "scenario_distances": _array_payload(self.scenario_distances),
            "scenario_ids": self.scenario_ids,
            "scenario_library_digest": self.scenario_library_digest,
            "scenario_probabilities": _array_payload(self.scenario_probabilities),
            "scenario_set_digest": self.scenario_set_digest,
            "schema_version": self.schema_version,
            "score": _array_payload(self.score),
            "selected_candidate_index": self.selected_candidate_index,
            "source_commit": self.source_commit,
            "starting_equity": self.starting_equity,
            "state_snapshot_digest": self.state_snapshot_digest,
            "terminal_equity": _array_payload(self.terminal_equity),
            "termination_codes": _array_payload(self.termination_codes),
            "termination_reasons": self.termination_reasons,
            "tie_candidate_indices": self.tie_candidate_indices,
            "train_start": self.train_start,
            "train_stop": self.train_stop,
            "trend_digest": self.trend_digest,
            "zero_candidate_index": self.zero_candidate_index,
            "expected_filled_turnover": _array_payload(self.expected_filled_turnover),
        }
        if include_result_digest:
            payload["result_digest"] = self.result_digest
        return payload


def evaluate_causal_scenario_actions(
    *,
    query: CausalQuerySnapshot,
    scenarios: CausalScenarioSet,
    config: CausalScenarioEvaluatorConfig,
    rollout_factory: ScenarioRolloutFactory,
    external_actions: Sequence[np.ndarray] = (),
) -> CausalScenarioEvaluationResult:
    if query.action_dimension != config.action_dimension:
        raise ValueError("query action dimension does not match config")
    if scenarios.scenario_count != config.scenario_count:
        raise ValueError("scenario count does not match config")
    raw_actions = generate_residual_candidates(
        query.baseline_target,
        external_actions=external_actions,
        max_candidates=config.max_candidates,
    )
    candidates = _project_candidates(
        query, raw_actions, rollout_factory, max_candidates=config.max_candidates
    )
    zero_index = next(
        index for index, candidate in enumerate(candidates) if candidate.is_zero
    )
    scenario_count = scenarios.scenario_count
    candidate_count = len(candidates)
    terminal = np.zeros((scenario_count, candidate_count), dtype=np.float64)
    gross = np.zeros_like(terminal)
    turnover = np.zeros_like(terminal)
    cost = np.zeros_like(terminal)
    fill_ratio = np.zeros_like(terminal)
    feasible = np.ones((scenario_count, candidate_count), dtype=np.bool_)
    reasons: list[list[str]] = [
        ["" for _ in candidates] for _ in scenarios.scenario_ids
    ]
    for scenario_index, scenario_id in enumerate(scenarios.scenario_ids):
        for candidate_index, candidate in enumerate(candidates):
            rollout = rollout_factory.create_rollout(query, scenario_index, scenario_id)
            evidence = rollout.run(
                candidate,
                horizon_decisions=config.horizon_decisions,
                zero_residual_after_first=True,
            )
            if not isinstance(evidence, ScenarioRolloutEvidence):
                raise ValueError("rollout must return ScenarioRolloutEvidence")
            if not evidence.feasible:
                raise ValueError("C1 requires every scenario rollout to be feasible")
            replayed = math.log(evidence.terminal_equity / query.starting_equity)
            if not math.isclose(
                replayed,
                evidence.reported_log_return,
                rel_tol=0.0,
                abs_tol=config.replay_tolerance,
            ):
                raise ValueError(
                    "scenario rollout log return does not match terminal equity"
                )
            terminal[scenario_index, candidate_index] = evidence.terminal_equity
            gross[scenario_index, candidate_index] = replayed
            turnover[scenario_index, candidate_index] = evidence.filled_turnover
            cost[scenario_index, candidate_index] = evidence.interval_cost
            fill_ratio[scenario_index, candidate_index] = evidence.fill_ratio
            reasons[scenario_index][candidate_index] = evidence.termination_reason
    vocabulary = tuple(sorted({reason for row in reasons for reason in row}))
    reason_map = {reason: index for index, reason in enumerate(vocabulary)}
    codes = np.asarray(
        [[reason_map[reason] for reason in row] for row in reasons], dtype=np.int64
    )
    advantages = gross - gross[:, zero_index][:, None]
    means = advantages.mean(axis=0)
    cvar = _loss_cvar(advantages, alpha=config.cvar_alpha)
    scores = means - config.cvar_penalty * cvar
    regrets = float(scores.max()) - scores
    expected_turnover = turnover.mean(axis=0)
    lower, upper = _bootstrap_mean_intervals(
        advantages,
        query_digest=query.query_digest,
        config_digest=config.digest,
        resamples=config.bootstrap_resamples,
        confidence_level=config.confidence_level,
    )
    tie = tuple(
        int(index)
        for index in np.flatnonzero(
            float(scores.max()) - scores <= config.score_tolerance
        )
    )
    selected = min(
        tie,
        key=lambda index: (
            float(expected_turnover[index]),
            float(np.abs(candidates[index].raw_action).sum()),
            0 if index == zero_index else 1,
            candidates[index].candidate_digest,
        ),
    )
    result_kwargs: dict[str, object] = {
        "config": config,
        "dataset_id": query.dataset_id,
        "fold_digest": query.fold_digest,
        "train_start": query.train_start,
        "train_stop": query.train_stop,
        "query_index": query.query_index,
        "query_timestamp_ns": query.query_timestamp_ns,
        "source_commit": query.source_commit,
        "query_digest": query.query_digest,
        "state_snapshot_digest": query.state_snapshot_digest,
        "observation_digest": query.observation_digest,
        "environment_digest": query.environment_digest,
        "action_spec_digest": query.action_spec_digest,
        "execution_policy_digest": query.execution_policy_digest,
        "risk_digest": query.risk_digest,
        "trend_digest": query.trend_digest,
        "starting_equity": query.starting_equity,
        "candidate_generator_digest": _candidate_generator_digest(
            query, raw_actions, external_actions, config.max_candidates
        ),
        "scenario_set_digest": scenarios.digest,
        "scenario_library_digest": scenarios.library_digest,
        "scenario_ids": scenarios.scenario_ids,
        "candidate_digests": tuple(
            candidate.candidate_digest for candidate in candidates
        ),
        "execution_intent_digests": tuple(
            candidate.execution_intent_digest for candidate in candidates
        ),
        "termination_reasons": vocabulary,
        "raw_candidate_actions": np.stack(
            [candidate.raw_action for candidate in candidates]
        ),
        "projected_targets": np.stack(
            [candidate.projected_target for candidate in candidates]
        ),
        "scenario_probabilities": scenarios.probabilities,
        "scenario_anchor_indices": scenarios.anchor_indices,
        "scenario_distances": scenarios.distances,
        "query_condition": scenarios.query_condition,
        "anchor_conditions": scenarios.anchor_conditions,
        "terminal_equity": terminal,
        "gross_log_returns": gross,
        "baseline_relative_advantages": advantages,
        "filled_turnover": turnover,
        "interval_cost": cost,
        "fill_ratio": fill_ratio,
        "feasible_mask": feasible,
        "termination_codes": codes,
        "mean_advantage": means,
        "loss_cvar": cvar,
        "score": scores,
        "regret": regrets,
        "confidence_lower": lower,
        "confidence_upper": upper,
        "expected_filled_turnover": expected_turnover,
        "selected_candidate_index": selected,
        "zero_candidate_index": zero_index,
        "tie_candidate_indices": tie,
        "result_digest": "0" * 64,
    }
    payload = _result_digest_payload(result_kwargs, config=config)
    result_digest = content_digest(payload)
    return CausalScenarioEvaluationResult(
        config=config,
        dataset_id=query.dataset_id,
        fold_digest=query.fold_digest,
        train_start=query.train_start,
        train_stop=query.train_stop,
        query_index=query.query_index,
        query_timestamp_ns=query.query_timestamp_ns,
        source_commit=query.source_commit,
        query_digest=query.query_digest,
        state_snapshot_digest=query.state_snapshot_digest,
        observation_digest=query.observation_digest,
        environment_digest=query.environment_digest,
        action_spec_digest=query.action_spec_digest,
        execution_policy_digest=query.execution_policy_digest,
        risk_digest=query.risk_digest,
        trend_digest=query.trend_digest,
        starting_equity=query.starting_equity,
        candidate_generator_digest=str(result_kwargs["candidate_generator_digest"]),
        scenario_set_digest=scenarios.digest,
        scenario_library_digest=scenarios.library_digest,
        scenario_ids=scenarios.scenario_ids,
        candidate_digests=tuple(candidate.candidate_digest for candidate in candidates),
        execution_intent_digests=tuple(
            candidate.execution_intent_digest for candidate in candidates
        ),
        termination_reasons=vocabulary,
        raw_candidate_actions=np.stack(
            [candidate.raw_action for candidate in candidates]
        ),
        projected_targets=np.stack(
            [candidate.projected_target for candidate in candidates]
        ),
        scenario_probabilities=scenarios.probabilities,
        scenario_anchor_indices=scenarios.anchor_indices,
        scenario_distances=scenarios.distances,
        query_condition=scenarios.query_condition,
        anchor_conditions=scenarios.anchor_conditions,
        terminal_equity=terminal,
        gross_log_returns=gross,
        baseline_relative_advantages=advantages,
        filled_turnover=turnover,
        interval_cost=cost,
        fill_ratio=fill_ratio,
        feasible_mask=feasible,
        termination_codes=codes,
        mean_advantage=means,
        loss_cvar=cvar,
        score=scores,
        regret=regrets,
        confidence_lower=lower,
        confidence_upper=upper,
        expected_filled_turnover=expected_turnover,
        selected_candidate_index=selected,
        zero_candidate_index=zero_index,
        tie_candidate_indices=tie,
        result_digest=result_digest,
    )


def _result_digest_payload(
    values: dict[str, object], *, config: CausalScenarioEvaluatorConfig
) -> dict[str, object]:
    array_names = (
        "raw_candidate_actions",
        "projected_targets",
        "scenario_probabilities",
        "scenario_anchor_indices",
        "scenario_distances",
        "query_condition",
        "anchor_conditions",
        "terminal_equity",
        "gross_log_returns",
        "baseline_relative_advantages",
        "filled_turnover",
        "interval_cost",
        "fill_ratio",
        "feasible_mask",
        "termination_codes",
        "mean_advantage",
        "loss_cvar",
        "score",
        "regret",
        "confidence_lower",
        "confidence_upper",
        "expected_filled_turnover",
    )
    payload: dict[str, object] = {
        key: value
        for key, value in values.items()
        if key not in array_names and key not in {"config", "result_digest"}
    }
    payload["config"] = config.digest_payload()
    payload["schema_version"] = CAUSAL_SCENARIO_EVALUATOR_SCHEMA
    for name in array_names:
        payload[name] = _array_payload(np.asarray(values[name]))
    return payload


__all__ = [
    "CAUSAL_SCENARIO_EVALUATOR_SCHEMA",
    "CausalQuerySnapshot",
    "CausalScenarioEvaluationResult",
    "CausalScenarioEvaluatorConfig",
    "CausalScenarioSet",
    "ProjectedResidualCandidate",
    "ScenarioRollout",
    "ScenarioRolloutEvidence",
    "ScenarioRolloutFactory",
    "evaluate_causal_scenario_actions",
    "generate_residual_candidates",
]
