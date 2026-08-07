"""Immutable contracts for causal-scenario C3 evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256, require_unique_non_empty

C3_CONFIG_SCHEMA: Final = "causal_scenario_c3_config_v1"
C3_REPLAY_IDENTITY_SCHEMA: Final = "causal_scenario_c3_replay_identity_v1"
C3_DECISION_SCHEMA: Final = "causal_scenario_c3_decision_v1"
C3_REALIZED_OUTCOME_SCHEMA: Final = "causal_scenario_c3_realized_outcome_v1"
C3_QUERY_COMPARISON_SCHEMA: Final = "causal_scenario_c3_query_comparison_v1"
_DAY_NS: Final = 86_400_000_000_000


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


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
    if shape is not None:
        if len(shape) != array.ndim or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, shape, strict=True)
        ):
            raise ValueError(f"{name} has an invalid shape")
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
    if shape is not None:
        if len(shape) != array.ndim or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, shape, strict=True)
        ):
            raise ValueError(f"{name} has an invalid shape")
    array.setflags(write=False)
    return array


def _array_payload(array: np.ndarray) -> dict[str, object]:
    return {
        "dtype": array.dtype.str,
        "shape": tuple(int(size) for size in array.shape),
        "values": array.tolist(),
    }


@dataclass(frozen=True, slots=True)
class CausalScenarioC3Config:
    horizon_decisions: int = 96
    scenario_count: int = 64
    random_comparator_count: int = 8
    bootstrap_block_days: int = 7
    ranking_tolerance: float = 1e-8
    required_folds: int = 6
    required_selection_days: int = 180
    bootstrap_resamples: int = 1_000
    schema_version: str = C3_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "horizon_decisions",
            "scenario_count",
            "random_comparator_count",
            "bootstrap_block_days",
            "required_folds",
            "required_selection_days",
            "bootstrap_resamples",
        ):
            object.__setattr__(self, field, _positive_int(field, getattr(self, field)))
        tolerance = _finite_float("ranking_tolerance", self.ranking_tolerance)
        if tolerance < 0.0:
            raise ValueError("ranking_tolerance must be non-negative")
        if self.schema_version != C3_CONFIG_SCHEMA:
            raise ValueError("unsupported causal scenario C3 config schema")
        object.__setattr__(self, "ranking_tolerance", tolerance)

    @property
    def policy_order(self) -> tuple[str, ...]:
        return (
            "trend",
            "scenario_oracle",
            "ppo_mean",
            "random_candidate",
            "perfect_information",
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "bootstrap_block_days": self.bootstrap_block_days,
                "bootstrap_resamples": self.bootstrap_resamples,
                "horizon_decisions": self.horizon_decisions,
                "policy_order": self.policy_order,
                "random_comparator_count": self.random_comparator_count,
                "ranking_tolerance": self.ranking_tolerance,
                "required_folds": self.required_folds,
                "required_selection_days": self.required_selection_days,
                "scenario_count": self.scenario_count,
                "schema_version": self.schema_version,
            }
        )


@dataclass(frozen=True, slots=True)
class C3ReplayIdentity:
    dataset_id: str
    fold_digest: str
    environment_digest: str
    action_spec_digest: str
    observation_digest: str
    execution_policy_digest: str
    risk_digest: str
    initial_state_digest: str
    query_index: int
    query_timestamp_ns: int
    realized_stop_index: int
    aum: float
    schema_version: str = C3_REPLAY_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "dataset_id",
            "fold_digest",
            "environment_digest",
            "action_spec_digest",
            "observation_digest",
            "execution_policy_digest",
            "risk_digest",
            "initial_state_digest",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(str(getattr(self, field)), field=field),
            )
        query_index = _non_negative_int("query_index", self.query_index)
        timestamp = _positive_int("query_timestamp_ns", self.query_timestamp_ns)
        stop = _positive_int("realized_stop_index", self.realized_stop_index)
        if stop <= query_index:
            raise ValueError("realized_stop_index must be greater than query_index")
        aum = _finite_float("aum", self.aum)
        if aum <= 0.0:
            raise ValueError("aum must be positive")
        if self.schema_version != C3_REPLAY_IDENTITY_SCHEMA:
            raise ValueError("unsupported C3 replay identity schema")
        object.__setattr__(self, "query_index", query_index)
        object.__setattr__(self, "query_timestamp_ns", timestamp)
        object.__setattr__(self, "realized_stop_index", stop)
        object.__setattr__(self, "aum", aum)

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_spec_digest": self.action_spec_digest,
            "aum": self.aum,
            "dataset_id": self.dataset_id,
            "environment_digest": self.environment_digest,
            "execution_policy_digest": self.execution_policy_digest,
            "fold_digest": self.fold_digest,
            "initial_state_digest": self.initial_state_digest,
            "observation_digest": self.observation_digest,
            "query_index": self.query_index,
            "query_timestamp_ns": self.query_timestamp_ns,
            "realized_stop_index": self.realized_stop_index,
            "risk_digest": self.risk_digest,
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


@dataclass(frozen=True, slots=True)
class PersistedScenarioDecision:
    dataset_id: str
    fold_digest: str
    query_index: int
    query_timestamp_ns: int
    state_snapshot_digest: str
    observation_digest: str
    environment_digest: str
    action_spec_digest: str
    execution_policy_digest: str
    risk_digest: str
    starting_equity: float
    realized_stop_index: int
    scenario_library_digest: str
    scenario_set_digest: str
    candidate_generator_digest: str
    value_result_digest: str
    candidate_digests: tuple[str, ...]
    raw_candidate_actions: np.ndarray
    projected_targets: np.ndarray
    score: np.ndarray
    regret: np.ndarray
    selected_candidate_index: int
    zero_candidate_index: int
    tie_candidate_indices: tuple[int, ...]
    selected_candidate_digest: str
    created_before_realized_replay: bool
    decision_digest: str
    schema_version: str = C3_DECISION_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "dataset_id",
            "fold_digest",
            "state_snapshot_digest",
            "observation_digest",
            "environment_digest",
            "action_spec_digest",
            "execution_policy_digest",
            "risk_digest",
            "scenario_library_digest",
            "scenario_set_digest",
            "candidate_generator_digest",
            "value_result_digest",
            "selected_candidate_digest",
            "decision_digest",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(str(getattr(self, field)), field=field),
            )
        if self.schema_version != C3_DECISION_SCHEMA:
            raise ValueError("unsupported C3 decision schema")
        query_index = _non_negative_int("query_index", self.query_index)
        timestamp = _positive_int("query_timestamp_ns", self.query_timestamp_ns)
        stop = _positive_int("realized_stop_index", self.realized_stop_index)
        if stop <= query_index:
            raise ValueError("realized_stop_index must be greater than query_index")
        equity = _finite_float("starting_equity", self.starting_equity)
        if equity <= 0.0:
            raise ValueError("starting_equity must be positive")
        candidate_digests = tuple(
            require_sha256(value, field="candidate_digests")
            for value in require_unique_non_empty(
                tuple(self.candidate_digests), field="candidate_digests"
            )
        )
        candidate_count = len(candidate_digests)
        raw = _readonly_float_array(
            "raw_candidate_actions",
            self.raw_candidate_actions,
            ndim=2,
            shape=(candidate_count, None),
        )
        projected = _readonly_float_array(
            "projected_targets",
            self.projected_targets,
            ndim=2,
            shape=raw.shape,
        )
        score = _readonly_float_array(
            "score", self.score, ndim=1, shape=(candidate_count,)
        )
        regret = _readonly_float_array(
            "regret", self.regret, ndim=1, shape=(candidate_count,)
        )
        expected_regret = float(score.max()) - score
        if not np.allclose(regret, expected_regret, rtol=0.0, atol=1e-12):
            raise ValueError("regret does not match score")
        selected = _non_negative_int(
            "selected_candidate_index", self.selected_candidate_index
        )
        zero = _non_negative_int("zero_candidate_index", self.zero_candidate_index)
        if selected >= candidate_count or zero >= candidate_count:
            raise ValueError("candidate index is outside candidate range")
        if self.selected_candidate_digest != candidate_digests[selected]:
            raise ValueError(
                "selected_candidate_digest does not match selected candidate"
            )
        if not math.isclose(
            float(score[selected]), float(score.max()), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("selected candidate does not maximize stored score")
        ties = tuple(
            _non_negative_int("tie_candidate_indices", index)
            for index in self.tie_candidate_indices
        )
        if not ties or len(set(ties)) != len(ties):
            raise ValueError("tie_candidate_indices must be unique and non-empty")
        if selected not in ties or any(index >= candidate_count for index in ties):
            raise ValueError("tie_candidate_indices do not include selected candidate")
        if any(
            not math.isclose(
                float(score[index]), float(score.max()), rel_tol=0.0, abs_tol=1e-12
            )
            for index in ties
        ):
            raise ValueError("tie_candidate_indices contain a non-maximum score")
        if not isinstance(self.created_before_realized_replay, bool):
            raise ValueError("created_before_realized_replay must be boolean")
        if not self.created_before_realized_replay:
            raise ValueError("C3 decision must be created before realized replay")
        object.__setattr__(self, "query_index", query_index)
        object.__setattr__(self, "query_timestamp_ns", timestamp)
        object.__setattr__(self, "realized_stop_index", stop)
        object.__setattr__(self, "starting_equity", equity)
        object.__setattr__(self, "candidate_digests", candidate_digests)
        object.__setattr__(self, "raw_candidate_actions", raw)
        object.__setattr__(self, "projected_targets", projected)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "regret", regret)
        object.__setattr__(self, "selected_candidate_index", selected)
        object.__setattr__(self, "zero_candidate_index", zero)
        object.__setattr__(self, "tie_candidate_indices", ties)
        if self.decision_digest != content_digest(self.digest_payload()):
            raise ValueError("decision_digest does not match C3 decision")

    @property
    def replay_identity(self) -> C3ReplayIdentity:
        return C3ReplayIdentity(
            dataset_id=self.dataset_id,
            fold_digest=self.fold_digest,
            environment_digest=self.environment_digest,
            action_spec_digest=self.action_spec_digest,
            observation_digest=self.observation_digest,
            execution_policy_digest=self.execution_policy_digest,
            risk_digest=self.risk_digest,
            initial_state_digest=self.state_snapshot_digest,
            query_index=self.query_index,
            query_timestamp_ns=self.query_timestamp_ns,
            realized_stop_index=self.realized_stop_index,
            aum=self.starting_equity,
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_spec_digest": self.action_spec_digest,
            "candidate_digests": self.candidate_digests,
            "candidate_generator_digest": self.candidate_generator_digest,
            "created_before_realized_replay": self.created_before_realized_replay,
            "dataset_id": self.dataset_id,
            "environment_digest": self.environment_digest,
            "execution_policy_digest": self.execution_policy_digest,
            "fold_digest": self.fold_digest,
            "observation_digest": self.observation_digest,
            "projected_targets": self.projected_targets.tolist(),
            "query_index": self.query_index,
            "query_timestamp_ns": self.query_timestamp_ns,
            "raw_candidate_actions": self.raw_candidate_actions.tolist(),
            "realized_stop_index": self.realized_stop_index,
            "regret": self.regret.tolist(),
            "risk_digest": self.risk_digest,
            "scenario_library_digest": self.scenario_library_digest,
            "scenario_set_digest": self.scenario_set_digest,
            "schema_version": self.schema_version,
            "score": self.score.tolist(),
            "selected_candidate_digest": self.selected_candidate_digest,
            "selected_candidate_index": self.selected_candidate_index,
            "starting_equity": self.starting_equity,
            "state_snapshot_digest": self.state_snapshot_digest,
            "tie_candidate_indices": self.tie_candidate_indices,
            "value_result_digest": self.value_result_digest,
            "zero_candidate_index": self.zero_candidate_index,
        }

    @property
    def selected_raw_residual(self) -> np.ndarray:
        return self.raw_candidate_actions[self.selected_candidate_index]

    @property
    def selected_submitted_target(self) -> np.ndarray:
        return self.projected_targets[self.selected_candidate_index]


class PerfectInformationComparisonStatus(StrEnum):
    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"
    NOT_EVALUATED = "not_evaluated"


class PerfectInformationComparisonReason(StrEnum):
    DOMINANCE_VERIFIED = "dominance_conditions_verified"
    PERIOD_MISMATCH = "period_mismatch"
    RETURN_MATRIX_MISMATCH = "return_matrix_mismatch"
    INITIAL_WEIGHTS_MISMATCH = "initial_weights_mismatch"
    AUM_MISMATCH = "aum_mismatch"
    EXPOSURE_NOT_RELAXED = "exposure_not_relaxed"
    COST_NOT_RELAXED = "cost_not_relaxed"
    BOUND_ORDER_VIOLATION = "bound_order_violation"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True)
class PerfectInformationComparison:
    status: PerfectInformationComparisonStatus
    reason: str
    bound_log_return: float | None
    causal_log_return: float | None
    gap: float | None
    compatibility_evidence_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PerfectInformationComparisonStatus):
            raise ValueError("status must be PerfectInformationComparisonStatus")
        try:
            reason = PerfectInformationComparisonReason(self.reason).value
        except ValueError as error:
            raise ValueError(
                "reason must be a supported Perfect Information reason"
            ) from error
        object.__setattr__(self, "reason", reason)
        evidence_digest = self.compatibility_evidence_digest
        if evidence_digest is not None:
            evidence_digest = require_sha256(
                evidence_digest, field="compatibility_evidence_digest"
            )
            object.__setattr__(self, "compatibility_evidence_digest", evidence_digest)
        if self.status is PerfectInformationComparisonStatus.COMPARABLE:
            if reason != PerfectInformationComparisonReason.DOMINANCE_VERIFIED.value:
                raise ValueError("comparable result requires dominance verification")
            if evidence_digest is None:
                raise ValueError("comparable result requires compatibility evidence")
            if self.bound_log_return is None or self.causal_log_return is None:
                raise ValueError("comparable results require both log returns")
            bound = _finite_float("bound_log_return", self.bound_log_return)
            causal = _finite_float("causal_log_return", self.causal_log_return)
            gap = _finite_float("gap", self.gap)
            if not math.isclose(gap, bound - causal, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("gap does not match comparable log returns")
            object.__setattr__(self, "bound_log_return", bound)
            object.__setattr__(self, "causal_log_return", causal)
            object.__setattr__(self, "gap", gap)
        else:
            if reason == PerfectInformationComparisonReason.DOMINANCE_VERIFIED.value:
                raise ValueError("dominance verification requires comparable status")
            if any(
                value is not None
                for value in (self.bound_log_return, self.causal_log_return, self.gap)
            ):
                raise ValueError(
                    "gap and log returns must be absent when not comparable"
                )
            if (
                self.status is PerfectInformationComparisonStatus.NOT_EVALUATED
                and reason != PerfectInformationComparisonReason.NOT_EVALUATED.value
            ):
                raise ValueError("not-evaluated status requires not_evaluated reason")

    @classmethod
    def comparable(
        cls,
        *,
        bound_log_return: float,
        causal_log_return: float,
        compatibility_evidence_digest: str,
    ) -> PerfectInformationComparison:
        return cls(
            status=PerfectInformationComparisonStatus.COMPARABLE,
            reason=PerfectInformationComparisonReason.DOMINANCE_VERIFIED.value,
            bound_log_return=bound_log_return,
            causal_log_return=causal_log_return,
            gap=float(bound_log_return) - float(causal_log_return),
            compatibility_evidence_digest=compatibility_evidence_digest,
        )

    @classmethod
    def not_comparable(
        cls,
        reason: PerfectInformationComparisonReason | str,
        *,
        compatibility_evidence_digest: str | None = None,
    ) -> PerfectInformationComparison:
        return cls(
            status=PerfectInformationComparisonStatus.NOT_COMPARABLE,
            reason=(
                reason.value
                if isinstance(reason, PerfectInformationComparisonReason)
                else reason
            ),
            bound_log_return=None,
            causal_log_return=None,
            gap=None,
            compatibility_evidence_digest=compatibility_evidence_digest,
        )

    @classmethod
    def not_evaluated(cls) -> PerfectInformationComparison:
        return cls(
            status=PerfectInformationComparisonStatus.NOT_EVALUATED,
            reason=PerfectInformationComparisonReason.NOT_EVALUATED.value,
            bound_log_return=None,
            causal_log_return=None,
            gap=None,
            compatibility_evidence_digest=None,
        )


@dataclass(frozen=True, slots=True)
class RealizedPolicyOutcome:
    policy_kind: str
    gross_log_return: float
    filled_turnover: float
    fees: float
    spread_cost: float
    impact_cost: float
    funding_paid: float
    borrow_paid: float
    fill_ratio: float
    fill_count: int
    pending_order_events: int
    cancel_replace_events: int
    max_drawdown: float
    terminal_equity: float
    termination_reason: str
    outcome_digest: str
    schema_version: str = C3_REALIZED_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        policy_kind = self.policy_kind.strip()
        if not policy_kind:
            raise ValueError("policy_kind must be non-empty")
        object.__setattr__(self, "policy_kind", policy_kind)
        for field in (
            "gross_log_return",
            "filled_turnover",
            "fees",
            "spread_cost",
            "impact_cost",
            "funding_paid",
            "borrow_paid",
            "fill_ratio",
            "max_drawdown",
            "terminal_equity",
        ):
            object.__setattr__(self, field, _finite_float(field, getattr(self, field)))
        for field in (
            "filled_turnover",
            "fees",
            "spread_cost",
            "impact_cost",
            "funding_paid",
            "borrow_paid",
            "max_drawdown",
        ):
            if getattr(self, field) < 0.0:
                raise ValueError(f"{field} must be non-negative")
        if not 0.0 <= self.fill_ratio <= 1.0:
            raise ValueError("fill_ratio must be in [0, 1]")
        if not 0.0 <= self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown must be in [0, 1]")
        if self.terminal_equity <= 0.0:
            raise ValueError("terminal_equity must be positive")
        for field in ("fill_count", "pending_order_events", "cancel_replace_events"):
            object.__setattr__(
                self,
                field,
                _non_negative_int(field, getattr(self, field)),
            )
        reason = self.termination_reason.strip()
        if not reason:
            raise ValueError("termination_reason must be non-empty")
        object.__setattr__(self, "termination_reason", reason)
        if self.schema_version != C3_REALIZED_OUTCOME_SCHEMA:
            raise ValueError("unsupported C3 realized outcome schema")
        digest = require_sha256(self.outcome_digest, field="outcome_digest")
        if digest != content_digest(self.digest_payload()):
            raise ValueError("outcome_digest does not match realized outcome")
        object.__setattr__(self, "outcome_digest", digest)

    @property
    def total_economic_cost(self) -> float:
        return float(
            self.fees
            + self.spread_cost
            + self.impact_cost
            + self.funding_paid
            + self.borrow_paid
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "borrow_paid": self.borrow_paid,
            "cancel_replace_events": self.cancel_replace_events,
            "fees": self.fees,
            "fill_count": self.fill_count,
            "fill_ratio": self.fill_ratio,
            "filled_turnover": self.filled_turnover,
            "funding_paid": self.funding_paid,
            "gross_log_return": self.gross_log_return,
            "impact_cost": self.impact_cost,
            "max_drawdown": self.max_drawdown,
            "pending_order_events": self.pending_order_events,
            "policy_kind": self.policy_kind,
            "schema_version": self.schema_version,
            "spread_cost": self.spread_cost,
            "terminal_equity": self.terminal_equity,
            "termination_reason": self.termination_reason,
        }


@dataclass(frozen=True, slots=True)
class CausalScenarioQueryComparison:
    decision_digest: str
    query_timestamp_ns: int
    replay_identity_digest: str
    execution_scenario: str
    prediction_result_digest: str
    predicted_score: np.ndarray
    predicted_mean_advantage: np.ndarray
    predicted_loss_cvar: np.ndarray
    predicted_expected_turnover: np.ndarray
    scenario_anchor_indices: np.ndarray
    scenario_distances: np.ndarray
    trend: RealizedPolicyOutcome
    scenario_oracle: RealizedPolicyOutcome
    ppo_mean: RealizedPolicyOutcome
    random_candidate: RealizedPolicyOutcome
    random_candidate_indices: tuple[int, ...]
    random_candidate_outcomes: tuple[RealizedPolicyOutcome, ...]
    random_realized_regrets: np.ndarray
    candidate_outcomes: tuple[RealizedPolicyOutcome, ...]
    realized_candidate_advantages: np.ndarray
    predicted_realized_spearman: float
    selected_realized_regret: float
    random_realized_regret: float
    perfect_information: PerfectInformationComparison
    schema_version: str = C3_QUERY_COMPARISON_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_digest",
            require_sha256(self.decision_digest, field="decision_digest"),
        )
        object.__setattr__(
            self,
            "replay_identity_digest",
            require_sha256(self.replay_identity_digest, field="replay_identity_digest"),
        )
        object.__setattr__(
            self,
            "prediction_result_digest",
            require_sha256(
                self.prediction_result_digest, field="prediction_result_digest"
            ),
        )
        object.__setattr__(
            self,
            "query_timestamp_ns",
            _positive_int("query_timestamp_ns", self.query_timestamp_ns),
        )
        execution_scenario = self.execution_scenario.strip()
        if not execution_scenario:
            raise ValueError("execution_scenario must be non-empty")
        object.__setattr__(self, "execution_scenario", execution_scenario)
        if self.schema_version != C3_QUERY_COMPARISON_SCHEMA:
            raise ValueError("unsupported C3 query comparison schema")
        for name in ("trend", "scenario_oracle", "ppo_mean", "random_candidate"):
            if not isinstance(getattr(self, name), RealizedPolicyOutcome):
                raise ValueError(f"{name} must be a realized outcome")
        outcomes = tuple(self.candidate_outcomes)
        if not outcomes or any(
            not isinstance(item, RealizedPolicyOutcome) for item in outcomes
        ):
            raise ValueError("candidate_outcomes must contain realized outcomes")
        candidate_count = len(outcomes)
        advantages = _readonly_float_array(
            "realized_candidate_advantages",
            self.realized_candidate_advantages,
            ndim=1,
            shape=(candidate_count,),
        )
        predicted_arrays = {}
        for name in (
            "predicted_score",
            "predicted_mean_advantage",
            "predicted_loss_cvar",
            "predicted_expected_turnover",
        ):
            predicted_arrays[name] = _readonly_float_array(
                name,
                getattr(self, name),
                ndim=1,
                shape=(candidate_count,),
            )
        if np.any(predicted_arrays["predicted_loss_cvar"] < 0.0):
            raise ValueError("predicted_loss_cvar must be non-negative")
        if np.any(predicted_arrays["predicted_expected_turnover"] < 0.0):
            raise ValueError("predicted_expected_turnover must be non-negative")
        anchors = _readonly_int_array(
            "scenario_anchor_indices", self.scenario_anchor_indices, ndim=1
        )
        distances = _readonly_float_array(
            "scenario_distances",
            self.scenario_distances,
            ndim=1,
            shape=anchors.shape,
        )
        if np.any(anchors < 0):
            raise ValueError("scenario_anchor_indices must be non-negative")
        if np.any(distances < 0.0):
            raise ValueError("scenario_distances must be non-negative")
        random_indices = tuple(
            _non_negative_int("random_candidate_indices", index)
            for index in self.random_candidate_indices
        )
        random_outcomes = tuple(self.random_candidate_outcomes)
        if not random_indices or len(random_indices) != len(random_outcomes):
            raise ValueError("random comparator evidence count mismatch")
        if any(index >= candidate_count for index in random_indices):
            raise ValueError("random comparator index is outside candidate range")
        if any(not isinstance(item, RealizedPolicyOutcome) for item in random_outcomes):
            raise ValueError("random comparators must contain realized outcomes")
        for index, outcome in zip(random_indices, random_outcomes, strict=True):
            if outcome.outcome_digest != outcomes[index].outcome_digest:
                raise ValueError(
                    "random comparator outcome does not match candidate index"
                )
        if self.random_candidate.outcome_digest != random_outcomes[0].outcome_digest:
            raise ValueError("random_candidate must be the first random comparator")
        random_regrets = _readonly_float_array(
            "random_realized_regrets",
            self.random_realized_regrets,
            ndim=1,
            shape=(len(random_indices),),
        )
        if np.any(random_regrets < 0.0):
            raise ValueError("random_realized_regrets must be non-negative")
        object.__setattr__(self, "candidate_outcomes", outcomes)
        object.__setattr__(self, "realized_candidate_advantages", advantages)
        for name, value in predicted_arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "scenario_anchor_indices", anchors)
        object.__setattr__(self, "scenario_distances", distances)
        object.__setattr__(self, "random_candidate_indices", random_indices)
        object.__setattr__(self, "random_candidate_outcomes", random_outcomes)
        object.__setattr__(self, "random_realized_regrets", random_regrets)
        spearman = _finite_float(
            "predicted_realized_spearman", self.predicted_realized_spearman
        )
        if not -1.0 <= spearman <= 1.0:
            raise ValueError("predicted_realized_spearman must be in [-1, 1]")
        object.__setattr__(self, "predicted_realized_spearman", spearman)
        selected_regret = _finite_float(
            "selected_realized_regret", self.selected_realized_regret
        )
        random_regret = _finite_float(
            "random_realized_regret", self.random_realized_regret
        )
        if selected_regret < 0.0 or random_regret < 0.0:
            raise ValueError("realized regrets must be non-negative")
        if not math.isclose(
            random_regret,
            float(random_regrets.mean()),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("random_realized_regret must equal mean random regret")
        object.__setattr__(self, "selected_realized_regret", selected_regret)
        object.__setattr__(self, "random_realized_regret", random_regret)
        if not isinstance(self.perfect_information, PerfectInformationComparison):
            raise ValueError("perfect_information has invalid type")

    @property
    def day_index(self) -> int:
        return self.query_timestamp_ns // _DAY_NS

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "candidate_outcomes": tuple(
                    outcome.outcome_digest for outcome in self.candidate_outcomes
                ),
                "decision_digest": self.decision_digest,
                "execution_scenario": self.execution_scenario,
                "perfect_information": {
                    "bound_log_return": self.perfect_information.bound_log_return,
                    "causal_log_return": self.perfect_information.causal_log_return,
                    "compatibility_evidence_digest": (
                        self.perfect_information.compatibility_evidence_digest
                    ),
                    "gap": self.perfect_information.gap,
                    "reason": self.perfect_information.reason,
                    "status": self.perfect_information.status.value,
                },
                "ppo_mean": self.ppo_mean.outcome_digest,
                "predicted_expected_turnover": _array_payload(
                    self.predicted_expected_turnover
                ),
                "predicted_loss_cvar": _array_payload(self.predicted_loss_cvar),
                "predicted_mean_advantage": _array_payload(
                    self.predicted_mean_advantage
                ),
                "predicted_realized_spearman": self.predicted_realized_spearman,
                "predicted_score": _array_payload(self.predicted_score),
                "prediction_result_digest": self.prediction_result_digest,
                "query_timestamp_ns": self.query_timestamp_ns,
                "random_candidate": self.random_candidate.outcome_digest,
                "random_candidate_indices": self.random_candidate_indices,
                "random_candidate_outcomes": tuple(
                    outcome.outcome_digest for outcome in self.random_candidate_outcomes
                ),
                "random_realized_regret": self.random_realized_regret,
                "random_realized_regrets": _array_payload(self.random_realized_regrets),
                "realized_candidate_advantages": _array_payload(
                    self.realized_candidate_advantages
                ),
                "replay_identity_digest": self.replay_identity_digest,
                "scenario_anchor_indices": _array_payload(self.scenario_anchor_indices),
                "scenario_distances": _array_payload(self.scenario_distances),
                "scenario_oracle": self.scenario_oracle.outcome_digest,
                "schema_version": self.schema_version,
                "selected_realized_regret": self.selected_realized_regret,
                "trend": self.trend.outcome_digest,
            }
        )
