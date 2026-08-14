"""Leakage-explicit evaluation artifacts for oracle teachers and cloned policies.

Oracle targets optimize with knowledge of the complete evaluation path.  Their
performance is therefore a hindsight diagnostic, never a production estimate.
Behavior-cloning performance is reported separately from causal held-out policy
rollouts; action agreement with the oracle remains a hindsight diagnostic.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.hierarchical_bc_metrics import (
    HierarchicalBehaviorCloningMetrics,
)

LEARNING_EVALUATION_SCHEMA: Final = "oracle_behavior_cloning_evaluation_v2"
BEHAVIOR_CLONING_GATE_SCHEMA: Final = "behavior_cloning_gate_evaluation_v2"
ORACLE_DIAGNOSTIC_SCOPE: Final = "hindsight_oracle_diagnostic"
CAUSAL_GENERALIZATION_SCOPE: Final = "causal_policy_generalization"
GATE_STATUSES: Final = frozenset({"passed", "failed", "insufficient_support"})


def _finite_vector(value: object, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a non-empty finite rank-one array")
    return result


def _evaluation_range(value: tuple[int, int], *, field: str) -> tuple[int, int]:
    if (
        len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], int)
        or not isinstance(value[1], int)
        or value[0] < 0
        or value[1] <= value[0] + 1
    ):
        raise ValueError(f"{field} must contain at least one decision")
    return value


def _compounded_return(step_returns: np.ndarray) -> float:
    return float(np.prod(1.0 + step_returns, dtype=np.float64) - 1.0)


def deterministic_bootstrap_upper_bound(
    values: object,
    *,
    confidence_level: float,
    resamples: int,
    seed_material: str,
) -> float:
    """Return a reproducible one-sided bootstrap upper bound for the mean."""

    sample = _finite_vector(values, field="bootstrap values")
    if np.any(sample < 0.0):
        raise ValueError("bootstrap regret values must be non-negative")
    if not math.isfinite(confidence_level) or not 0.5 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence_level must be within (0.5, 1)")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 1_000
    ):
        raise ValueError("bootstrap resamples must be an integer of at least 1000")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("bootstrap seed_material must be non-empty")
    seed = int(content_digest({"seed_material": seed_material})[:16], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(resamples, len(sample)))
    means = sample[indices].mean(axis=1, dtype=np.float64)
    return float(np.quantile(means, confidence_level, method="higher"))


def deterministic_bootstrap_lower_bound(
    values: object,
    *,
    confidence_level: float,
    resamples: int,
    seed_material: str,
) -> float:
    """Return a reproducible one-sided bootstrap lower bound for the mean."""

    sample = _finite_vector(values, field="bootstrap values")
    if not math.isfinite(confidence_level) or not 0.5 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence_level must be within (0.5, 1)")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 1_000
    ):
        raise ValueError("bootstrap resamples must be an integer of at least 1000")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("bootstrap seed_material must be non-empty")
    seed = int(content_digest({"seed_material": seed_material})[:16], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(resamples, len(sample)))
    means = sample[indices].mean(axis=1, dtype=np.float64)
    return float(np.quantile(means, 1.0 - confidence_level, method="lower"))


@dataclass(frozen=True, slots=True)
class ActionPathCollapseEvidence:
    """Explain where submitted target changes disappeared before execution."""

    decision_count: int
    action_dimension_count: int
    active_dimension_count: int
    inactive_dimension_count: int
    proposal_distance_count: int
    submitted_change_count: int
    downstream_no_trade_suppression_count: int
    execution_rejection_count: int
    executed_change_count: int
    trade_count: int
    constant_submitted_actions: bool
    execution_rejection_reason_counts: tuple[tuple[str, int], ...] = ()
    risk_projection_reason_counts: tuple[tuple[str, int], ...] = ()
    hard_risk_violation: bool = False

    def __post_init__(self) -> None:
        if self.decision_count <= 0 or self.action_dimension_count <= 0:
            raise ValueError("action-path evidence dimensions must be positive")
        total_dimensions = self.decision_count * self.action_dimension_count
        if self.active_dimension_count < 0 or self.inactive_dimension_count < 0:
            raise ValueError("action-path active counts must be non-negative")
        if (
            self.active_dimension_count + self.inactive_dimension_count
            != total_dimensions
        ):
            raise ValueError("action-path active counts do not cover all dimensions")
        for name in (
            "proposal_distance_count",
            "submitted_change_count",
            "downstream_no_trade_suppression_count",
            "execution_rejection_count",
            "executed_change_count",
            "trade_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.proposal_distance_count > self.active_dimension_count:
            raise ValueError("proposal distance count exceeds active support")
        for name in (
            "submitted_change_count",
            "downstream_no_trade_suppression_count",
            "executed_change_count",
        ):
            if getattr(self, name) > self.decision_count:
                raise ValueError(f"{name} exceeds decision count")
        if not isinstance(self.constant_submitted_actions, bool):
            raise ValueError("constant_submitted_actions must be a boolean")
        for field in (
            "execution_rejection_reason_counts",
            "risk_projection_reason_counts",
        ):
            raw = tuple(getattr(self, field))
            normalized: list[tuple[str, int]] = []
            seen: set[str] = set()
            for item in raw:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError(f"{field} entries must be reason/count pairs")
                reason, count = item
                if not isinstance(reason, str) or not reason.strip() or reason in seen:
                    raise ValueError(f"{field} reasons must be non-empty and unique")
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"{field} counts must be positive integers")
                seen.add(reason)
                normalized.append((reason, count))
            object.__setattr__(self, field, tuple(sorted(normalized)))
        if (
            sum(count for _, count in self.execution_rejection_reason_counts)
            != self.execution_rejection_count
        ):
            raise ValueError(
                "execution_rejection_reason_counts must match execution_rejection_count"
            )
        if not isinstance(self.hard_risk_violation, bool):
            raise ValueError("hard_risk_violation must be a boolean")

    @property
    def proposal_distance_rate(self) -> float:
        if self.active_dimension_count == 0:
            return 0.0
        return self.proposal_distance_count / self.active_dimension_count

    @property
    def inactive_mask_rate(self) -> float:
        total = self.active_dimension_count + self.inactive_dimension_count
        return 0.0 if total == 0 else self.inactive_dimension_count / total

    @property
    def submitted_change_rate(self) -> float:
        return self.submitted_change_count / self.decision_count

    @property
    def executed_change_rate(self) -> float:
        return self.executed_change_count / self.decision_count

    @property
    def execution_rejection_rate(self) -> float:
        if self.proposal_distance_count == 0:
            return 0.0
        return self.execution_rejection_count / self.proposal_distance_count

    @property
    def downstream_no_trade_suppression_rate(self) -> float:
        if self.submitted_change_count == 0:
            return 0.0
        return self.downstream_no_trade_suppression_count / self.submitted_change_count

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "downstream_no_trade_suppression_rate": (
                self.downstream_no_trade_suppression_rate
            ),
            "executed_change_rate": self.executed_change_rate,
            "execution_rejection_rate": self.execution_rejection_rate,
            "inactive_mask_rate": self.inactive_mask_rate,
            "proposal_distance_rate": self.proposal_distance_rate,
            "submitted_change_rate": self.submitted_change_rate,
        }


@dataclass(frozen=True, slots=True)
class BehaviorCloningGateThresholds:
    """Explicit thresholds for mandatory BC admission evidence."""

    minimum_composed_loss_relative_improvement: float
    minimum_gate_precision: float
    minimum_gate_recall: float
    maximum_active_target_rmse: float
    minimum_activity_ratio: float
    maximum_activity_ratio: float
    minimum_teacher_positive_support: int
    minimum_causal_holdout_trades: int
    maximum_causal_holdout_regret: float
    minimum_causal_holdout_episodes: int = 1
    maximum_causal_holdout_regret_upper_bound: float | None = None
    minimum_causal_holdout_net_return_lower_bound: float = -1.0

    def __post_init__(self) -> None:
        fractions = (
            self.minimum_composed_loss_relative_improvement,
            self.minimum_gate_precision,
            self.minimum_gate_recall,
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions
        ):
            raise ValueError("BC gate fractions must be finite and within [0, 1]")
        if (
            not math.isfinite(self.maximum_active_target_rmse)
            or self.maximum_active_target_rmse < 0.0
        ):
            raise ValueError("maximum_active_target_rmse must be non-negative")
        if (
            not math.isfinite(self.minimum_activity_ratio)
            or not math.isfinite(self.maximum_activity_ratio)
            or self.minimum_activity_ratio < 0.0
            or self.maximum_activity_ratio < self.minimum_activity_ratio
        ):
            raise ValueError("BC activity-ratio bounds are invalid")
        for name in (
            "minimum_teacher_positive_support",
            "minimum_causal_holdout_trades",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not math.isfinite(self.maximum_causal_holdout_regret)
            or self.maximum_causal_holdout_regret < 0.0
        ):
            raise ValueError("maximum_causal_holdout_regret must be non-negative")
        if (
            isinstance(self.minimum_causal_holdout_episodes, bool)
            or not isinstance(self.minimum_causal_holdout_episodes, int)
            or self.minimum_causal_holdout_episodes <= 0
        ):
            raise ValueError("minimum_causal_holdout_episodes must be positive")
        upper = self.maximum_causal_holdout_regret_upper_bound
        if upper is not None and (not math.isfinite(upper) or upper < 0.0):
            raise ValueError(
                "maximum_causal_holdout_regret_upper_bound must be non-negative"
            )
        lower = self.minimum_causal_holdout_net_return_lower_bound
        if not math.isfinite(lower) or lower < -1.0:
            raise ValueError(
                "minimum_causal_holdout_net_return_lower_bound must be finite "
                "and at least -1"
            )


@dataclass(frozen=True, slots=True)
class BehaviorCloningGateMetric:
    """One explainable required gate decision."""

    name: str
    status: str
    observed: float | int | bool | None
    comparison: str
    threshold: float | int | bool | tuple[float, float] | None
    support: int | None
    minimum_support: int | None
    reason: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.status not in GATE_STATUSES or not self.reason:
            raise ValueError("BC gate metric identity is invalid")
        if self.comparison not in {">=", "<=", "==", "between", "is_false"}:
            raise ValueError("BC gate metric comparison is invalid")
        if self.support is not None and self.support < 0:
            raise ValueError("BC gate metric support must be non-negative")
        if self.minimum_support is not None and self.minimum_support < 0:
            raise ValueError("BC gate minimum support must be non-negative")
        if not isinstance(self.required, bool) or not self.required:
            raise ValueError("BC gate metrics in v2 are mandatory")

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True, slots=True)
class BehaviorCloningGateGroup:
    """Mandatory teacher-reconstruction or causal non-collapse gate group."""

    name: str
    metrics: tuple[BehaviorCloningGateMetric, ...]
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.metrics or not self.required:
            raise ValueError("BC gate group must be named, required, and non-empty")
        names = tuple(metric.name for metric in self.metrics)
        if len(set(names)) != len(names):
            raise ValueError("BC gate metric names must be unique within a group")

    @property
    def passed(self) -> bool:
        return all(metric.passed for metric in self.metrics)

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": [asdict(metric) for metric in self.metrics],
            "name": self.name,
            "passed": self.passed,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class BehaviorCloningGateEvaluation:
    """Separate mandatory reconstruction and causal non-collapse decisions."""

    teacher_reconstruction_gate: BehaviorCloningGateGroup
    causal_non_collapse_gate: BehaviorCloningGateGroup
    schema_version: str = BEHAVIOR_CLONING_GATE_SCHEMA

    def __post_init__(self) -> None:
        if self.teacher_reconstruction_gate.name != "teacher_reconstruction_gate":
            raise ValueError("teacher reconstruction gate name is invalid")
        if self.causal_non_collapse_gate.name != "causal_non_collapse_gate":
            raise ValueError("causal non-collapse gate name is invalid")
        if self.schema_version != BEHAVIOR_CLONING_GATE_SCHEMA:
            raise ValueError("unsupported behavior cloning gate schema")

    @property
    def passed(self) -> bool:
        return (
            self.teacher_reconstruction_gate.passed
            and self.causal_non_collapse_gate.passed
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "causal_non_collapse_gate": self.causal_non_collapse_gate.to_dict(),
            "passed": self.passed,
            "schema_version": self.schema_version,
            "teacher_reconstruction_gate": self.teacher_reconstruction_gate.to_dict(),
        }

    def require_passed(self) -> None:
        if self.passed:
            return
        for group in (
            self.teacher_reconstruction_gate,
            self.causal_non_collapse_gate,
        ):
            for metric in group.metrics:
                if not metric.passed:
                    raise RuntimeError(
                        f"behavior cloning failed {group.name}: {metric.reason}"
                    )
        raise RuntimeError("behavior cloning failed an unknown mandatory gate")


@dataclass(frozen=True, slots=True)
class PathPerformanceMetrics:
    """Execution-path metrics with closed trades distinct from traded steps."""

    step_count: int
    traded_step_count: int
    trade_count: int
    gross_return: float
    net_return: float
    reward_total: float
    reward_mean: float
    trade_win_rate: float | None
    positive_step_rate: float
    turnover_total: float
    turnover_mean: float
    cost_total: float
    cost_mean: float
    maximum_drawdown: float

    def __post_init__(self) -> None:
        if (
            self.step_count <= 0
            or not 0 <= self.traded_step_count <= self.step_count
            or self.trade_count < 0
        ):
            raise ValueError("path metric counts are invalid")
        numeric = (
            self.gross_return,
            self.net_return,
            self.reward_total,
            self.reward_mean,
            self.positive_step_rate,
            self.turnover_total,
            self.turnover_mean,
            self.cost_total,
            self.cost_mean,
            self.maximum_drawdown,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("path metrics must be finite")
        if self.trade_win_rate is not None and not (
            math.isfinite(self.trade_win_rate) and 0.0 <= self.trade_win_rate <= 1.0
        ):
            raise ValueError("trade_win_rate must be null or within [0, 1]")


def evaluate_path_performance(
    *,
    gross_step_returns: object,
    net_step_returns: object,
    rewards: object,
    turnover: object,
    costs: object,
    closed_trade_pnls: object | None = None,
    closed_trade_count: int | None = None,
    winning_trade_count: int | None = None,
    trade_epsilon: float = 1e-12,
) -> PathPerformanceMetrics:
    """Summarize an already executed path without reconstructing accounting.

    Gross and net returns are deliberately supplied independently so the caller
    preserves the exact simulator accounting contract (fees, funding, borrow,
    slippage, and other costs need not be approximated here).
    """

    gross = _finite_vector(gross_step_returns, field="gross_step_returns")
    net = _finite_vector(net_step_returns, field="net_step_returns")
    reward = _finite_vector(rewards, field="rewards")
    traded_turnover = _finite_vector(turnover, field="turnover")
    paid_cost = _finite_vector(costs, field="costs")
    if (
        len({len(gross), len(net), len(reward), len(traded_turnover), len(paid_cost)})
        != 1
    ):
        raise ValueError("path evaluation arrays must have equal lengths")
    if np.any(gross <= -1.0) or np.any(net <= -1.0):
        raise ValueError("step returns must be greater than -1")
    if np.any(traded_turnover < 0.0) or np.any(paid_cost < 0.0):
        raise ValueError("turnover and costs must be non-negative")
    if not math.isfinite(trade_epsilon) or trade_epsilon < 0.0:
        raise ValueError("trade_epsilon must be finite and non-negative")

    traded_step_mask = traded_turnover > trade_epsilon
    if closed_trade_pnls is not None:
        if closed_trade_count is not None or winning_trade_count is not None:
            raise ValueError(
                "closed trade PnLs and aggregate counts are mutually exclusive"
            )
        trade_pnls = np.asarray(closed_trade_pnls, dtype=np.float64)
        if trade_pnls.ndim != 1 or not np.isfinite(trade_pnls).all():
            raise ValueError("closed_trade_pnls must be a finite rank-one array")
        resolved_trade_count = len(trade_pnls)
        resolved_winning_count = int(np.count_nonzero(trade_pnls > 0.0))
    else:
        if closed_trade_count is None and winning_trade_count is None:
            resolved_trade_count = 0
            resolved_winning_count = 0
        elif (
            isinstance(closed_trade_count, bool)
            or isinstance(winning_trade_count, bool)
            or not isinstance(closed_trade_count, int)
            or not isinstance(winning_trade_count, int)
            or closed_trade_count < 0
            or not 0 <= winning_trade_count <= closed_trade_count
        ):
            raise ValueError(
                "closed/winning trade counts must be reconciled non-negative integers"
            )
        else:
            resolved_trade_count = closed_trade_count
            resolved_winning_count = winning_trade_count
    equity = np.concatenate(
        (np.ones(1, dtype=np.float64), np.cumprod(1.0 + net, dtype=np.float64))
    )
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / peak
    return PathPerformanceMetrics(
        step_count=len(net),
        traded_step_count=int(np.count_nonzero(traded_step_mask)),
        trade_count=resolved_trade_count,
        gross_return=_compounded_return(gross),
        net_return=_compounded_return(net),
        reward_total=float(np.sum(reward, dtype=np.float64)),
        reward_mean=float(np.mean(reward, dtype=np.float64)),
        trade_win_rate=(
            None
            if resolved_trade_count == 0
            else resolved_winning_count / resolved_trade_count
        ),
        positive_step_rate=float(np.mean(net > 0.0, dtype=np.float64)),
        turnover_total=float(np.sum(traded_turnover, dtype=np.float64)),
        turnover_mean=float(np.mean(traded_turnover, dtype=np.float64)),
        cost_total=float(np.sum(paid_cost, dtype=np.float64)),
        cost_mean=float(np.mean(paid_cost, dtype=np.float64)),
        maximum_drawdown=float(np.max(drawdown)),
    )


@dataclass(frozen=True, slots=True)
class OracleTeacherEvaluation:
    evaluation_range: tuple[int, int]
    performance: PathPerformanceMetrics
    scope: str = ORACLE_DIAGNOSTIC_SCOPE
    uses_future_information: bool = True
    eligible_for_production_generalization: bool = False
    schema_version: str = LEARNING_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        start, stop = _evaluation_range(self.evaluation_range, field="evaluation_range")
        if self.performance.step_count != stop - start - 1:
            raise ValueError("oracle metrics do not cover the exact evaluation range")
        if (
            self.scope != ORACLE_DIAGNOSTIC_SCOPE
            or not self.uses_future_information
            or self.eligible_for_production_generalization
        ):
            raise ValueError(
                "oracle evaluation must remain a hindsight-only diagnostic"
            )
        if self.schema_version != LEARNING_EVALUATION_SCHEMA:
            raise ValueError("unsupported learning evaluation schema")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BehaviorCloningHoldoutEvaluation:
    teacher_train_range: tuple[int, int]
    heldout_range: tuple[int, int]
    oracle_reference: OracleTeacherEvaluation
    causal_policy_performance: PathPerformanceMetrics
    action_agreement_rate: float
    action_mae: float
    action_rmse: float
    oracle_minus_policy_gross_return: float
    oracle_minus_policy_net_return: float
    heldout_oracle_regret: float
    oracle_minus_policy_reward: float
    causal_policy_evidence: ActionPathCollapseEvidence | None = None
    policy_scope: str = CAUSAL_GENERALIZATION_SCOPE
    policy_uses_future_information: bool = False
    oracle_comparison_is_production_evidence: bool = False
    schema_version: str = LEARNING_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        train_start, train_stop = _evaluation_range(
            self.teacher_train_range, field="teacher_train_range"
        )
        heldout_start, heldout_stop = _evaluation_range(
            self.heldout_range, field="heldout_range"
        )
        del train_start
        # Ranges are bar-half-open and contain decisions [start, stop - 1).
        # Adjacent decision partitions therefore share the boundary bar:
        # train=(start, d + 1), heldout=(d, stop).
        if heldout_start < train_stop - 1:
            raise ValueError(
                "heldout decisions must not overlap teacher training decisions"
            )
        if self.oracle_reference.evaluation_range != self.heldout_range:
            raise ValueError("oracle reference must cover the exact heldout range")
        if (
            self.causal_policy_performance.step_count
            != heldout_stop - heldout_start - 1
        ):
            raise ValueError("policy metrics do not cover the exact heldout range")
        for field, value in (
            ("action_agreement_rate", self.action_agreement_rate),
            ("action_mae", self.action_mae),
            ("action_rmse", self.action_rmse),
            ("oracle_minus_policy_gross_return", self.oracle_minus_policy_gross_return),
            ("oracle_minus_policy_net_return", self.oracle_minus_policy_net_return),
            ("heldout_oracle_regret", self.heldout_oracle_regret),
            ("oracle_minus_policy_reward", self.oracle_minus_policy_reward),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if not 0.0 <= self.action_agreement_rate <= 1.0:
            raise ValueError("action_agreement_rate must be within [0, 1]")
        if (
            self.action_mae < 0.0
            or self.action_rmse < 0.0
            or self.heldout_oracle_regret < 0.0
        ):
            raise ValueError("action errors and oracle regret must be non-negative")
        if self.causal_policy_evidence is not None:
            evidence = self.causal_policy_evidence
            if evidence.decision_count != self.causal_policy_performance.step_count:
                raise ValueError("causal collapse evidence does not cover holdout")
            if evidence.trade_count != self.causal_policy_performance.trade_count:
                raise ValueError("causal collapse evidence trade count mismatch")
            if (
                evidence.executed_change_count
                != self.causal_policy_performance.traded_step_count
            ):
                raise ValueError("causal collapse evidence execution mismatch")
        if (
            self.policy_scope != CAUSAL_GENERALIZATION_SCOPE
            or self.policy_uses_future_information
            or self.oracle_comparison_is_production_evidence
        ):
            raise ValueError("BC holdout scope must remain causally explicit")
        if self.schema_version != LEARNING_EVALUATION_SCHEMA:
            raise ValueError("unsupported learning evaluation schema")

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["causal_policy_evidence"] = (
            None
            if self.causal_policy_evidence is None
            else self.causal_policy_evidence.to_dict()
        )
        return payload


def evaluate_behavior_cloning_holdout(
    *,
    teacher_train_range: tuple[int, int],
    heldout_range: tuple[int, int],
    oracle_actions: object,
    policy_actions: object,
    oracle_performance: PathPerformanceMetrics,
    causal_policy_performance: PathPerformanceMetrics,
    causal_policy_evidence: ActionPathCollapseEvidence | None = None,
    action_tolerance: float = 0.05,
) -> BehaviorCloningHoldoutEvaluation:
    """Compare causal held-out BC actions with a hindsight oracle reference.

    The policy rollout must use only observations available at each decision.
    Oracle actions may use the full held-out path, so agreement and regret are
    diagnostic ceilings and are explicitly excluded from production evidence.
    """

    oracle = np.asarray(oracle_actions, dtype=np.float64)
    policy = np.asarray(policy_actions, dtype=np.float64)
    if (
        oracle.ndim != 2
        or policy.shape != oracle.shape
        or oracle.size == 0
        or not np.isfinite(oracle).all()
        or not np.isfinite(policy).all()
    ):
        raise ValueError(
            "oracle and policy actions must be equal finite rank-two arrays"
        )
    start, stop = _evaluation_range(heldout_range, field="heldout_range")
    expected_count = stop - start - 1
    if len(oracle) != expected_count:
        raise ValueError("actions do not cover the exact heldout range")
    if not math.isfinite(action_tolerance) or action_tolerance < 0.0:
        raise ValueError("action_tolerance must be finite and non-negative")

    difference = oracle - policy
    oracle_evaluation = OracleTeacherEvaluation(
        evaluation_range=heldout_range,
        performance=oracle_performance,
    )
    gross_gap = oracle_performance.gross_return - causal_policy_performance.gross_return
    net_gap = oracle_performance.net_return - causal_policy_performance.net_return
    reward_gap = (
        oracle_performance.reward_total - causal_policy_performance.reward_total
    )
    return BehaviorCloningHoldoutEvaluation(
        teacher_train_range=teacher_train_range,
        heldout_range=heldout_range,
        oracle_reference=oracle_evaluation,
        causal_policy_performance=causal_policy_performance,
        action_agreement_rate=float(
            np.mean(np.all(np.abs(difference) <= action_tolerance, axis=1))
        ),
        action_mae=float(np.mean(np.abs(difference), dtype=np.float64)),
        action_rmse=float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
        oracle_minus_policy_gross_return=float(gross_gap),
        oracle_minus_policy_net_return=float(net_gap),
        heldout_oracle_regret=float(max(0.0, net_gap)),
        oracle_minus_policy_reward=float(reward_gap),
        causal_policy_evidence=causal_policy_evidence,
    )


def _gate_metric(
    *,
    name: str,
    observed: float | int | bool | None,
    comparison: str,
    threshold: float | int | bool | tuple[float, float] | None,
    support: int | None,
    minimum_support: int | None,
    passed: bool,
    failure_reason: str,
) -> BehaviorCloningGateMetric:
    if minimum_support is not None and (support is None or support < minimum_support):
        status = "insufficient_support"
        reason = (
            f"{name} has support {support}; minimum required support is "
            f"{minimum_support}"
        )
    elif observed is None:
        status = "insufficient_support"
        reason = f"{name} is unavailable"
    elif passed:
        status = "passed"
        reason = f"{name} passed"
    else:
        status = "failed"
        reason = failure_reason
    return BehaviorCloningGateMetric(
        name=name,
        status=status,
        observed=observed,
        comparison=comparison,
        threshold=threshold,
        support=support,
        minimum_support=minimum_support,
        reason=reason,
    )


def evaluate_behavior_cloning_gates(
    *,
    initial_composed_loss: float | None,
    final_composed_loss: float | None,
    reconstruction_metrics: HierarchicalBehaviorCloningMetrics | None,
    holdout: BehaviorCloningHoldoutEvaluation | None,
    thresholds: BehaviorCloningGateThresholds,
) -> BehaviorCloningGateEvaluation:
    """Build two required gate groups without hindsight agreement as evidence."""

    composed_improvement: float | None = None
    if (
        initial_composed_loss is not None
        and final_composed_loss is not None
        and math.isfinite(initial_composed_loss)
        and math.isfinite(final_composed_loss)
        and initial_composed_loss >= 0.0
        and final_composed_loss >= 0.0
    ):
        composed_improvement = (initial_composed_loss - final_composed_loss) / max(
            initial_composed_loss, float(np.finfo(np.float64).eps)
        )
    support = (
        None
        if reconstruction_metrics is None
        else reconstruction_metrics.positive_support
    )
    active_target_rmse = (
        None
        if reconstruction_metrics is None
        else reconstruction_metrics.active_target_rmse
    )
    activity_ratio = (
        None
        if reconstruction_metrics is None
        else reconstruction_metrics.activity_ratio
    )
    teacher_metrics = (
        _gate_metric(
            name="composed_loss_relative_improvement",
            observed=composed_improvement,
            comparison=">=",
            threshold=thresholds.minimum_composed_loss_relative_improvement,
            support=support,
            minimum_support=thresholds.minimum_teacher_positive_support,
            passed=(
                composed_improvement is not None
                and composed_improvement
                >= thresholds.minimum_composed_loss_relative_improvement
            ),
            failure_reason="composed-loss improvement is below the required threshold",
        ),
        _gate_metric(
            name="gate_precision",
            observed=None
            if reconstruction_metrics is None
            else reconstruction_metrics.gate_precision,
            comparison=">=",
            threshold=thresholds.minimum_gate_precision,
            support=(
                None
                if reconstruction_metrics is None
                else reconstruction_metrics.predicted_positive_support
            ),
            minimum_support=thresholds.minimum_teacher_positive_support,
            passed=(
                reconstruction_metrics is not None
                and reconstruction_metrics.gate_precision
                >= thresholds.minimum_gate_precision
            ),
            failure_reason="gate precision is below the required threshold",
        ),
        _gate_metric(
            name="gate_recall",
            observed=None
            if reconstruction_metrics is None
            else reconstruction_metrics.gate_recall,
            comparison=">=",
            threshold=thresholds.minimum_gate_recall,
            support=support,
            minimum_support=thresholds.minimum_teacher_positive_support,
            passed=(
                reconstruction_metrics is not None
                and reconstruction_metrics.gate_recall >= thresholds.minimum_gate_recall
            ),
            failure_reason="gate recall is below the required threshold",
        ),
        _gate_metric(
            name="active_target_rmse",
            observed=active_target_rmse,
            comparison="<=",
            threshold=thresholds.maximum_active_target_rmse,
            support=support,
            minimum_support=thresholds.minimum_teacher_positive_support,
            passed=(
                active_target_rmse is not None
                and active_target_rmse <= thresholds.maximum_active_target_rmse
            ),
            failure_reason="active target RMSE exceeds the required maximum",
        ),
        _gate_metric(
            name="activity_ratio",
            observed=activity_ratio,
            comparison="between",
            threshold=(
                thresholds.minimum_activity_ratio,
                thresholds.maximum_activity_ratio,
            ),
            support=support,
            minimum_support=thresholds.minimum_teacher_positive_support,
            passed=(
                activity_ratio is not None
                and thresholds.minimum_activity_ratio
                <= activity_ratio
                <= thresholds.maximum_activity_ratio
            ),
            failure_reason=(
                "policy/teacher activity ratio is outside the required bounds "
                f"[{thresholds.minimum_activity_ratio}, "
                f"{thresholds.maximum_activity_ratio}]"
            ),
        ),
        *tuple(
            _gate_metric(
                name=name,
                observed=None if reconstruction_metrics is None else value,
                comparison="is_false",
                threshold=False,
                support=support,
                minimum_support=thresholds.minimum_teacher_positive_support,
                passed=reconstruction_metrics is not None and not value,
                failure_reason=f"{name.replace('_', '-')} detected",
            )
            for name, value in (
                (
                    "constant_action_collapse",
                    False
                    if reconstruction_metrics is None
                    else reconstruction_metrics.constant_action_collapse,
                ),
                (
                    "all_hold_collapse",
                    False
                    if reconstruction_metrics is None
                    else reconstruction_metrics.all_hold_collapse,
                ),
                (
                    "all_trade_collapse",
                    False
                    if reconstruction_metrics is None
                    else reconstruction_metrics.all_trade_collapse,
                ),
            )
        ),
    )
    evidence = None if holdout is None else holdout.causal_policy_evidence
    causal_support = support
    executed_changes = None if evidence is None else evidence.executed_change_count
    submitted_changes = None if evidence is None else evidence.submitted_change_count
    constant_actions = None if evidence is None else evidence.constant_submitted_actions
    causal_regret = (
        None
        if holdout is None
        else max(0.0, -holdout.causal_policy_performance.net_return)
    )
    causal_records = () if holdout is None else tuple(getattr(holdout, "records", ()))
    causal_episode_support = (
        len(causal_records) if causal_records else (0 if holdout is None else 1)
    )
    causal_net_return_lower = (
        None
        if holdout is None
        else float(
            getattr(
                holdout,
                "causal_net_return_lower_confidence_bound",
                holdout.causal_policy_performance.net_return,
            )
        )
    )
    causal_regret_upper = None
    if holdout is not None and causal_records:
        cash_regrets = np.asarray(
            [
                max(0.0, -record.causal_policy_performance.net_return)
                for record in causal_records
            ],
            dtype=np.float64,
        )
        causal_regret_upper = deterministic_bootstrap_upper_bound(
            cash_regrets,
            confidence_level=float(
                getattr(holdout, "bootstrap_confidence_level", 0.95)
            ),
            resamples=int(getattr(holdout, "bootstrap_resamples", 2_000)),
            seed_material=content_digest(
                {
                    "scope": "causal_cash_regret",
                    "values": cash_regrets.tolist(),
                }
            ),
        )
    elif holdout is not None:
        causal_regret_upper = getattr(holdout, "heldout_oracle_regret", None)
    upper_threshold = (
        thresholds.maximum_causal_holdout_regret
        if thresholds.maximum_causal_holdout_regret_upper_bound is None
        else thresholds.maximum_causal_holdout_regret_upper_bound
    )
    zero_trade_failure = (
        "zero-trade collapse: causal holdout executed no target changes despite "
        "nonzero teacher change support"
    )
    causal_metrics = (
        _gate_metric(
            name="executed_change_count",
            observed=executed_changes,
            comparison=">=",
            threshold=thresholds.minimum_causal_holdout_trades,
            support=causal_support,
            minimum_support=thresholds.minimum_teacher_positive_support,
            passed=(
                executed_changes is not None
                and executed_changes >= thresholds.minimum_causal_holdout_trades
            ),
            failure_reason=(
                zero_trade_failure
                if executed_changes == 0
                else "causal holdout executed-change count is below the minimum"
            ),
        ),
        _gate_metric(
            name="submitted_change_count",
            observed=submitted_changes,
            comparison=">=",
            threshold=1,
            support=causal_support,
            minimum_support=thresholds.minimum_teacher_positive_support,
            passed=submitted_changes is not None and submitted_changes >= 1,
            failure_reason="constant submitted actions caused zero-change collapse",
        ),
        _gate_metric(
            name="constant_submitted_actions",
            observed=constant_actions,
            comparison="is_false",
            threshold=False,
            support=causal_support,
            minimum_support=thresholds.minimum_teacher_positive_support,
            passed=constant_actions is not None and not constant_actions,
            failure_reason="constant submitted actions detected on causal holdout",
        ),
        _gate_metric(
            name="causal_regret_upper_confidence_bound",
            observed=causal_regret_upper,
            comparison="<=",
            threshold=upper_threshold,
            support=causal_episode_support,
            minimum_support=thresholds.minimum_causal_holdout_episodes,
            passed=(
                causal_regret_upper is not None
                and causal_regret_upper <= upper_threshold
            ),
            failure_reason=(
                "causal holdout regret upper confidence bound exceeds the limit"
            ),
        ),
        _gate_metric(
            name="causal_net_return_lower_confidence_bound",
            observed=causal_net_return_lower,
            comparison=">=",
            threshold=(thresholds.minimum_causal_holdout_net_return_lower_bound),
            support=causal_episode_support,
            minimum_support=thresholds.minimum_causal_holdout_episodes,
            passed=(
                causal_net_return_lower is not None
                and causal_net_return_lower
                >= thresholds.minimum_causal_holdout_net_return_lower_bound
            ),
            failure_reason=(
                "causal after-cost net-return lower confidence bound is below "
                "the required floor"
            ),
        ),
        _gate_metric(
            name="cash_baseline_after_cost_regret",
            observed=causal_regret,
            comparison="<=",
            threshold=thresholds.maximum_causal_holdout_regret,
            support=None if evidence is None else evidence.decision_count,
            minimum_support=1,
            passed=(
                causal_regret is not None
                and causal_regret <= thresholds.maximum_causal_holdout_regret
            ),
            failure_reason="causal after-cost result breaches catastrophic regret limit",
        ),
    )
    return BehaviorCloningGateEvaluation(
        teacher_reconstruction_gate=BehaviorCloningGateGroup(
            name="teacher_reconstruction_gate",
            metrics=teacher_metrics,
        ),
        causal_non_collapse_gate=BehaviorCloningGateGroup(
            name="causal_non_collapse_gate",
            metrics=causal_metrics,
        ),
    )


def write_learning_evaluation(
    path: str | Path,
    evaluation: (
        OracleTeacherEvaluation
        | BehaviorCloningHoldoutEvaluation
        | BehaviorCloningGateEvaluation
    ),
) -> str:
    """Atomically write canonical JSON and return its content digest."""

    output = Path(path)
    payload: Mapping[str, object] = evaluation.to_dict()
    digest = content_digest(payload)
    document = {**payload, "artifact_digest": digest}
    encoded = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    return digest


__all__ = [
    "BEHAVIOR_CLONING_GATE_SCHEMA",
    "CAUSAL_GENERALIZATION_SCOPE",
    "LEARNING_EVALUATION_SCHEMA",
    "ORACLE_DIAGNOSTIC_SCOPE",
    "ActionPathCollapseEvidence",
    "BehaviorCloningGateEvaluation",
    "BehaviorCloningGateGroup",
    "BehaviorCloningGateMetric",
    "BehaviorCloningGateThresholds",
    "BehaviorCloningHoldoutEvaluation",
    "OracleTeacherEvaluation",
    "PathPerformanceMetrics",
    "deterministic_bootstrap_lower_bound",
    "deterministic_bootstrap_upper_bound",
    "evaluate_behavior_cloning_gates",
    "evaluate_behavior_cloning_holdout",
    "evaluate_path_performance",
    "write_learning_evaluation",
]
