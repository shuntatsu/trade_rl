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

LEARNING_EVALUATION_SCHEMA: Final = "oracle_behavior_cloning_evaluation_v1"
ORACLE_DIAGNOSTIC_SCOPE: Final = "hindsight_oracle_diagnostic"
CAUSAL_GENERALIZATION_SCOPE: Final = "causal_policy_generalization"


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
    if len({len(gross), len(net), len(reward), len(traded_turnover), len(paid_cost)}) != 1:
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
            raise ValueError("oracle evaluation must remain a hindsight-only diagnostic")
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
        if self.causal_policy_performance.step_count != heldout_stop - heldout_start - 1:
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
        if self.action_mae < 0.0 or self.action_rmse < 0.0 or self.heldout_oracle_regret < 0.0:
            raise ValueError("action errors and oracle regret must be non-negative")
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
        return asdict(self)


def evaluate_behavior_cloning_holdout(
    *,
    teacher_train_range: tuple[int, int],
    heldout_range: tuple[int, int],
    oracle_actions: object,
    policy_actions: object,
    oracle_performance: PathPerformanceMetrics,
    causal_policy_performance: PathPerformanceMetrics,
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
        raise ValueError("oracle and policy actions must be equal finite rank-two arrays")
    start, stop = _evaluation_range(heldout_range, field="heldout_range")
    expected_count = stop - start - 1
    if len(oracle) != expected_count:
        raise ValueError("actions do not cover the exact heldout range")
    if (
        not math.isfinite(action_tolerance)
        or action_tolerance < 0.0
    ):
        raise ValueError("action_tolerance must be finite and non-negative")

    difference = oracle - policy
    oracle_evaluation = OracleTeacherEvaluation(
        evaluation_range=heldout_range,
        performance=oracle_performance,
    )
    gross_gap = oracle_performance.gross_return - causal_policy_performance.gross_return
    net_gap = oracle_performance.net_return - causal_policy_performance.net_return
    reward_gap = oracle_performance.reward_total - causal_policy_performance.reward_total
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
    )


def write_learning_evaluation(
    path: str | Path,
    evaluation: OracleTeacherEvaluation | BehaviorCloningHoldoutEvaluation,
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
    "CAUSAL_GENERALIZATION_SCOPE",
    "LEARNING_EVALUATION_SCHEMA",
    "ORACLE_DIAGNOSTIC_SCOPE",
    "BehaviorCloningHoldoutEvaluation",
    "OracleTeacherEvaluation",
    "PathPerformanceMetrics",
    "evaluate_behavior_cloning_holdout",
    "evaluate_path_performance",
    "write_learning_evaluation",
]
