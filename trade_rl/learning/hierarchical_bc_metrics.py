"""Deterministic reconstruction metrics for hierarchical behavior cloning."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.hierarchical_teacher_labels import (
    HierarchicalTeacherLabels,
    TeacherActionEvent,
)


@dataclass(frozen=True, slots=True)
class HierarchicalBehaviorCloningMetrics:
    """Teacher-reconstruction quality and explicit collapse evidence."""

    active_support: int
    positive_support: int
    predicted_positive_support: int
    gate_precision: float
    gate_recall: float
    gate_f1: float
    active_target_rmse: float | None
    composed_rmse: float | None
    teacher_activity_rate: float
    policy_activity_rate: float
    activity_ratio: float | None
    event_recalls: tuple[float | None, float | None, float | None, float | None]
    constant_action_collapse: bool
    all_hold_collapse: bool
    all_trade_collapse: bool
    insufficient_target_support: bool

    @property
    def digest(self) -> str:
        return content_digest(
            {
                **asdict(self),
                "schema_version": "hierarchical_behavior_cloning_metrics_v1",
            }
        )


@dataclass(frozen=True, slots=True)
class HierarchicalBehaviorCloningLosses:
    """Normalized Gate, Target, composed, and configured weighted losses."""

    gate: float
    target: float
    composed: float
    weighted: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"hierarchical {name} loss must be finite and non-negative"
                )


def _selected(value: np.ndarray, indices: np.ndarray | None) -> np.ndarray:
    array = np.asarray(value)
    return array if indices is None else array[indices]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not bool(np.any(mask)):
        return None
    error = (
        np.asarray(prediction, dtype=np.float64)[mask]
        - np.asarray(target, dtype=np.float64)[mask]
    )
    return float(np.sqrt(np.mean(np.square(error))))


def hierarchical_bc_metrics(
    *,
    gate_probabilities: np.ndarray,
    proposal_actions: np.ndarray,
    composed_actions: np.ndarray,
    labels: HierarchicalTeacherLabels,
    gate_threshold: float = 0.5,
    indices: np.ndarray | None = None,
    constant_tolerance: float = 1e-6,
) -> HierarchicalBehaviorCloningMetrics:
    """Measure event reconstruction without allowing hold-majority masking."""

    if (
        isinstance(gate_threshold, bool)
        or not math.isfinite(float(gate_threshold))
        or not 0.0 < float(gate_threshold) < 1.0
    ):
        raise ValueError("gate_threshold must be within (0, 1)")
    if (
        isinstance(constant_tolerance, bool)
        or not math.isfinite(float(constant_tolerance))
        or float(constant_tolerance) < 0.0
    ):
        raise ValueError("constant_tolerance must be finite and non-negative")
    gate = np.asarray(gate_probabilities, dtype=np.float64)
    proposal = np.asarray(proposal_actions, dtype=np.float64)
    composed = np.asarray(composed_actions, dtype=np.float64)
    if not (gate.ndim == proposal.ndim == composed.ndim == 2):
        raise ValueError("hierarchical BC predictions must be rank-two")
    if not (gate.shape == proposal.shape == composed.shape):
        raise ValueError("hierarchical BC predictions must have equal shapes")
    if not (
        np.isfinite(gate).all()
        and np.isfinite(proposal).all()
        and np.isfinite(composed).all()
    ):
        raise ValueError("hierarchical BC predictions must be finite")
    if np.any(gate < 0.0) or np.any(gate > 1.0):
        raise ValueError("gate probabilities must be within [0, 1]")
    if np.any(np.abs(proposal) > 1.0) or np.any(np.abs(composed) > 1.0):
        raise ValueError("hierarchical BC actions must be within [-1, 1]")

    active = _selected(labels.active_mask, indices)
    teacher_positive = _selected(labels.gate_labels, indices) & active
    teacher_targets = _selected(labels.target_actions, indices)
    events = _selected(labels.events, indices)
    expected_shape = active.shape
    if gate.shape != expected_shape:
        raise ValueError("hierarchical BC predictions do not match selected labels")

    predicted_positive = (gate >= float(gate_threshold)) & active
    true_positive = int(np.count_nonzero(predicted_positive & teacher_positive))
    positive_support = int(np.count_nonzero(teacher_positive))
    predicted_support = int(np.count_nonzero(predicted_positive))
    active_support = int(np.count_nonzero(active))
    precision = _safe_ratio(true_positive, predicted_support)
    recall = _safe_ratio(true_positive, positive_support)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    teacher_rate = _safe_ratio(positive_support, active_support)
    policy_rate = _safe_ratio(predicted_support, active_support)
    if teacher_rate > 0.0:
        activity_ratio: float | None = policy_rate / teacher_rate
    elif policy_rate == 0.0:
        activity_ratio = 1.0
    else:
        activity_ratio = None

    event_recalls: list[float | None] = []
    for event in (
        TeacherActionEvent.ENTER,
        TeacherActionEvent.RESIZE,
        TeacherActionEvent.EXIT,
        TeacherActionEvent.REVERSE,
    ):
        event_mask = active & (events == int(event))
        support = int(np.count_nonzero(event_mask))
        event_recalls.append(
            None
            if support == 0
            else float(np.count_nonzero(predicted_positive & event_mask) / support)
        )

    if composed.shape[0] <= 1:
        constant_action = True
    else:
        active_columns = np.any(active, axis=0)
        if not bool(np.any(active_columns)):
            constant_action = True
        else:
            spans = np.ptp(composed[:, active_columns], axis=0)
            constant_action = bool(np.all(spans <= float(constant_tolerance)))

    return HierarchicalBehaviorCloningMetrics(
        active_support=active_support,
        positive_support=positive_support,
        predicted_positive_support=predicted_support,
        gate_precision=precision,
        gate_recall=recall,
        gate_f1=f1,
        active_target_rmse=_rmse(proposal, teacher_targets, teacher_positive),
        composed_rmse=_rmse(composed, teacher_targets, active),
        teacher_activity_rate=teacher_rate,
        policy_activity_rate=policy_rate,
        activity_ratio=activity_ratio,
        event_recalls=(
            event_recalls[0],
            event_recalls[1],
            event_recalls[2],
            event_recalls[3],
        ),
        constant_action_collapse=constant_action,
        all_hold_collapse=active_support > 0 and predicted_support == 0,
        all_trade_collapse=active_support > 0 and predicted_support == active_support,
        insufficient_target_support=positive_support == 0,
    )


__all__ = [
    "HierarchicalBehaviorCloningLosses",
    "HierarchicalBehaviorCloningMetrics",
    "hierarchical_bc_metrics",
]
