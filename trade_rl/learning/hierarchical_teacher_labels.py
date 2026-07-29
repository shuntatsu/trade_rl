"""Immutable hierarchical labels derived from chronological teacher actions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

HIERARCHICAL_TEACHER_LABEL_SCHEMA: Final = "hierarchical_teacher_labels_v1"


class TeacherActionEvent(IntEnum):
    """Material teacher action change relative to the effective current weight."""

    HOLD = 0
    ENTER = 1
    RESIZE = 2
    EXIT = 3
    REVERSE = 4


@dataclass(frozen=True, slots=True)
class HierarchicalTeacherDiagnostics:
    """Deterministic class-distribution and hold-run diagnostics."""

    active_dimension_count: int
    gate_positive_count: int
    event_counts: tuple[int, int, int, int, int]
    longest_hold_run_by_action: tuple[int, ...]
    target_long_count: int
    target_flat_count: int
    target_short_count: int

    @property
    def gate_positive_rate(self) -> float:
        if self.active_dimension_count == 0:
            return 0.0
        return self.gate_positive_count / self.active_dimension_count


def _positive_finite_threshold(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("change_threshold must be positive finite")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("change_threshold must be positive finite")
    return threshold


def _rank_two_finite_weights(value: np.ndarray, *, field: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2:
        raise ValueError(f"{field} must be rank-two")
    if raw.shape[0] <= 0 or raw.shape[1] <= 0:
        raise ValueError(f"{field} must not be empty")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{field} must be numeric")
    array = np.asarray(raw, dtype=np.float32).copy(order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError(f"{field} must be within [-1, 1]")
    return array


def _rank_two_boolean_mask(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2:
        raise ValueError("active_mask must be rank-two")
    if raw.dtype != np.dtype(np.bool_):
        raise ValueError("active_mask must be boolean")
    return raw.copy(order="C")


def _classify_events(
    *,
    current_weights: np.ndarray,
    target_actions: np.ndarray,
    active_mask: np.ndarray,
    change_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    changed = active_mask & (
        np.abs(target_actions - current_weights) >= change_threshold
    )
    enter = (
        changed
        & (np.abs(current_weights) < change_threshold)
        & (np.abs(target_actions) >= change_threshold)
    )
    exit_ = (
        changed
        & (np.abs(current_weights) >= change_threshold)
        & (np.abs(target_actions) < change_threshold)
    )
    reverse = changed & ~enter & ~exit_ & (current_weights * target_actions < 0.0)
    resize = changed & ~(enter | exit_ | reverse)

    events = np.full(
        target_actions.shape,
        int(TeacherActionEvent.HOLD),
        dtype=np.uint8,
    )
    events[enter] = int(TeacherActionEvent.ENTER)
    events[resize] = int(TeacherActionEvent.RESIZE)
    events[exit_] = int(TeacherActionEvent.EXIT)
    events[reverse] = int(TeacherActionEvent.REVERSE)
    return changed, events


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "shape": tuple(int(size) for size in array.shape),
    }


def _label_digest(
    *,
    gate_labels: np.ndarray,
    current_weights: np.ndarray,
    target_actions: np.ndarray,
    active_mask: np.ndarray,
    events: np.ndarray,
    source_teacher_digest: str,
    change_threshold: float,
) -> str:
    return content_digest(
        {
            "active_mask": _array_identity(active_mask),
            "change_threshold": change_threshold,
            "current_weights": _array_identity(current_weights),
            "events": _array_identity(events),
            "gate_labels": _array_identity(gate_labels),
            "schema_version": HIERARCHICAL_TEACHER_LABEL_SCHEMA,
            "source_teacher_digest": source_teacher_digest,
            "target_actions": _array_identity(target_actions),
        }
    )


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


@dataclass(frozen=True, slots=True)
class HierarchicalTeacherLabels:
    """Read-only chronological Gate + Target labels with bound provenance."""

    gate_labels: np.ndarray
    current_weights: np.ndarray
    target_actions: np.ndarray
    active_mask: np.ndarray
    events: np.ndarray
    source_teacher_digest: str
    change_threshold: float
    label_config_digest: str
    schema_version: str = HIERARCHICAL_TEACHER_LABEL_SCHEMA

    def __post_init__(self) -> None:
        source_teacher_digest = require_sha256(
            self.source_teacher_digest,
            field="source_teacher_digest",
        )
        label_config_digest = require_sha256(
            self.label_config_digest,
            field="label_config_digest",
        )
        threshold = _positive_finite_threshold(self.change_threshold)
        current_weights = _rank_two_finite_weights(
            self.current_weights,
            field="current_weights",
        )
        target_actions = _rank_two_finite_weights(
            self.target_actions,
            field="teacher_targets",
        )
        active_mask = _rank_two_boolean_mask(self.active_mask)
        gate_labels = _rank_two_boolean_mask(self.gate_labels)
        events_raw = np.asarray(self.events)
        if events_raw.ndim != 2 or not np.issubdtype(events_raw.dtype, np.integer):
            raise ValueError("events must be a rank-two integer array")
        events = np.asarray(events_raw, dtype=np.uint8).copy(order="C")
        shape = target_actions.shape
        if not (
            current_weights.shape
            == active_mask.shape
            == gate_labels.shape
            == events.shape
            == shape
        ):
            raise ValueError("hierarchical teacher label arrays must have equal shapes")
        valid_events = np.isin(events, [int(event) for event in TeacherActionEvent])
        if not bool(valid_events.all()):
            raise ValueError("events contain an unsupported teacher action event")
        expected_gate, expected_events = _classify_events(
            current_weights=current_weights,
            target_actions=target_actions,
            active_mask=active_mask,
            change_threshold=threshold,
        )
        if not np.array_equal(gate_labels, expected_gate) or not np.array_equal(
            events, expected_events
        ):
            raise ValueError("hierarchical teacher labels are inconsistent")
        if self.schema_version != HIERARCHICAL_TEACHER_LABEL_SCHEMA:
            raise ValueError("unsupported hierarchical teacher label schema")
        expected_digest = _label_digest(
            gate_labels=gate_labels,
            current_weights=current_weights,
            target_actions=target_actions,
            active_mask=active_mask,
            events=events,
            source_teacher_digest=source_teacher_digest,
            change_threshold=threshold,
        )
        if label_config_digest != expected_digest:
            raise ValueError("hierarchical teacher label digest mismatch")
        for array in (
            gate_labels,
            current_weights,
            target_actions,
            active_mask,
            events,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "gate_labels", gate_labels)
        object.__setattr__(self, "current_weights", current_weights)
        object.__setattr__(self, "target_actions", target_actions)
        object.__setattr__(self, "active_mask", active_mask)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "change_threshold", threshold)
        object.__setattr__(self, "source_teacher_digest", source_teacher_digest)
        object.__setattr__(self, "label_config_digest", label_config_digest)

    @property
    def sample_count(self) -> int:
        return int(self.target_actions.shape[0])

    @property
    def action_count(self) -> int:
        return int(self.target_actions.shape[1])

    @property
    def diagnostics(self) -> HierarchicalTeacherDiagnostics:
        active_events = self.events[self.active_mask]
        event_counts = tuple(
            int(np.count_nonzero(active_events == int(event)))
            for event in TeacherActionEvent
        )
        hold = self.active_mask & (self.events == int(TeacherActionEvent.HOLD))
        threshold = self.change_threshold
        active_targets = self.target_actions[self.active_mask]
        return HierarchicalTeacherDiagnostics(
            active_dimension_count=int(np.count_nonzero(self.active_mask)),
            gate_positive_count=int(np.count_nonzero(self.gate_labels)),
            event_counts=(
                event_counts[0],
                event_counts[1],
                event_counts[2],
                event_counts[3],
                event_counts[4],
            ),
            longest_hold_run_by_action=tuple(
                _longest_true_run(hold[:, index]) for index in range(self.action_count)
            ),
            target_long_count=int(np.count_nonzero(active_targets >= threshold)),
            target_flat_count=int(np.count_nonzero(np.abs(active_targets) < threshold)),
            target_short_count=int(np.count_nonzero(active_targets <= -threshold)),
        )


def build_hierarchical_teacher_labels(
    *,
    teacher_targets: np.ndarray,
    current_weights: np.ndarray,
    active_mask: np.ndarray,
    change_threshold: float,
    source_teacher_digest: str,
) -> HierarchicalTeacherLabels:
    """Derive immutable labels without deleting or reordering chronological rows."""

    source_digest = require_sha256(
        source_teacher_digest,
        field="source_teacher_digest",
    )
    threshold = _positive_finite_threshold(change_threshold)
    targets = _rank_two_finite_weights(teacher_targets, field="teacher_targets")
    current = _rank_two_finite_weights(current_weights, field="current_weights")
    active = _rank_two_boolean_mask(active_mask)
    if not (targets.shape == current.shape == active.shape):
        raise ValueError(
            "teacher_targets, current_weights, and active_mask need equal shapes"
        )
    gate_labels, events = _classify_events(
        current_weights=current,
        target_actions=targets,
        active_mask=active,
        change_threshold=threshold,
    )
    digest = _label_digest(
        gate_labels=gate_labels,
        current_weights=current,
        target_actions=targets,
        active_mask=active,
        events=events,
        source_teacher_digest=source_digest,
        change_threshold=threshold,
    )
    return HierarchicalTeacherLabels(
        gate_labels=gate_labels,
        current_weights=current,
        target_actions=targets,
        active_mask=active,
        events=events,
        source_teacher_digest=source_digest,
        change_threshold=threshold,
        label_config_digest=digest,
    )


__all__ = [
    "HIERARCHICAL_TEACHER_LABEL_SCHEMA",
    "HierarchicalTeacherDiagnostics",
    "HierarchicalTeacherLabels",
    "TeacherActionEvent",
    "build_hierarchical_teacher_labels",
]
