from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.hierarchical_teacher_labels import (
    TeacherActionEvent,
    build_hierarchical_teacher_labels,
)


def _build(
    current: np.ndarray,
    target: np.ndarray,
    *,
    active: np.ndarray | None = None,
    threshold: float = 0.01,
):
    return build_hierarchical_teacher_labels(
        teacher_targets=target,
        current_weights=current,
        active_mask=np.ones_like(target, dtype=np.bool_) if active is None else active,
        change_threshold=threshold,
        source_teacher_digest="a" * 64,
    )


def test_labels_classify_enter_exit_reverse_without_reordering() -> None:
    current = np.array([[0.0], [0.4], [0.4], [-0.3]], dtype=np.float32)
    target = np.array([[0.4], [0.0], [-0.3], [-0.3]], dtype=np.float32)

    labels = _build(current, target)

    assert labels.gate_labels[:, 0].tolist() == [True, True, True, False]
    assert labels.events[:, 0].tolist() == [
        TeacherActionEvent.ENTER,
        TeacherActionEvent.EXIT,
        TeacherActionEvent.REVERSE,
        TeacherActionEvent.HOLD,
    ]
    np.testing.assert_array_equal(labels.current_weights, current)
    np.testing.assert_array_equal(labels.target_actions, target)


def test_effectively_flat_enter_exit_take_precedence_over_reverse() -> None:
    current = np.array([[0.005], [0.4]], dtype=np.float32)
    target = np.array([[-0.4], [-0.005]], dtype=np.float32)

    labels = _build(current, target, threshold=0.01)

    assert labels.gate_labels[:, 0].tolist() == [True, True]
    assert labels.events[:, 0].tolist() == [
        TeacherActionEvent.ENTER,
        TeacherActionEvent.EXIT,
    ]


def test_labels_classify_resize_and_mask_inactive_dimensions() -> None:
    current = np.array([[0.2, -0.2], [0.2, 0.0]], dtype=np.float32)
    target = np.array([[0.5, -0.2], [0.21, 0.7]], dtype=np.float32)
    active = np.array([[True, True], [True, False]], dtype=np.bool_)

    labels = _build(current, target, active=active)

    assert labels.events.tolist() == [
        [TeacherActionEvent.RESIZE, TeacherActionEvent.HOLD],
        [TeacherActionEvent.HOLD, TeacherActionEvent.HOLD],
    ]
    assert labels.gate_labels.tolist() == [[True, False], [False, False]]
    np.testing.assert_array_equal(labels.target_actions, target)


def test_labels_copy_and_freeze_every_array() -> None:
    current = np.array([[0.0], [0.2]], dtype=np.float32)
    target = np.array([[0.3], [0.0]], dtype=np.float32)
    active = np.ones_like(target, dtype=np.bool_)

    labels = _build(current, target, active=active)
    current[:] = -0.9
    target[:] = 0.9
    active[:] = False

    np.testing.assert_allclose(labels.current_weights[:, 0], [0.0, 0.2], atol=1e-7)
    np.testing.assert_allclose(labels.target_actions[:, 0], [0.3, 0.0], atol=1e-7)
    np.testing.assert_array_equal(labels.active_mask[:, 0], [True, True])
    for value in (
        labels.gate_labels,
        labels.current_weights,
        labels.target_actions,
        labels.active_mask,
        labels.events,
    ):
        assert value.flags.writeable is False
        with pytest.raises(ValueError):
            value.flat[0] = value.flat[0]


def test_label_digest_is_deterministic_and_binds_threshold_and_arrays() -> None:
    current = np.array([[0.0], [0.2]], dtype=np.float32)
    target = np.array([[0.3], [0.0]], dtype=np.float32)

    first = _build(current, target, threshold=0.01)
    second = _build(current.copy(), target.copy(), threshold=0.01)
    changed_threshold = _build(current, target, threshold=0.02)
    changed_target = _build(current, np.array([[0.4], [0.0]], dtype=np.float32))

    assert first.label_config_digest == second.label_config_digest
    assert first.label_config_digest != changed_threshold.label_config_digest
    assert first.label_config_digest != changed_target.label_config_digest
    assert len(first.label_config_digest) == 64


def test_diagnostics_report_distribution_and_hold_runs() -> None:
    current = np.array([[0.0], [0.4], [0.4], [0.4], [0.0]], dtype=np.float32)
    target = np.array([[0.4], [0.4], [0.4], [0.0], [0.0]], dtype=np.float32)

    labels = _build(current, target)

    assert labels.diagnostics.active_dimension_count == 5
    assert labels.diagnostics.gate_positive_count == 2
    assert labels.diagnostics.event_counts == (3, 1, 0, 1, 0)
    assert labels.diagnostics.longest_hold_run_by_action == (2,)
    assert labels.diagnostics.target_long_count == 3
    assert labels.diagnostics.target_flat_count == 2
    assert labels.diagnostics.target_short_count == 0


@pytest.mark.parametrize(
    ("current", "target", "active", "threshold", "digest", "message"),
    [
        (
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            np.ones((2, 1), dtype=np.bool_),
            0.01,
            "a" * 64,
            "equal shapes",
        ),
        (
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.bool_),
            0.01,
            "a" * 64,
            "rank-two",
        ),
        (
            np.array([[np.nan]], dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.bool_),
            0.01,
            "a" * 64,
            "finite",
        ),
        (
            np.zeros((1, 1), dtype=np.float32),
            np.array([[1.1]], dtype=np.float32),
            np.ones((1, 1), dtype=np.bool_),
            0.01,
            "a" * 64,
            "within",
        ),
        (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            0.01,
            "a" * 64,
            "boolean",
        ),
        (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.bool_),
            0.0,
            "a" * 64,
            "positive finite",
        ),
        (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.bool_),
            0.01,
            "invalid",
            "SHA-256",
        ),
    ],
)
def test_label_validation_fails_closed(
    current: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    threshold: float,
    digest: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_hierarchical_teacher_labels(
            teacher_targets=target,
            current_weights=current,
            active_mask=active,
            change_threshold=threshold,
            source_teacher_digest=digest,
        )
