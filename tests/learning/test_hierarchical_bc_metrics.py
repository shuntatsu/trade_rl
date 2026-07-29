from __future__ import annotations

import numpy as np

from trade_rl.learning.hierarchical_bc_metrics import hierarchical_bc_metrics
from trade_rl.learning.hierarchical_teacher_labels import (
    build_hierarchical_teacher_labels,
)


def _labels():
    current = np.array(
        [[0.0, 0.0], [0.4, 0.0], [0.4, -0.3], [0.0, -0.3]],
        dtype=np.float32,
    )
    target = np.array(
        [[0.4, 0.0], [0.4, -0.3], [0.0, 0.2], [0.0, -0.3]],
        dtype=np.float32,
    )
    return build_hierarchical_teacher_labels(
        teacher_targets=target,
        current_weights=current,
        active_mask=np.ones_like(target, dtype=np.bool_),
        change_threshold=0.05,
        source_teacher_digest="a" * 64,
    )


def test_all_hold_prediction_is_reported_as_collapse() -> None:
    labels = _labels()
    metrics = hierarchical_bc_metrics(
        gate_probabilities=np.zeros((4, 2), dtype=np.float32),
        proposal_actions=np.zeros((4, 2), dtype=np.float32),
        composed_actions=labels.current_weights,
        labels=labels,
        gate_threshold=0.5,
    )

    assert metrics.all_hold_collapse is True
    assert metrics.all_trade_collapse is False
    assert metrics.gate_recall == 0.0
    assert metrics.positive_support == 4
    assert metrics.predicted_positive_support == 0


def test_metrics_report_event_quality_activity_and_rmse() -> None:
    labels = _labels()
    gate = labels.gate_labels.astype(np.float32)
    metrics = hierarchical_bc_metrics(
        gate_probabilities=gate,
        proposal_actions=labels.target_actions,
        composed_actions=labels.target_actions,
        labels=labels,
    )

    assert metrics.gate_precision == 1.0
    assert metrics.gate_recall == 1.0
    assert metrics.gate_f1 == 1.0
    assert metrics.active_target_rmse == 0.0
    assert metrics.composed_rmse == 0.0
    assert metrics.activity_ratio == 1.0
    assert metrics.event_recalls == (1.0, None, 1.0, 1.0)
    assert metrics.constant_action_collapse is False


def test_all_trade_and_constant_action_are_independent_collapse_flags() -> None:
    labels = _labels()
    metrics = hierarchical_bc_metrics(
        gate_probabilities=np.ones((4, 2), dtype=np.float32),
        proposal_actions=np.zeros((4, 2), dtype=np.float32),
        composed_actions=np.zeros((4, 2), dtype=np.float32),
        labels=labels,
    )

    assert metrics.all_trade_collapse is True
    assert metrics.all_hold_collapse is False
    assert metrics.constant_action_collapse is True
    assert len(metrics.digest) == 64
