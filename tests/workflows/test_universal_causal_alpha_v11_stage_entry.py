from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v11 import CausalAlphaV11StudyArm
from trade_rl.workflows import universal_causal_alpha_v11_stage_entry as stage


def _metric(*, net_return: float = 0.01) -> SimpleNamespace:
    trace = SimpleNamespace(
        requested_targets=np.asarray([[0.0], [0.1]]),
        projected_targets=np.asarray([[0.0], [0.1]]),
        realized_weights=np.asarray([[0.0], [0.1]]),
        gross_returns=np.asarray([0.0, 0.02]),
        net_returns=np.asarray([0.0, net_return]),
        costs=np.asarray([0.0, 0.001]),
        turnovers=np.asarray([0.0, 0.1]),
    )
    lifecycle = SimpleNamespace(
        transition_classes=("flat", "entry"),
        execution_intent_targets=np.asarray([[0.0], [0.1]]),
        final_risk_targets=np.asarray([[0.0], [0.1]]),
    )
    base = SimpleNamespace(
        gross_return=0.02,
        net_return=net_return,
        total_execution_cost=0.001,
        turnover_per_day=0.1,
    )
    return SimpleNamespace(v6_metric=base, step_trace=trace, lifecycle_trace=lifecycle)


def test_v11_leaf_schema_requires_candidate_policy_and_trace_digests() -> None:
    assert stage._REPLAY_LEAF_SCHEMA == "causal_alpha_v11_replay_leaf_v1"


def test_v11_control_rejects_any_r21_economic_drift() -> None:
    with pytest.raises(ValueError, match="behavior-neutral control drifted"):
        stage._require_control_equivalence(_metric(), _metric(net_return=0.02))


def test_v11_stage_config_binds_one_study_arm() -> None:
    first = stage.causal_alpha_v11_stage_config_digest(
        source_config_digest="a" * 64,
        study_arm=CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2,
    )
    second = stage.causal_alpha_v11_stage_config_digest(
        source_config_digest="a" * 64,
        study_arm=CausalAlphaV11StudyArm.AFTER_COST_ENTRY,
    )

    assert first != second
