from __future__ import annotations

from dataclasses import fields

import trade_rl.evaluation.experiments as experiments
from trade_rl.evaluation.experiments.contracts import (
    PPO_HOLDING_DURATION_HORIZONS,
    PPO_HOLDING_DURATION_MAX_DRAWDOWN,
    PPO_HOLDING_DURATION_SEED_COUNT,
    StudyPlan,
    StudyProtocol,
)


def test_ppo_holding_protocol_is_explicit_and_versioned() -> None:
    assert experiments.StudyProtocol is StudyProtocol
    assert StudyProtocol.PPO_HOLDING_DURATION.value == "ppo_holding_duration_v1"
    assert PPO_HOLDING_DURATION_HORIZONS == (72, 168, 336, 504)
    assert PPO_HOLDING_DURATION_SEED_COUNT == 5
    assert PPO_HOLDING_DURATION_MAX_DRAWDOWN == 0.20
    assert "protocol" in {field.name for field in fields(StudyPlan)}
