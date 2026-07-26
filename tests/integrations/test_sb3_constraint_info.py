from __future__ import annotations

import numpy as np
import pytest

from trade_rl.integrations.sb3_training import _compact_training_info
from trade_rl.rl.environment_constraints import (
    ActionPathDiagnostics,
    ConstraintCostVector,
)


def test_compact_training_info_preserves_constraint_telemetry() -> None:
    action_path = ActionPathDiagnostics.from_stages(
        policy_target=np.array([0.8, -0.4]),
        pretrade_target=np.array([0.5, -0.4]),
        feasible_target=np.array([0.4, -0.3]),
        submitted_order_target=np.array([0.4, -0.2]),
        filled_weight=np.array([0.25, -0.1]),
    )
    costs = ConstraintCostVector(
        drawdown_excess=0.01,
        drawdown_stop_event=0.0,
        margin_deficit_fraction=0.002,
        forced_liquidation_event=0.0,
        gross_exposure_request_excess=0.2,
        daily_turnover=1.5,
        execution_cost_fraction=0.003,
        funding_credit_fraction=0.0004,
    )
    heavy = object()
    info: dict[str, object] = {
        "action_path": action_path,
        "constraint_costs": costs,
        "action_path_policy_to_filled_l1": action_path.policy_to_filled_l1,
        "constraint_cost_execution_fraction": costs.execution_cost_fraction,
        "hybrid_execution": heavy,
        "shadow_execution": heavy,
        "hybrid_liquidation": heavy,
        "shadow_liquidation": heavy,
    }

    compact = _compact_training_info(info)

    assert compact["action_path"] is action_path
    assert compact["constraint_costs"] is costs
    assert compact["action_path_policy_to_filled_l1"] == pytest.approx(0.85)
    assert compact["constraint_cost_execution_fraction"] == pytest.approx(0.003)
    assert "hybrid_execution" not in compact
    assert "shadow_execution" not in compact
    assert "hybrid_liquidation" not in compact
    assert "shadow_liquidation" not in compact
