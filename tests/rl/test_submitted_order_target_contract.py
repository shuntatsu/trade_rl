from __future__ import annotations

import numpy as np

from trade_rl.rl.environment_constraints import ActionPathDiagnostics
from trade_rl.rl.environment_info import EnvironmentInfoBuilder


def test_action_path_info_exposes_submitted_order_target_vector() -> None:
    diagnostics = ActionPathDiagnostics.from_stages(
        policy_target=np.array([0.8, -0.4]),
        pretrade_target=np.array([0.5, -0.4]),
        feasible_target=np.array([0.4, -0.3]),
        submitted_order_target=np.array([0.4, -0.2]),
        filled_weight=np.array([0.25, -0.1]),
    )

    info = EnvironmentInfoBuilder._action_path_info(diagnostics)

    np.testing.assert_array_equal(
        info["submitted_order_target"],
        diagnostics.submitted_order_target,
    )
    assert info["submitted_order_target"] is not diagnostics.submitted_order_target
