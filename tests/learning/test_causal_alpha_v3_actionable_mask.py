from __future__ import annotations

import numpy as np

from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3TargetConfig,
    causal_alpha_v3_target_path,
)


def test_v3_target_compiler_holds_when_signal_is_unactionable() -> None:
    config = CausalAlphaV3TargetConfig(
        target_magnitudes=(0.0, 0.1, 0.2),
        uncertainty_multiplier=0.0,
        execution_cost_multiplier=1.0,
        edge_margin=0.0,
        alpha_rebalance_decisions=1,
        strong_reversal_threshold=0.01,
        max_target_delta=0.2,
    )

    path = causal_alpha_v3_target_path(
        np.asarray([0.2, -0.2, -0.2], dtype=np.float64),
        uncertainties=np.zeros(3, dtype=np.float64),
        one_way_cost_rates=np.zeros(3, dtype=np.float64),
        liquidity_weight_caps=np.ones(3, dtype=np.float64),
        actionable_mask=np.asarray([True, False, True]),
        config=config,
        initial_weight=0.0,
    )

    assert path.targets[0] > 0.0
    assert path.targets[1] == path.targets[0]
    assert path.reasons[1] == "unactionable_hold"
    assert path.targets[2] < path.targets[1]
