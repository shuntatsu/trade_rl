from __future__ import annotations

import numpy as np

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


def test_pretrade_config_contains_no_strategy_hysteresis_fields() -> None:
    fields = set(PreTradeRiskConfig.__dataclass_fields__)
    assert fields.isdisjoint({"entry_threshold", "exit_threshold", "no_trade_band"})


def test_valid_economic_targets_pass_through_without_entry_exit_judgment() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0, max_turnover=2.0)
    )
    cases = (
        (np.array([0.08]), np.array([0.0])),
        (np.array([0.04]), np.array([0.20])),
        (np.array([0.02]), np.array([0.20])),
        (np.array([-0.08]), np.array([0.20])),
    )

    for target, current in cases:
        result = risk.constrain(target, current=current, drawdown=0.0)
        np.testing.assert_array_equal(result.weights, target)
