from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.learning import rollout_evaluation
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


def _environment() -> object:
    return SimpleNamespace(
        pre_trade_risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.10,
                max_turnover=2.0,
                drawdown_start=0.10,
                drawdown_stop=0.20,
            )
        ),
        # The environment may have a worse end-of-step drawdown after market
        # movement. The projection oracle uses the risk scale applied before
        # execution, not a retroactive cap derived from this later state.
        hybrid=SimpleNamespace(max_drawdown=0.20),
    )


def test_hard_risk_projection_uses_applied_pretrade_scale() -> None:
    environment = _environment()
    projected_weights = np.asarray([0.06], dtype=np.float64)

    assert not rollout_evaluation._hard_risk_projection_violation(
        environment,
        projected_weights,
        applied_risk_scale=1.0,
    )
    assert rollout_evaluation._hard_risk_projection_violation(
        environment,
        projected_weights,
        applied_risk_scale=0.5,
    )
