from __future__ import annotations

import numpy as np
import pytest

from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def _portfolio_risk() -> PreTradeRiskConfig:
    return PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=1.0,
        drawdown_stop=1.0,
    )


def test_per_symbol_ppo_accounts_do_not_represent_shared_cash_gross_constraint(
) -> None:
    dataset = pooled_market()
    execution_cost = ExecutionCostConfig.zero()
    risk_config = _portfolio_risk()

    independent_weights: list[float] = []
    for symbol_index in (0, 1):
        env = PPOTradingEnv(
            dataset,
            feature_indices=(0,),
            symbol_indices=(symbol_index,),
            start_index=0,
            stop_index=3,
            gross_budget=0.5,
            initial_capital=1_000.0,
            execution_cost=execution_cost,
            risk_config=risk_config,
        )
        env.reset(seed=17 + symbol_index)
        _, _, _, _, info = env.step(2)
        independent_weights.append(abs(float(info["realized_weight"])))

    shared = run_shared_cash_replay(
        dataset,
        (
            ConstantIntentStrategy(PositionIntent.LONG),
            ConstantIntentStrategy(PositionIntent.LONG),
        ),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk=PreTradeRisk(risk_config),
    )

    independent_total_gross = float(np.sum(independent_weights))
    shared_first_target_gross = float(np.sum(np.abs(shared.decisions[0].target_weights)))

    assert independent_total_gross == pytest.approx(1.0)
    assert shared_first_target_gross == pytest.approx(0.5)
    assert independent_total_gross > shared_first_target_gross
    assert "max_gross" in shared.decisions[0].risk_reasons
