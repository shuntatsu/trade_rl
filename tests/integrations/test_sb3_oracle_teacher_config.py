from __future__ import annotations

from types import SimpleNamespace

from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig


def test_oracle_teacher_config_for_environment_matches_training_contract() -> None:
    from trade_rl.integrations.sb3_runtime import oracle_teacher_config_for_environment

    execution = ExecutionCostConfig(
        fee_rate=0.0007,
        spread_rate=0.0003,
        impact_rate=0.0004,
        max_participation_rate=0.03,
    )
    portfolio = PortfolioRiskConfig(max_abs_weight=0.8, max_net_exposure=0.9)
    risk = PreTradeRiskConfig(
        max_gross=0.9,
        max_abs_weight=0.7,
        entry_threshold=0.11,
        exit_threshold=0.04,
        no_trade_band=0.03,
    )
    environment = SimpleNamespace(
        config=SimpleNamespace(
            execution_cost=execution,
            signal_delay_decisions=1,
        ),
        portfolio_risk=SimpleNamespace(config=portfolio),
        pre_trade_risk=SimpleNamespace(config=risk),
        initial_capital=123_456.0,
    )

    config = oracle_teacher_config_for_environment(environment)

    assert config.execution_cost == execution
    assert config.portfolio_risk == portfolio
    assert config.max_gross == risk.max_gross
    assert config.max_abs_weight == risk.max_abs_weight
    assert config.entry_threshold == risk.entry_threshold
    assert config.exit_threshold == risk.exit_threshold
    assert config.no_trade_band == risk.no_trade_band
    assert config.reference_portfolio_value == 123_456.0
    assert config.signal_delay_decisions == 1
