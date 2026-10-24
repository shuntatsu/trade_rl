from __future__ import annotations

from dataclasses import fields
from inspect import getsource, signature
from pathlib import Path

import trade_rl.evaluation as evaluation
from trade_rl.evaluation import replay as replay_module
from trade_rl.evaluation.replay import (
    ScriptedReplayAction,
    SharedCashReplayDecision,
    SharedCashReplayLedgerEvidence,
    run_scripted_shared_cash_replay,
    run_shared_cash_replay,
)
from trade_rl.simulation.execution import MarketExecutor
from trade_rl.simulation.orders.reconciliation import reconcile_target
from trade_rl.simulation.targets.execution import execute_target_statefully
from trade_rl.strategies.forecasts.controller import CostAwareForecastIntentController
from trade_rl.strategies.forecasts.lightgbm import LightGBMForecastStrategy
from trade_rl.strategies.forecasts.ridge import RidgeForecastStrategy
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_duration import constrain_intent_for_minimum_hold
from trade_rl.strategies.rules.adaptive import RegimeAdaptiveStrategy


def test_shared_cash_replay_binds_age_and_terminal_settlement_inputs() -> None:
    parameters = signature(run_shared_cash_replay).parameters
    decision_fields = {field.name for field in fields(SharedCashReplayDecision)}
    ledger_fields = {field.name for field in fields(SharedCashReplayLedgerEvidence)}

    assert "minimum_hold_bars" in parameters
    assert "settle_terminal_position" in parameters
    assert "position_age_bars_before" in decision_fields
    assert "position_age_bars_after" in decision_fields
    assert "minimum_hold_suppressed" in decision_fields
    assert "decisions" in ledger_fields


def test_scripted_exact_hold_stays_inside_shared_execution_owners() -> None:
    assert (
        run_scripted_shared_cash_replay is replay_module.run_scripted_shared_cash_replay
    )
    assert "ScriptedReplayAction" not in evaluation.__all__
    assert "run_scripted_shared_cash_replay" not in evaluation.__all__
    assert ScriptedReplayAction.HOLD_QUANTITY.value == "hold_quantity"
    assert "actions_by_symbol" in signature(run_scripted_shared_cash_replay).parameters
    for owner in (
        MarketExecutor.execute_interval,
        execute_target_statefully,
        reconcile_target,
    ):
        assert "hold_current_quantity_symbols" in signature(owner).parameters

    strategy_decide = signature(SingleSymbolStrategy.decide)
    assert tuple(strategy_decide.parameters) == ("self", "observation")


def test_adaptive_exit_contract_uses_execution_return_and_can_bypass_hold() -> None:
    observation_fields = {field.name for field in fields(StrategyObservation)}
    hold_parameters = signature(constrain_intent_for_minimum_hold).parameters
    adaptive_decide = getsource(RegimeAdaptiveStrategy.decide)

    assert "gross_position_return" in observation_fields
    assert "current_position_quantity" in observation_fields
    assert "allow_protective_exit" in hold_parameters
    assert "observation.current_position_quantity" in adaptive_decide


def test_cost_aware_forecast_contract_receives_optional_fill_budget() -> None:
    observation_fields = {field.name: field for field in fields(StrategyObservation)}
    observation_builder_parameters = signature(replay_module._observation).parameters
    controller_decide = getsource(CostAwareForecastIntentController.decide_for_position)

    budget_field = observation_fields["intent_gross_budget"]
    cached_target_field = observation_fields["same_intent_target_weight"]
    assert budget_field.default is None
    assert cached_target_field.default is None
    assert "intent_gross_budget" in observation_builder_parameters
    assert "same_intent_target_weight" in observation_builder_parameters
    assert "intent_gross_budget=gross_budget" in getsource(
        replay_module.run_single_symbol_replay
    )
    assert "intent_gross_budget=gross_budget" in getsource(
        replay_module.run_shared_cash_replay
    )
    assert "current_position_quantity" in controller_decide
    assert "current_weight" in controller_decide
    assert "same_intent_target_weight" in controller_decide
    assert "current_intent" in controller_decide
    for strategy_type in (RidgeForecastStrategy, LightGBMForecastStrategy):
        strategy_decide = getsource(strategy_type.decide)
        assert "observation.intent_gross_budget is not None" in strategy_decide
        assert "observation.same_intent_target_weight" in strategy_decide
        assert "decide_for_position" in strategy_decide


def test_adaptive_exit_fill_state_invariant_is_documented_with_oracle() -> None:
    repository_root = Path(__file__).parents[2]
    lean_core = (repository_root / "docs/architecture/lean-core.md").read_text()
    package_boundaries = (
        repository_root / "docs/architecture/package-boundaries.md"
    ).read_text()
    assurance = (
        repository_root / "docs/architecture/research-assurance.md"
    ).read_text()

    assert "current_position_quantity" in lean_core
    assert "missed fill and a mark recovery below the trigger" in lean_core
    assert "signed filled quantity" in package_boundaries
    assert (
        "test_adaptive_protective_exit_stays_latched_after_missed_fill_and_recovery"
        in assurance
    )
    assert "cannot guarantee that the next order fills" in assurance
