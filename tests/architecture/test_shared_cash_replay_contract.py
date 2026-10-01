from __future__ import annotations

from dataclasses import fields
from inspect import signature

from trade_rl.evaluation.replay import (
    SharedCashReplayDecision,
    SharedCashReplayLedgerEvidence,
    run_shared_cash_replay,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_duration import constrain_intent_for_minimum_hold


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


def test_adaptive_exit_contract_uses_execution_return_and_can_bypass_hold() -> None:
    observation_fields = {field.name for field in fields(StrategyObservation)}
    hold_parameters = signature(constrain_intent_for_minimum_hold).parameters

    assert "gross_position_return" in observation_fields
    assert "allow_protective_exit" in hold_parameters
