from __future__ import annotations

from dataclasses import fields
from inspect import signature

from trade_rl.evaluation.replay import (
    SharedCashReplayDecision,
    SharedCashReplayLedgerEvidence,
    run_shared_cash_replay,
)


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
