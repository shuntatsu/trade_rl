from __future__ import annotations

from dataclasses import fields
from inspect import getsource, signature
from pathlib import Path

from trade_rl.evaluation.replay import (
    SharedCashReplayDecision,
    SharedCashReplayLedgerEvidence,
    run_shared_cash_replay,
)
from trade_rl.strategies.interface import StrategyObservation
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


def test_adaptive_exit_contract_uses_execution_return_and_can_bypass_hold() -> None:
    observation_fields = {field.name for field in fields(StrategyObservation)}
    hold_parameters = signature(constrain_intent_for_minimum_hold).parameters
    adaptive_decide = getsource(RegimeAdaptiveStrategy.decide)

    assert "gross_position_return" in observation_fields
    assert "current_position_quantity" in observation_fields
    assert "allow_protective_exit" in hold_parameters
    assert "observation.current_position_quantity" in adaptive_decide


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
