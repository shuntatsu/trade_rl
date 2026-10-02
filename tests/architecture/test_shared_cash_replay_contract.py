from __future__ import annotations

from dataclasses import fields, replace
from inspect import getsource, signature
from pathlib import Path

from trade_rl.evaluation.replay import (
    SharedCashLedgerIntervalEvidence,
    SharedCashReplayDecision,
    SharedCashReplayLedgerEvidence,
    run_shared_cash_replay,
)
from trade_rl.evaluation.runs import artifact as candidate_artifact
from trade_rl.simulation.diagnostics.accounting_transition import (
    AccountingStateSnapshot,
    AccountingTransitionEvidence,
)
from trade_rl.simulation.stateful import symbol_fills
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


def test_ledger_evidence_preserves_positional_schema_version() -> None:
    evidence = SharedCashReplayLedgerEvidence(
        "dataset",
        "digest",
        0,
        1,
        (),
        (),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        (),
        (),
        "shared_cash_replay_ledger_v2",
    )

    assert evidence.schema_version == "shared_cash_replay_ledger_v2"
    assert evidence.decisions == ()


def test_v3_accounting_evidence_is_opt_in_and_keeps_legacy_mapping_shape() -> None:
    interval_fields = {field.name for field in fields(SharedCashLedgerIntervalEvidence)}
    snapshot_fields = {field.name for field in fields(AccountingStateSnapshot)}
    transition_fields = {field.name for field in fields(AccountingTransitionEvidence)}

    assert "capture_accounting_evidence" in signature(run_shared_cash_replay).parameters
    assert "accounting_transitions" in interval_fields
    assert {
        "cash",
        "exact_quantities",
        "mark_prices",
        "contract_multipliers",
    } <= snapshot_fields
    assert {
        "sequence",
        "processing_index",
        "transition_type",
        "state_before",
        "state_after",
        "evidence",
        "order_event_sequence",
    } <= transition_fields

    legacy = SharedCashReplayLedgerEvidence(
        "dataset",
        "digest",
        0,
        1,
        (),
        (),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        (),
        (),
        "shared_cash_replay_ledger_v2",
    )
    v3 = replace(
        legacy,
        schema_version="shared_cash_replay_ledger_v3",
        initial_mark_prices=(100.0,),
        contract_multipliers=(1.0,),
    )
    legacy_mapping = legacy.to_mapping()
    v3_mapping = v3.to_mapping()

    assert "initial_mark_prices" not in legacy_mapping
    assert "contract_multipliers" not in legacy_mapping
    assert set(v3_mapping) == set(legacy_mapping) | {
        "initial_mark_prices",
        "contract_multipliers",
    }


def test_v3_fill_accounting_binds_exact_accepted_and_applied_quantities() -> None:
    fill_source = getsource(symbol_fills)
    validator_source = getsource(
        candidate_artifact._validate_v10_accounting_transitions
    )

    assert '"filled_quantity_exact": str(' in fill_source
    assert '"filled_lot_size": float(allocation.lot_size)' in fill_source
    assert '"filled_lot_count": allocation.filled_lot_count' in fill_source
    assert '"book_applied_quantity_exact": str(applied_quantity)' in fill_source
    assert '"filled_quantity_exact"' in validator_source
    assert "exact_fill_quantity" in validator_source
    assert '"book_applied_quantity_exact"' in validator_source
    assert "expected_applied_quantity" in validator_source
    assert "_project_exact_quantity" in validator_source


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
    assert "inactive asset's lifecycle settlement" in lean_core
    assert "signed filled quantity" in package_boundaries
    assert "inactive assetのlifecycle settlement" in package_boundaries
    assert (
        "test_adaptive_protective_exit_stays_latched_after_missed_fill_and_recovery"
        in assurance
    )
    assert "cannot guarantee that the next order fills" in assurance
