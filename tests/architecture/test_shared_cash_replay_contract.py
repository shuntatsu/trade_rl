from __future__ import annotations

from dataclasses import fields, replace
from inspect import getsource, signature

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
