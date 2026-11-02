from __future__ import annotations

import json
import math
from dataclasses import replace
from hashlib import sha256

import pytest

from tools.tier_d_nonrl.task18_receipts import (
    ArmAttemptProgress,
    ArmReceipt,
    ExternalArmVerificationReport,
    ExternalSourceVerificationReport,
    RunReceipt,
    SourceBinding,
    SourceFilePin,
    diagnostic_combined_delta,
    diagnostic_pair_delta,
    evaluate_run_receipt,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol

SOURCE = SourceBinding(
    commit_sha="a" * 40,
    files=(
        SourceFilePin("tools/tier_d_nonrl/task18_recipe.py", "b" * 64),
        SourceFilePin("tools/tier_d_nonrl/task18_receipts.py", "c" * 64),
    ),
    import_origin_manifest_sha256="d" * 64,
)


def _complete_run() -> RunReceipt:
    protocol = build_task18_protocol()
    run_id = "task18-run-test-001"
    source_report = ExternalSourceVerificationReport(
        commit_sha=SOURCE.commit_sha,
        tracked_tree_clean=True,
        source_files=SOURCE.files,
        import_origin_manifest_sha256=SOURCE.import_origin_manifest_sha256,
        imported_module_count=5,
        verification_report_sha256="e" * 64,
    )
    arms = tuple(
        _complete_arm_receipt(
            arm=arm,
            run_id=run_id,
            protocol=protocol,
            source_report=source_report,
        )
        for arm in protocol.arms
    )
    return RunReceipt(
        run_id=run_id,
        protocol_digest=protocol.adapted_protocol_digest,
        historical_protocol_digest=protocol.historical_reference.protocol_digest,
        dataset_id=protocol.historical_reference.dataset_id,
        manifest_sha256=protocol.historical_reference.manifest_sha256,
        arrays_sha256=protocol.historical_reference.arrays_sha256,
        source_binding=SOURCE,
        source_report=source_report,
        runtime_completed=True,
        arms=arms,
    )


def _complete_arm_receipt(
    *, arm, run_id: str, protocol, source_report: ExternalSourceVerificationReport
) -> ArmReceipt:
    terminal_equity = 100_100.0 if arm.role == "NONRL" else 100_000.0
    ledger_digest = sha256(f"ledger:{arm.arm_id}".encode()).hexdigest()
    result_digest = sha256(f"result:{arm.arm_id}".encode()).hexdigest()
    evidence = ExternalArmVerificationReport(
        run_id=run_id,
        protocol_digest=protocol.adapted_protocol_digest,
        source_report_binding_sha256=source_report.binding_sha256,
        arm_id=arm.arm_id,
        pair_id=arm.pair_id,
        role=arm.role,
        configuration=arm.configuration,
        symbol=arm.symbol,
        scenario=arm.scenario,
        dataset_id=protocol.historical_reference.dataset_id,
        native_ledger_sha256=ledger_digest,
        arm_result_sha256=result_digest,
        verification_report_sha256="f" * 64,
        expected_advances=protocol.index_plan.native_advances_per_arm,
        observed_advances=protocol.index_plan.native_advances_per_arm,
        initial_equity_usdt=100_000.0,
        terminal_equity_usdt=terminal_equity,
        max_drawdown=0.05,
        terminal_settlement_complete=True,
        terminal_flat=True,
        pending_order_count=0,
        cost_oracle_passed=True,
    )
    return ArmReceipt(
        arm_id=arm.arm_id,
        pair_id=arm.pair_id,
        role=arm.role,
        configuration=arm.configuration,
        symbol=arm.symbol,
        scenario=arm.scenario,
        protocol_digest=protocol.adapted_protocol_digest,
        dataset_id=protocol.historical_reference.dataset_id,
        manifest_sha256=protocol.historical_reference.manifest_sha256,
        arrays_sha256=protocol.historical_reference.arrays_sha256,
        source_binding_sha256=SOURCE.manifest_sha256,
        status="COMPLETED",
        initial_equity_usdt=100_000.0,
        terminal_equity_usdt=terminal_equity,
        max_drawdown=0.05,
        terminal_flat=True,
        pending_order_count=0,
        native_ledger_sha256=ledger_digest,
        arm_result_sha256=result_digest,
        external_report=evidence,
    )


def test_complete_receipt_reports_match_without_claiming_verification() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()

    result = evaluate_run_receipt(run, protocol)

    assert result.declaration_consistent
    assert result.source_report_matches
    assert result.arm_reports_match
    assert result.roster_complete
    assert result.declared_runtime_completed
    assert result.declared_arm_statuses_complete
    assert not hasattr(result, "software_valid")
    assert not hasattr(result, "terminal_integrity_valid")
    assert not hasattr(result, "drawdown_guardrails_valid")
    assert not hasattr(result, "pair_metrics_available")
    assert result.terminal_claims_pass_diagnostic
    assert result.drawdown_claims_pass_diagnostic
    assert result.diagnostic_pair_arithmetic_available
    assert result.winner_status == "NOT_DECLARED"
    assert result.adoption_status == "NOT_ESTABLISHED"
    assert diagnostic_pair_delta(run, protocol, "PAIR-01") == pytest.approx(0.001)
    assert diagnostic_combined_delta(
        run, protocol, "H24_RIDGE_A1", "BASEL0"
    ) == pytest.approx(0.001)


def test_duplicate_or_missing_arm_invalidates_complete_roster() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()

    duplicate = RunReceipt(**{**run.__dict__, "arms": run.arms[:-1] + (run.arms[0],)})
    missing = RunReceipt(**{**run.__dict__, "arms": run.arms[:-1]})

    assert not evaluate_run_receipt(duplicate, protocol).roster_complete
    assert not evaluate_run_receipt(missing, protocol).roster_complete
    with pytest.raises(ValueError, match="complete declaration and matching reports"):
        diagnostic_pair_delta(missing, protocol, "PAIR-01")


def test_malformed_arm_identity_fails_closed_without_hashing_bad_value() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    malformed = ArmReceipt(**{**run.arms[0].__dict__, "arm_id": []})
    invalid = RunReceipt(**{**run.__dict__, "arms": (malformed,) + run.arms[1:]})

    result = evaluate_run_receipt(invalid, protocol)

    assert not result.roster_complete
    assert not result.diagnostic_pair_arithmetic_available


def test_wrong_source_or_dataset_binding_fails_closed() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    wrong_source_arm = ArmReceipt(
        **{**run.arms[0].__dict__, "source_binding_sha256": "d" * 64}
    )
    wrong_source = RunReceipt(
        **{**run.__dict__, "arms": (wrong_source_arm,) + run.arms[1:]}
    )
    wrong_data = RunReceipt(**{**run.__dict__, "dataset_id": "e" * 64})

    wrong_source_result = evaluate_run_receipt(wrong_source, protocol)
    wrong_data_result = evaluate_run_receipt(wrong_data, protocol)

    assert not wrong_source_result.declaration_consistent
    assert not wrong_source_result.terminal_claims_pass_diagnostic
    assert not wrong_source_result.drawdown_claims_pass_diagnostic
    assert not wrong_data_result.declaration_consistent
    assert not wrong_data_result.terminal_claims_pass_diagnostic
    assert not wrong_data_result.drawdown_claims_pass_diagnostic


def test_nonfinite_metrics_and_missing_arm_evidence_are_not_software_valid() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    broken = ArmReceipt(
        **{
            **run.arms[0].__dict__,
            "terminal_equity_usdt": math.nan,
            "external_report": None,
        }
    )
    invalid = RunReceipt(**{**run.__dict__, "arms": (broken,) + run.arms[1:]})

    result = evaluate_run_receipt(invalid, protocol)

    assert not result.diagnostic_pair_arithmetic_available
    assert not result.declared_metrics_well_formed
    assert not result.terminal_claims_pass_diagnostic
    assert not result.drawdown_claims_pass_diagnostic


def test_terminal_flatness_and_drawdown_are_separate_guardrails() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    open_arm = ArmReceipt(
        **{
            **run.arms[0].__dict__,
            "terminal_flat": False,
            "pending_order_count": 1,
            "external_report": replace(
                run.arms[0].external_report,
                terminal_flat=False,
                pending_order_count=1,
            ),
        }
    )
    open_run = RunReceipt(**{**run.__dict__, "arms": (open_arm,) + run.arms[1:]})
    risk_arm = ArmReceipt(
        **{
            **run.arms[0].__dict__,
            "max_drawdown": 0.21,
            "external_report": replace(
                run.arms[0].external_report,
                max_drawdown=0.21,
            ),
        }
    )
    risk_run = RunReceipt(**{**run.__dict__, "arms": (risk_arm,) + run.arms[1:]})

    open_result = evaluate_run_receipt(open_run, protocol)
    risk_result = evaluate_run_receipt(risk_run, protocol)

    assert not open_result.terminal_claims_pass_diagnostic
    assert not open_result.diagnostic_pair_arithmetic_available
    assert risk_result.terminal_claims_pass_diagnostic
    assert not risk_result.drawdown_claims_pass_diagnostic
    assert risk_result.adoption_status == "NOT_ESTABLISHED"


def test_incomplete_runtime_cannot_pass_endpoint_or_drawdown_gates() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    failed = ArmReceipt(
        **{
            **run.arms[-1].__dict__,
            "status": "FAILED",
            "error_code": "NATIVE_EXECUTION_FAILED",
            "reason_code": "PROCESS_FAILED",
            "external_report": None,
        }
    )
    incomplete = RunReceipt(**{**run.__dict__, "arms": run.arms[:-1] + (failed,)})

    result = evaluate_run_receipt(incomplete, protocol)

    assert result.roster_complete
    assert not result.declared_arm_statuses_complete
    assert not result.terminal_claims_pass_diagnostic
    assert not result.drawdown_claims_pass_diagnostic
    assert not result.diagnostic_pair_arithmetic_available

    interrupted = RunReceipt(**{**run.__dict__, "runtime_completed": False})
    interrupted_result = evaluate_run_receipt(interrupted, protocol)

    assert interrupted_result.declared_arm_statuses_complete
    assert not interrupted_result.declared_runtime_completed
    assert not interrupted_result.terminal_claims_pass_diagnostic
    assert not interrupted_result.drawdown_claims_pass_diagnostic


def test_unverified_declarations_and_caller_flags_cannot_prove_source_or_arm() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    unverified_arms = tuple(replace(arm, external_report=None) for arm in run.arms)
    unverified = replace(run, source_report=None, arms=unverified_arms)

    result = evaluate_run_receipt(unverified, protocol)

    assert result.declaration_consistent
    assert not result.source_report_matches
    assert not result.arm_reports_match
    assert not result.diagnostic_pair_arithmetic_available
    assert "receipt_verified" not in unverified.arms[0].to_mapping()
    assert "source_verified" not in result.__dict__
    assert "software_valid" not in result.__dict__


def test_source_report_binding_requires_clean_exact_manifest_claims() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    dirty = replace(run.source_report, tracked_tree_clean=False)
    missing_imports = replace(run.source_report, import_origin_manifest_sha256="e" * 64)
    mismatched_source = replace(
        run.source_report,
        source_files=(SourceFilePin("tools/other.py", "f" * 64),),
    )

    for snapshot in (dirty, missing_imports, mismatched_source):
        result = evaluate_run_receipt(replace(run, source_report=snapshot), protocol)
        assert result.declaration_consistent
        assert not result.source_report_matches
        assert not result.diagnostic_pair_arithmetic_available


def test_arm_verification_must_bind_actual_native_ledger_and_facts() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    first = run.arms[0]
    stale_evidence = replace(
        first.external_report,
        observed_advances=first.external_report.observed_advances - 1,
    )
    stale_arm = replace(first, external_report=stale_evidence)
    stale_run = replace(run, arms=(stale_arm,) + run.arms[1:])

    result = evaluate_run_receipt(stale_run, protocol)

    assert result.declaration_consistent
    assert result.source_report_matches
    assert not result.arm_reports_match
    assert not result.diagnostic_pair_arithmetic_available

    wrong_ledger_evidence = replace(
        first.external_report,
        native_ledger_sha256="0" * 64,
    )
    wrong_ledger_arm = replace(first, external_report=wrong_ledger_evidence)
    wrong_ledger_run = replace(run, arms=(wrong_ledger_arm,) + run.arms[1:])
    wrong_ledger_result = evaluate_run_receipt(wrong_ledger_run, protocol)
    assert not wrong_ledger_result.arm_reports_match
    assert not wrong_ledger_result.diagnostic_pair_arithmetic_available


def test_failed_raw_arm_is_retained_without_authoritative_metrics() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    failed = replace(
        run.arms[-1],
        status="FAILED",
        error_code="NATIVE_EXECUTION_FAILED",
        reason_code="PROCESS_FAILED",
        external_report=None,
    )
    incomplete = replace(run, arms=run.arms[:-1] + (failed,))

    mapping = incomplete.to_mapping()
    result = evaluate_run_receipt(incomplete, protocol)

    assert mapping["arms"][-1]["status"] == "FAILED"
    assert mapping["arms"][-1]["error_code"] == "NATIVE_EXECUTION_FAILED"
    assert not result.declared_arm_statuses_complete
    assert not result.terminal_claims_pass_diagnostic
    assert not result.drawdown_claims_pass_diagnostic
    assert not result.diagnostic_pair_arithmetic_available

    all_failed = replace(
        run,
        arms=tuple(
            replace(
                arm,
                status="FAILED",
                error_code="NATIVE_EXECUTION_FAILED",
                reason_code="PROCESS_FAILED",
                external_report=None,
            )
            for arm in run.arms
        ),
    )
    all_failed_result = evaluate_run_receipt(all_failed, protocol)
    assert not all_failed_result.declared_arm_statuses_complete
    assert not all_failed_result.arm_reports_match
    assert not all_failed_result.diagnostic_pair_arithmetic_available


def test_prelaunch_receipt_has_explicit_not_attempted_slot_for_every_arm() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    slots = tuple(
        replace(
            arm,
            status="NOT_ATTEMPTED",
            initial_equity_usdt=None,
            terminal_equity_usdt=None,
            max_drawdown=None,
            terminal_flat=None,
            pending_order_count=None,
            native_ledger_sha256=None,
            arm_result_sha256=None,
            external_report=None,
            error_code=None,
            reason_code="NOT_STARTED",
            progress=None,
        )
        for arm in run.arms
    )
    prelaunch = replace(run, source_report=None, runtime_completed=False, arms=slots)

    evaluated = evaluate_run_receipt(prelaunch, protocol)
    encoded = json.dumps(prelaunch.to_mapping(), sort_keys=True)
    decoded = json.loads(encoded)

    assert len(decoded["arms"]) == 48
    assert decoded["schema"] == "task18_run_receipt_v3"
    assert decoded["source_report"] is None
    assert all(slot["status"] == "NOT_ATTEMPTED" for slot in decoded["arms"])
    assert all(slot["reason_code"] == "NOT_STARTED" for slot in decoded["arms"])
    assert all(slot["terminal_equity_usdt"] is None for slot in decoded["arms"])
    assert all(slot["progress"] is None for slot in decoded["arms"])
    assert evaluated.roster_complete
    assert not evaluated.declared_runtime_completed
    assert not evaluated.declared_arm_statuses_complete
    assert not evaluated.terminal_claims_pass_diagnostic
    assert not evaluated.drawdown_claims_pass_diagnostic
    assert not evaluated.diagnostic_pair_arithmetic_available
    with pytest.raises(ValueError, match="complete declaration and matching reports"):
        diagnostic_pair_delta(prelaunch, protocol, "PAIR-01")


def test_not_attempted_slot_cannot_carry_completion_only_facts() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    contradictory_slot = replace(
        run.arms[0],
        status="NOT_ATTEMPTED",
        reason_code="NOT_STARTED",
    )
    contradictory = replace(run, arms=(contradictory_slot,) + run.arms[1:])

    evaluated = evaluate_run_receipt(contradictory, protocol)

    assert not evaluated.declared_metrics_well_formed
    assert not evaluated.declared_arm_statuses_complete
    assert not evaluated.terminal_claims_pass_diagnostic
    assert not evaluated.drawdown_claims_pass_diagnostic
    assert not evaluated.diagnostic_pair_arithmetic_available


def test_interrupted_arm_preserves_known_progress_and_unknown_facts_as_none() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    interrupted_arm = replace(
        run.arms[0],
        status="INTERRUPTED",
        initial_equity_usdt=None,
        terminal_equity_usdt=None,
        max_drawdown=None,
        terminal_flat=None,
        pending_order_count=None,
        native_ledger_sha256=None,
        arm_result_sha256=None,
        external_report=None,
        error_code=None,
        reason_code="PROCESS_INTERRUPTED",
        progress=ArmAttemptProgress(
            native_advances_observed=37,
            last_native_index=2940,
            last_observed_equity_usdt=None,
            max_drawdown_observed=None,
        ),
    )
    remaining = tuple(
        replace(
            arm,
            status="NOT_ATTEMPTED",
            initial_equity_usdt=None,
            terminal_equity_usdt=None,
            max_drawdown=None,
            terminal_flat=None,
            pending_order_count=None,
            native_ledger_sha256=None,
            arm_result_sha256=None,
            external_report=None,
            error_code=None,
            reason_code="RUN_INTERRUPTED_BEFORE_START",
            progress=None,
        )
        for arm in run.arms[1:]
    )
    interrupted = replace(
        run,
        runtime_completed=False,
        arms=(interrupted_arm,) + remaining,
    )

    evaluated = evaluate_run_receipt(interrupted, protocol)
    decoded = json.loads(json.dumps(interrupted.to_mapping(), sort_keys=True))
    interrupted_mapping = decoded["arms"][0]

    assert len(decoded["arms"]) == 48
    assert interrupted_mapping["status"] == "INTERRUPTED"
    assert interrupted_mapping["reason_code"] == "PROCESS_INTERRUPTED"
    assert interrupted_mapping["progress"] == {
        "native_advances_observed": 37,
        "last_native_index": 2940,
        "last_observed_equity_usdt": None,
        "max_drawdown_observed": None,
    }
    assert interrupted_mapping["terminal_equity_usdt"] is None
    assert all(slot["status"] == "NOT_ATTEMPTED" for slot in decoded["arms"][1:])
    assert evaluated.roster_complete
    assert not evaluated.declared_arm_statuses_complete
    assert not evaluated.terminal_claims_pass_diagnostic
    assert not evaluated.drawdown_claims_pass_diagnostic
    assert not evaluated.diagnostic_pair_arithmetic_available


def test_attempt_slots_require_status_specific_reason_codes() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    first = run.arms[0]
    not_attempted = replace(
        first,
        status="NOT_ATTEMPTED",
        reason_code=None,
        external_report=None,
    )
    wrong_reason = replace(
        first,
        status="INTERRUPTED",
        reason_code="NOT_STARTED",
        external_report=None,
    )
    invalid = replace(run, arms=(not_attempted,) + run.arms[1:])
    mismatched = replace(run, arms=(wrong_reason,) + run.arms[1:])

    for receipt in (invalid, mismatched):
        evaluated = evaluate_run_receipt(receipt, protocol)
        assert not evaluated.declared_metrics_well_formed
        assert not evaluated.declared_arm_statuses_complete
        assert not evaluated.diagnostic_pair_arithmetic_available


def test_running_arm_snapshot_cannot_be_treated_as_completed() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    running = replace(
        run.arms[0],
        status="RUNNING",
        reason_code="RUN_IN_PROGRESS",
        initial_equity_usdt=None,
        terminal_equity_usdt=None,
        max_drawdown=None,
        terminal_flat=None,
        pending_order_count=None,
        native_ledger_sha256=None,
        arm_result_sha256=None,
        external_report=None,
        error_code=None,
        progress=ArmAttemptProgress(native_advances_observed=12),
    )
    snapshot = replace(run, runtime_completed=False, arms=(running,) + run.arms[1:])

    evaluated = evaluate_run_receipt(snapshot, protocol)

    assert evaluated.roster_complete
    assert not evaluated.declared_runtime_completed
    assert not evaluated.declared_arm_statuses_complete
    assert not evaluated.terminal_claims_pass_diagnostic
    assert not evaluated.drawdown_claims_pass_diagnostic
    assert not evaluated.diagnostic_pair_arithmetic_available


def test_malformed_progress_type_fails_closed() -> None:
    protocol = build_task18_protocol()
    run = _complete_run()
    malformed = replace(run.arms[0], progress={"native_advances_observed": 1})  # type: ignore[arg-type]
    receipt = replace(run, arms=(malformed,) + run.arms[1:])

    evaluated = evaluate_run_receipt(receipt, protocol)

    assert not evaluated.declared_metrics_well_formed
    assert not evaluated.diagnostic_pair_arithmetic_available


def test_source_manifest_rejects_path_escape_and_duplicate_paths() -> None:
    with pytest.raises(ValueError, match="repo-relative"):
        SourceBinding(
            commit_sha="a" * 40,
            files=(SourceFilePin("../outside.py", "b" * 64),),
        )
    with pytest.raises(ValueError, match="duplicate"):
        SourceBinding(
            commit_sha="a" * 40,
            files=(
                SourceFilePin("tools/a.py", "b" * 64),
                SourceFilePin("tools/a.py", "c" * 64),
            ),
        )


@pytest.mark.parametrize(
    "path",
    (
        "tools//nested.py",
        "tools/./nested.py",
        "C:/outside.py",
        "tools/\x00nested.py",
    ),
)
def test_source_manifest_rejects_noncanonical_or_windows_escape_paths(
    path: str,
) -> None:
    with pytest.raises(ValueError, match="repo-relative"):
        SourceFilePin(path, "b" * 64)


def test_source_binding_rejects_mutable_pin_collections() -> None:
    with pytest.raises(ValueError, match="immutable"):
        SourceBinding(
            commit_sha="a" * 40,
            files=[SourceFilePin("tools/a.py", "b" * 64)],  # type: ignore[arg-type]
        )


def test_native_cost_oracle_failure_blocks_receipt_arithmetic() -> None:
    run = _complete_run()
    arm = run.arms[0]
    assert arm.external_report is not None
    altered = replace(
        arm, external_report=replace(arm.external_report, cost_oracle_passed=False)
    )
    modified = replace(run, arms=(altered,) + run.arms[1:])
    evaluated = evaluate_run_receipt(modified, build_task18_protocol())
    assert not evaluated.declared_metrics_well_formed
    assert not evaluated.diagnostic_pair_arithmetic_available
    assert evaluated.adoption_status == "NOT_ESTABLISHED"


def test_noncompleted_arm_cannot_claim_terminal_verification_report() -> None:
    run = _complete_run()
    arm = replace(run.arms[0], status="INTERRUPTED", reason_code="PROCESS_INTERRUPTED")
    evaluated = evaluate_run_receipt(
        replace(run, runtime_completed=False, arms=(arm,) + run.arms[1:]),
        build_task18_protocol(),
    )
    assert not evaluated.declared_metrics_well_formed
    assert not evaluated.diagnostic_pair_arithmetic_available
