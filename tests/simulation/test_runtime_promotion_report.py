import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation import runtime_promotion
from trade_rl.simulation.runtime_promotion import (
    EXECUTION_PROMOTION_REPORT_SCHEMA,
    ExecutionPromotionEvidence,
    ExecutionPromotionReport,
    RuntimeMode,
    RuntimePromotionDecision,
    build_execution_promotion_report,
)


def _evidence(*, performance_approved: bool) -> ExecutionPromotionEvidence:
    return ExecutionPromotionEvidence(
        capability_passed=True,
        causal_bridge_passed=True,
        funding_passed=True,
        terminal_flat_passed=True,
        exact_parity_passed=False,
        determinism_passed=True,
        performance_approved=performance_approved,
    )


def test_promotion_report_is_deterministic_and_fail_closed() -> None:
    first = build_execution_promotion_report(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=_evidence(performance_approved=False),
    )
    second = build_execution_promotion_report(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=_evidence(performance_approved=False),
    )

    assert first == second
    assert len(first.digest) == 64
    assert first.decision.allowed is False
    assert first.decision.missing == (
        "exact_parity_passed",
        "performance_approved",
    )
    assert first.to_mapping()["requested_mode"] == "nautilus_authoritative"
    assert first.to_mapping()["allowed"] is False


def test_promotion_report_rejects_decision_inconsistent_with_evidence() -> None:
    evidence = _evidence(performance_approved=False)
    forged_decision = RuntimePromotionDecision(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        allowed=True,
        missing=(),
    )
    forged_payload = {
        "allowed": True,
        "evidence": {
            "capability_passed": True,
            "causal_bridge_passed": True,
            "funding_passed": True,
            "terminal_flat_passed": True,
            "exact_parity_passed": False,
            "determinism_passed": True,
            "performance_approved": False,
        },
        "missing": (),
        "requested_mode": "nautilus_authoritative",
        "schema_version": EXECUTION_PROMOTION_REPORT_SCHEMA,
    }

    with pytest.raises(ValueError, match="decision does not match evidence"):
        ExecutionPromotionReport(
            digest=content_digest(forged_payload),
            requested_mode=RuntimeMode.NAUTILUS_AUTHORITATIVE,
            evidence=evidence,
            decision=forged_decision,
        )


def test_promotion_report_round_trips_mapping_and_rejects_tampering() -> None:
    report = build_execution_promotion_report(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=_evidence(performance_approved=False),
    )

    loaded = ExecutionPromotionReport.from_mapping(report.to_mapping())
    assert loaded == report

    tampered = report.to_mapping()
    tampered["allowed"] = True
    with pytest.raises(ValueError, match="promotion report is invalid"):
        ExecutionPromotionReport.from_mapping(tampered)


def test_promotion_report_persists_immutably(tmp_path) -> None:
    report = build_execution_promotion_report(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=_evidence(performance_approved=False),
    )
    path = tmp_path / "execution-promotion.json"

    assert hasattr(runtime_promotion, "write_execution_promotion_report")
    assert hasattr(runtime_promotion, "load_execution_promotion_report")
    runtime_promotion.write_execution_promotion_report(path, report)
    assert runtime_promotion.load_execution_promotion_report(path) == report

    same_path = runtime_promotion.write_execution_promotion_report(path, report)
    assert same_path == path

    different = build_execution_promotion_report(
        requested=RuntimeMode.DUAL_SHADOW,
        evidence=_evidence(performance_approved=False),
    )
    with pytest.raises(
        FileExistsError, match="refusing to overwrite immutable evidence"
    ):
        runtime_promotion.write_execution_promotion_report(path, different)


def test_selection_proposal_binds_exact_promotion_report_digest() -> None:
    report = build_execution_promotion_report(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=_evidence(performance_approved=False),
    )
    proposal = SelectionProposal.create(
        walk_forward_run_digest="1" * 64,
        gate_evidence_digest="2" * 64,
        execution_sensitivity_digest="3" * 64,
        dataset_id="4" * 64,
        selected_configuration="candidate-a",
        candidate_config_digest="5" * 64,
        seeds=(7, 11),
        git_commit="a" * 40,
        dependency_digest="6" * 64,
        resume_checkpoint_digests=(),
        execution_evidence_digest=report.digest,
    )

    proposal.require_execution_evidence_digest(report.digest)
    assert proposal.execution_evidence_digest == report.digest
