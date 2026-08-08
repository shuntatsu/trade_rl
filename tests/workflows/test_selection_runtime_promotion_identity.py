from __future__ import annotations

import json

import pytest

from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionEvidence,
    RuntimeMode,
    build_execution_promotion_report,
)
from trade_rl.workflows.runtime_promotion_binding import (
    require_selection_execution_promotion,
)


def _allowed_report():
    return build_execution_promotion_report(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=ExecutionPromotionEvidence(
            capability_passed=True,
            causal_bridge_passed=True,
            funding_passed=True,
            terminal_flat_passed=True,
            exact_parity_passed=True,
            determinism_passed=True,
            performance_approved=True,
        ),
        representative_evidence_digest="8" * 64,
        performance_evidence_digest="9" * 64,
    )


def _proposal(*, execution_digest: str, runtime_digest: str) -> SelectionProposal:
    return SelectionProposal.create(
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
        execution_evidence_digest=execution_digest,
        runtime_promotion_report_digest=runtime_digest,
    )


def test_selection_proposal_keeps_execution_and_runtime_promotion_evidence_distinct() -> (
    None
):
    report = _allowed_report()
    execution_digest = "e" * 64
    proposal = _proposal(
        execution_digest=execution_digest,
        runtime_digest=report.digest,
    )

    assert proposal.execution_evidence_digest == execution_digest
    assert proposal.runtime_promotion_report_digest == report.digest
    proposal.require_execution_evidence_digest(execution_digest)
    proposal.require_runtime_promotion_report_digest(report.digest)

    serialized = json.loads(json.dumps(proposal.to_mapping()))
    restored = SelectionProposal.from_mapping(serialized)
    assert restored == proposal


def test_runtime_binding_never_accepts_legacy_execution_evidence_digest() -> None:
    report = _allowed_report()
    proposal = _proposal(
        execution_digest=report.digest,
        runtime_digest="f" * 64,
    )

    with pytest.raises(
        ValueError,
        match="selection proposal runtime promotion report digest mismatch",
    ):
        require_selection_execution_promotion(
            proposal=proposal,
            report=report,
            required_mode=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        )
