from __future__ import annotations

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


def _report(*, requested: RuntimeMode, allowed: bool):
    return build_execution_promotion_report(
        requested=requested,
        evidence=ExecutionPromotionEvidence(
            capability_passed=True,
            causal_bridge_passed=True,
            funding_passed=True,
            terminal_flat_passed=True,
            exact_parity_passed=allowed,
            determinism_passed=True,
            performance_approved=allowed,
        ),
    )


def _proposal(execution_evidence_digest: str) -> SelectionProposal:
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
        execution_evidence_digest=execution_evidence_digest,
    )


def test_selection_binding_accepts_exact_allowed_promotion_report() -> None:
    report = _report(requested=RuntimeMode.NAUTILUS_AUTHORITATIVE, allowed=True)
    proposal = _proposal(report.digest)

    require_selection_execution_promotion(
        proposal=proposal,
        report=report,
        required_mode=RuntimeMode.NAUTILUS_AUTHORITATIVE,
    )


def test_selection_binding_rejects_denied_promotion_report() -> None:
    report = _report(requested=RuntimeMode.NAUTILUS_AUTHORITATIVE, allowed=False)
    proposal = _proposal(report.digest)

    with pytest.raises(ValueError, match="execution promotion is not allowed"):
        require_selection_execution_promotion(
            proposal=proposal,
            report=report,
            required_mode=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        )


def test_selection_binding_rejects_wrong_mode_or_digest() -> None:
    dual_shadow = _report(requested=RuntimeMode.DUAL_SHADOW, allowed=True)
    proposal = _proposal(dual_shadow.digest)

    with pytest.raises(ValueError, match="execution promotion mode mismatch"):
        require_selection_execution_promotion(
            proposal=proposal,
            report=dual_shadow,
            required_mode=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        )

    authoritative = _report(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        allowed=True,
    )
    with pytest.raises(
        ValueError,
        match="selection proposal execution evidence digest mismatch",
    ):
        require_selection_execution_promotion(
            proposal=proposal,
            report=authoritative,
            required_mode=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        )
