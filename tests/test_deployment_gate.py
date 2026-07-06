from pathlib import Path

from mars_lite.server.deployment_gate import DeploymentEvidence, DeploymentGate


def test_deployment_gate_blocks_production_without_canary():
    gate = DeploymentGate()
    evidence = DeploymentEvidence(stage="production", shadow_passed=True)

    decision = gate.evaluate(evidence)

    assert decision.allowed is False
    assert "canary" in decision.reason


def test_deployment_gate_allows_ordered_shadow_canary_production():
    gate = DeploymentGate()

    assert gate.evaluate(DeploymentEvidence(stage="shadow")).allowed is True
    assert (
        gate.evaluate(DeploymentEvidence(stage="canary", shadow_passed=True)).allowed
        is True
    )
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="production",
                shadow_passed=True,
                canary_passed=True,
                approval_ticket="APPROVED-1",
            )
        ).allowed
        is True
    )


def test_required_runbooks_exist():
    for path in [
        Path("docs/runbook_incident_response.md"),
        Path("docs/runbook_compliance.md"),
        Path("docs/model_decision_log.md"),
    ]:
        assert path.exists()
