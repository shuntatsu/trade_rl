from pathlib import Path

from mars_lite.server.deployment_gate import DeploymentEvidence, DeploymentGate


def test_deployment_gate_blocks_production_without_canary():
    gate = DeploymentGate()
    evidence = DeploymentEvidence(
        stage="production",
        shadow_passed=True,
        model_version="1.0.0",
        git_commit="a" * 40,
        drift_report_passed=True,
        approval_ticket="PROD-123",
    )

    decision = gate.evaluate(evidence)

    assert decision.allowed is False
    assert "canary" in decision.reason


def test_deployment_gate_allows_ordered_shadow_canary_production():
    gate = DeploymentGate()

    assert gate.evaluate(DeploymentEvidence(stage="shadow")).allowed is True
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="canary",
                shadow_passed=True,
                model_version="1.0.0",
                git_commit="a" * 40,
                drift_report_passed=True,
            )
        ).allowed
        is True
    )
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="production",
                shadow_passed=True,
                canary_passed=True,
                model_version="1.0.0",
                git_commit="a" * 40,
                drift_report_passed=True,
                approval_ticket="PROD-123",
            )
        ).allowed
        is True
    )


def test_deployment_gate_active_incidents_blocks_all_stages():
    gate = DeploymentGate()

    # shadow
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="shadow",
                active_incidents=True,
            )
        ).allowed
        is False
    )

    # canary
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="canary",
                shadow_passed=True,
                model_version="1.0.0",
                git_commit="a" * 40,
                drift_report_passed=True,
                active_incidents=True,
            )
        ).allowed
        is False
    )

    # production
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="production",
                shadow_passed=True,
                canary_passed=True,
                model_version="1.0.0",
                git_commit="a" * 40,
                drift_report_passed=True,
                approval_ticket="PROD-123",
                active_incidents=True,
            )
        ).allowed
        is False
    )


def test_deployment_gate_git_commit_sha1_validation():
    gate = DeploymentGate()

    valid_commit = "a" * 40
    invalid_commit_short = "a" * 39
    invalid_commit_long = "a" * 41
    invalid_commit_chars = "z" * 40  # Wait, SHA-1 is hexadecimal [a-fA-F0-9]. 'z' is invalid.

    # Canary validation
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="canary",
                shadow_passed=True,
                model_version="1.0.0",
                git_commit=valid_commit,
                drift_report_passed=True,
            )
        ).allowed
        is True
    )

    for invalid in [None, "", invalid_commit_short, invalid_commit_long, invalid_commit_chars]:
        assert (
            gate.evaluate(
                DeploymentEvidence(
                    stage="canary",
                    shadow_passed=True,
                    model_version="1.0.0",
                    git_commit=invalid,
                    drift_report_passed=True,
                )
            ).allowed
            is False
        )

    # Production validation
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="production",
                shadow_passed=True,
                canary_passed=True,
                model_version="1.0.0",
                git_commit=valid_commit,
                drift_report_passed=True,
                approval_ticket="PROD-123",
            )
        ).allowed
        is True
    )

    for invalid in [None, "", invalid_commit_short, invalid_commit_long, invalid_commit_chars]:
        assert (
            gate.evaluate(
                DeploymentEvidence(
                    stage="production",
                    shadow_passed=True,
                    canary_passed=True,
                    model_version="1.0.0",
                    git_commit=invalid,
                    drift_report_passed=True,
                    approval_ticket="PROD-123",
                )
            ).allowed
            is False
        )


def test_deployment_gate_approval_ticket_validation():
    gate = DeploymentGate()

    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="production",
                shadow_passed=True,
                canary_passed=True,
                model_version="1.0.0",
                git_commit="a" * 40,
                drift_report_passed=True,
                approval_ticket="PROD-12345",
            )
        ).allowed
        is True
    )

    for invalid_ticket in [None, "", "PROD-", "PROD-abc", "APPROVED-1", "PROD-12a"]:
        assert (
            gate.evaluate(
                DeploymentEvidence(
                    stage="production",
                    shadow_passed=True,
                    canary_passed=True,
                    model_version="1.0.0",
                    git_commit="a" * 40,
                    drift_report_passed=True,
                    approval_ticket=invalid_ticket,
                )
            ).allowed
            is False
        )


def test_deployment_gate_model_version_and_drift_report_validation():
    gate = DeploymentGate()

    # Canary missing model version
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="canary",
                shadow_passed=True,
                model_version=None,
                git_commit="a" * 40,
                drift_report_passed=True,
            )
        ).allowed
        is False
    )

    # Canary missing drift report
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="canary",
                shadow_passed=True,
                model_version="1.0.0",
                git_commit="a" * 40,
                drift_report_passed=False,
            )
        ).allowed
        is False
    )

    # Production missing model version
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="production",
                shadow_passed=True,
                canary_passed=True,
                model_version=None,
                git_commit="a" * 40,
                drift_report_passed=True,
                approval_ticket="PROD-123",
            )
        ).allowed
        is False
    )

    # Production missing drift report
    assert (
        gate.evaluate(
            DeploymentEvidence(
                stage="production",
                shadow_passed=True,
                canary_passed=True,
                model_version="1.0.0",
                git_commit="a" * 40,
                drift_report_passed=False,
                approval_ticket="PROD-123",
            )
        ).allowed
        is False
    )


def test_required_runbooks_exist():
    for path in [
        Path("docs/runbook_incident_response.md"),
        Path("docs/runbook_compliance.md"),
        Path("docs/model_decision_log.md"),
        Path("docs/runbook_model_rollback.md"),
    ]:
        assert path.exists()


def test_deployment_gate_security_hardening():
    gate = DeploymentGate()

    # 1. git_commit 改行コードインジェクションのテスト
    for stage in ["canary", "production"]:
        for invalid_commit in ["a" * 40 + "\n", "\n" + "a" * 40, "a" * 40 + "\r\n"]:
            evidence = DeploymentEvidence(
                stage=stage,
                shadow_passed=True,
                canary_passed=True,
                model_version="1.0.0",
                git_commit=invalid_commit,
                drift_report_passed=True,
                approval_ticket="PROD-123",
            )
            assert gate.evaluate(evidence).allowed is False

    # 2. approval_ticket 改行コードインジェクションのテスト
    for invalid_ticket in ["PROD-123\n", "\nPROD-123", "PROD-123\r\n"]:
        evidence = DeploymentEvidence(
            stage="production",
            shadow_passed=True,
            canary_passed=True,
            model_version="1.0.0",
            git_commit="a" * 40,
            drift_report_passed=True,
            approval_ticket=invalid_ticket,
        )
        assert gate.evaluate(evidence).allowed is False

    # 3. model_version の不正文字テスト
    invalid_versions = [
        "1.0.0 ",
        " 1.0.0",
        "1.0.0\n",
        "1.0.0\r\n",
        "1.0.0-beta!",
        "1.0.0/2",
        "1.0.0$",
        "v1_0_0@",
    ]
    for stage in ["canary", "production"]:
        for invalid_version in invalid_versions:
            evidence = DeploymentEvidence(
                stage=stage,
                shadow_passed=True,
                canary_passed=True,
                model_version=invalid_version,
                git_commit="a" * 40,
                drift_report_passed=True,
                approval_ticket="PROD-123",
            )
            decision = gate.evaluate(evidence)
            assert decision.allowed is False

