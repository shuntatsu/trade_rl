import os

import pytest

from mars_lite.server.deployment_gate import DeploymentEvidence, DeploymentGate


def test_adversarial_git_commit_newline_bypass():
    # 修正確認: 末尾の改行コードが強固にブロックされること
    gate = DeploymentGate()

    # 40文字 + 改行
    bad_commit = "a" * 40 + "\n"
    evidence = DeploymentEvidence(
        stage="canary",
        shadow_passed=True,
        model_version="1.0.0",
        git_commit=bad_commit,
        drift_report_passed=True,
    )

    # セキュリティ改善により、改行コード付きのコミットハッシュは拒否される
    decision = gate.evaluate(evidence)
    assert decision.allowed is False  # ブロック成功を確認


def test_adversarial_approval_ticket_newline_bypass():
    # 修正確認: 末尾の改行コードが強固にブロックされること
    gate = DeploymentGate()

    # PROD-123 + 改行
    bad_ticket = "PROD-123\n"
    evidence = DeploymentEvidence(
        stage="production",
        shadow_passed=True,
        canary_passed=True,
        model_version="1.0.0",
        git_commit="a" * 40,
        drift_report_passed=True,
        approval_ticket=bad_ticket,
    )

    # セキュリティ改善により、改行コード付きの承認チケットは拒否される
    decision = gate.evaluate(evidence)
    assert decision.allowed is False  # ブロック成功を確認


def test_adversarial_spaces_in_ticket():
    gate = DeploymentGate()

    # 前後にスペースがある場合
    for ticket in [" PROD-123", "PROD-123 ", "PROD- 123", "PROD-123\t"]:
        evidence = DeploymentEvidence(
            stage="production",
            shadow_passed=True,
            canary_passed=True,
            model_version="1.0.0",
            git_commit="a" * 40,
            drift_report_passed=True,
            approval_ticket=ticket,
        )
        decision = gate.evaluate(evidence)
        # model_version のバリデーションにより、不正な文字や極端な長さは False になる
        assert decision.allowed is False


def test_adversarial_model_version_abuse():
    gate = DeploymentGate()

    # 1. 特殊文字やコマンドインジェクション試行は正規表現 [a-zA-Z0-9_\-\.]+ により安全にブロックされる
    special_version = "v1.0.0; DROP TABLE models; --"
    evidence_special = DeploymentEvidence(
        stage="canary",
        shadow_passed=True,
        model_version=special_version,
        git_commit="a" * 40,
        drift_report_passed=True,
    )
    decision_special = gate.evaluate(evidence_special)
    assert decision_special.allowed is False
    assert "model version" in decision_special.reason

    # 2. 【指摘事項】極端に長い文字列（文字種が正しい場合）に対する長さの上限チェックが存在しない
    # 正規表現 [a-zA-Z0-9_\-\.]+ では長さが無制限であるため、10万文字の 'X' でも通過してしまう
    huge_version = "X" * 100000
    evidence_huge = DeploymentEvidence(
        stage="canary",
        shadow_passed=True,
        model_version=huge_version,
        git_commit="a" * 40,
        drift_report_passed=True,
    )
    decision_huge = gate.evaluate(evidence_huge)
    # 修正後: 長さ制限（50文字）によりブロックされること
    assert decision_huge.allowed is False
    assert (
        decision_huge.reason == "model version exceeds maximum length of 50 characters"
    )


def test_adversarial_approval_ticket_abuse():
    gate = DeploymentGate()

    # 【指摘事項】approval_ticket (PROD-\d+) に対しても、数字の長さの上限チェックが存在しない
    # PROD- の後ろに10万個の数字が並んでいても通過してしまう
    huge_ticket = "PROD-" + "1" * 100000
    evidence_huge = DeploymentEvidence(
        stage="production",
        shadow_passed=True,
        canary_passed=True,
        model_version="1.0.0",
        git_commit="a" * 40,
        drift_report_passed=True,
        approval_ticket=huge_ticket,
    )
    decision_huge = gate.evaluate(evidence_huge)
    # 修正後: 長さ制限（20文字）によりブロックされること
    assert decision_huge.allowed is False
    assert (
        decision_huge.reason
        == "approval ticket exceeds maximum length of 20 characters"
    )


def test_adversarial_unknown_stages():
    gate = DeploymentGate()

    # 存在しないステージ
    for stage in ["production ", "PRODUCTION", "test", "staging", ""]:
        evidence = DeploymentEvidence(
            stage=stage,
            shadow_passed=True,
            canary_passed=True,
            model_version="1.0.0",
            git_commit="a" * 40,
            drift_report_passed=True,
            approval_ticket="PROD-123",
        )
        decision = gate.evaluate(evidence)
        assert decision.allowed is False
        assert "unknown deployment stage" in decision.reason


def test_adversarial_active_incidents_truthy():
    gate = DeploymentGate()

    # active_incidents が bool 以外の Truthy 値の場合
    for incident_val in [1, "true", [1], "yes"]:
        evidence = DeploymentEvidence(
            stage="shadow",
            active_incidents=incident_val,
        )
        decision = gate.evaluate(evidence)
        # Python の if evidence.active_incidents: により True と判定されブロックされる（安全）
        assert decision.allowed is False
        assert "blocked due to active incidents" in decision.reason


def test_deploy_yml_parsing_simulation():
    # deploy.yml で環境変数から読み込む際のパースシミュレーション
    env_ticket = "   "
    approval_ticket = env_ticket or None

    assert approval_ticket == "   "

    gate = DeploymentGate()
    evidence = DeploymentEvidence(
        stage="production",
        shadow_passed=True,
        canary_passed=True,
        model_version="1.0.0",
        git_commit="a" * 40,
        drift_report_passed=True,
        approval_ticket=approval_ticket,
    )
    decision = gate.evaluate(evidence)
    assert decision.allowed is False

    # SHADOW_PASSED のパース
    env_shadow = "true\n"
    shadow_passed = env_shadow.lower() == "true"
    assert shadow_passed is False  # 末尾改行があると False になる

    # ACTIVE_INCIDENTS のパース
    active_incidents_input = "false"
    active_incidents = active_incidents_input.lower() == "true"
    assert active_incidents is False
