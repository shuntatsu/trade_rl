import hashlib
import json
from pathlib import Path

import pytest

from mars_lite.pipeline.release_eligibility import derive_release_eligibility
from mars_lite.pipeline.release_risk import ReleaseRiskPolicy
from mars_lite.server.deployment_gate import (
    DeploymentEvidence,
    DeploymentGate,
    load_evidence_bundle,
)
from mars_lite.serving.candidate import create_candidate_bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eligibility():
    return derive_release_eligibility(
        forced=False,
        skip_p0=False,
        skip_pbt=False,
        skip_wf=False,
        skip_gate=False,
        sealed_holdout_used=True,
        p0_passed=True,
        signal_gate_passed=True,
        walk_forward_passed=True,
        gate2_passed=True,
        significance_passed=None,
    )


def _risk() -> ReleaseRiskPolicy:
    return ReleaseRiskPolicy(
        max_leverage=1.0,
        max_single_weight=0.5,
        max_net_exposure=1.0,
        max_worst_case_notional=100_000.0,
        min_order_notional=10.0,
        symbol_liquidity_caps={"BTCUSDT": 50_000.0},
        forbidden_symbols=(),
    )


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    source = tmp_path / "model-source.zip"
    source.write_bytes(b"model")
    serving = create_candidate_bundle(
        destination=root / "serving_candidate",
        model_source=source,
        version="v1.0.0",
        git_sha="a" * 40,
        symbols=("BTCUSDT",),
        feature_names=("ret",),
        global_feature_names=(),
        feature_norm="none",
        feature_mask=None,
        observation_dim=5,
        observation_schema_version=1,
        post_processor={},
        run_config={
            "base_timeframe": "1h",
            "observation_progress_mode": "zero",
        },
        metrics={"gate2": {"passed": True}},
        guardrails={},
        risk_policy=_risk(),
        release_eligibility=_eligibility(),
    )
    manifest = serving / "manifest.json"
    identity = {
        "model_version": "v1.0.0",
        "git_commit": "a" * 40,
        "artifact_sha256": _sha(manifest),
    }
    reports = {
        "shadow.json": {
            "run_id": "shadow-run",
            **identity,
            "oos_sharpe": 1.0,
            "baseline_sharpe": 1.0,
            "max_drawdown": 0.1,
        },
        "drift.json": {
            "report_id": "drift-run",
            **identity,
            "psi_score": 0.1,
            "ks_p_value": 0.5,
        },
        "incident.json": {
            "report_id": "incident-run",
            **identity,
            "active_incidents": False,
        },
    }
    for name, payload in reports.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    candidate = {
        "model_version": identity["model_version"],
        "git_commit": identity["git_commit"],
        "artifact_path": "serving_candidate/manifest.json",
        "artifact_sha256": identity["artifact_sha256"],
        "shadow_report_sha256": _sha(root / "shadow.json"),
        "drift_report_sha256": _sha(root / "drift.json"),
        "incident_report_sha256": _sha(root / "incident.json"),
    }
    (root / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "stage",
    ["production ", "PRODUCTION", "test", "staging", ""],
)
def test_unknown_stage_is_rejected(stage):
    decision = DeploymentGate().evaluate(DeploymentEvidence(stage=stage))
    assert decision.allowed is False
    assert "unknown deployment stage" in decision.reason


def test_candidate_newline_and_command_injection_are_rejected(tmp_path):
    root = _bundle(tmp_path)
    for case_index, (field, value) in enumerate(
        (
            ("git_commit", "a" * 40 + "\n"),
            ("model_version", "v1.0.0; rm -rf /"),
            ("model_version", "X" * 100_000),
        )
    ):
        payload = json.loads((root / "candidate.json").read_text())
        payload[field] = value
        (root / "candidate.json").write_text(json.dumps(payload), encoding="utf-8")
        evidence = load_evidence_bundle(root, "canary")
        assert DeploymentGate().evaluate(evidence).allowed is False
        root = _bundle(tmp_path / f"case-{case_index}")


def test_production_ticket_injection_is_rejected(tmp_path):
    root = _bundle(tmp_path)
    evidence = load_evidence_bundle(root, "canary")
    assert DeploymentGate().evaluate(evidence).allowed
    for ticket in ["PROD-123\n", " PROD-123", "PROD-123 ", "PROD-abc"]:
        result = DeploymentGate().evaluate(
            DeploymentEvidence(
                stage="production",
                artifact_root=evidence.artifact_root,
                candidate=evidence.candidate,
                shadow_report=evidence.shadow_report,
                drift_report=evidence.drift_report,
                incident_report=evidence.incident_report,
                approval_ticket=ticket,
                environment_approver="risk-manager",
            )
        )
        assert result.allowed is False


def test_report_digest_cannot_be_replaced_by_well_formed_fake_hash(tmp_path):
    root = _bundle(tmp_path)
    candidate = json.loads((root / "candidate.json").read_text())
    candidate["shadow_report_sha256"] = "b" * 64
    (root / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="shadow report SHA-256 mismatch"):
        load_evidence_bundle(root, "canary")


def test_manifest_sha_cannot_reference_another_well_formed_hash(tmp_path):
    root = _bundle(tmp_path)
    candidate = json.loads((root / "candidate.json").read_text())
    candidate["artifact_sha256"] = "b" * 64
    (root / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    evidence = load_evidence_bundle(root, "canary")
    decision = DeploymentGate().evaluate(evidence)
    assert decision.allowed is False
    assert "serving manifest SHA-256 mismatch" in decision.reason
