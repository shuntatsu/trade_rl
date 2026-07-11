import hashlib
import json
from pathlib import Path

from mars_lite.server.deployment_gate import (
    DeploymentEvidence,
    DeploymentGate,
    load_evidence_bundle,
    main,
)
from mars_lite.serving.candidate import create_candidate_bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(tmp_path: Path, *, production: bool = False) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    source = tmp_path / "model-source.zip"
    source.write_bytes(b"model-bytes")
    serving = create_candidate_bundle(
        destination=root / "serving_candidate",
        model_source=source,
        version="1.0.0",
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
        pre_trade={},
    )
    manifest = serving / "manifest.json"
    manifest_hash = _sha(manifest)
    identity = {
        "model_version": "1.0.0",
        "git_commit": "a" * 40,
        "artifact_sha256": manifest_hash,
    }
    shadow = {
        "run_id": "shadow-1",
        **identity,
        "oos_sharpe": 1.2,
        "baseline_sharpe": 1.0,
        "max_drawdown": 0.10,
    }
    drift = {
        "report_id": "drift-1",
        **identity,
        "psi_score": 0.10,
        "ks_p_value": 0.20,
    }
    incident = {
        "report_id": "incident-1",
        **identity,
        "active_incidents": False,
    }
    canary = {
        "run_id": "canary-1",
        "parent_shadow_run_id": "shadow-1",
        **identity,
        "capital_cap_usd": 5_000.0,
        "duration_days": 14,
        "max_loss_pct": 0.02,
        "mean_slippage_bps": 5.0,
    }
    for name, payload in (
        ("shadow.json", shadow),
        ("drift.json", drift),
        ("incident.json", incident),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    if production:
        (root / "canary.json").write_text(json.dumps(canary), encoding="utf-8")
    candidate = {
        "model_version": "1.0.0",
        "git_commit": "a" * 40,
        "artifact_path": "serving_candidate/manifest.json",
        "artifact_sha256": manifest_hash,
        "shadow_report_sha256": _sha(root / "shadow.json"),
        "drift_report_sha256": _sha(root / "drift.json"),
        "incident_report_sha256": _sha(root / "incident.json"),
        "canary_report_sha256": _sha(root / "canary.json") if production else None,
    }
    (root / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    return root


def test_shadow_is_first_stage():
    assert DeploymentGate().evaluate(DeploymentEvidence(stage="shadow")).allowed


def test_canary_requires_bundle_not_boolean_fallback():
    decision = DeploymentGate().evaluate(DeploymentEvidence(stage="canary"))
    assert decision.allowed is False
    assert "bundle" in decision.reason


def test_verified_bundle_allows_canary(tmp_path):
    root = _write_bundle(tmp_path)
    evidence = load_evidence_bundle(root, "canary")
    assert DeploymentGate().evaluate(evidence).allowed is True


def test_verified_bundle_allows_production(tmp_path):
    root = _write_bundle(tmp_path, production=True)
    evidence = load_evidence_bundle(
        root,
        "production",
        approval_ticket="PROD-123",
        environment_approver="risk-manager",
    )
    assert DeploymentGate().evaluate(evidence).allowed is True


def test_serving_bundle_file_tamper_is_blocked(tmp_path):
    root = _write_bundle(tmp_path)
    (root / "serving_candidate" / "model.zip").write_bytes(b"tampered")
    evidence = load_evidence_bundle(root, "canary")
    decision = DeploymentGate().evaluate(evidence)
    assert decision.allowed is False
    assert "serving bundle validation failed" in decision.reason
    assert "digest mismatch" in decision.reason


def test_report_tamper_is_blocked_before_parsing(tmp_path):
    root = _write_bundle(tmp_path)
    payload = json.loads((root / "shadow.json").read_text())
    payload["oos_sharpe"] = 99.0
    (root / "shadow.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_evidence_bundle(root, "canary")
    except ValueError as exc:
        assert "shadow report SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered report must be rejected")


def test_cross_model_report_reuse_is_blocked(tmp_path):
    root = _write_bundle(tmp_path)
    candidate_payload = json.loads((root / "candidate.json").read_text())
    shadow_payload = json.loads((root / "shadow.json").read_text())
    shadow_payload["model_version"] = "old-model"
    (root / "shadow.json").write_text(json.dumps(shadow_payload), encoding="utf-8")
    candidate_payload["shadow_report_sha256"] = _sha(root / "shadow.json")
    (root / "candidate.json").write_text(
        json.dumps(candidate_payload), encoding="utf-8"
    )
    evidence = load_evidence_bundle(root, "canary")
    decision = DeploymentGate().evaluate(evidence)
    assert decision.allowed is False
    assert "model version does not match" in decision.reason


def test_canary_must_reference_verified_shadow_run(tmp_path):
    root = _write_bundle(tmp_path, production=True)
    candidate_payload = json.loads((root / "candidate.json").read_text())
    canary_payload = json.loads((root / "canary.json").read_text())
    canary_payload["parent_shadow_run_id"] = "other-shadow"
    (root / "canary.json").write_text(json.dumps(canary_payload), encoding="utf-8")
    candidate_payload["canary_report_sha256"] = _sha(root / "canary.json")
    (root / "candidate.json").write_text(
        json.dumps(candidate_payload), encoding="utf-8"
    )
    evidence = load_evidence_bundle(
        root,
        "production",
        approval_ticket="PROD-123",
        environment_approver="risk-manager",
    )
    decision = DeploymentGate().evaluate(evidence)
    assert decision.allowed is False
    assert "verified shadow run" in decision.reason


def test_active_incident_blocks_deployment(tmp_path):
    root = _write_bundle(tmp_path)
    candidate_payload = json.loads((root / "candidate.json").read_text())
    incident_payload = json.loads((root / "incident.json").read_text())
    incident_payload["active_incidents"] = True
    (root / "incident.json").write_text(json.dumps(incident_payload), encoding="utf-8")
    candidate_payload["incident_report_sha256"] = _sha(root / "incident.json")
    (root / "candidate.json").write_text(
        json.dumps(candidate_payload), encoding="utf-8"
    )
    evidence = load_evidence_bundle(root, "canary")
    decision = DeploymentGate().evaluate(evidence)
    assert decision.allowed is False
    assert "active incidents" in decision.reason


def test_arbitrary_artifact_path_is_blocked(tmp_path):
    root = _write_bundle(tmp_path)
    candidate_payload = json.loads((root / "candidate.json").read_text())
    candidate_payload["artifact_path"] = "serving_candidate/model.zip"
    candidate_payload["artifact_sha256"] = _sha(
        root / "serving_candidate" / "model.zip"
    )
    (root / "candidate.json").write_text(
        json.dumps(candidate_payload), encoding="utf-8"
    )
    evidence = load_evidence_bundle(root, "canary")
    decision = DeploymentGate().evaluate(evidence)
    assert decision.allowed is False
    assert "serving_candidate/manifest.json" in decision.reason


def test_cli_returns_nonzero_for_missing_bundle(capsys):
    result = main(["--stage", "canary", "--bundle-dir", "/missing"])
    assert result == 1
    output = json.loads(capsys.readouterr().out)
    assert output["allowed"] is False
