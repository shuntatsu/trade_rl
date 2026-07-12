from pathlib import Path

import yaml

from mars_lite.pipeline.release_eligibility import derive_release_eligibility
from mars_lite.pipeline.release_risk import ReleaseRiskPolicy
from mars_lite.serving.audit_store import AuditStore
from mars_lite.serving.candidate import create_candidate_bundle
from mars_lite.serving.registry import ModelRegistry
from mars_lite.serving.runtime import RuntimeComponents, ServingRuntime


class _Model:
    def predict(self, observation, deterministic=True):
        return [0.0], None


def _components(bundle):
    return RuntimeComponents(
        model=_Model(),
        decide=lambda raw, state, returns, trend: (raw, {}),
        apply_guardrails=lambda target, current, state, age, features: (
            target,
            {"action": "proceed", "triggered": []},
        ),
        evaluate_risk=lambda target, state, symbols: {"approved": True},
        include_observation_risk_state=False,
    )


def _eligibility():
    return derive_release_eligibility(
        forced=False,
        skip_p0=False,
        skip_pbt=False,
        skip_wf=False,
        skip_gate=False,
        sealed_holdout_used=True,
        p0_passed=True,
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


def test_approved_candidate_identity_becomes_served_identity(tmp_path: Path) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")
    candidate = create_candidate_bundle(
        destination=tmp_path / "candidate",
        model_source=model,
        version="v1",
        git_sha="a" * 40,
        symbols=("BTCUSDT",),
        feature_names=("ret",),
        global_feature_names=(),
        feature_norm="none",
        feature_mask=None,
        observation_dim=5,
        observation_schema_version=1,
        post_processor={},
        run_config={"observation_progress_mode": "zero"},
        metrics={"gate2": {"passed": True}},
        guardrails={},
        risk_policy=_risk(),
        release_eligibility=_eligibility(),
    )
    registry = ModelRegistry(tmp_path / "registry")
    approved = registry.register(candidate)
    registry.activate("v1", evidence_identity="github-actions:123:production")
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(tmp_path / "audit.sqlite3"),
        component_factory=_components,
        release_git_sha="a" * 40,
        strict_release_binding=True,
    )
    assert runtime.refresh() is True
    readiness = runtime.readiness()
    assert readiness.active_version == approved.version
    assert readiness.bundle_digest == approved.bundle_digest
    assert readiness.release_git_sha == approved.git_sha


def test_deploy_workflow_orders_gate_activation_and_verification() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["gate"]
    assert job["runs-on"] == ["self-hosted", "trade-rl-deploy"]

    steps = job["steps"]
    names = [step.get("name") for step in steps if isinstance(step, dict)]
    gate_index = names.index("Evaluate deployment gate")
    target_index = names.index("Validate deployment target")
    activation_index = names.index("Register and atomically activate approved bundle")
    verification_index = names.index("Verify served identity")
    assert gate_index < target_index < activation_index < verification_index

    binding = next(
        step
        for step in steps
        if step.get("name") == "Download and bind immutable evidence"
    )
    binding_script = binding["run"]
    assert 'expected_artifact = "serving_candidate/manifest.json"' in binding_script
    assert "candidate artifact digest does not match serving manifest" in binding_script
    assert "MODEL_VERSION=" in binding_script
    assert "BUNDLE_DIGEST=" in binding_script
    assert "RELEASE_GIT_SHA=" in binding_script

    target = next(
        step for step in steps if step.get("name") == "Validate deployment target"
    )
    target_script = target["run"]
    assert "TRADE_RL_REGISTRY_DIR must be an absolute path" in target_script
    assert "TRADE_RL_SERVING_READY_URL" in target_script

    activation = next(
        step
        for step in steps
        if step.get("name") == "Register and atomically activate approved bundle"
    )
    activation_script = activation["run"]
    assert "scripts/manage_registry.py" in activation_script
    assert "register deployment_bundle/serving_candidate" in activation_script
    assert 'activate "$MODEL_VERSION"' in activation_script

    verification = next(
        step for step in steps if step.get("name") == "Verify served identity"
    )
    verify_script = verification["run"]
    assert "scripts/verify_served_identity.py" in verify_script
    assert '--version "$MODEL_VERSION"' in verify_script
    assert '--digest "$BUNDLE_DIGEST"' in verify_script
    assert '--release-git-sha "$RELEASE_GIT_SHA"' in verify_script
