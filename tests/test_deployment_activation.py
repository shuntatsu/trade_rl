from pathlib import Path

import yaml

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
        pre_trade={},
    )
    registry = ModelRegistry(tmp_path / "registry")
    approved = registry.register(candidate)
    registry.activate("v1", evidence_identity="github-actions:123:production")
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(tmp_path / "audit.sqlite3"),
        component_factory=_components,
    )
    assert runtime.refresh() is True
    readiness = runtime.readiness()
    assert readiness.active_version == approved.version
    assert readiness.bundle_digest == approved.bundle_digest


def test_deploy_workflow_gates_before_registry_activation() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["gate"]["steps"]
    names = [step.get("name") for step in steps if isinstance(step, dict)]
    gate_index = names.index("Evaluate deployment gate")
    activation_index = names.index("Register and atomically activate approved bundle")
    assert activation_index > gate_index
    activation = next(
        step
        for step in steps
        if step.get("name") == "Register and atomically activate approved bundle"
    )
    script = activation["run"]
    assert "scripts/manage_registry.py" in script
    assert "register deployment_bundle/serving_candidate" in script
    assert "activate \"$MODEL_VERSION\"" in script
