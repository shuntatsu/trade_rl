import json
from pathlib import Path

import numpy as np
import pytest

from mars_lite.serving.audit_store import AuditStore
from mars_lite.serving.bundle import build_manifest
from mars_lite.serving.contracts import InferenceRequest, InferenceState
from mars_lite.serving.registry import ModelRegistry
from mars_lite.serving.runtime import (
    FeatureSnapshot,
    RuntimeComponents,
    ServingRuntime,
)


def _eligibility() -> dict[str, object]:
    return {
        "eligible": True,
        "forced": False,
        "skipped_gates": [],
        "optimization_steps_skipped": [],
        "sealed_holdout_used": True,
        "required_gates": {
            "p0": "passed",
            "walk_forward": "passed",
            "gate2": "passed",
            "significance": "not_required",
        },
    }


def _risk(symbols: tuple[str, ...]) -> dict[str, object]:
    return {
        "guardrails": {},
        "pre_trade": {
            "max_leverage": 1.0,
            "max_single_weight": 0.5,
            "max_net_exposure": 1.0,
            "max_worst_case_notional": 100_000.0,
            "min_order_notional": 10.0,
            "symbol_liquidity_caps": {
                symbol: 50_000.0 for symbol in symbols
            },
            "forbidden_symbols": [],
        },
    }


def create_bundle(
    root: Path,
    version: str,
    payload: bytes,
    symbols: tuple[str, ...] = ("BTCUSDT",),
    *,
    git_sha: str = "a" * 40,
) -> Path:
    root.mkdir()
    (root / "model.zip").write_bytes(payload)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": version,
                "git_sha": git_sha,
                "model_kind": "single",
                "symbols": list(symbols),
                "observation_schema_version": 1,
                "observation_progress_mode": "zero",
                "observation_dim": len(symbols) * 2 + 4,
                "run_config": {},
                "release_eligibility": _eligibility(),
            }
        ),
        encoding="utf-8",
    )
    (root / "preprocessing.json").write_text(
        '{"feature_names":["ret"],"global_feature_names":["market"],'
        '"feature_norm":"none","feature_mask":[true],"post_mask_dim":1}',
        encoding="utf-8",
    )
    (root / "risk.json").write_text(
        json.dumps(_risk(symbols)), encoding="utf-8"
    )
    build_manifest(root)
    return root


class RecordingModel:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64)
        self.observations = []

    def predict(self, observation, deterministic=True):
        self.observations.append(np.asarray(observation).copy())
        return self.action.copy(), None


class ComponentFactory:
    def __init__(self, action):
        self.models = []
        self.action = action
        self.turnovers = []

    def __call__(self, bundle):
        model = RecordingModel(self.action)
        self.models.append(model)

        def decide(raw_action, state, recent_returns, htf_trend):
            return np.asarray(raw_action, dtype=np.float64), {"vol_scale": 1.0}

        def guard(target, current, state, data_age_hours, features):
            self.turnovers.append(float(np.abs(target - current).sum()))
            return target, {"action": "proceed", "triggered": []}

        def risk(target, state, symbols):
            return {"approved": True}

        return RuntimeComponents(
            model=model,
            decide=decide,
            apply_guardrails=guard,
            evaluate_risk=risk,
            include_observation_risk_state=False,
            serving_progress=0.0,
        )


def test_corrupt_new_bundle_keeps_old_runtime_active(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    v1 = create_bundle(tmp_path / "v1", "v1", b"one")
    v2 = create_bundle(tmp_path / "v2", "v2", b"two")
    registry.register(v1)
    registry.activate("v1", evidence_identity="run-1")
    factory = ComponentFactory([0.0])
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(tmp_path / "audit.sqlite3"),
        component_factory=factory,
    )
    assert runtime.refresh() is True
    assert runtime.readiness().active_version == "v1"

    registry.register(v2)
    registry.activate("v2", evidence_identity="run-2")
    (registry.version_dir("v2") / "model.zip").write_bytes(b"corrupt")

    assert runtime.refresh() is False
    ready = runtime.readiness()
    assert ready.status == "degraded"
    assert ready.active_version == "v1"


def test_strict_runtime_rejects_bundle_from_different_git_sha(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    previous = create_bundle(
        tmp_path / "v1", "v1", b"one", git_sha="a" * 40
    )
    mismatched = create_bundle(
        tmp_path / "v2", "v2", b"two", git_sha="b" * 40
    )
    registry.register(previous)
    registry.activate("v1", evidence_identity="run-1")
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(tmp_path / "audit.sqlite3"),
        component_factory=ComponentFactory([0.0]),
        release_git_sha="a" * 40,
        strict_release_binding=True,
    )
    assert runtime.refresh() is True

    registry.register(mismatched)
    registry.activate("v2", evidence_identity="run-2")

    assert runtime.refresh() is False
    ready = runtime.readiness()
    assert runtime.active_version == "v1"
    assert ready.status == "degraded"
    assert ready.active_version == "v1"
    assert ready.release_git_sha == "a" * 40
    assert "git sha mismatch" in str(ready.reason).lower()


def test_strict_runtime_loads_matching_git_sha(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    candidate = create_bundle(
        tmp_path / "v1", "v1", b"one", git_sha="a" * 40
    )
    registered = registry.register(candidate)
    registry.activate("v1", evidence_identity="run-1")
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(tmp_path / "audit.sqlite3"),
        component_factory=ComponentFactory([0.0]),
        release_git_sha="A" * 40,
        strict_release_binding=True,
    )

    assert runtime.refresh() is True
    ready = runtime.readiness()
    assert ready.active_version == "v1"
    assert ready.bundle_digest == registered.bundle_digest
    assert ready.release_git_sha == "a" * 40


def test_strict_runtime_requires_valid_release_git_sha(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")

    with pytest.raises(ValueError, match="release_git_sha"):
        ServingRuntime(
            registry=registry,
            audit_store=AuditStore(tmp_path / "audit.sqlite3"),
            strict_release_binding=True,
        )
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        ServingRuntime(
            registry=registry,
            audit_store=AuditStore(tmp_path / "audit-2.sqlite3"),
            release_git_sha="not-a-sha",
            strict_release_binding=True,
        )


def test_current_weights_are_in_policy_observation_before_predict(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    candidate = create_bundle(
        tmp_path / "v1", "v1", b"one", symbols=("BTCUSDT", "ETHUSDT")
    )
    registry.register(candidate)
    registry.activate("v1", evidence_identity="run-1")
    factory = ComponentFactory([0.3, -0.2])
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(tmp_path / "audit.sqlite3"),
        component_factory=factory,
    )
    assert runtime.refresh() is True
    request = InferenceRequest(
        request_id="req-1",
        market_snapshot_id="snap-1",
        state=InferenceState(
            current_weights={"BTCUSDT": 0.2, "ETHUSDT": -0.1},
            portfolio_value=90.0,
            day_start_value=100.0,
            peak_value=100.0,
            consecutive_losses=0,
            turnover_mean=0.1,
            turnover_std=0.02,
        ),
    )
    snapshot = FeatureSnapshot(
        snapshot_id="snap-1",
        symbols=("BTCUSDT", "ETHUSDT"),
        feature_names=("ret",),
        global_feature_names=("market",),
        feature_history=np.zeros((5, 2, 1), dtype=np.float64),
        global_features=np.array([0.5], dtype=np.float64),
        close_history=np.ones((5, 2), dtype=np.float64),
        data_age_hours=0.1,
    )
    response = runtime.infer(request, snapshot)
    assert response.status == "ok"
    recorded = factory.models[-1].observations[-1]
    assert recorded[1] == np.float32(0.2)
    assert recorded[3] == np.float32(-0.1)
    assert factory.turnovers[-1] == pytest.approx(0.2)


def test_duplicate_request_fails_closed(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    candidate = create_bundle(tmp_path / "v1", "v1", b"one")
    registry.register(candidate)
    registry.activate("v1", evidence_identity="run-1")
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(tmp_path / "audit.sqlite3"),
        component_factory=ComponentFactory([0.1]),
    )
    runtime.refresh()
    request = InferenceRequest(
        request_id="req-1",
        market_snapshot_id="snap-1",
        state=InferenceState(
            current_weights={"BTCUSDT": 0.0},
            portfolio_value=100.0,
            day_start_value=100.0,
            peak_value=100.0,
            consecutive_losses=0,
            turnover_mean=0.0,
            turnover_std=1.0,
        ),
    )
    snapshot = FeatureSnapshot(
        snapshot_id="snap-1",
        symbols=("BTCUSDT",),
        feature_names=("ret",),
        global_feature_names=("market",),
        feature_history=np.zeros((5, 1, 1), dtype=np.float64),
        global_features=np.array([0.5], dtype=np.float64),
        close_history=np.ones((5, 1), dtype=np.float64),
        data_age_hours=0.1,
    )
    assert runtime.infer(request, snapshot).status == "ok"
    duplicate = runtime.infer(request, snapshot)
    assert duplicate.status == "rejected"
    assert "duplicate request" in duplicate.reasons[0]


def test_zero_risk_scales_are_preserved_in_observation(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    candidate = create_bundle(tmp_path / "v1", "v1", b"one")
    metadata_path = candidate / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["observation_dim"] = 10
    metadata["run_config"] = {"obs_risk_state": True}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    build_manifest(candidate)
    registry.register(candidate)
    registry.activate("v1", evidence_identity="run-1")
    model = RecordingModel([0.1])
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(tmp_path / "audit.sqlite3"),
        component_factory=lambda bundle: RuntimeComponents(
            model=model,
            decide=lambda raw, state, recent, htf: (raw, {}),
            apply_guardrails=lambda target, current, state, age, features: (
                target,
                {"action": "proceed", "triggered": []},
            ),
            evaluate_risk=lambda target, state, symbols: {"approved": True},
            include_observation_risk_state=True,
            serving_progress=0.0,
        ),
    )
    assert runtime.refresh() is True
    request = InferenceRequest(
        request_id="req-zero",
        market_snapshot_id="snap-zero",
        state=InferenceState(
            current_weights={"BTCUSDT": 0.0},
            portfolio_value=100.0,
            day_start_value=100.0,
            peak_value=100.0,
            consecutive_losses=0,
            turnover_mean=0.0,
            turnover_std=1.0,
            vol_scale=0.0,
            dd_scale=0.0,
            disagreement_scale=0.0,
            est_port_vol=0.0,
        ),
    )
    snapshot = FeatureSnapshot(
        snapshot_id="snap-zero",
        symbols=("BTCUSDT",),
        feature_names=("ret",),
        global_feature_names=("market",),
        feature_history=np.zeros((5, 1, 1)),
        global_features=np.array([0.0]),
        close_history=np.ones((5, 1)),
        data_age_hours=0.1,
    )
    assert runtime.infer(request, snapshot).status == "ok"
    assert model.observations[-1][-4:].tolist() == [0.0, 0.0, 0.0, 0.0]
