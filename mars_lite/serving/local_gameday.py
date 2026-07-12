"""Deterministic exchange-free failure drill for the local serving stack."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from fastapi.testclient import TestClient

from mars_lite.pipeline.release_eligibility import derive_release_eligibility
from mars_lite.pipeline.release_risk import ReleaseRiskPolicy
from mars_lite.server.signal_server import create_app
from mars_lite.serving.audit_store import AuditStore
from mars_lite.serving.candidate import create_candidate_bundle
from mars_lite.serving.market_time import resolve_completed_bar_endpoint
from mars_lite.serving.registry import ModelRegistry
from mars_lite.serving.runtime import FeatureSnapshot, RuntimeComponents, ServingRuntime
from mars_lite.serving.snapshot_identity import compute_snapshot_id
from mars_lite.trading.guardrails import (
    GuardrailConfig,
    GuardrailState,
    apply_guardrails,
    evaluate_guardrails,
)

_RELEASE_SHA = "a" * 40
_AUTH_TOKEN = "local-gameday-token"
_SCENARIO_NAMES = (
    "healthy_activation",
    "content_mutation_identity",
    "timeframe_freshness",
    "stale_data_fail_closed",
    "replay_rejection",
    "bundle_rejection_preserves_healthy_runtime",
    "rollback",
)


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Policy:
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return np.asarray([0.25], dtype=np.float64), None


class _FixedProvider:
    def __init__(self, snapshot: FeatureSnapshot) -> None:
        self.snapshot = snapshot

    def get_snapshot(self) -> FeatureSnapshot:
        return self.snapshot


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


def _risk_policy() -> ReleaseRiskPolicy:
    return ReleaseRiskPolicy(
        max_leverage=1.0,
        max_single_weight=0.5,
        max_net_exposure=1.0,
        max_worst_case_notional=100_000.0,
        min_order_notional=10.0,
        symbol_liquidity_caps={"BTCUSDT": 50_000.0},
        forbidden_symbols=(),
    )


def _create_bundle(root: Path, version: str, *, git_sha: str, payload: bytes) -> Path:
    model_source = root / f"{version}.zip"
    model_source.write_bytes(payload)
    return create_candidate_bundle(
        destination=root / f"candidate-{version}",
        model_source=model_source,
        version=version,
        git_sha=git_sha,
        symbols=("BTCUSDT",),
        feature_names=("ret",),
        global_feature_names=(),
        feature_norm="none",
        feature_mask=None,
        observation_dim=5,
        observation_schema_version=1,
        post_processor={},
        run_config={
            "observation_progress_mode": "zero",
            "base_timeframe": "1h",
        },
        metrics={},
        guardrails={"max_data_age_hours": 2.0},
        risk_policy=_risk_policy(),
        release_eligibility=_eligibility(),
    )


def _component_factory(bundle) -> RuntimeComponents:
    guard_config = GuardrailConfig(**dict(bundle.risk.get("guardrails") or {}))

    def decide(raw_action, state, recent_returns, htf_trend):
        return np.asarray(raw_action, dtype=np.float64), {"source": "fixed-policy"}

    def guard(target, current, state, data_age_hours, features):
        result = evaluate_guardrails(
            weights=np.asarray(target, dtype=np.float64),
            portfolio_value=state.portfolio_value,
            turnover=float(np.abs(target - current).sum()),
            data_age_hours=data_age_hours,
            features=np.asarray(features, dtype=np.float64).reshape(-1),
            state=GuardrailState(
                day_start_value=state.day_start_value,
                peak_value=state.peak_value,
                consecutive_losses=state.consecutive_losses,
                turnover_mean=state.turnover_mean,
                turnover_std=state.turnover_std,
            ),
            config=guard_config,
        )
        return apply_guardrails(
            np.asarray(target, dtype=np.float64), result
        ), result.to_dict()

    def risk(target, state, symbols):
        return {"approved": True}

    return RuntimeComponents(
        model=_Policy(),
        decide=decide,
        apply_guardrails=guard,
        evaluate_risk=risk,
        include_observation_risk_state=False,
        serving_progress=0.0,
    )


def _snapshot(*, snapshot_id: str, data_age_hours: float) -> FeatureSnapshot:
    return FeatureSnapshot(
        snapshot_id=snapshot_id,
        symbols=("BTCUSDT",),
        feature_names=("ret",),
        global_feature_names=(),
        feature_history=np.asarray([[[0.1]], [[0.2]]], dtype=np.float64),
        global_features=np.empty(0, dtype=np.float64),
        close_history=np.asarray([[100.0], [101.0]], dtype=np.float64),
        data_age_hours=data_age_hours,
    )


def _request_payload(request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "market_snapshot_id": "latest",
        "state": {
            "current_weights": {"BTCUSDT": 0.0},
            "portfolio_value": 100.0,
            "day_start_value": 100.0,
            "peak_value": 100.0,
            "consecutive_losses": 0,
            "turnover_mean": 0.0,
            "turnover_std": 1.0,
            "pending_orders": [],
        },
    }


def _client(runtime: ServingRuntime, snapshot: FeatureSnapshot) -> TestClient:
    return TestClient(
        create_app(
            runtime=runtime,
            feature_provider=_FixedProvider(snapshot),
            auth_token=_AUTH_TOKEN,
        )
    )


def _post_signal(client: TestClient, request_id: str) -> tuple[int, dict[str, Any]]:
    response = client.post(
        "/api/signal/latest",
        headers={"Authorization": f"Bearer {_AUTH_TOKEN}"},
        json=_request_payload(request_id),
    )
    return response.status_code, response.json()


def _scenario(name: str, action: Callable[[], Mapping[str, Any]]) -> ScenarioResult:
    try:
        return ScenarioResult(name=name, passed=True, details=dict(action()))
    except Exception as exc:
        return ScenarioResult(
            name=name,
            passed=False,
            details={"error": f"{type(exc).__name__}: {exc}"},
        )


def _run(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(root / "registry")
    audit = AuditStore(root / "audit.sqlite3")
    valid_v1 = _create_bundle(root, "v1", git_sha=_RELEASE_SHA, payload=b"one")
    valid_v2 = _create_bundle(root, "v2", git_sha=_RELEASE_SHA, payload=b"two")
    mismatched = _create_bundle(root, "vbad", git_sha="b" * 40, payload=b"bad")
    registered_v1 = registry.register(valid_v1)
    registered_v2 = registry.register(valid_v2)
    registry.register(mismatched)
    registry.activate("v1", evidence_identity="local-gameday:v1")
    runtime = ServingRuntime(
        registry=registry,
        audit_store=audit,
        component_factory=_component_factory,
        release_git_sha=_RELEASE_SHA,
        strict_release_binding=True,
    )
    fresh = _snapshot(snapshot_id="fresh-snapshot", data_age_hours=0.25)

    def healthy_activation() -> Mapping[str, Any]:
        if not runtime.refresh():
            raise AssertionError("healthy runtime refresh failed")
        response = _client(runtime, fresh).get("/ready")
        body = response.json()
        if response.status_code != 200:
            raise AssertionError(f"readiness returned {response.status_code}")
        expected = {
            "status": "ready",
            "active_version": "v1",
            "bundle_digest": registered_v1.bundle_digest,
            "release_git_sha": _RELEASE_SHA,
        }
        for key, value in expected.items():
            if body.get(key) != value:
                raise AssertionError(f"readiness {key} mismatch")
        return expected

    def content_mutation_identity() -> Mapping[str, Any]:
        timestamps = np.asarray(
            ["2026-07-12T08:00", "2026-07-12T09:00"], dtype="datetime64[ns]"
        )
        common = {
            "bundle_digest": registered_v1.bundle_digest,
            "base_timeframe": "1h",
            "timestamps": timestamps,
            "symbols": ("BTCUSDT",),
            "feature_names": ("ret",),
            "global_feature_names": (),
            "global_features": np.empty(0, dtype=np.float64),
            "close_history": np.asarray([[100.0], [101.0]], dtype=np.float64),
        }
        original = compute_snapshot_id(
            **common,
            feature_history=np.asarray([[[0.1]], [[0.2]]], dtype=np.float64),
        )
        mutated = compute_snapshot_id(
            **common,
            feature_history=np.asarray([[[0.1]], [[0.3]]], dtype=np.float64),
        )
        if original == mutated:
            raise AssertionError("content mutation did not change snapshot identity")
        return {"identity_changed": True}

    def timeframe_freshness() -> Mapping[str, Any]:
        cases = {
            "1h": (
                ["2026-07-12T08:00", "2026-07-12T09:00"],
                "2026-07-12T09:30",
                1,
                0.5,
            ),
            "4h": (
                ["2026-07-12T00:00", "2026-07-12T04:00"],
                "2026-07-12T07:00",
                1,
                3.0,
            ),
            "1d": (
                ["2026-07-10T00:00", "2026-07-11T00:00"],
                "2026-07-12T12:00",
                2,
                12.0,
            ),
        }
        details: dict[str, Any] = {}
        for timeframe, (timestamps, now, expected_end, expected_age) in cases.items():
            endpoint = resolve_completed_bar_endpoint(
                np.asarray(timestamps, dtype="datetime64[ns]"),
                base_timeframe=timeframe,
                now_utc=np.datetime64(now, "ns"),
            )
            if endpoint.end_exclusive != expected_end:
                raise AssertionError(f"{timeframe} completed endpoint mismatch")
            if endpoint.data_age_hours != expected_age:
                raise AssertionError(f"{timeframe} age mismatch")
            details[timeframe] = {
                "end_exclusive": endpoint.end_exclusive,
                "data_age_hours": endpoint.data_age_hours,
            }
        return details

    def stale_data_fail_closed() -> Mapping[str, Any]:
        stale = _snapshot(snapshot_id="stale-snapshot", data_age_hours=3.0)
        status_code, body = _post_signal(_client(runtime, stale), "stale-request")
        target = body.get("target_weights") or {}
        if status_code != 200 or body.get("status") != "ok":
            raise AssertionError("stale request did not produce a guarded response")
        if any(abs(float(value)) > 0 for value in target.values()):
            raise AssertionError("stale data returned actionable exposure")
        reasons = tuple(body.get("reasons") or ())
        if not any("stale data" in str(reason) for reason in reasons):
            raise AssertionError("stale-data guardrail reason was not reported")
        return {"actionable_exposure": False, "reason": "stale data"}

    def replay_rejection() -> Mapping[str, Any]:
        client = _client(runtime, fresh)
        first_code, first = _post_signal(client, "replay-request")
        second_code, second = _post_signal(client, "replay-request")
        if first_code != 200 or first.get("status") != "ok":
            raise AssertionError("first replay probe request failed")
        if second_code != 200 or second.get("status") != "rejected":
            raise AssertionError("duplicate request was not rejected")
        events = audit.list_events(limit=100)
        if not any(
            event["event_type"] == "replay" and event["request_id"] == "replay-request"
            for event in events
        ):
            raise AssertionError("replay audit event was not recorded")
        return {"first_status": "ok", "second_status": "rejected", "audited": True}

    def bundle_rejection_preserves_healthy_runtime() -> Mapping[str, Any]:
        registry.activate("vbad", evidence_identity="local-gameday:vbad")
        if runtime.refresh():
            raise AssertionError("Git-SHA-mismatched bundle was accepted")
        readiness = runtime.readiness()
        if (
            readiness.status != "degraded"
            or readiness.active_version != "v1"
            or readiness.bundle_digest != registered_v1.bundle_digest
        ):
            raise AssertionError("healthy in-memory runtime was not preserved")
        return {
            "status": "degraded",
            "active_version": "v1",
            "healthy_runtime_preserved": True,
        }

    def rollback() -> Mapping[str, Any]:
        registry.activate("v2", evidence_identity="local-gameday:v2")
        if not runtime.refresh() or runtime.active_version != "v2":
            raise AssertionError("second valid version did not load")
        registry.rollback(target_version="v1")
        if not runtime.refresh():
            raise AssertionError("rollback refresh failed")
        readiness = runtime.readiness()
        if (
            readiness.active_version != "v1"
            or readiness.bundle_digest != registered_v1.bundle_digest
        ):
            raise AssertionError("rollback identity mismatch")
        if registered_v2.bundle_digest == registered_v1.bundle_digest:
            raise AssertionError("test versions unexpectedly share one digest")
        return {
            "active_version": "v1",
            "bundle_digest": registered_v1.bundle_digest,
        }

    actions = (
        healthy_activation,
        content_mutation_identity,
        timeframe_freshness,
        stale_data_fail_closed,
        replay_rejection,
        bundle_rejection_preserves_healthy_runtime,
        rollback,
    )
    results = [
        _scenario(name, action) for name, action in zip(_SCENARIO_NAMES, actions)
    ]
    return {
        "passed": all(result.passed for result in results),
        "scenarios": [result.to_dict() for result in results],
    }


def run_local_gameday(root: Path | None = None) -> dict[str, Any]:
    """Run all local scenarios and return a deterministic machine-readable summary."""

    if root is not None:
        return _run(Path(root))
    with tempfile.TemporaryDirectory(prefix="trade-rl-gameday-") as temporary:
        return _run(Path(temporary))


def exit_code_for_summary(summary: Mapping[str, Any]) -> int:
    return 0 if summary.get("passed") is True else 1
