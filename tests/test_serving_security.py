import numpy as np
from fastapi.testclient import TestClient

from mars_lite.server.signal_server import create_app
from mars_lite.serving.contracts import InferenceResponse
from mars_lite.serving.runtime import FeatureSnapshot, ReadinessState


class FakeRuntime:
    def refresh(self):
        return True

    def readiness(self):
        return ReadinessState("ready", "v1", "digest")

    def infer(self, request, snapshot):
        return InferenceResponse(
            status="ok",
            request_id=request.request_id,
            market_snapshot_id=request.market_snapshot_id,
            model_version="v1",
            bundle_digest="digest",
            target_weights={"BTCUSDT": 0.1},
            pre_trade_risk={"approved": True},
        )


class FakeProvider:
    def get_snapshot(self):
        return FeatureSnapshot(
            snapshot_id="snap-1",
            symbols=("BTCUSDT",),
            feature_names=("ret",),
            global_feature_names=("market",),
            feature_history=np.zeros((5, 1, 1)),
            global_features=np.array([0.0]),
            close_history=np.ones((5, 1)),
            data_age_hours=0.1,
        )


def valid_request():
    return {
        "request_id": "req-1",
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


def make_client():
    app = create_app(
        runtime=FakeRuntime(),
        feature_provider=FakeProvider(),
        auth_token="test-token",
        allowed_origins=("https://platform.example",),
    )
    return TestClient(app)


def test_serving_app_exposes_only_read_only_routes() -> None:
    client = make_client()
    paths = {route.path for route in client.app.routes}
    assert "/health" in paths
    assert "/ready" in paths
    assert "/api/signal/latest" in paths
    forbidden = ("delete", "training", "promote", "rollback", "/api/models")
    assert not any(any(word in path.lower() for word in forbidden) for path in paths)


def test_signal_requires_bearer_token() -> None:
    client = make_client()
    assert client.post("/api/signal/latest", json=valid_request()).status_code == 401
    assert (
        client.post(
            "/api/signal/latest",
            headers={"Authorization": "Bearer wrong"},
            json=valid_request(),
        ).status_code
        == 403
    )
    response = client.post(
        "/api/signal/latest",
        headers={"Authorization": "Bearer test-token"},
        json=valid_request(),
    )
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"


def test_cors_does_not_use_wildcard() -> None:
    client = make_client()
    response = client.options(
        "/api/signal/latest",
        headers={
            "Origin": "https://platform.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers["access-control-allow-origin"] == "https://platform.example"
