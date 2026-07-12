import numpy as np
from fastapi.testclient import TestClient

from mars_lite.server.signal_server import create_app, parse_inference_request
from mars_lite.serving.contracts import InferenceResponse
from mars_lite.serving.runtime import FeatureSnapshot, ReadinessState


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


class FakeRuntime:
    def __init__(self, status="ok"):
        self.status = status

    def refresh(self):
        return True

    def readiness(self):
        return ReadinessState(
            "ready", "v1", "digest", release_git_sha="a" * 40
        )

    def infer(self, request, snapshot):
        return InferenceResponse(
            status=self.status,
            request_id=request.request_id,
            market_snapshot_id=request.market_snapshot_id,
            model_version="v1",
            bundle_digest="digest",
            target_weights={"BTCUSDT": 0.1} if self.status == "ok" else None,
            reasons=() if self.status == "ok" else ("not ready",),
            pre_trade_risk={"approved": self.status == "ok"},
        )


def payload():
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


def client(status="ok"):
    return TestClient(
        create_app(
            runtime=FakeRuntime(status),
            feature_provider=FakeProvider(),
            auth_token="token",
        )
    )


def test_health_and_readiness_are_public() -> None:
    api = client()
    assert api.get("/health").json() == {"status": "ok"}
    ready = api.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["active_version"] == "v1"
    assert ready.json()["bundle_digest"] == "digest"
    assert ready.json()["release_git_sha"] == "a" * 40


def test_no_signal_returns_503() -> None:
    response = client("no_signal").post(
        "/api/signal/latest",
        headers={"Authorization": "Bearer token"},
        json=payload(),
    )
    assert response.status_code == 503
    assert response.json()["status"] == "no_signal"


def test_malformed_pending_order_is_422() -> None:
    body = payload()
    body["state"]["pending_orders"] = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "notional": 10,
            "reduce_only": "false",
        }
    ]
    response = client().post(
        "/api/signal/latest",
        headers={"Authorization": "Bearer token"},
        json=body,
    )
    assert response.status_code == 422


def test_request_parser_preserves_state() -> None:
    parsed = parse_inference_request(payload())
    assert parsed.state.portfolio_value == 100.0
    assert parsed.state.current_weights == {"BTCUSDT": 0.0}


def test_market_snapshot_is_bound_by_server() -> None:
    body = payload()
    body["market_snapshot_id"] = "stale-snapshot"
    response = client().post(
        "/api/signal/latest",
        headers={"Authorization": "Bearer token"},
        json=body,
    )
    assert response.status_code == 422
    assert "market_snapshot_id" in response.json()["detail"]
