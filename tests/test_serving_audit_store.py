from pathlib import Path

import pytest

from mars_lite.serving.audit_store import AuditStore
from mars_lite.serving.contracts import InferenceState


def test_inference_state_rejects_invalid_account_values() -> None:
    state = InferenceState(
        current_weights={"BTCUSDT": 0.1},
        portfolio_value=100.0,
        day_start_value=0.0,
        peak_value=110.0,
        consecutive_losses=0,
        turnover_mean=0.1,
        turnover_std=0.02,
        pending_orders=(),
    )
    with pytest.raises(ValueError, match="day_start_value"):
        state.validate(("BTCUSDT",))


def test_duplicate_request_id_is_rejected(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    assert store.claim_request("req-1", "hash-1") is True
    assert store.claim_request("req-1", "hash-1") is False


def test_request_id_reuse_with_different_payload_is_rejected(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    assert store.claim_request("req-1", "hash-1") is True
    with pytest.raises(ValueError, match="different payload"):
        store.claim_request("req-1", "hash-2")


def test_audit_event_is_persisted(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.append_event(
        event_type="inference",
        request_id="req-1",
        model_version="v1",
        bundle_digest="abc",
        payload={"status": "ok"},
    )
    events = store.list_events(limit=10)
    assert events[0]["event_type"] == "inference"
    assert events[0]["payload"] == {"status": "ok"}


def test_observation_risk_state_is_required_when_bundle_uses_it() -> None:
    state = InferenceState(
        current_weights={"BTCUSDT": 0.1},
        portfolio_value=100.0,
        day_start_value=100.0,
        peak_value=110.0,
        consecutive_losses=0,
        turnover_mean=0.1,
        turnover_std=0.02,
        pending_orders=(),
    )
    with pytest.raises(ValueError, match="vol_scale"):
        state.validate(("BTCUSDT",), require_observation_risk_state=True)
