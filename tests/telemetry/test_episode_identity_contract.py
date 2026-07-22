from __future__ import annotations

import pytest

from trade_rl.telemetry import TrainingTelemetryRecord


def _record(sequence: int, *, episode_id: int | None = None) -> TrainingTelemetryRecord:
    values: dict[str, object] = {
        "sequence": sequence,
        "recorded_at": "2026-07-22T12:00:00+00:00",
        "global_step": sequence * 32,
        "environment_step": sequence,
        "seed": 7,
        "environment_id": 0,
        "event_type": "rollout",
        "market_index": 100 + sequence,
        "market_time": "2026-07-22T11:55:00.000000000",
        "symbol": "BTCUSDT",
        "open": 67_500.0,
        "high": 67_900.0,
        "low": 67_400.0,
        "close": 67_842.3,
        "action": (0.4,),
        "executed_target": (0.4,),
        "weights_before": (0.2,),
        "weights_after": (0.4,),
        "portfolio_value": 101_342.85,
        "baseline_portfolio_value": 100_400.0,
        "reward": 0.214,
        "drawdown": 0.0086,
        "interval_cost": 4.25,
        "interval_return": 0.0012,
        "risk_reasons": (),
        "emergency_deleverage": False,
        "terminated": False,
        "truncated": False,
    }
    if episode_id is not None:
        values["episode_id"] = episode_id
    return TrainingTelemetryRecord(**values)  # type: ignore[arg-type]


def test_episode_id_round_trip_is_explicit() -> None:
    original = _record(1, episode_id=9)

    restored = TrainingTelemetryRecord.from_json_dict(original.to_json_dict())

    assert restored.episode_id == 9
    assert original.to_json_dict()["episode_id"] == 9


def test_legacy_record_without_episode_id_remains_readable() -> None:
    payload = _record(1).to_json_dict()
    payload.pop("episode_id", None)

    restored = TrainingTelemetryRecord.from_json_dict(payload)

    assert restored.episode_id is None


@pytest.mark.parametrize("episode_id", (-1, True))
def test_invalid_episode_id_fails_closed(episode_id: object) -> None:
    payload = _record(1).to_json_dict()
    payload["episode_id"] = episode_id

    with pytest.raises(ValueError, match="episode_id"):
        TrainingTelemetryRecord.from_json_dict(payload)
