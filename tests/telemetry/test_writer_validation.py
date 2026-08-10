from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.telemetry import TrainingTelemetryRecord, TrainingTelemetryWriter


def _record(
    *, terminated: bool = False, truncated: bool = False
) -> TrainingTelemetryRecord:
    return TrainingTelemetryRecord(
        sequence=1,
        recorded_at="2026-07-21T08:00:00+00:00",
        global_step=32,
        environment_step=1,
        seed=7,
        environment_id=0,
        event_type="episode_end",
        market_index=101,
        market_time="2026-07-21T07:55:00.000000000",
        symbol="BTCUSDT",
        open=67_500.0,
        high=67_900.0,
        low=67_400.0,
        close=67_842.3,
        action=(0.4,),
        executed_target=(0.4,),
        weights_before=(0.2,),
        weights_after=(0.4,),
        portfolio_value=101_342.85,
        baseline_portfolio_value=100_400.0,
        reward=0.214,
        drawdown=0.0086,
        interval_cost=4.25,
        interval_return=0.0012,
        risk_reasons=(),
        emergency_deleverage=False,
        terminated=terminated,
        truncated=truncated,
    )


@pytest.mark.parametrize("flush_every", (0, -1, True))
def test_writer_rejects_non_positive_or_boolean_flush_interval(
    tmp_path: Path,
    flush_every: int,
) -> None:
    with pytest.raises(ValueError, match="flush_every must be positive"):
        TrainingTelemetryWriter(
            tmp_path / "training-telemetry.jsonl",
            flush_every=flush_every,
        )


def test_record_rejects_simultaneous_termination_and_truncation() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _record(terminated=True, truncated=True)


def test_legacy_v1_dual_flags_are_normalized_as_time_limit_truncation() -> None:
    payload = _record().to_json_dict()
    payload["terminated"] = True
    payload["truncated"] = True

    decoded = TrainingTelemetryRecord.from_json_dict(payload)

    assert decoded.terminated is False
    assert decoded.truncated is True
