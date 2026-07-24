from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trade_rl.telemetry import (
    TrainingTelemetryRecord,
    TrainingTelemetryWriter,
    read_training_telemetry,
)


def _record(sequence: int) -> TrainingTelemetryRecord:
    return TrainingTelemetryRecord(
        sequence=sequence,
        recorded_at="2026-07-22T09:00:00+00:00",
        global_step=sequence * 32,
        environment_step=sequence,
        seed=7,
        environment_id=0,
        event_type="rollout",
        market_index=100 + sequence,
        market_time="2026-07-22T08:55:00.000000000",
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
        terminated=False,
        truncated=False,
    )


@pytest.mark.skipif(os.name == "nt", reason="Windows prevents replacing an open file")
def test_strict_index_flush_rejects_a_detached_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    writer = TrainingTelemetryWriter(path, flush_every=100)
    try:
        writer.append(_record(1))
        replacement.write_text(
            json.dumps(_record(1).to_json_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(replacement, path)

        descriptor = writer._fd
        assert descriptor is not None
        with pytest.raises(RuntimeError, match="stream identity changed"):
            writer._flush_index_unlocked(descriptor)
    finally:
        writer.close()

    page = read_training_telemetry(path, after_sequence=0, limit=10)
    assert [item.sequence for item in page.items] == [1]
