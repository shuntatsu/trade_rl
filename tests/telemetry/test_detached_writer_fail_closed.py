from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trade_rl.rl.training_telemetry import TrainingTelemetryRecord
from trade_rl.telemetry.training import TrainingTelemetryWriter, read_training_telemetry


def _record(sequence: int) -> TrainingTelemetryRecord:
    return TrainingTelemetryRecord(
        sequence=sequence,
        run_id="run-1",
        candidate="candidate-a",
        fold_id="fold-0",
        seed=7,
        checkpoint_step=sequence * 100,
        wall_time_seconds=float(sequence),
        episode_return=None,
        episode_length=None,
        equity=100.0,
        drawdown=0.0,
        gross_exposure=0.5,
        turnover=0.1,
        transaction_cost=0.01,
        funding_cost=0.0,
        borrow_cost=0.0,
        cash_interest=0.0,
        reward=0.2,
        reward_components={"growth": 0.2},
        action_mean=(0.1,),
        action_std=(0.2,),
        positions=(0.5,),
        closed_trades=None,
        winning_trades=None,
        losing_trades=None,
        breakeven_trades=None,
        trade_win_rate=None,
        average_realized_pnl=None,
        profit_factor=None,
        net_realized_pnl=None,
        gross_profit=None,
        gross_loss=None,
        open_positions=None,
        learning_rate=0.001,
        entropy=0.03,
        policy_loss=-0.1,
        value_loss=0.4,
        explained_variance=0.5,
        approx_kl=0.01,
        clip_fraction=0.02,
        normalizer={"count": 1.0},
        schema_version="training_telemetry_v2",
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
