from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.integrations.training_telemetry import (
    TrainingTelemetrySampler,
    environment_market_snapshot,
)
from trade_rl.telemetry.training import read_training_telemetry


def info(
    step: int,
    *,
    before: tuple[float, ...] = (0.20,),
    after: tuple[float, ...] = (0.20,),
    risk_reasons: tuple[str, ...] = (),
    emergency: bool = False,
    terminated: bool = False,
) -> dict[str, object]:
    return {
        "decision_step_index": step,
        "telemetry_market_index": 100 + step,
        "telemetry_market_time": "2026-07-21T08:00:00.000000000",
        "telemetry_symbol": "BTCUSDT",
        "telemetry_ohlc": (67_500.0, 67_900.0, 67_400.0, 67_842.3),
        "telemetry_weights_before": np.asarray(before, dtype=np.float64),
        "telemetry_weights_after": np.asarray(after, dtype=np.float64),
        "submitted_target": np.asarray(after, dtype=np.float64),
        "executed_target": np.asarray(after, dtype=np.float64),
        "portfolio_value_after": 101_342.85,
        "baseline_portfolio_value_after": 100_400.0,
        "reward_total_scaled": 0.214,
        "drawdown_after": 0.0086,
        "interval_cost": 4.25,
        "interval_net_return": 0.0012,
        "telemetry_risk_reasons": risk_reasons,
        "emergency_deleverage": emergency,
        "hybrid_terminated": terminated,
    }


def test_environment_market_snapshot_aggregates_exact_decision_interval() -> None:
    timestamps = np.datetime64("2026-07-21T08:00:00", "ns") + np.arange(
        6
    ) * np.timedelta64(5, "m")
    open_price = np.column_stack((np.arange(100.0, 106.0), np.arange(200.0, 206.0)))
    close = open_price + 0.5
    dataset = SimpleNamespace(
        n_bars=6,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        open=open_price,
        high=open_price + np.asarray([2.0, 3.0]),
        low=open_price - np.asarray([1.0, 2.0]),
        close=close,
    )
    environment = SimpleNamespace(
        unwrapped=SimpleNamespace(dataset=dataset, current_index=5)
    )

    snapshot = environment_market_snapshot(environment, bars_advanced=2)

    assert snapshot["telemetry_symbol"] == "BTCUSDT"
    assert snapshot["telemetry_market_index"] == 5
    assert snapshot["telemetry_market_time"] == "2026-07-21T08:25:00.000000000"
    assert snapshot["telemetry_ohlc"] == pytest.approx((104.0, 107.0, 103.0, 105.5))


def test_sampler_skips_unimportant_steps_and_preserves_position_risk_and_terminal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    sampler = TrainingTelemetrySampler(path, seed=7, sample_every=32)

    assert (
        sampler.consume(
            global_step=1,
            actions=np.asarray([[0.1]], dtype=np.float32),
            rewards=np.asarray([0.0], dtype=np.float32),
            dones=np.asarray([False]),
            infos=(info(1),),
        )
        == 0
    )
    assert (
        sampler.consume(
            global_step=2,
            actions=np.asarray([[0.4]], dtype=np.float32),
            rewards=np.asarray([0.2], dtype=np.float32),
            dones=np.asarray([False]),
            infos=(info(2, before=(0.2,), after=(0.4,)),),
        )
        == 1
    )
    assert (
        sampler.consume(
            global_step=3,
            actions=np.asarray([[0.2]], dtype=np.float32),
            rewards=np.asarray([-0.1], dtype=np.float32),
            dones=np.asarray([False]),
            infos=(info(3, risk_reasons=("drawdown",)),),
        )
        == 1
    )
    assert (
        sampler.consume(
            global_step=4,
            actions=np.asarray([[0.0]], dtype=np.float32),
            rewards=np.asarray([-1.0], dtype=np.float32),
            dones=np.asarray([True]),
            infos=(info(4, terminated=True),),
        )
        == 1
    )
    sampler.close()

    page = read_training_telemetry(path, limit=10)
    assert [item.event_type for item in page.items] == [
        "position",
        "risk",
        "episode_end",
    ]
    assert [item.sequence for item in page.items] == [1, 2, 3]
    assert page.items[0].action == pytest.approx((0.4,))


def test_sampler_records_regular_interval_and_disables_itself_after_writer_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    sampler = TrainingTelemetrySampler(path, seed=1, sample_every=2)

    assert (
        sampler.consume(
            global_step=2,
            actions=np.asarray([[0.1]], dtype=np.float32),
            rewards=np.asarray([0.1], dtype=np.float32),
            dones=np.asarray([False]),
            infos=(info(2),),
        )
        == 1
    )
    sampler.close()

    assert (
        sampler.consume(
            global_step=4,
            actions=np.asarray([[0.1]], dtype=np.float32),
            rewards=np.asarray([0.1], dtype=np.float32),
            dones=np.asarray([False]),
            infos=(info(4),),
        )
        == 0
    )
    assert sampler.disabled is True
