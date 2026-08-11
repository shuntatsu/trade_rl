from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from trade_rl.operations.universal_training_monitor import (
    inspect_universal_training_generation,
)

NOW = datetime(2026, 8, 12, 12, 1, tzinfo=UTC)


def _generation(root: Path, *, stale: bool = False, nonfinite_log: bool = False) -> Path:
    member = root / "ppo" / "seed-0"
    member.mkdir(parents=True)
    updated = NOW - (timedelta(hours=1) if stale else timedelta(minutes=1))
    (member / "training-heartbeat.json").write_text(
        json.dumps(
            {
                "algorithm": "ppo",
                "global_step": 4096,
                "phase": "training",
                "schema_version": "training_heartbeat_v1",
                "seed": 0,
                "updated_at": updated.isoformat(),
                "scalars": {},
            }
        ),
        encoding="utf-8",
    )
    (member / "telemetry.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "global_step": step,
                    "symbol": "BTCUSDT",
                    "reward": reward,
                    "portfolio_value": 100_000 + step,
                    "baseline_portfolio_value": 100_000,
                    "drawdown": drawdown,
                    "interval_cost": 0.001,
                }
            )
            for step, reward, drawdown in ((1, -0.4, 0.2), (2, -0.2, 0.18), (3, 0.1, 0.15), (4, 0.3, 0.1))
        )
        + "\n",
        encoding="utf-8",
    )
    writer = SummaryWriter(str(member / "tensorboard"))
    for step, reward, growth, drawdown in (
        (1, -0.4, -0.01, 0.20),
        (2, -0.2, 0.0, 0.18),
        (3, 0.1, 0.01, 0.15),
        (4, 0.3, 0.02, 0.10),
    ):
        writer.add_scalar("trade_rl/reward_mean", reward, step)
        writer.add_scalar("trade_rl/reward_growth_raw_mean", growth, step)
        writer.add_scalar("trade_rl/drawdown_mean", drawdown, step)
    writer.close()
    (member / "checkpoints" / "step-4096").mkdir(parents=True)
    (member / "checkpoints" / "step-4096" / "manifest.json").write_text("{}", encoding="utf-8")
    if nonfinite_log:
        (root / "container.log").write_text("CUDA out of memory", encoding="utf-8")
    return root


def test_monitor_reports_reward_components_and_training_health(tmp_path: Path) -> None:
    snapshot = inspect_universal_training_generation(_generation(tmp_path), now=NOW)
    member = snapshot.members[0]
    assert member.reward_total.direction == "improving"
    assert member.reward_growth.direction == "improving"
    assert member.drawdown.direction == "improving"
    assert member.telemetry_records == 4
    assert member.per_symbol_counts == {"BTCUSDT": 4}
    assert snapshot.status == "healthy"


def test_monitor_fails_closed_on_stale_heartbeat(tmp_path: Path) -> None:
    snapshot = inspect_universal_training_generation(
        _generation(tmp_path, stale=True), now=NOW
    )
    assert snapshot.status == "failed"
    assert any("stale" in item.lower() for item in snapshot.findings)


def test_monitor_fails_closed_on_oom_log(tmp_path: Path) -> None:
    snapshot = inspect_universal_training_generation(
        _generation(tmp_path, nonfinite_log=True), now=NOW
    )
    assert snapshot.status == "failed"
    assert any("oom" in item.lower() for item in snapshot.findings)
