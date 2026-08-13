from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.operations.universal_training_monitor import (
    inspect_universal_training_generation,
)
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateEpisodeMetricsV2,
)
from trade_rl.workflows.universal_causal_alpha_teacher import (
    write_causal_alpha_selection_checkpoint_metric_v2,
)

NOW = datetime(2026, 8, 12, 12, 1, tzinfo=UTC)


def _generation(
    root: Path, *, stale: bool = False, nonfinite_log: bool = False
) -> Path:
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
                    "interval_gross_return": reward + 0.01,
                    "baseline_excess_return": reward - 0.05,
                    "filled_turnover": 5.0 - step,
                    "fill_count": 5 - step,
                    "target_delta_l1": 0.5 - step * 0.05,
                    "sign_flip_count": 5 - step,
                    "command_target_delta_l1": 0.6 - step * 0.05,
                    "command_target_sign_flip_count": 5 - step,
                    "gross_pnl": reward * 100.0,
                    "net_pnl": reward * 90.0,
                }
            )
            for step, reward, drawdown in (
                (1, -0.4, 0.2),
                (2, -0.2, 0.18),
                (3, 0.1, 0.15),
                (4, 0.3, 0.1),
            )
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
    (member / "checkpoints" / "step-4096" / "manifest.json").write_text(
        "{}", encoding="utf-8"
    )
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
    assert member.telemetry_trends["baseline_excess_return"].direction == "improving"
    assert member.telemetry_trends["filled_turnover"].direction == "improving"
    assert member.telemetry_trends["gross_pnl"].count == 4
    assert member.telemetry_trends["net_pnl"].count == 4
    assert member.telemetry_trends["sign_flip_count"].direction == "improving"
    assert member.telemetry_trends["command_target_delta_l1"].direction == "improving"
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


def test_telemetry_trend_sampling_stays_bounded(monkeypatch, tmp_path: Path) -> None:
    import trade_rl.operations.universal_training_monitor as module

    monkeypatch.setattr(module, "MAX_TELEMETRY_TREND_POINTS", 3, raising=False)
    member = tmp_path / "member"
    member.mkdir()
    (member / "telemetry.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "global_step": step,
                    "reward": float(step),
                    "symbol": "BTCUSDT",
                }
            )
            + "\n"
            for step in range(10)
        ),
        encoding="utf-8",
    )

    count, _, points, nonfinite = module._telemetry(member)

    assert count == 10
    assert nonfinite == 0
    assert len(points["reward"]) <= 3
    assert points["reward"][0] == (0, 0.0)
    assert points["reward"][-1] == (9, 9.0)


def test_monitor_reports_teacher_progress_before_member_heartbeat(
    tmp_path: Path,
) -> None:
    teacher_root = tmp_path / "_shared-causal-teacher"
    teacher_root.mkdir()
    progress = {
        "completed_replays": 17,
        "phase": "causal_teacher_selection",
        "symbol": "BTCUSDT",
        "total_replays": 100,
    }
    (teacher_root / "causal-teacher-progress.json").write_text(
        json.dumps(progress),
        encoding="utf-8",
    )

    snapshot = inspect_universal_training_generation(tmp_path, now=NOW)

    assert snapshot.status == "incomplete"
    assert snapshot.teacher_progress == progress
    assert snapshot.findings == ("causal teacher selection in progress 17/100",)


def test_monitor_summarizes_bounded_v2_teacher_checkpoint_window(
    monkeypatch, tmp_path: Path
) -> None:
    import trade_rl.operations.universal_training_monitor as module

    monkeypatch.setattr(
        module, "MAX_CAUSAL_TEACHER_CHECKPOINT_BYTES", 12_000, raising=False
    )
    teacher_root = tmp_path / "_shared-causal-teacher"
    teacher_root.mkdir()
    (teacher_root / "causal-teacher-progress.json").write_text(
        json.dumps(
            {
                "completed_replays": 4,
                "phase": "causal_teacher_selection_v2",
                "total_replays": 100,
            }
        ),
        encoding="utf-8",
    )
    signal = evaluate_causal_alpha_signal_diagnostics(
        np.asarray([-0.02, -0.01, 0.01, 0.02]),
        np.asarray([-0.01, -0.02, 0.02, 0.01]),
    )
    metric = CausalAlphaCandidateEpisodeMetricsV2(
        candidate_digest=content_digest("candidate"),
        symbol="BTCUSDT",
        episode_index=0,
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.3,
        total_execution_cost=4.0,
        trade_count=2,
        signal_24h=signal,
        signal_72h=signal,
        cost_suppressed_change_count=3,
        submitted_change_count=2,
        strong_reversal_count=1,
        command_sign_flip_count=1,
        execution_rejection_count=1,
        execution_rejection_reason_counts=(("minimum_notional", 1),),
        risk_projection_reason_counts=(("no_trade_band", 2),),
        hard_risk_violation=False,
    )
    checkpoint = teacher_root / "causal-teacher-selection-checkpoint-v2.jsonl"
    grid_digest = content_digest("grid")
    for episode_index in range(4):
        write_causal_alpha_selection_checkpoint_metric_v2(
            checkpoint,
            replace(metric, episode_index=episode_index, digest=""),
            grid_digest=grid_digest,
        )

    snapshot = inspect_universal_training_generation(tmp_path, now=NOW)

    summary = snapshot.teacher_checkpoint_summary
    assert summary is not None
    assert summary["window_bytes"] <= 12_000
    candidate = summary["candidates"][metric.candidate_digest]
    assert candidate["record_count"] >= 1
    assert candidate["net_return_mean"] == 0.01
    assert candidate["turnover_per_day_mean"] == 0.3
    assert candidate["cost_suppressed_change_count"] >= 3
    assert candidate["execution_rejection_reason_counts"] == {
        "minimum_notional": candidate["record_count"]
    }
    assert candidate["explained_execution_no_fill_count"] == 0
    assert candidate["unexplained_execution_rejection_count"] == candidate[
        "record_count"
    ]
    assert candidate["signal_24h_pearson_mean"] == signal.pearson_correlation
