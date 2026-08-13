"""Read-only health and reward-trend inspection for Universal generations."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trade_rl.workflows.universal_causal_alpha_teacher import (
    _causal_alpha_candidate_metric_v2_from_payload,
)

REWARD_TAGS = (
    "trade_rl/reward_mean",
    "trade_rl/reward_growth_raw_mean",
    "trade_rl/reward_absolute_component_mean",
    "trade_rl/reward_excess_component_mean",
    "trade_rl/reward_baseline_penalty_weighted_mean",
    "trade_rl/reward_drawdown_penalty_weighted_mean",
    "trade_rl/reward_projection_penalty_weighted_mean",
    "trade_rl/reward_terminal_penalty_weighted_mean",
    "trade_rl/reward_margin_penalty_weighted_mean",
    "trade_rl/reward_total_raw_mean",
    "trade_rl/portfolio_value_mean",
    "trade_rl/baseline_portfolio_value_mean",
    "trade_rl/drawdown_mean",
    "trade_rl/interval_cost_mean",
    "trade_rl/rolling_growth_gap_mean",
)
TRAIN_TAGS = (
    "train/approx_kl",
    "train/explained_variance",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/std",
)
TELEMETRY_TREND_FIELDS = (
    "reward",
    "portfolio_value",
    "baseline_portfolio_value",
    "drawdown",
    "interval_cost",
    "interval_return",
    "filled_turnover",
    "fill_count",
    "interval_gross_return",
    "baseline_excess_return",
    "target_delta_l1",
    "sign_flip_count",
    "command_target_delta_l1",
    "command_target_sign_flip_count",
    "gross_pnl",
    "net_pnl",
)
TELEMETRY_LOWER_IS_BETTER = {
    "drawdown",
    "interval_cost",
    "filled_turnover",
    "target_delta_l1",
    "sign_flip_count",
    "command_target_delta_l1",
    "command_target_sign_flip_count",
}
MAX_TELEMETRY_TREND_POINTS = 4_096
MAX_CAUSAL_TEACHER_CHECKPOINT_BYTES = 8 * 1024 * 1024


@dataclass(slots=True)
class _TeacherCandidateAggregate:
    command_sign_flip_count: int = 0
    cost_suppressed_change_count: int = 0
    execution_rejection_count: int = 0
    execution_rejection_reason_counts: Counter[str] = field(default_factory=Counter)
    gross_returns: list[float] = field(default_factory=list)
    hard_risk_violation: bool = False
    net_returns: list[float] = field(default_factory=list)
    risk_projection_reason_counts: Counter[str] = field(default_factory=Counter)
    signal_24h_direction_accuracy: list[float] = field(default_factory=list)
    signal_24h_pearson: list[float] = field(default_factory=list)
    signal_24h_rank: list[float] = field(default_factory=list)
    strong_reversal_count: int = 0
    submitted_change_count: int = 0
    total_execution_cost: float = 0.0
    total_trade_count: int = 0
    turnover_per_day: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScalarTrend:
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    first_window_mean: float | None = None
    last_window_mean: float | None = None
    slope_per_100k_steps: float | None = None
    direction: str = "missing"


@dataclass(frozen=True, slots=True)
class UniversalTrainingMemberSnapshot:
    algorithm: str
    seed: int
    phase: str
    global_step: int
    heartbeat_updated_at: str
    status: str
    reward_total: ScalarTrend
    reward_growth: ScalarTrend
    drawdown: ScalarTrend
    scalar_trends: dict[str, ScalarTrend]
    telemetry_trends: dict[str, ScalarTrend]
    telemetry_records: int
    per_symbol_counts: dict[str, int]
    checkpoint_count: int
    reward_tag_count: int
    nonfinite_count: int
    findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UniversalTrainingSnapshot:
    generation_root: Path
    inspected_at: str
    status: str
    members: tuple[UniversalTrainingMemberSnapshot, ...]
    teacher_progress: dict[str, object] | None
    teacher_checkpoint_summary: dict[str, object] | None
    findings: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["generation_root"] = str(self.generation_root)
        return payload


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("heartbeat updated_at is invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("heartbeat updated_at must be timezone-aware")
    return parsed.astimezone(UTC)


def _trend(
    points: list[tuple[int, float]], *, lower_is_better: bool = False
) -> ScalarTrend:
    finite = [(step, value) for step, value in points if math.isfinite(value)]
    if not finite:
        return ScalarTrend()
    values = [value for _, value in finite]
    width = max(1, math.ceil(len(values) * 0.2))
    first = statistics.fmean(values[:width])
    last = statistics.fmean(values[-width:])
    delta = last - first
    tolerance = max(1e-12, statistics.fmean(abs(value) for value in values) * 1e-3)
    if abs(delta) <= tolerance:
        direction = "flat"
    elif (delta < 0) == lower_is_better:
        direction = "improving"
    else:
        direction = "worsening"
    step_delta = finite[-1][0] - finite[0][0]
    slope = None if step_delta <= 0 else (values[-1] - values[0]) * 100_000 / step_delta
    return ScalarTrend(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        first_window_mean=first,
        last_window_mean=last,
        slope_per_100k_steps=slope,
        direction=direction,
    )


def _tensorboard_scalars(
    member: Path,
) -> tuple[dict[str, list[tuple[int, float]]], int]:
    values: dict[str, list[tuple[int, float]]] = {}
    nonfinite = 0
    directories = sorted(
        {path.parent for path in member.rglob("events.out.tfevents.*")}
    )
    for directory in directories:
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", ()):
            points = values.setdefault(tag, [])
            for event in accumulator.Scalars(tag):
                value = float(event.value)
                if not math.isfinite(value):
                    nonfinite += 1
                points.append((int(event.step), value))
    for points in values.values():
        points.sort(key=lambda item: item[0])
    return values, nonfinite


def _telemetry(
    member: Path,
) -> tuple[int, dict[str, int], dict[str, list[tuple[int, float]]], int]:
    path = member / "telemetry.jsonl"
    if not path.is_file():
        return 0, {}, {}, 0
    count = 0
    nonfinite = 0
    symbols: Counter[str] = Counter()
    trends: dict[str, list[tuple[int, float]]] = {}
    with path.open("r", encoding="utf-8") as telemetry_file:
        for line in telemetry_file:
            if not line.strip():
                continue
            count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                nonfinite += 1
                continue
            symbol = payload.get("symbol")
            if isinstance(symbol, str) and symbol:
                symbols[symbol] += 1
            raw_step = payload.get("global_step")
            step = (
                int(raw_step)
                if isinstance(raw_step, int) and not isinstance(raw_step, bool)
                else count
            )
            for key in TELEMETRY_TREND_FIELDS:
                value = payload.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                number = float(value)
                if not math.isfinite(number):
                    nonfinite += 1
                    continue
                points = trends.setdefault(key, [])
                points.append((step, number))
                overflow = len(points) - MAX_TELEMETRY_TREND_POINTS
                if overflow > 0:
                    del points[1 : 1 + overflow]
    return count, dict(sorted(symbols.items())), trends, nonfinite


def _member_snapshot(member: Path, *, now: datetime) -> UniversalTrainingMemberSnapshot:
    heartbeat_path = member / "training-heartbeat.json"
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    updated = _parse_time(payload.get("updated_at"))
    findings: list[str] = []
    if now - updated > timedelta(minutes=30):
        findings.append("stale training heartbeat")
    scalars, scalar_nonfinite = _tensorboard_scalars(member)
    telemetry_records, symbol_counts, telemetry_points, telemetry_nonfinite = (
        _telemetry(member)
    )
    nonfinite = scalar_nonfinite + telemetry_nonfinite
    if nonfinite:
        findings.append(f"non-finite evidence count={nonfinite}")
    checkpoints = tuple((member / "checkpoints").glob("step-*/manifest.json"))
    phase = str(payload.get("phase", "unknown"))
    if phase == "completed" and not checkpoints:
        findings.append("completed member has no checkpoint manifest")
    trends = {
        tag: _trend(
            points,
            lower_is_better=tag
            in {"trade_rl/drawdown_mean", "trade_rl/interval_cost_mean"},
        )
        for tag, points in scalars.items()
    }
    telemetry_trends = {
        field: _trend(
            points,
            lower_is_better=field in TELEMETRY_LOWER_IS_BETTER,
        )
        for field, points in telemetry_points.items()
    }
    reward_total = trends.get("trade_rl/reward_mean", ScalarTrend())
    reward_growth = trends.get("trade_rl/reward_growth_raw_mean", ScalarTrend())
    drawdown = trends.get("trade_rl/drawdown_mean", ScalarTrend())
    status = "failed" if findings else "healthy"
    return UniversalTrainingMemberSnapshot(
        algorithm=str(payload.get("algorithm", member.parent.name)),
        seed=int(payload.get("seed", member.name.removeprefix("seed-"))),
        phase=phase,
        global_step=int(payload.get("global_step", 0)),
        heartbeat_updated_at=updated.isoformat(),
        status=status,
        reward_total=reward_total,
        reward_growth=reward_growth,
        drawdown=drawdown,
        scalar_trends=trends,
        telemetry_trends=telemetry_trends,
        telemetry_records=telemetry_records,
        per_symbol_counts=symbol_counts,
        checkpoint_count=len(checkpoints),
        reward_tag_count=sum(tag in scalars for tag in REWARD_TAGS),
        nonfinite_count=nonfinite,
        findings=tuple(findings),
    )


def _causal_teacher_checkpoint_summary(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    start = max(0, size - MAX_CAUSAL_TEACHER_CHECKPOINT_BYTES)
    with path.open("rb") as checkpoint:
        checkpoint.seek(start)
        if start > 0:
            checkpoint.readline()
        data = checkpoint.read(MAX_CAUSAL_TEACHER_CHECKPOINT_BYTES)
    rows = tuple(line for line in data.splitlines() if line.strip())
    candidates: dict[str, _TeacherCandidateAggregate] = {}
    grid_digest: str | None = None
    for line in rows:
        raw = json.loads(line)
        if raw.get("schema_version") != "causal_alpha_selection_checkpoint_metric_v2":
            raise ValueError("causal teacher checkpoint schema mismatch")
        row_grid = raw.get("grid_digest")
        if not isinstance(row_grid, str) or len(row_grid) != 64:
            raise ValueError("causal teacher checkpoint grid digest is invalid")
        if grid_digest is None:
            grid_digest = row_grid
        elif grid_digest != row_grid:
            raise ValueError("causal teacher checkpoint grid digest drifted")
        metric = _causal_alpha_candidate_metric_v2_from_payload(raw)
        aggregate = candidates.setdefault(
            metric.candidate_digest, _TeacherCandidateAggregate()
        )
        aggregate.gross_returns.append(metric.gross_return)
        aggregate.net_returns.append(metric.net_return)
        aggregate.turnover_per_day.append(metric.turnover_per_day)
        aggregate.signal_24h_direction_accuracy.append(
            metric.signal_24h.direction_accuracy
        )
        if metric.signal_24h.pearson_correlation is not None:
            aggregate.signal_24h_pearson.append(metric.signal_24h.pearson_correlation)
        if metric.signal_24h.rank_correlation is not None:
            aggregate.signal_24h_rank.append(metric.signal_24h.rank_correlation)
        aggregate.command_sign_flip_count += metric.command_sign_flip_count
        aggregate.cost_suppressed_change_count += metric.cost_suppressed_change_count
        aggregate.execution_rejection_count += metric.execution_rejection_count
        aggregate.strong_reversal_count += metric.strong_reversal_count
        aggregate.submitted_change_count += metric.submitted_change_count
        aggregate.total_execution_cost += metric.total_execution_cost
        aggregate.total_trade_count += metric.trade_count
        aggregate.hard_risk_violation |= metric.hard_risk_violation
        aggregate.execution_rejection_reason_counts.update(
            dict(metric.execution_rejection_reason_counts)
        )
        aggregate.risk_projection_reason_counts.update(
            dict(metric.risk_projection_reason_counts)
        )

    summaries: dict[str, object] = {}
    for digest, aggregate in sorted(candidates.items()):
        summaries[digest] = {
            "command_sign_flip_count": aggregate.command_sign_flip_count,
            "cost_suppressed_change_count": aggregate.cost_suppressed_change_count,
            "execution_rejection_count": aggregate.execution_rejection_count,
            "execution_rejection_reason_counts": dict(
                sorted(aggregate.execution_rejection_reason_counts.items())
            ),
            "gross_return_mean": statistics.fmean(aggregate.gross_returns),
            "hard_risk_violation": aggregate.hard_risk_violation,
            "net_return_lower_tail": min(aggregate.net_returns),
            "net_return_mean": statistics.fmean(aggregate.net_returns),
            "record_count": len(aggregate.net_returns),
            "risk_projection_reason_counts": dict(
                sorted(aggregate.risk_projection_reason_counts.items())
            ),
            "signal_24h_direction_accuracy_mean": statistics.fmean(
                aggregate.signal_24h_direction_accuracy
            ),
            "signal_24h_pearson_mean": (
                None
                if not aggregate.signal_24h_pearson
                else statistics.fmean(aggregate.signal_24h_pearson)
            ),
            "signal_24h_rank_mean": (
                None
                if not aggregate.signal_24h_rank
                else statistics.fmean(aggregate.signal_24h_rank)
            ),
            "strong_reversal_count": aggregate.strong_reversal_count,
            "submitted_change_count": aggregate.submitted_change_count,
            "total_execution_cost": aggregate.total_execution_cost,
            "total_trade_count": aggregate.total_trade_count,
            "turnover_per_day_mean": statistics.fmean(aggregate.turnover_per_day),
        }
    return {
        "candidates": summaries,
        "grid_digest": grid_digest,
        "record_count": len(rows),
        "schema_version": "causal_teacher_checkpoint_summary_v1",
        "window_bytes": len(data),
        "window_start_offset": start,
    }


def inspect_universal_training_generation(
    root: str | Path,
    *,
    now: datetime | None = None,
    container_state: dict[str, Any] | None = None,
    container_log: str | None = None,
) -> UniversalTrainingSnapshot:
    """Inspect current immutable evidence without changing the generation."""

    generation = Path(root)
    current = (now or datetime.now(UTC)).astimezone(UTC)
    findings: list[str] = []
    log = container_log
    if log is None and (generation / "container.log").is_file():
        log = (generation / "container.log").read_text(
            encoding="utf-8", errors="replace"
        )
    if log:
        if re.search(r"out of memory|oomkilled|\boom\b", log, re.IGNORECASE):
            findings.append("OOM evidence found in container log")
        if re.search(r"traceback|\bnan\b|\binf(?:inity)?\b", log, re.IGNORECASE):
            findings.append("traceback or non-finite evidence found in container log")
    if container_state:
        if bool(container_state.get("OOMKilled")):
            findings.append("container state reports OOMKilled")
        if (
            container_state.get("Status") == "exited"
            and int(container_state.get("ExitCode", 0)) != 0
        ):
            findings.append(
                f"container exited with code {container_state.get('ExitCode')}"
            )
    teacher_progress: dict[str, object] | None = None
    teacher_checkpoint_summary: dict[str, object] | None = None
    teacher_progress_path = (
        generation / "_shared-causal-teacher" / "causal-teacher-progress.json"
    )
    if teacher_progress_path.is_file():
        try:
            raw_teacher_progress = json.loads(
                teacher_progress_path.read_text(encoding="utf-8")
            )
            if not isinstance(raw_teacher_progress, dict):
                raise TypeError("teacher progress must be an object")
            teacher_progress = raw_teacher_progress
        except (OSError, TypeError, json.JSONDecodeError) as error:
            findings.append(f"invalid causal teacher progress: {error}")
    teacher_checkpoint_path = (
        generation
        / "_shared-causal-teacher"
        / "causal-teacher-selection-checkpoint-v2.jsonl"
    )
    if teacher_checkpoint_path.is_file():
        try:
            teacher_checkpoint_summary = _causal_teacher_checkpoint_summary(
                teacher_checkpoint_path
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            findings.append(f"invalid causal teacher checkpoint: {error}")
    members: list[UniversalTrainingMemberSnapshot] = []
    for heartbeat in sorted(generation.glob("*/seed-*/training-heartbeat.json")):
        try:
            members.append(_member_snapshot(heartbeat.parent, now=current))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            findings.append(f"invalid member evidence {heartbeat.parent}: {error}")
    if not members:
        if teacher_progress is None:
            findings.append("no member heartbeat evidence")
        else:
            completed = teacher_progress.get("completed_replays", 0)
            total = teacher_progress.get("total_replays", 0)
            findings.append(f"causal teacher selection in progress {completed}/{total}")
    findings.extend(item for member in members for item in member.findings)
    status = "healthy"
    if findings:
        status = "failed" if members or len(findings) > 1 else "incomplete"
    return UniversalTrainingSnapshot(
        generation_root=generation,
        inspected_at=current.isoformat(),
        status=status,
        members=tuple(members),
        teacher_progress=teacher_progress,
        teacher_checkpoint_summary=teacher_checkpoint_summary,
        findings=tuple(findings),
    )


__all__ = [
    "REWARD_TAGS",
    "TRAIN_TAGS",
    "ScalarTrend",
    "UniversalTrainingMemberSnapshot",
    "UniversalTrainingSnapshot",
    "inspect_universal_training_generation",
]
