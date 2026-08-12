"""Read-only health and reward-trend inspection for Universal generations."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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


def _telemetry(member: Path) -> tuple[int, dict[str, int], int]:
    path = member / "telemetry.jsonl"
    if not path.is_file():
        return 0, {}, 0
    count = 0
    nonfinite = 0
    symbols: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
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
        for key in (
            "reward",
            "portfolio_value",
            "baseline_portfolio_value",
            "drawdown",
            "interval_cost",
        ):
            value = payload.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                nonfinite += 1
    return count, dict(sorted(symbols.items())), nonfinite


def _member_snapshot(member: Path, *, now: datetime) -> UniversalTrainingMemberSnapshot:
    heartbeat_path = member / "training-heartbeat.json"
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    updated = _parse_time(payload.get("updated_at"))
    findings: list[str] = []
    if now - updated > timedelta(minutes=30):
        findings.append("stale training heartbeat")
    scalars, scalar_nonfinite = _tensorboard_scalars(member)
    telemetry_records, symbol_counts, telemetry_nonfinite = _telemetry(member)
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
        telemetry_records=telemetry_records,
        per_symbol_counts=symbol_counts,
        checkpoint_count=len(checkpoints),
        reward_tag_count=sum(tag in scalars for tag in REWARD_TAGS),
        nonfinite_count=nonfinite,
        findings=tuple(findings),
    )


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
    members: list[UniversalTrainingMemberSnapshot] = []
    for heartbeat in sorted(generation.glob("*/seed-*/training-heartbeat.json")):
        try:
            members.append(_member_snapshot(heartbeat.parent, now=current))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            findings.append(f"invalid member evidence {heartbeat.parent}: {error}")
    if not members:
        findings.append("no member heartbeat evidence")
    findings.extend(item for member in members for item in member.findings)
    status = "healthy"
    if findings:
        status = "failed" if members or len(findings) > 1 else "incomplete"
    return UniversalTrainingSnapshot(
        generation_root=generation,
        inspected_at=current.isoformat(),
        status=status,
        members=tuple(members),
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
