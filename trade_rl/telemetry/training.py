"""Validated append-only telemetry for training-rollout visualization."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, Self, cast

TELEMETRY_SCHEMA_VERSION = "training_telemetry_v1"
TelemetryEventType = Literal[
    "rollout",
    "position",
    "risk",
    "episode_end",
    "checkpoint",
    "gap",
]
_EVENT_TYPES = {
    "rollout",
    "position",
    "risk",
    "episode_end",
    "checkpoint",
    "gap",
}
_MAX_READ_LIMIT = 2_000


def _finite(value: float | None, *, field: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


def _finite_tuple(values: tuple[float, ...], *, field: str) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{field} values must be finite")


def _required_int(raw: dict[str, Any], name: str, *, minimum: int = 0) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"telemetry {name} is invalid")
    return value


def _optional_int(raw: dict[str, Any], name: str) -> int | None:
    value = raw.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"telemetry {name} is invalid")
    return value


def _required_string(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"telemetry {name} is invalid")
    return value


def _optional_string(raw: dict[str, Any], name: str) -> str | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"telemetry {name} is invalid")
    return value


def _optional_float(raw: dict[str, Any], name: str) -> float | None:
    value = raw.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"telemetry {name} is invalid")
    resolved = float(value)
    _finite(resolved, field=name)
    return resolved


def _float_tuple(raw: dict[str, Any], name: str) -> tuple[float, ...]:
    value = raw.get(name)
    if not isinstance(value, list):
        raise ValueError(f"telemetry {name} is invalid")
    resolved: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"telemetry {name} is invalid")
        resolved.append(float(item))
    result = tuple(resolved)
    _finite_tuple(result, field=name)
    return result


def _string_tuple(raw: dict[str, Any], name: str) -> tuple[str, ...]:
    value = raw.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"telemetry {name} is invalid")
    return tuple(cast(list[str], value))


@dataclass(frozen=True)
class TrainingTelemetryRecord:
    """One JSON-native, identity-scoped training visualization record."""

    sequence: int
    recorded_at: str
    global_step: int
    environment_step: int
    seed: int
    environment_id: int
    event_type: TelemetryEventType
    market_index: int | None
    market_time: str | None
    symbol: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    action: tuple[float, ...]
    executed_target: tuple[float, ...]
    weights_before: tuple[float, ...]
    weights_after: tuple[float, ...]
    portfolio_value: float | None
    baseline_portfolio_value: float | None
    reward: float | None
    drawdown: float | None
    interval_cost: float | None
    interval_return: float | None
    risk_reasons: tuple[str, ...]
    emergency_deleverage: bool
    terminated: bool
    truncated: bool
    schema_version: str = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported telemetry schema")
        for name, integer_value, minimum in (
            ("sequence", self.sequence, 1),
            ("global_step", self.global_step, 0),
            ("environment_step", self.environment_step, 0),
            ("seed", self.seed, 0),
            ("environment_id", self.environment_id, 0),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < minimum
            ):
                raise ValueError(f"{name} is invalid")
        if self.market_index is not None and (
            isinstance(self.market_index, bool)
            or not isinstance(self.market_index, int)
            or self.market_index < 0
        ):
            raise ValueError("market_index is invalid")
        try:
            recorded = datetime.fromisoformat(self.recorded_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("recorded_at is invalid") from error
        if recorded.tzinfo is None:
            raise ValueError("recorded_at must include a timezone")
        if self.market_time is not None and not self.market_time:
            raise ValueError("market_time is invalid")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.event_type not in _EVENT_TYPES:
            raise ValueError("event_type is invalid")
        for field, metric_value in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("portfolio_value", self.portfolio_value),
            ("baseline_portfolio_value", self.baseline_portfolio_value),
            ("reward", self.reward),
            ("drawdown", self.drawdown),
            ("interval_cost", self.interval_cost),
            ("interval_return", self.interval_return),
        ):
            _finite(metric_value, field=field)
        for field, values in (
            ("action", self.action),
            ("executed_target", self.executed_target),
            ("weights_before", self.weights_before),
            ("weights_after", self.weights_after),
        ):
            _finite_tuple(values, field=field)
        if self.high is not None and self.low is not None and self.high < self.low:
            raise ValueError("high cannot be below low")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "recorded_at": self.recorded_at,
            "global_step": self.global_step,
            "environment_step": self.environment_step,
            "seed": self.seed,
            "environment_id": self.environment_id,
            "event_type": self.event_type,
            "market_index": self.market_index,
            "market_time": self.market_time,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "action": list(self.action),
            "executed_target": list(self.executed_target),
            "weights_before": list(self.weights_before),
            "weights_after": list(self.weights_after),
            "portfolio_value": self.portfolio_value,
            "baseline_portfolio_value": self.baseline_portfolio_value,
            "reward": self.reward,
            "drawdown": self.drawdown,
            "interval_cost": self.interval_cost,
            "interval_return": self.interval_return,
            "risk_reasons": list(self.risk_reasons),
            "emergency_deleverage": self.emergency_deleverage,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    @classmethod
    def from_json_dict(cls, value: object) -> Self:
        if not isinstance(value, dict):
            raise ValueError("telemetry record must be an object")
        raw = cast(dict[str, Any], value)
        if raw.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported telemetry schema")
        event_type = _required_string(raw, "event_type")
        if event_type not in _EVENT_TYPES:
            raise ValueError("event_type is invalid")
        return cls(
            sequence=_required_int(raw, "sequence", minimum=1),
            recorded_at=_required_string(raw, "recorded_at"),
            global_step=_required_int(raw, "global_step"),
            environment_step=_required_int(raw, "environment_step"),
            seed=_required_int(raw, "seed"),
            environment_id=_required_int(raw, "environment_id"),
            event_type=cast(TelemetryEventType, event_type),
            market_index=_optional_int(raw, "market_index"),
            market_time=_optional_string(raw, "market_time"),
            symbol=_required_string(raw, "symbol"),
            open=_optional_float(raw, "open"),
            high=_optional_float(raw, "high"),
            low=_optional_float(raw, "low"),
            close=_optional_float(raw, "close"),
            action=_float_tuple(raw, "action"),
            executed_target=_float_tuple(raw, "executed_target"),
            weights_before=_float_tuple(raw, "weights_before"),
            weights_after=_float_tuple(raw, "weights_after"),
            portfolio_value=_optional_float(raw, "portfolio_value"),
            baseline_portfolio_value=_optional_float(raw, "baseline_portfolio_value"),
            reward=_optional_float(raw, "reward"),
            drawdown=_optional_float(raw, "drawdown"),
            interval_cost=_optional_float(raw, "interval_cost"),
            interval_return=_optional_float(raw, "interval_return"),
            risk_reasons=_string_tuple(raw, "risk_reasons"),
            emergency_deleverage=bool(raw.get("emergency_deleverage", False)),
            terminated=bool(raw.get("terminated", False)),
            truncated=bool(raw.get("truncated", False)),
        )


@dataclass(frozen=True, slots=True)
class TrainingTelemetryPage:
    items: tuple[TrainingTelemetryRecord, ...]
    next_sequence: int
    truncated: bool
    malformed_lines: int
    sequence_gaps: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class TrainingTelemetryStatus:
    available: bool
    record_count: int
    last_sequence: int
    malformed_lines: int
    size_bytes: int


class TrainingTelemetryWriter:
    """Thread-safe append-only JSONL writer with monotonic sequence checks."""

    def __init__(self, path: Path, *, flush_every: int = 32) -> None:
        if isinstance(flush_every, bool) or flush_every <= 0:
            raise ValueError("flush_every must be positive")
        self.path = Path(path)
        self.flush_every = int(flush_every)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] | None = self.path.open(
            "a", encoding="utf-8", newline="\n"
        )
        self._lock = threading.Lock()
        self._pending = 0
        self._last_sequence = training_telemetry_status(self.path).last_sequence

    def append(self, record: TrainingTelemetryRecord) -> None:
        with self._lock:
            if self._handle is None:
                raise RuntimeError("telemetry writer is closed")
            if record.sequence <= self._last_sequence:
                raise ValueError("telemetry sequence must strictly increase")
            payload = json.dumps(
                record.to_json_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            self._handle.write(payload + "\n")
            self._last_sequence = record.sequence
            self._pending += 1
            if self._pending >= self.flush_every:
                self._handle.flush()
                self._pending = 0

    def flush(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            if self._handle is None:
                return
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._pending = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def read_training_telemetry(
    path: Path,
    *,
    after_sequence: int = 0,
    limit: int = 512,
) -> TrainingTelemetryPage:
    if isinstance(after_sequence, bool) or after_sequence < 0:
        raise ValueError("after_sequence must be non-negative")
    if isinstance(limit, bool) or limit <= 0 or limit > _MAX_READ_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_READ_LIMIT}")
    resolved = Path(path)
    if not resolved.is_file():
        return TrainingTelemetryPage((), after_sequence, False, 0, ())

    items: list[TrainingTelemetryRecord] = []
    malformed_lines = 0
    sequence_gaps: list[tuple[int, int]] = []
    previous_sequence: int | None = None
    truncated = False
    with resolved.open("r", encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = TrainingTelemetryRecord.from_json_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError):
                malformed_lines += 1
                continue
            if previous_sequence is not None:
                if record.sequence <= previous_sequence:
                    malformed_lines += 1
                    continue
                if record.sequence > previous_sequence + 1:
                    sequence_gaps.append((previous_sequence + 1, record.sequence - 1))
            previous_sequence = record.sequence
            if record.sequence <= after_sequence:
                continue
            if len(items) >= limit:
                truncated = True
                break
            items.append(record)

    next_sequence = items[-1].sequence if items else after_sequence
    return TrainingTelemetryPage(
        items=tuple(items),
        next_sequence=next_sequence,
        truncated=truncated,
        malformed_lines=malformed_lines,
        sequence_gaps=tuple(sequence_gaps),
    )


def training_telemetry_status(path: Path) -> TrainingTelemetryStatus:
    resolved = Path(path)
    if not resolved.is_file():
        return TrainingTelemetryStatus(False, 0, 0, 0, 0)
    record_count = 0
    malformed_lines = 0
    last_sequence = 0
    with resolved.open("r", encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = TrainingTelemetryRecord.from_json_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError):
                malformed_lines += 1
                continue
            if record.sequence <= last_sequence:
                malformed_lines += 1
                continue
            last_sequence = record.sequence
            record_count += 1
    return TrainingTelemetryStatus(
        available=True,
        record_count=record_count,
        last_sequence=last_sequence,
        malformed_lines=malformed_lines,
        size_bytes=resolved.stat().st_size,
    )


__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryEventType",
    "TrainingTelemetryPage",
    "TrainingTelemetryRecord",
    "TrainingTelemetryStatus",
    "TrainingTelemetryWriter",
    "read_training_telemetry",
    "training_telemetry_status",
]
