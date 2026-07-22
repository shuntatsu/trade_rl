"""Strict append-only telemetry parsing with an identity-bound sparse index."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

from trade_rl.telemetry import training as _training

_INDEX_SCHEMA = "training_telemetry_index_v1"
_INDEX_STRIDE = 64
_BaseRecord = _training.TrainingTelemetryRecord
_BaseWriter = _training.TrainingTelemetryWriter
Pair = tuple[int, int]


def _required_bool(raw: dict[str, Any], name: str) -> bool:
    if name not in raw or not isinstance(raw[name], bool):
        raise ValueError(f"telemetry {name} must be a boolean")
    return cast(bool, raw[name])


class StrictTrainingTelemetryRecord(_BaseRecord):
    """Training record whose JSON flags never use truthiness coercion."""

    @classmethod
    def from_json_dict(cls, value: object) -> Self:
        if not isinstance(value, dict):
            raise ValueError("telemetry record must be an object")
        raw = cast(dict[str, Any], value)
        for name in ("emergency_deleverage", "terminated", "truncated"):
            _required_bool(raw, name)
        return super().from_json_dict(value)


@dataclass(slots=True)
class _TelemetryIndex:
    device: int
    inode: int
    indexed_size: int = 0
    last_scan_start: int = 0
    record_count: int = 0
    last_sequence: int = 0
    malformed_lines: int = 0
    sequence_gaps: list[Pair] = field(default_factory=list)
    checkpoints: list[Pair] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": _INDEX_SCHEMA,
            "device": self.device,
            "inode": self.inode,
            "indexed_size": self.indexed_size,
            "last_scan_start": self.last_scan_start,
            "record_count": self.record_count,
            "last_sequence": self.last_sequence,
            "malformed_lines": self.malformed_lines,
            "sequence_gaps": self.sequence_gaps,
            "checkpoints": self.checkpoints,
        }


def _index_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.index.json")


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"telemetry index {field_name} is invalid")
    return value


def _pairs(value: object, *, field_name: str) -> list[Pair]:
    if not isinstance(value, list):
        raise ValueError(f"telemetry index {field_name} is invalid")
    result: list[Pair] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in item)
            or any(part < 0 for part in item)
        ):
            raise ValueError(f"telemetry index {field_name} is invalid")
        result.append((int(item[0]), int(item[1])))
    return result


def _load_index(path: Path) -> _TelemetryIndex | None:
    index_path = _index_path(path)
    if not index_path.is_file() or index_path.is_symlink():
        return None
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != _INDEX_SCHEMA:
            return None
        return _TelemetryIndex(
            device=_non_negative_int(raw.get("device"), field_name="device"),
            inode=_non_negative_int(raw.get("inode"), field_name="inode"),
            indexed_size=_non_negative_int(
                raw.get("indexed_size"), field_name="indexed_size"
            ),
            last_scan_start=_non_negative_int(
                raw.get("last_scan_start"), field_name="last_scan_start"
            ),
            record_count=_non_negative_int(
                raw.get("record_count"), field_name="record_count"
            ),
            last_sequence=_non_negative_int(
                raw.get("last_sequence"), field_name="last_sequence"
            ),
            malformed_lines=_non_negative_int(
                raw.get("malformed_lines"), field_name="malformed_lines"
            ),
            sequence_gaps=_pairs(raw.get("sequence_gaps"), field_name="sequence_gaps"),
            checkpoints=_pairs(raw.get("checkpoints"), field_name="checkpoints"),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_index(path: Path, index: _TelemetryIndex) -> None:
    destination = _index_path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = json.dumps(
        index.to_json_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(destination)


def _parse_record(raw_line: bytes) -> StrictTrainingTelemetryRecord:
    return StrictTrainingTelemetryRecord.from_json_dict(
        json.loads(raw_line.decode("utf-8"))
    )


def _refresh_index(path: Path) -> _TelemetryIndex | None:
    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        return None
    stat = resolved.stat()
    existing = _load_index(resolved)
    if (
        existing is None
        or existing.device != int(stat.st_dev)
        or existing.inode != int(stat.st_ino)
        or stat.st_size < existing.indexed_size
    ):
        index = _TelemetryIndex(device=int(stat.st_dev), inode=int(stat.st_ino))
    else:
        index = existing
    index.last_scan_start = index.indexed_size

    with resolved.open("rb") as handle:
        handle.seek(index.indexed_size)
        while True:
            raw_line = handle.readline()
            if not raw_line or not raw_line.endswith(b"\n"):
                break
            end_offset = handle.tell()
            index.indexed_size = end_offset
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = _parse_record(line)
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
                index.malformed_lines += 1
                continue
            if record.sequence <= index.last_sequence:
                index.malformed_lines += 1
                continue
            if index.last_sequence and record.sequence > index.last_sequence + 1:
                index.sequence_gaps.append(
                    (index.last_sequence + 1, record.sequence - 1)
                )
            index.record_count += 1
            index.last_sequence = record.sequence
            if index.record_count == 1 or index.record_count % _INDEX_STRIDE == 0:
                index.checkpoints.append((record.sequence, end_offset))
    _write_index(resolved, index)
    return index


def _seek_offset(index: _TelemetryIndex, after_sequence: int) -> Pair:
    selected: Pair = (0, 0)
    for sequence, offset in index.checkpoints:
        if sequence > after_sequence:
            break
        selected = (sequence, offset)
    return selected


def read_indexed_training_telemetry(
    path: Path,
    *,
    after_sequence: int = 0,
    limit: int = 512,
) -> _training.TrainingTelemetryPage:
    if isinstance(after_sequence, bool) or after_sequence < 0:
        raise ValueError("after_sequence must be non-negative")
    if isinstance(limit, bool) or limit <= 0 or limit > 2_000:
        raise ValueError("limit must be between 1 and 2000")
    resolved = Path(path)
    index = _refresh_index(resolved)
    if index is None:
        return _training.TrainingTelemetryPage((), after_sequence, False, 0, ())

    checkpoint_sequence, offset = _seek_offset(index, after_sequence)
    previous_sequence = checkpoint_sequence or None
    items: list[StrictTrainingTelemetryRecord] = []
    truncated = False
    with resolved.open("rb") as handle:
        handle.seek(offset)
        while True:
            raw_line = handle.readline()
            if not raw_line or not raw_line.endswith(b"\n"):
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = _parse_record(line)
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if previous_sequence is not None and record.sequence <= previous_sequence:
                continue
            previous_sequence = record.sequence
            if record.sequence <= after_sequence:
                continue
            if len(items) >= limit:
                truncated = True
                break
            items.append(record)
    next_sequence = items[-1].sequence if items else after_sequence
    return _training.TrainingTelemetryPage(
        items=tuple(items),
        next_sequence=next_sequence,
        truncated=truncated,
        malformed_lines=index.malformed_lines,
        sequence_gaps=tuple(index.sequence_gaps),
    )


def indexed_training_telemetry_status(
    path: Path,
) -> _training.TrainingTelemetryStatus:
    resolved = Path(path)
    index = _refresh_index(resolved)
    if index is None:
        return _training.TrainingTelemetryStatus(False, 0, 0, 0, 0)
    return _training.TrainingTelemetryStatus(
        available=True,
        record_count=index.record_count,
        last_sequence=index.last_sequence,
        malformed_lines=index.malformed_lines,
        size_bytes=resolved.stat().st_size,
    )


class IndexedTrainingTelemetryWriter(_BaseWriter):
    """Existing append writer paired with the indexed status implementation."""


__all__ = [
    "IndexedTrainingTelemetryWriter",
    "StrictTrainingTelemetryRecord",
    "indexed_training_telemetry_status",
    "read_indexed_training_telemetry",
]
