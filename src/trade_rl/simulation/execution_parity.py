"""Canonical exact comparison for legacy versus candidate execution traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class CanonicalExecutionRecord:
    """Integer-valued execution identity shared across runtime implementations."""

    sequence: int
    event_type: str
    timestamp_ns: int
    price_ticks: int
    quantity_lots: int
    fee_minor: int
    funding_minor: int
    position_lots: int
    equity_minor: int
    terminal_reason: str | None


@dataclass(frozen=True, slots=True)
class ExecutionParityMismatch:
    sequence: int | None
    field: str
    legacy_value: Any
    candidate_value: Any


@dataclass(frozen=True, slots=True)
class ExecutionParityReport:
    matches: bool
    legacy_digest: str
    candidate_digest: str
    mismatches: tuple[ExecutionParityMismatch, ...]


def execution_trace_digest(records: Iterable[CanonicalExecutionRecord]) -> str:
    """Return deterministic SHA-256 over canonical integer execution records."""

    payload = [asdict(record) for record in records]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_execution_traces(
    *,
    legacy: Iterable[CanonicalExecutionRecord],
    candidate: Iterable[CanonicalExecutionRecord],
) -> ExecutionParityReport:
    """Compare exact execution identity, reporting every differing canonical field."""

    legacy_records = tuple(legacy)
    candidate_records = tuple(candidate)
    mismatches: list[ExecutionParityMismatch] = []
    comparable = min(len(legacy_records), len(candidate_records))
    comparable_fields = tuple(field.name for field in fields(CanonicalExecutionRecord))

    for index in range(comparable):
        legacy_record = legacy_records[index]
        candidate_record = candidate_records[index]
        for field_name in comparable_fields:
            legacy_value = getattr(legacy_record, field_name)
            candidate_value = getattr(candidate_record, field_name)
            if legacy_value == candidate_value:
                continue
            sequence = (
                legacy_record.sequence
                if legacy_record.sequence == candidate_record.sequence
                else None
            )
            mismatches.append(
                ExecutionParityMismatch(
                    sequence=sequence,
                    field=field_name,
                    legacy_value=legacy_value,
                    candidate_value=candidate_value,
                )
            )

    if len(legacy_records) != len(candidate_records):
        mismatches.append(
            ExecutionParityMismatch(
                sequence=None,
                field="trace_length",
                legacy_value=len(legacy_records),
                candidate_value=len(candidate_records),
            )
        )

    legacy_digest = execution_trace_digest(legacy_records)
    candidate_digest = execution_trace_digest(candidate_records)
    return ExecutionParityReport(
        matches=not mismatches and legacy_digest == candidate_digest,
        legacy_digest=legacy_digest,
        candidate_digest=candidate_digest,
        mismatches=tuple(mismatches),
    )


__all__ = [
    "CanonicalExecutionRecord",
    "ExecutionParityMismatch",
    "ExecutionParityReport",
    "compare_execution_traces",
    "execution_trace_digest",
]
