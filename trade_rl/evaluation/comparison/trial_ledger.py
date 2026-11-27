"""Append-only ledger of every trial, with a strict canonical JSONL codec.

Multiple-testing corrections are only as honest as the trial count behind
them. This module records each trial attempt, including failed or abandoned
ones and the reason, so a family's multiplicity ``N`` can be counted from one
validated source. It carries no research method names, thresholds or rosters:
``family``, ``method`` and ``evaluation_data_label`` are free text that the
core validates only for shape and never interprets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Final

from trade_rl.artifacts.canonical import canonical_json_bytes, strict_json_loads
from trade_rl.evaluation.comparison.multiple_testing import (
    SharpeMoments,
    sharpe_moments_from_returns,
)

TRIAL_RECORD_SCHEMA: Final = "trial_record_v1"
MAX_LEDGER_BYTES: Final = 64 * 1024 * 1024

_ID_MAX = 256
_LABEL_MAX = 512
_REASON_MAX = 1024
_RECORD_FIELDS = frozenset(
    {
        "schema",
        "trial_id",
        "family",
        "method",
        "config_digest",
        "status",
        "reason",
        "performance",
        "returns",
        "evaluation_data_label",
    }
)
_PERFORMANCE_FIELDS = frozenset({"sharpe", "n_obs", "skew", "kurtosis"})
_HEX_DIGITS = frozenset("0123456789abcdef")


class TrialLedgerError(ValueError):
    """The ledger or its serialised form violates the trial contract."""


class TrialStatus(StrEnum):
    """Outcome of one trial attempt."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


def _text(value: object, *, name: str, max_length: int) -> str:
    if not isinstance(value, str) or not value:
        raise TrialLedgerError(f"{name} must be a non-empty string")
    if value != value.strip() or not value.isprintable() or len(value) > max_length:
        raise TrialLedgerError(
            f"{name} must be trimmed printable text of at most {max_length} characters"
        )
    return value


def _validate_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not frozenset(value) <= _HEX_DIGITS
    ):
        raise TrialLedgerError("config_digest must be 64 lowercase hex characters")
    return value


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One immutable trial attempt.

    A COMPLETED trial carries exactly one of ``performance`` (per-period
    summary statistics) or ``returns`` (the per-period return series, from
    which the statistics are derived). FAILED and ABANDONED trials carry
    neither and must give a ``reason``. ``evaluation_data_label`` is optional
    free text describing the data the numbers came from; the core does not
    interpret it, and recording it makes no claim about the data's status.
    """

    trial_id: str
    family: str
    method: str
    config_digest: str
    status: TrialStatus
    reason: str | None = None
    performance: SharpeMoments | None = None
    returns: tuple[float, ...] | None = None
    evaluation_data_label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trial_id", _text(self.trial_id, name="trial_id", max_length=_ID_MAX)
        )
        object.__setattr__(
            self, "family", _text(self.family, name="family", max_length=_ID_MAX)
        )
        object.__setattr__(
            self, "method", _text(self.method, name="method", max_length=_ID_MAX)
        )
        object.__setattr__(self, "config_digest", _validate_digest(self.config_digest))
        try:
            status = TrialStatus(self.status)
        except ValueError as exc:
            raise TrialLedgerError("status must be a TrialStatus") from exc
        object.__setattr__(self, "status", status)
        if self.reason is not None:
            object.__setattr__(
                self,
                "reason",
                _text(self.reason, name="reason", max_length=_REASON_MAX),
            )
        if self.evaluation_data_label is not None:
            object.__setattr__(
                self,
                "evaluation_data_label",
                _text(
                    self.evaluation_data_label,
                    name="evaluation_data_label",
                    max_length=_LABEL_MAX,
                ),
            )
        if status is TrialStatus.COMPLETED:
            self._validate_completed()
        else:
            if self.reason is None:
                raise TrialLedgerError(f"{status.value} trials require a reason")
            if self.performance is not None or self.returns is not None:
                raise TrialLedgerError(
                    f"{status.value} trials must not carry performance or returns"
                )

    def _validate_completed(self) -> None:
        if (self.performance is None) == (self.returns is None):
            raise TrialLedgerError(
                "COMPLETED trials need exactly one of performance or returns"
            )
        if self.performance is not None and not isinstance(
            self.performance, SharpeMoments
        ):
            raise TrialLedgerError("performance must be SharpeMoments")
        if self.returns is not None:
            raw_items: object = self.returns
            if isinstance(raw_items, (str, bytes)) or not isinstance(
                raw_items, Sequence
            ):
                raise TrialLedgerError("returns must be a sequence of floats")
            values: list[float] = []
            for item in raw_items:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise TrialLedgerError("returns must contain finite real numbers")
                number = float(item)
                if not math.isfinite(number):
                    raise TrialLedgerError("returns must contain finite real numbers")
                values.append(number)
            try:
                sharpe_moments_from_returns(values)
            except ValueError as exc:
                raise TrialLedgerError(f"returns are unusable: {exc}") from exc
            object.__setattr__(self, "returns", tuple(values))

    def moments(self) -> SharpeMoments:
        """Per-period SR statistics of a COMPLETED trial."""

        if self.status is not TrialStatus.COMPLETED:
            raise TrialLedgerError("only COMPLETED trials have performance")
        if self.performance is not None:
            return self.performance
        assert self.returns is not None
        return sharpe_moments_from_returns(self.returns)


class TrialLedger:
    """Append-only, order-preserving collection of unique trial records.

    Appending an identical record again is an idempotent no-op; appending a
    different record under an existing ``trial_id`` is refused. Records are
    never removed or replaced, so a family's trial count cannot shrink.
    """

    __slots__ = ("_by_id", "_order")

    def __init__(self, records: Iterable[TrialRecord] = ()) -> None:
        self._by_id: dict[str, TrialRecord] = {}
        self._order: list[TrialRecord] = []
        for record in records:
            self.append(record)

    def append(self, record: TrialRecord) -> bool:
        """Add ``record``; return ``False`` when an identical one was present."""

        if not isinstance(record, TrialRecord):
            raise TrialLedgerError("ledger accepts only TrialRecord values")
        existing = self._by_id.get(record.trial_id)
        if existing is not None:
            if existing == record:
                return False
            raise TrialLedgerError(
                f"trial_id {record.trial_id!r} already recorded with different content"
            )
        self._by_id[record.trial_id] = record
        self._order.append(record)
        return True

    @property
    def records(self) -> tuple[TrialRecord, ...]:
        return tuple(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self) -> Iterator[TrialRecord]:
        return iter(tuple(self._order))

    def __contains__(self, trial_id: object) -> bool:
        return trial_id in self._by_id

    def get(self, trial_id: str) -> TrialRecord | None:
        return self._by_id.get(trial_id)

    def family_records(self, family: str) -> tuple[TrialRecord, ...]:
        return tuple(item for item in self._order if item.family == family)

    def family_counts(self) -> dict[str, int]:
        """Trials per family, FAILED and ABANDONED included (a fresh dict)."""

        counts: dict[str, int] = {}
        for item in self._order:
            counts[item.family] = counts.get(item.family, 0) + 1
        return dict(sorted(counts.items()))

    def digest(self) -> str:
        """SHA-256 hex of the canonical JSONL bytes (order-sensitive)."""

        return sha256(encode_trial_ledger_jsonl(self)).hexdigest()


def _record_payload(record: TrialRecord) -> dict[str, object]:
    performance: dict[str, object] | None = None
    if record.performance is not None:
        performance = {
            "sharpe": record.performance.sharpe,
            "n_obs": record.performance.n_obs,
            "skew": record.performance.skew,
            "kurtosis": record.performance.kurtosis,
        }
    return {
        "schema": TRIAL_RECORD_SCHEMA,
        "trial_id": record.trial_id,
        "family": record.family,
        "method": record.method,
        "config_digest": record.config_digest,
        "status": record.status.value,
        "reason": record.reason,
        "performance": performance,
        "returns": None if record.returns is None else list(record.returns),
        "evaluation_data_label": record.evaluation_data_label,
    }


def encode_trial_ledger_jsonl(ledger: TrialLedger) -> bytes:
    """Deterministic canonical JSONL: one sorted-key compact object per line.

    Lines keep append order and each ends with ``\\n``; an empty ledger is
    empty bytes.
    """

    lines = [canonical_json_bytes(_record_payload(item)) for item in ledger.records]
    return b"".join(line + b"\n" for line in lines)


def _float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TrialLedgerError(f"{name} must be a finite JSON float")
    return value


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload[key]
    if value is None:
        return None
    return _text(value, name=key, max_length=_LABEL_MAX)


def _decode_performance(value: object) -> SharpeMoments | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _PERFORMANCE_FIELDS:
        raise TrialLedgerError("performance fields do not match the contract")
    n_obs = value["n_obs"]
    if type(n_obs) is not int:
        raise TrialLedgerError("performance.n_obs must be a JSON integer")
    try:
        return SharpeMoments(
            sharpe=_float(value["sharpe"], name="performance.sharpe"),
            n_obs=n_obs,
            skew=_float(value["skew"], name="performance.skew"),
            kurtosis=_float(value["kurtosis"], name="performance.kurtosis"),
        )
    except ValueError as exc:
        raise TrialLedgerError(f"invalid performance: {exc}") from exc


def _decode_record(payload: object) -> TrialRecord:
    if not isinstance(payload, dict):
        raise TrialLedgerError("each ledger line must be a JSON object")
    keys = set(payload)
    if keys != _RECORD_FIELDS:
        unknown = sorted(keys - _RECORD_FIELDS)
        missing = sorted(_RECORD_FIELDS - keys)
        raise TrialLedgerError(
            f"trial record fields do not match the contract: "
            f"unknown={unknown} missing={missing}"
        )
    if payload["schema"] != TRIAL_RECORD_SCHEMA:
        raise TrialLedgerError("unsupported trial record schema")
    raw_returns = payload["returns"]
    returns: tuple[float, ...] | None = None
    if raw_returns is not None:
        if not isinstance(raw_returns, list):
            raise TrialLedgerError("returns must be a JSON array or null")
        returns = tuple(_float(item, name="returns item") for item in raw_returns)
    raw_status = payload["status"]
    if not isinstance(raw_status, str):
        raise TrialLedgerError("status must be a string")
    try:
        status = TrialStatus(raw_status)
    except ValueError as exc:
        raise TrialLedgerError("unknown trial status") from exc
    raw_reason = payload["reason"]
    return TrialRecord(
        trial_id=_text(payload["trial_id"], name="trial_id", max_length=_ID_MAX),
        family=_text(payload["family"], name="family", max_length=_ID_MAX),
        method=_text(payload["method"], name="method", max_length=_ID_MAX),
        config_digest=_validate_digest(payload["config_digest"]),
        status=status,
        reason=(
            None
            if raw_reason is None
            else _text(raw_reason, name="reason", max_length=_REASON_MAX)
        ),
        performance=_decode_performance(payload["performance"]),
        returns=returns,
        evaluation_data_label=_optional_text(payload, "evaluation_data_label"),
    )


def decode_trial_ledger_jsonl(
    data: bytes, *, expected_digest: str | None = None
) -> TrialLedger:
    """Strictly decode canonical ledger bytes.

    Refuses duplicate JSON keys at any depth, unknown or missing fields,
    non-finite numbers, malformed digests, duplicate ``trial_id`` lines and
    any line that is not byte-identical to the canonical encoding of the
    record it decodes to. ``expected_digest`` (``TrialLedger.digest()`` of the
    intended ledger) additionally binds the content; without it the codec
    cannot detect a consistent rewrite of the whole file.
    """

    if not isinstance(data, bytes):
        raise TrialLedgerError("ledger data must be bytes")
    if len(data) > MAX_LEDGER_BYTES:
        raise TrialLedgerError("ledger data exceeds the size limit")
    if expected_digest is not None:
        _validate_digest(expected_digest)
    if data and not data.endswith(b"\n"):
        raise TrialLedgerError("ledger must end with a newline")
    ledger = TrialLedger()
    if data:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrialLedgerError("ledger is not valid UTF-8") from exc
        for number, line in enumerate(text[:-1].split("\n"), start=1):
            if not line:
                raise TrialLedgerError(f"line {number}: blank line")
            try:
                payload = strict_json_loads(line)
            except (ValueError, RecursionError, json.JSONDecodeError) as exc:
                raise TrialLedgerError(f"line {number}: invalid JSON: {exc}") from exc
            try:
                record = _decode_record(payload)
            except TrialLedgerError as exc:
                raise TrialLedgerError(f"line {number}: {exc}") from exc
            if canonical_json_bytes(_record_payload(record)).decode("utf-8") != line:
                raise TrialLedgerError(f"line {number}: not canonical JSON")
            if not ledger.append(record):
                raise TrialLedgerError(f"line {number}: duplicate trial_id")
    if expected_digest is not None and ledger.digest() != expected_digest:
        raise TrialLedgerError("ledger digest does not match the expected digest")
    return ledger


__all__ = [
    "MAX_LEDGER_BYTES",
    "TRIAL_RECORD_SCHEMA",
    "TrialLedger",
    "TrialLedgerError",
    "TrialRecord",
    "TrialStatus",
    "decode_trial_ledger_jsonl",
    "encode_trial_ledger_jsonl",
]
