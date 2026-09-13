"""Sealed pre-P&L protocol for canonical bookDepth capacity calibration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "book_depth_capacity_calibration_protocol_v1"
_CANONICAL_DATASET_ID = (
    "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
)
_CANONICAL_STUDY_DIGEST = (
    "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
)
_CANONICAL_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
)
_CALIBRATION_START = datetime(2021, 1, 1, tzinfo=UTC)
_CALIBRATION_STOP_EXCLUSIVE = datetime(2023, 1, 1, tzinfo=UTC)
_EVALUATION_START = datetime(2023, 1, 1, tzinfo=UTC)
_SAMPLE_MONTH_DAYS = (1, 15)
_CAPACITY_BANDS = (-1, 1)
_QUANTILE = 0.05
_BOOK_UTILIZATION_FRACTION = 0.10
_PARTICIPATION_CEILING = 0.05
_MIN_VALID_DAYS_PER_SYMBOL = 30
_VOLUME_ALIGNMENT = "previous_completed_1h"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "canonical_dataset_id",
        "canonical_study_digest",
        "symbols",
        "calibration_start",
        "calibration_stop_exclusive",
        "evaluation_start",
        "sample_month_days",
        "capacity_bands",
        "quantile",
        "book_utilization_fraction",
        "participation_ceiling",
        "min_valid_days_per_symbol",
        "volume_alignment",
    }
)


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_datetime(value: object, *, field: str) -> datetime:
    text = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO datetime") from error
    return _aware_utc(parsed, field=field)


def _float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be a finite number")
    return resolved


def _int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    values = tuple(_text(item, field=field) for item in value)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique non-empty values")
    return values


def _int_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    values = tuple(_int(item, field=field) for item in value)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique integer values")
    return values


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class BookDepthCapacityCalibrationProtocol:
    """Immutable canonical M2 capacity-calibration preregistration."""

    canonical_dataset_id: str
    canonical_study_digest: str
    symbols: tuple[str, ...]
    calibration_start: datetime
    calibration_stop_exclusive: datetime
    evaluation_start: datetime
    sample_month_days: tuple[int, ...]
    capacity_bands: tuple[int, ...]
    quantile: float
    book_utilization_fraction: float
    participation_ceiling: float
    min_valid_days_per_symbol: int
    volume_alignment: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        dataset_id = _sha256(self.canonical_dataset_id, field="canonical_dataset_id")
        study_digest = _sha256(
            self.canonical_study_digest,
            field="canonical_study_digest",
        )
        symbols = tuple(self.symbols)
        calibration_start = _aware_utc(
            self.calibration_start,
            field="calibration_start",
        )
        calibration_stop = _aware_utc(
            self.calibration_stop_exclusive,
            field="calibration_stop_exclusive",
        )
        evaluation_start = _aware_utc(
            self.evaluation_start,
            field="evaluation_start",
        )
        sample_month_days = tuple(self.sample_month_days)
        capacity_bands = tuple(self.capacity_bands)
        quantile = _float(self.quantile, field="quantile")
        utilization = _float(
            self.book_utilization_fraction,
            field="book_utilization_fraction",
        )
        ceiling = _float(self.participation_ceiling, field="participation_ceiling")
        min_valid_days = _int(
            self.min_valid_days_per_symbol,
            field="min_valid_days_per_symbol",
        )
        volume_alignment = _text(self.volume_alignment, field="volume_alignment")
        schema_version = _text(self.schema_version, field="schema_version")

        if not calibration_start < calibration_stop:
            raise ValueError("calibration_start must be before calibration_stop_exclusive")
        if calibration_stop > evaluation_start:
            raise ValueError("calibration_stop_exclusive cannot exceed evaluation_start")

        actual = (
            dataset_id,
            study_digest,
            symbols,
            calibration_start,
            calibration_stop,
            evaluation_start,
            sample_month_days,
            capacity_bands,
            quantile,
            utilization,
            ceiling,
            min_valid_days,
            volume_alignment,
            schema_version,
        )
        preregistered = (
            _CANONICAL_DATASET_ID,
            _CANONICAL_STUDY_DIGEST,
            _CANONICAL_SYMBOLS,
            _CALIBRATION_START,
            _CALIBRATION_STOP_EXCLUSIVE,
            _EVALUATION_START,
            _SAMPLE_MONTH_DAYS,
            _CAPACITY_BANDS,
            _QUANTILE,
            _BOOK_UTILIZATION_FRACTION,
            _PARTICIPATION_CEILING,
            _MIN_VALID_DAYS_PER_SYMBOL,
            _VOLUME_ALIGNMENT,
            _SCHEMA_VERSION,
        )
        if actual != preregistered:
            raise ValueError("capacity calibration fields differ from preregistered protocol")

        object.__setattr__(self, "canonical_dataset_id", dataset_id)
        object.__setattr__(self, "canonical_study_digest", study_digest)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "calibration_start", calibration_start)
        object.__setattr__(self, "calibration_stop_exclusive", calibration_stop)
        object.__setattr__(self, "evaluation_start", evaluation_start)
        object.__setattr__(self, "sample_month_days", sample_month_days)
        object.__setattr__(self, "capacity_bands", capacity_bands)
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "book_utilization_fraction", utilization)
        object.__setattr__(self, "participation_ceiling", ceiling)
        object.__setattr__(self, "min_valid_days_per_symbol", min_valid_days)
        object.__setattr__(self, "volume_alignment", volume_alignment)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def planned_days(self) -> tuple[datetime, ...]:
        days: list[datetime] = []
        year = self.calibration_start.year
        month = self.calibration_start.month
        while (year, month) < (
            self.calibration_stop_exclusive.year,
            self.calibration_stop_exclusive.month,
        ):
            for day in self.sample_month_days:
                candidate = datetime(year, month, day, tzinfo=UTC)
                if self.calibration_start <= candidate < self.calibration_stop_exclusive:
                    days.append(candidate)
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        return tuple(days)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonical_dataset_id": self.canonical_dataset_id,
            "canonical_study_digest": self.canonical_study_digest,
            "symbols": list(self.symbols),
            "calibration_start": _iso_utc(self.calibration_start),
            "calibration_stop_exclusive": _iso_utc(self.calibration_stop_exclusive),
            "evaluation_start": _iso_utc(self.evaluation_start),
            "sample_month_days": list(self.sample_month_days),
            "capacity_bands": list(self.capacity_bands),
            "quantile": self.quantile,
            "book_utilization_fraction": self.book_utilization_fraction,
            "participation_ceiling": self.participation_ceiling,
            "min_valid_days_per_symbol": self.min_valid_days_per_symbol,
            "volume_alignment": self.volume_alignment,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_m2_book_depth_capacity_protocol() -> BookDepthCapacityCalibrationProtocol:
    """Return the frozen canonical M2 bookDepth capacity preregistration."""

    return BookDepthCapacityCalibrationProtocol(
        canonical_dataset_id=_CANONICAL_DATASET_ID,
        canonical_study_digest=_CANONICAL_STUDY_DIGEST,
        symbols=_CANONICAL_SYMBOLS,
        calibration_start=_CALIBRATION_START,
        calibration_stop_exclusive=_CALIBRATION_STOP_EXCLUSIVE,
        evaluation_start=_EVALUATION_START,
        sample_month_days=_SAMPLE_MONTH_DAYS,
        capacity_bands=_CAPACITY_BANDS,
        quantile=_QUANTILE,
        book_utilization_fraction=_BOOK_UTILIZATION_FRACTION,
        participation_ceiling=_PARTICIPATION_CEILING,
        min_valid_days_per_symbol=_MIN_VALID_DAYS_PER_SYMBOL,
        volume_alignment=_VOLUME_ALIGNMENT,
    )


def load_book_depth_capacity_calibration_protocol(
    path: str | Path,
) -> BookDepthCapacityCalibrationProtocol:
    """Load one strict JSON preregistration without accepting unknown fields."""

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number is forbidden: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load capacity calibration protocol: {source}") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("capacity calibration protocol must be a JSON object")
    payload = cast(dict[str, object], raw)
    keys = set(payload)
    missing = sorted(_PAYLOAD_FIELDS - keys)
    unknown = sorted(keys - _PAYLOAD_FIELDS)
    if missing or unknown:
        raise ValueError(
            "capacity calibration protocol keys differ from contract: "
            f"missing={missing}, unknown={unknown}"
        )
    return BookDepthCapacityCalibrationProtocol(
        schema_version=_text(payload["schema_version"], field="schema_version"),
        canonical_dataset_id=_sha256(
            payload["canonical_dataset_id"],
            field="canonical_dataset_id",
        ),
        canonical_study_digest=_sha256(
            payload["canonical_study_digest"],
            field="canonical_study_digest",
        ),
        symbols=_string_tuple(payload["symbols"], field="symbols"),
        calibration_start=_parse_datetime(
            payload["calibration_start"],
            field="calibration_start",
        ),
        calibration_stop_exclusive=_parse_datetime(
            payload["calibration_stop_exclusive"],
            field="calibration_stop_exclusive",
        ),
        evaluation_start=_parse_datetime(
            payload["evaluation_start"],
            field="evaluation_start",
        ),
        sample_month_days=_int_tuple(
            payload["sample_month_days"],
            field="sample_month_days",
        ),
        capacity_bands=_int_tuple(payload["capacity_bands"], field="capacity_bands"),
        quantile=_float(payload["quantile"], field="quantile"),
        book_utilization_fraction=_float(
            payload["book_utilization_fraction"],
            field="book_utilization_fraction",
        ),
        participation_ceiling=_float(
            payload["participation_ceiling"],
            field="participation_ceiling",
        ),
        min_valid_days_per_symbol=_int(
            payload["min_valid_days_per_symbol"],
            field="min_valid_days_per_symbol",
        ),
        volume_alignment=_text(
            payload["volume_alignment"],
            field="volume_alignment",
        ),
    )


__all__ = [
    "BookDepthCapacityCalibrationProtocol",
    "canonical_m2_book_depth_capacity_protocol",
    "load_book_depth_capacity_calibration_protocol",
]
