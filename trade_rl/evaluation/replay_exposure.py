"""Opt-in native replay mark evidence and offline realized gross verification.

Only observer facts; no replay, market-data read, new accounting book, or
economic authorization. The caller must independently establish the ledger
digest and expected identity before comparing any reported exposure statistic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.simulation.accounting import (
    EconomicTerminationReason,
    _require_representable_position_values,
)
from trade_rl.simulation.quantities import parse_quantity, project_quantity

EXPOSURE_LEDGER_SCHEMA = "shared_cash_portfolio_replay_ledger_v2"


@dataclass(frozen=True, slots=True)
class PreIntervalMarkEvidence:
    """Actual book marks and multipliers at the pre-execution frontier."""

    start_index: int
    timestamp_ns: int
    mark_prices: tuple[float, ...]
    contract_multipliers: tuple[float, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "start_index": self.start_index,
            "timestamp_ns": self.timestamp_ns,
            "mark_prices": self.mark_prices,
            "contract_multipliers": self.contract_multipliers,
        }


@dataclass(frozen=True, slots=True)
class ReplayExposureEvidence:
    """Ordered roster and native pre-interval mark facts."""

    symbols: tuple[str, ...]
    rows: tuple[PreIntervalMarkEvidence, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "symbols": self.symbols,
            "rows": tuple(row.to_mapping() for row in self.rows),
        }


def _number(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite native number")
    if type(value) is int and abs(value) > 2**53:
        raise ValueError(f"{name} must be binary64-compatible")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a native integer")
    return value


def _vector(value: object, *, size: int, name: str) -> np.ndarray:
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must match the ordered symbol roster")
    result = np.asarray([_number(v, name=name) for v in value], dtype=np.float64)
    if np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate ledger field")
        result[key] = value
    return result


def mean_pre_interval_gross_from_ledger(
    ledger: bytes,
    *,
    expected_ledger_digest: str,
    expected_dataset_id: str,
    expected_execution_policy_digest: str,
    expected_symbols: tuple[str, ...],
    expected_start_index: int,
    expected_stop_index: int,
) -> float:
    """Read canonical v2 bytes and recompute the observed positive-equity mean.

    The external digest authenticates neither the market nor the author; it must
    be established by the caller's independent receipt/identity verification.
    Validate exposure-specific identity, row alignment, marks, multipliers,
    exact quantities and pre-equity closure. Use the native binary64 position
    valuation to reconcile the book, not a second book. The reporting metric uses
    the frozen consumer's nearest float conversion of exact quantities and
    math.fsum, distinct from conservatively projected book weights. Include
    completed terminal-reserve intervals, exclude nonpositive pre-equity, and
    refuse a missing positive sample. This is not a full economic ledger audit.
    """
    for name, value in (
        ("ledger digest", expected_ledger_digest),
        ("dataset id", expected_dataset_id),
        ("execution policy digest", expected_execution_policy_digest),
    ):
        if type(value) is not str:
            raise ValueError(f"{name} must be a digest")
        require_sha256(value, field=name)
    if (
        type(ledger) is not bytes
        or hashlib.sha256(ledger).hexdigest() != expected_ledger_digest
    ):
        raise ValueError("ledger digest mismatch")
    if (
        type(expected_symbols) is not tuple
        or not expected_symbols
        or any(type(s) is not str or not s or s != s.strip() for s in expected_symbols)
        or len(set(expected_symbols)) != len(expected_symbols)
    ):
        raise ValueError("expected symbol roster must be unique and nonempty")
    start = _integer(expected_start_index, name="start index")
    stop = _integer(expected_stop_index, name="stop index")
    if not 0 <= start < stop:
        raise ValueError("invalid expected interval range")
    try:
        value = json.loads(ledger, object_pairs_hook=_unique_object)
        if canonical_json_bytes(value) != ledger:
            raise ValueError("ledger must use canonical JSON bytes")
    except (
        TypeError,
        OverflowError,
        RecursionError,
        UnicodeError,
        ValueError,
    ) as error:
        raise ValueError("invalid canonical ledger") from error
    if type(value) is not dict or (
        value.get("schema_version") != EXPOSURE_LEDGER_SCHEMA
        or value.get("dataset_id") != expected_dataset_id
        or value.get("execution_policy_digest") != expected_execution_policy_digest
        or _integer(value.get("start_index"), name="ledger start") != start
        or _integer(value.get("stop_index"), name="ledger stop") != stop
    ):
        raise ValueError("ledger identity or exposure schema mismatch")
    sidecar = value.get("pre_interval_exposure")
    intervals = value.get("intervals")
    if (
        type(sidecar) is not dict
        or set(sidecar) != {"symbols", "rows"}
        or sidecar.get("symbols") != list(expected_symbols)
        or type(sidecar.get("rows")) is not list
        or type(intervals) is not list
        or not 0 < len(intervals) <= stop - start
        or len(sidecar["rows"]) != len(intervals)
    ):
        raise ValueError("exposure row/roster mismatch")
    if "termination_reason" not in value:
        raise ValueError("missing terminal reason")
    terminal = value["termination_reason"]
    if terminal is not None and (
        type(terminal) is not str
        or terminal not in {reason.value for reason in EconomicTerminationReason}
    ):
        raise ValueError("invalid terminal reason")
    if terminal is None and len(intervals) != stop - start:
        raise ValueError("incomplete unterminated ledger")
    previous_time: int | None = None
    multipliers_before: np.ndarray | None = None
    gross: list[float] = []
    size = len(expected_symbols)
    for offset, (row, interval) in enumerate(
        zip(sidecar["rows"], intervals, strict=True)
    ):
        index = start + offset
        if (
            type(row) is not dict
            or set(row)
            != {"start_index", "timestamp_ns", "mark_prices", "contract_multipliers"}
            or type(interval) is not dict
            or _integer(row.get("start_index"), name="row index") != index
            or _integer(interval.get("start_index"), name="interval start") != index
            or _integer(interval.get("next_index"), name="interval next") != index + 1
            or "termination_reason" not in interval
            or interval["termination_reason"]
            != (terminal if offset == len(intervals) - 1 else None)
        ):
            raise ValueError("exposure interval binding mismatch")
        timestamp = _integer(row.get("timestamp_ns"), name="timestamp")
        if not -(2**63) < timestamp < 2**63 or (
            previous_time is not None and timestamp <= previous_time
        ):
            raise ValueError("invalid exposure clock")
        previous_time = timestamp
        marks = _vector(row.get("mark_prices"), size=size, name="marks")
        multipliers = _vector(
            row.get("contract_multipliers"), size=size, name="multipliers"
        )
        if multipliers_before is not None and not np.array_equal(
            multipliers_before, multipliers
        ):
            raise ValueError("contract multipliers changed within ledger")
        multipliers_before = multipliers
        quantities = interval.get("exact_quantities_before")
        if (
            type(quantities) is not list
            or len(quantities) != size
            or any(type(q) is not str for q in quantities)
        ):
            raise ValueError("exact quantities must match symbol roster")
        exact = tuple(parse_quantity(q) for q in quantities)
        projected = np.asarray([project_quantity(q) for q in exact], dtype=np.float64)
        if any(q != 0 and v == 0.0 for q, v in zip(exact, projected, strict=True)):
            raise ValueError("nonzero exact quantity cannot be represented")
        positions = _require_representable_position_values(
            quantities=projected, marks=marks, multipliers=multipliers
        )
        cash = _number(interval.get("cash_before"), name="cash")
        equity = _number(interval.get("portfolio_value_before"), name="equity")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if float(cash + positions.sum()) != equity:
                raise ValueError("pre-interval equity does not reconcile")
            if equity > 0.0:
                try:
                    exposure = math.fsum(
                        abs(float(q)) * float(mark) * float(multiplier) / equity
                        for q, mark, multiplier in zip(
                            exact, marks, multipliers, strict=True
                        )
                    )
                except OverflowError as error:
                    raise ValueError("gross exposure is non-finite") from error
                if not math.isfinite(exposure):
                    raise ValueError("gross exposure is non-finite")
                gross.append(exposure)
    if not gross:
        raise ValueError("no positive-equity completed interval")
    try:
        result = math.fsum(gross) / len(gross)
    except OverflowError as error:
        raise ValueError("mean gross exposure is non-finite") from error
    if not math.isfinite(result):
        raise ValueError("mean gross exposure is non-finite")
    return result


__all__ = [
    "PreIntervalMarkEvidence",
    "ReplayExposureEvidence",
    "mean_pre_interval_gross_from_ledger",
]
