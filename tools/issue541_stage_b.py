"""Execution-only audit helpers for Issue #541 Stage B.

This module does not implement trading logic.  It observes the canonical
``MarketExecutor.execute_interval`` boundary during the one-shot successor
baseline run and fails closed if the sealed causal-capacity contract is not
being respected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

_TOLERANCE = 1e-12


class StageBCapacityAuditError(RuntimeError):
    """Raised when Stage B execution differs from the sealed capacity contract."""


@dataclass(slots=True)
class StageBCapacityAudit:
    """Accumulate fail-closed capacity evidence across execution intervals."""

    dataset_id: str
    symbols: tuple[str, ...]
    capacity_caps: tuple[float, ...]
    execution_intervals_checked: int = 0
    capacity_violation_count: int = 0
    processing_bar_capacity_violation_count: int = 0
    runtime_overlay_violation_count: int = 0
    _max_participation_by_symbol: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or len(self.dataset_id) != 64:
            raise ValueError("dataset_id must be a SHA-256 string")
        try:
            int(self.dataset_id, 16)
        except ValueError as error:
            raise ValueError("dataset_id must be a SHA-256 string") from error

        symbols = tuple(self.symbols)
        caps = tuple(float(value) for value in self.capacity_caps)
        if not symbols or len(symbols) != len(set(symbols)) or any(not item for item in symbols):
            raise ValueError("symbols must be unique non-empty strings")
        if len(caps) != len(symbols):
            raise ValueError("capacity_caps must align with symbols")
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in caps):
            raise ValueError("capacity_caps must be finite and within (0, 1]")

        self.symbols = symbols
        self.capacity_caps = caps
        self._max_participation_by_symbol = np.zeros(len(symbols), dtype=np.float64)

    def record_interval(
        self,
        *,
        dataset_id: str,
        processing_bar_volume_capacity: bool,
        runtime_max_participation_rate: float,
        participation_by_symbol: np.ndarray,
    ) -> None:
        """Validate and record one canonical execution interval."""

        if dataset_id != self.dataset_id:
            raise StageBCapacityAuditError("Dataset identity differs from sealed authority")
        if processing_bar_volume_capacity:
            self.processing_bar_capacity_violation_count += 1
            raise StageBCapacityAuditError(
                "previous-completed-bar capacity mode is not active"
            )
        if (
            not math.isfinite(runtime_max_participation_rate)
            or runtime_max_participation_rate != 1.0
        ):
            self.runtime_overlay_violation_count += 1
            raise StageBCapacityAuditError(
                "runtime participation overlay differs from zero-overlay authority"
            )

        participation = np.asarray(participation_by_symbol, dtype=np.float64).reshape(-1)
        if participation.shape != (len(self.symbols),):
            raise StageBCapacityAuditError(
                "participation evidence does not match sealed symbol roster"
            )
        if not np.isfinite(participation).all() or np.any(participation < 0.0):
            raise StageBCapacityAuditError(
                "participation evidence must be finite and non-negative"
            )

        caps = np.asarray(self.capacity_caps, dtype=np.float64)
        tolerance = np.maximum(_TOLERANCE, np.abs(caps) * 1e-12)
        if np.any(participation > caps + tolerance):
            self.capacity_violation_count += 1
            raise StageBCapacityAuditError(
                "execution participation exceeds sealed cap"
            )

        self.execution_intervals_checked += 1
        self._max_participation_by_symbol = np.maximum(
            self._max_participation_by_symbol,
            participation,
        )

    @property
    def max_participation_by_symbol(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._max_participation_by_symbol)

    def to_payload(self) -> dict[str, object]:
        if self.execution_intervals_checked <= 0:
            raise StageBCapacityAuditError("capacity audit observed no execution intervals")
        return {
            "schema_version": "issue541_stage_b_capacity_audit_v1",
            "status": "PASS",
            "dataset_id": self.dataset_id,
            "symbols": list(self.symbols),
            "capacity_caps": list(self.capacity_caps),
            "execution_intervals_checked": self.execution_intervals_checked,
            "capacity_violation_count": self.capacity_violation_count,
            "processing_bar_capacity_violation_count": (
                self.processing_bar_capacity_violation_count
            ),
            "runtime_overlay_violation_count": self.runtime_overlay_violation_count,
            "max_participation_by_symbol": list(self.max_participation_by_symbol),
            "previous_completed_bar_capacity": True,
            "runtime_participation_overlay_absent": True,
        }


def stage_b_suite_decision(
    *,
    source_profitable_core_strategies: Iterable[str],
) -> str:
    """Return the frozen suite decision for the sealed empty source roster."""

    roster = tuple(source_profitable_core_strategies)
    if any(not isinstance(item, str) or not item for item in roster):
        raise StageBCapacityAuditError(
            "source-profitable roster must contain non-empty strategy names"
        )
    if roster:
        raise StageBCapacityAuditError(
            "source-profitable roster differs from sealed empty authority"
        )
    return "NO_PROFITABLE_CORE_BASELINE"


__all__ = [
    "StageBCapacityAudit",
    "StageBCapacityAuditError",
    "stage_b_suite_decision",
]
