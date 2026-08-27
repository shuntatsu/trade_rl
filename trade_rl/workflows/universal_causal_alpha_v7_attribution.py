"""Reconciled simulator attribution for Causal Alpha V7."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v7 import (
    CausalAlphaV7Candidate,
    CausalAlphaV7TargetPath,
)
from trade_rl.learning.rollout_evaluation import ActionPathEvaluation

_BOUNDARY_SCHEMA: Final = "causal_alpha_v7_attribution_boundaries_v1"
_EVIDENCE_SCHEMA: Final = "causal_alpha_v7_attribution_v1"
_QUARTILE_KEYS: Final = ("q1", "q2", "q3", "q4")
_RECONCILIATION_TOLERANCE: Final = 1e-12


def _boundaries(value: object, *, field: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        resolved = tuple(float(item) for item in array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"V7 {field} boundaries are invalid") from error
    if (
        len(resolved) != 3
        or not all(math.isfinite(item) for item in resolved)
        or any(left >= right for left, right in zip(resolved, resolved[1:]))
    ):
        raise ValueError(f"V7 {field} boundaries must be finite and strictly increasing")
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV7AttributionBoundaries:
    """Causal calibration-tail quartile boundaries frozen before Selection."""

    confidence: tuple[float, float, float]
    realized_volatility: tuple[float, float, float]
    liquidity: tuple[float, float, float]
    calibration_range_digest: str
    schema_version: str = _BOUNDARY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        confidence = _boundaries(self.confidence, field="confidence")
        volatility = _boundaries(
            self.realized_volatility,
            field="realized volatility",
        )
        liquidity = _boundaries(self.liquidity, field="liquidity")
        require_sha256(
            self.calibration_range_digest,
            field="V7 attribution calibration range digest",
        )
        if self.schema_version != _BOUNDARY_SCHEMA:
            raise ValueError("unsupported V7 attribution boundary schema")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "realized_volatility", volatility)
        object.__setattr__(self, "liquidity", liquidity)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 attribution boundary digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "calibration_range_digest": self.calibration_range_digest,
            "confidence": self.confidence,
            "liquidity": self.liquidity,
            "realized_volatility": self.realized_volatility,
            "schema_version": self.schema_version,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV7AttributionCell:
    dimension: str
    key: str
    support: int
    gross_log_return: float
    net_log_return: float
    execution_cost: float
    exposure_hours: float

    def __post_init__(self) -> None:
        if not self.dimension or not self.key:
            raise ValueError("V7 attribution cell identity is invalid")
        if isinstance(self.support, bool) or not isinstance(self.support, int) or self.support <= 0:
            raise ValueError("V7 attribution cell support must be positive")
        values = (
            self.gross_log_return,
            self.net_log_return,
            self.execution_cost,
            self.exposure_hours,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("V7 attribution cell values must be finite")
        if self.execution_cost < 0.0 or self.exposure_hours < 0.0:
            raise ValueError("V7 attribution cost/exposure must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "execution_cost": self.execution_cost,
            "exposure_hours": self.exposure_hours,
            "gross_log_return": self.gross_log_return,
            "key": self.key,
            "net_log_return": self.net_log_return,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV7AttributionEvidence:
    candidate: CausalAlphaV7Candidate
    target_path_digest: str
    boundaries_digest: str
    step_economics_digest: str
    decision_count: int
    gross_log_return: float
    net_log_return: float
    total_execution_cost: float
    total_exposure_hours: float
    cells: tuple[CausalAlphaV7AttributionCell, ...]
    schema_version: str = _EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV7Candidate(self.candidate)
        for name in (
            "target_path_digest",
            "boundaries_digest",
            "step_economics_digest",
        ):
            require_sha256(getattr(self, name), field=f"V7 attribution {name}")
        if (
            isinstance(self.decision_count, bool)
            or not isinstance(self.decision_count, int)
            or self.decision_count <= 0
        ):
            raise ValueError("V7 attribution decision count is invalid")
        totals = (
            self.gross_log_return,
            self.net_log_return,
            self.total_execution_cost,
            self.total_exposure_hours,
        )
        if not all(math.isfinite(value) for value in totals):
            raise ValueError("V7 attribution totals must be finite")
        if self.total_execution_cost < 0.0 or self.total_exposure_hours < 0.0:
            raise ValueError("V7 attribution total cost/exposure must be non-negative")
        cells = tuple(self.cells)
        if not cells or cells != tuple(sorted(cells, key=lambda cell: (cell.dimension, cell.key))):
            raise ValueError("V7 attribution cells are not canonical")
        dimensions = tuple(sorted({cell.dimension for cell in cells}))
        for dimension in dimensions:
            selected = tuple(cell for cell in cells if cell.dimension == dimension)
            if sum(cell.support for cell in selected) != self.decision_count:
                raise ValueError("V7 attribution support does not reconcile")
            for observed, expected in (
                (sum(cell.gross_log_return for cell in selected), self.gross_log_return),
                (sum(cell.net_log_return for cell in selected), self.net_log_return),
                (sum(cell.execution_cost for cell in selected), self.total_execution_cost),
                (sum(cell.exposure_hours for cell in selected), self.total_exposure_hours),
            ):
                if not math.isclose(
                    observed,
                    expected,
                    rel_tol=_RECONCILIATION_TOLERANCE,
                    abs_tol=_RECONCILIATION_TOLERANCE,
                ):
                    raise ValueError("V7 attribution economics do not reconcile")
        if self.schema_version != _EVIDENCE_SCHEMA:
            raise ValueError("unsupported V7 attribution schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "cells", cells)
        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected_digest:
            raise ValueError("V7 attribution digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def dimensions(self) -> tuple[str, ...]:
        return tuple(sorted({cell.dimension for cell in self.cells}))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "boundaries_digest": self.boundaries_digest,
            "candidate": self.candidate.value,
            "cells": tuple(cell.to_payload() for cell in self.cells),
            "decision_count": self.decision_count,
            "dimensions": self.dimensions,
            "gross_log_return": self.gross_log_return,
            "net_log_return": self.net_log_return,
            "schema_version": self.schema_version,
            "step_economics_digest": self.step_economics_digest,
            "target_path_digest": self.target_path_digest,
            "total_execution_cost": self.total_execution_cost,
            "total_exposure_hours": self.total_exposure_hours,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _vector(
    value: object,
    *,
    rows: int,
    field: str,
    non_negative: bool = True,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    invalid_sign = non_negative and np.any(array < 0.0)
    if array.shape != (rows,) or not np.isfinite(array).all() or invalid_sign:
        constraint = " finite non-negative" if non_negative else " finite"
        raise ValueError(f"V7 attribution {field} must be aligned{constraint}")
    return array


def _quartile_keys(values: np.ndarray, boundaries: tuple[float, float, float]) -> np.ndarray:
    indices = np.searchsorted(np.asarray(boundaries), values, side="right")
    return np.asarray([_QUARTILE_KEYS[index] for index in indices], dtype=object)


def _cells(
    *,
    dimension: str,
    keys: np.ndarray,
    gross: np.ndarray,
    net: np.ndarray,
    costs: np.ndarray,
    exposure: np.ndarray,
) -> tuple[CausalAlphaV7AttributionCell, ...]:
    result: list[CausalAlphaV7AttributionCell] = []
    for key in sorted({str(value) for value in keys}):
        mask = keys == key
        result.append(
            CausalAlphaV7AttributionCell(
                dimension=dimension,
                key=key,
                support=int(np.count_nonzero(mask)),
                gross_log_return=float(np.sum(gross[mask], dtype=np.float64)),
                net_log_return=float(np.sum(net[mask], dtype=np.float64)),
                execution_cost=float(np.sum(costs[mask], dtype=np.float64)),
                exposure_hours=float(np.sum(exposure[mask], dtype=np.float64)),
            )
        )
    return tuple(result)


def build_causal_alpha_v7_attribution(
    *,
    target_path: CausalAlphaV7TargetPath,
    evaluation: ActionPathEvaluation,
    confidence: object,
    realized_volatility: object,
    liquidity: object,
    boundaries: CausalAlphaV7AttributionBoundaries,
    step_hours: float,
) -> CausalAlphaV7AttributionEvidence:
    """Aggregate step economics into fixed, independently reconciled dimensions."""

    if not isinstance(target_path, CausalAlphaV7TargetPath):
        raise TypeError("V7 attribution target path is invalid")
    if not isinstance(evaluation, ActionPathEvaluation):
        raise TypeError("V7 attribution evaluation is invalid")
    if not isinstance(boundaries, CausalAlphaV7AttributionBoundaries):
        raise TypeError("V7 attribution boundaries are invalid")
    if not math.isfinite(step_hours) or step_hours <= 0.0:
        raise ValueError("V7 attribution step_hours must be positive")
    economics = evaluation.step_economics
    if economics is None:
        raise ValueError("V7 attribution requires simulator step economics")
    v6 = target_path.v6_target_path
    rows = int(v6.targets.size)
    if rows != evaluation.performance.step_count or evaluation.actions.shape != (rows, 1):
        raise ValueError("V7 attribution path and evaluation are not aligned")
    if not np.allclose(
        evaluation.actions[:, 0],
        v6.targets,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("V7 attribution evaluated actions differ from target path")
    confidence_values = _vector(confidence, rows=rows, field="confidence")
    volatility_values = _vector(
        realized_volatility,
        rows=rows,
        field="realized volatility",
    )
    liquidity_values = _vector(
        liquidity,
        rows=rows,
        field="liquidity",
        non_negative=False,
    )
    gross = np.log1p(economics.gross_returns)
    net = np.log1p(economics.net_returns)
    costs = economics.costs
    exposure = np.abs(v6.targets) * step_hours
    exposure_keys = np.where(v6.targets > 1e-12, "long", np.where(v6.targets < -1e-12, "short", "flat"))
    dimensions = {
        "confidence_quartile": _quartile_keys(confidence_values, boundaries.confidence),
        "exposure": exposure_keys,
        "liquidity_quartile": _quartile_keys(liquidity_values, boundaries.liquidity),
        "slow_state": np.asarray([state.value for state in v6.slow_states], dtype=object),
        "transition": np.asarray(v6.reasons, dtype=object),
        "volatility_quartile": _quartile_keys(
            volatility_values,
            boundaries.realized_volatility,
        ),
    }
    cells = tuple(
        cell
        for dimension in sorted(dimensions)
        for cell in _cells(
            dimension=dimension,
            keys=dimensions[dimension],
            gross=gross,
            net=net,
            costs=costs,
            exposure=exposure,
        )
    )
    step_digest = content_and_arrays_digest(
        {"schema_version": "causal_alpha_v7_step_economics_v1"},
        (
            ("gross_returns", economics.gross_returns),
            ("net_returns", economics.net_returns),
            ("costs", economics.costs),
            ("turnover", economics.turnover),
        ),
    )
    gross_total = float(np.sum(gross, dtype=np.float64))
    net_total = float(np.sum(net, dtype=np.float64))
    if not math.isclose(
        gross_total,
        math.log1p(evaluation.performance.gross_return),
        rel_tol=_RECONCILIATION_TOLERANCE,
        abs_tol=_RECONCILIATION_TOLERANCE,
    ) or not math.isclose(
        net_total,
        math.log1p(evaluation.performance.net_return),
        rel_tol=_RECONCILIATION_TOLERANCE,
        abs_tol=_RECONCILIATION_TOLERANCE,
    ):
        raise ValueError("V7 attribution step returns do not match performance")
    return CausalAlphaV7AttributionEvidence(
        candidate=target_path.candidate,
        target_path_digest=target_path.digest,
        boundaries_digest=boundaries.digest,
        step_economics_digest=step_digest,
        decision_count=rows,
        gross_log_return=gross_total,
        net_log_return=net_total,
        total_execution_cost=float(np.sum(costs, dtype=np.float64)),
        total_exposure_hours=float(np.sum(exposure, dtype=np.float64)),
        cells=cells,
    )


__all__ = [
    "CausalAlphaV7AttributionBoundaries",
    "CausalAlphaV7AttributionCell",
    "CausalAlphaV7AttributionEvidence",
    "build_causal_alpha_v7_attribution",
]
