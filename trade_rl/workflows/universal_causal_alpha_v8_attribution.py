"""Reconciled simulator attribution for Causal Alpha V8."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v7 import (
    CausalAlphaV7Candidate,
    CausalAlphaV7TargetPath,
)
from trade_rl.learning.causal_alpha_v8 import (
    CausalAlphaV8Candidate,
    CausalAlphaV8TargetPath,
)
from trade_rl.learning.rollout_evaluation import ActionPathEvaluation
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionBoundaries,
    CausalAlphaV7AttributionCell,
    build_causal_alpha_v7_attribution,
)

_EVIDENCE_SCHEMA: Final = "causal_alpha_v8_attribution_v1"
_RECONCILIATION_TOLERANCE: Final = 1e-12
_V7_CANDIDATE_BY_V8: Final = {
    CausalAlphaV8Candidate.V7_CONTROL: CausalAlphaV7Candidate.V6_CONTROL,
    CausalAlphaV8Candidate.ROBUST_CONTRARIAN: (
        CausalAlphaV7Candidate.SYMMETRIC_CONTRARIAN
    ),
    CausalAlphaV8Candidate.ROBUST_CALIBRATED: (
        CausalAlphaV7Candidate.CAUSAL_CALIBRATED
    ),
}


@dataclass(frozen=True, slots=True)
class CausalAlphaV8AttributionEvidence:
    """V8-owned attribution evidence over the shared fixed dimensions."""

    candidate: CausalAlphaV8Candidate
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
        candidate = CausalAlphaV8Candidate(self.candidate)
        for name in (
            "target_path_digest",
            "boundaries_digest",
            "step_economics_digest",
        ):
            require_sha256(getattr(self, name), field=f"V8 attribution {name}")
        if (
            isinstance(self.decision_count, bool)
            or not isinstance(self.decision_count, int)
            or self.decision_count <= 0
        ):
            raise ValueError("V8 attribution decision count is invalid")
        totals = (
            self.gross_log_return,
            self.net_log_return,
            self.total_execution_cost,
            self.total_exposure_hours,
        )
        if not all(math.isfinite(value) for value in totals):
            raise ValueError("V8 attribution totals must be finite")
        if self.total_execution_cost < 0.0 or self.total_exposure_hours < 0.0:
            raise ValueError("V8 attribution total cost/exposure must be non-negative")
        cells = tuple(self.cells)
        if not cells or cells != tuple(
            sorted(cells, key=lambda cell: (cell.dimension, cell.key))
        ):
            raise ValueError("V8 attribution cells are not canonical")
        for dimension in sorted({cell.dimension for cell in cells}):
            selected = tuple(cell for cell in cells if cell.dimension == dimension)
            if sum(cell.support for cell in selected) != self.decision_count:
                raise ValueError("V8 attribution support does not reconcile")
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
                    raise ValueError("V8 attribution economics do not reconcile")
        if self.schema_version != _EVIDENCE_SCHEMA:
            raise ValueError("unsupported V8 attribution schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "cells", cells)
        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected_digest:
            raise ValueError("V8 attribution digest mismatch")
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


def build_causal_alpha_v8_attribution(
    *,
    target_path: CausalAlphaV8TargetPath,
    evaluation: ActionPathEvaluation,
    confidence: object,
    realized_volatility: object,
    liquidity: object,
    boundaries: CausalAlphaV7AttributionBoundaries,
    step_hours: float,
) -> CausalAlphaV8AttributionEvidence:
    """Reuse fixed V7 bins while binding evidence to the V8 target identity."""

    if not isinstance(target_path, CausalAlphaV8TargetPath):
        raise TypeError("V8 attribution target path is invalid")
    compatibility_path = CausalAlphaV7TargetPath(
        candidate=_V7_CANDIDATE_BY_V8[target_path.candidate],
        v6_target_path=target_path.v6_target_path,
        source_forecast_digest=target_path.source_forecast_digest,
        calibration_fit_digest=target_path.calibration_fit_digest,
    )
    source = build_causal_alpha_v7_attribution(
        target_path=compatibility_path,
        evaluation=evaluation,
        confidence=confidence,
        realized_volatility=realized_volatility,
        liquidity=liquidity,
        boundaries=boundaries,
        step_hours=step_hours,
    )
    return CausalAlphaV8AttributionEvidence(
        candidate=target_path.candidate,
        target_path_digest=target_path.digest,
        boundaries_digest=source.boundaries_digest,
        step_economics_digest=source.step_economics_digest,
        decision_count=source.decision_count,
        gross_log_return=source.gross_log_return,
        net_log_return=source.net_log_return,
        total_execution_cost=source.total_execution_cost,
        total_exposure_hours=source.total_exposure_hours,
        cells=source.cells,
    )


__all__ = [
    "CausalAlphaV8AttributionEvidence",
    "build_causal_alpha_v8_attribution",
]
