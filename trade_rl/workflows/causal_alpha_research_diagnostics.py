"""Read-only diagnostics for historical causal-alpha selection checkpoints.

This module deliberately does not participate in resume or promotion.  It can
inspect an older v2 checkpoint whose generator identity differs from the
currently imported implementation, including legacy checkpoints created before
that identity was recorded.  Every row must agree on both grid identity and
generator-identity availability.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateEpisodeMetricsV2,
)
from trade_rl.workflows.universal_causal_alpha_teacher import (
    causal_alpha_candidate_metric_v2_from_payload,
)

_CHECKPOINT_SCHEMA = "causal_alpha_selection_checkpoint_metric_v2"
_REPORT_SCHEMA = "causal_alpha_research_diagnostic_report_v1"
_SNAPSHOT_SCHEMA = "causal_alpha_diagnostic_checkpoint_v2"
_GENERATOR_IDENTITY_PRESENT = "present"
_GENERATOR_IDENTITY_UNAVAILABLE_LEGACY = "unavailable_legacy"


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"causal alpha diagnostic {field} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class CausalAlphaDiagnosticCheckpointV2:
    """One internally consistent historical v2 checkpoint snapshot."""

    grid_digest: str
    generator_code_digest: str | None
    metrics: tuple[CausalAlphaCandidateEpisodeMetricsV2, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        grid = _require_digest(self.grid_digest, field="grid digest")
        generator = (
            None
            if self.generator_code_digest is None
            else _require_digest(
                self.generator_code_digest,
                field="generator code digest",
            )
        )
        metrics = tuple(self.metrics)
        if not metrics:
            raise ValueError("causal alpha diagnostic checkpoint is empty")
        identities = tuple(
            (item.candidate_digest, item.symbol, item.episode_index) for item in metrics
        )
        if len(set(identities)) != len(identities):
            raise ValueError("causal alpha diagnostic checkpoint contains duplicates")
        expected = content_digest(
            {
                "generator_code_digest": generator,
                "generator_identity_status": self.generator_identity_status,
                "grid_digest": grid,
                "metric_digests": tuple(item.digest for item in metrics),
                "schema_version": _SNAPSHOT_SCHEMA,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha diagnostic checkpoint digest mismatch")
        object.__setattr__(self, "grid_digest", grid)
        object.__setattr__(self, "generator_code_digest", generator)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "digest", expected)

    @property
    def row_count(self) -> int:
        return len(self.metrics)

    @property
    def generator_identity_status(self) -> str:
        if self.generator_code_digest is None:
            return _GENERATOR_IDENTITY_UNAVAILABLE_LEGACY
        return _GENERATOR_IDENTITY_PRESENT


@dataclass(frozen=True, slots=True)
class CausalAlphaDiagnosticCandidateSummary:
    candidate_digest: str
    replay_count: int
    mean_gross_return: float
    mean_net_return: float
    worst_net_return: float
    mean_turnover_per_day: float
    total_execution_cost: float
    total_trade_count: int

    def __post_init__(self) -> None:
        _require_digest(self.candidate_digest, field="candidate digest")
        if self.replay_count <= 0 or self.total_trade_count < 0:
            raise ValueError("causal alpha diagnostic candidate counts are invalid")
        values = (
            self.mean_gross_return,
            self.mean_net_return,
            self.worst_net_return,
            self.mean_turnover_per_day,
            self.total_execution_cost,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("causal alpha diagnostic candidate metric is non-finite")
        if self.mean_turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("causal alpha diagnostic candidate cost metric is invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_digest": self.candidate_digest,
            "mean_gross_return": self.mean_gross_return,
            "mean_net_return": self.mean_net_return,
            "mean_turnover_per_day": self.mean_turnover_per_day,
            "replay_count": self.replay_count,
            "total_execution_cost": self.total_execution_cost,
            "total_trade_count": self.total_trade_count,
            "worst_net_return": self.worst_net_return,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaResearchReport:
    checkpoint_digest: str
    grid_digest: str
    generator_code_digest: str | None
    row_count: int
    unique_prediction_episode_count: int
    duplicate_signal_row_count: int
    candidates: tuple[CausalAlphaDiagnosticCandidateSummary, ...]
    promotion_eligible: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        for field in ("checkpoint_digest", "grid_digest"):
            _require_digest(getattr(self, field), field=field.replace("_", " "))
        if self.generator_code_digest is not None:
            _require_digest(
                self.generator_code_digest,
                field="generator code digest",
            )
        if self.row_count <= 0:
            raise ValueError("causal alpha diagnostic report must contain rows")
        if not 0 < self.unique_prediction_episode_count <= self.row_count:
            raise ValueError(
                "causal alpha diagnostic unique prediction count is invalid"
            )
        if self.duplicate_signal_row_count != (
            self.row_count - self.unique_prediction_episode_count
        ):
            raise ValueError("causal alpha diagnostic duplicate count is inconsistent")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("causal alpha diagnostic report has no candidates")
        if self.promotion_eligible:
            raise ValueError("causal alpha historical diagnostics cannot be promotable")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha diagnostic report digest mismatch")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_count": len(self.candidates),
            "candidates": tuple(item.to_payload() for item in self.candidates),
            "checkpoint_digest": self.checkpoint_digest,
            "duplicate_signal_row_count": self.duplicate_signal_row_count,
            "generator_code_digest": self.generator_code_digest,
            "generator_identity_status": self.generator_identity_status,
            "grid_digest": self.grid_digest,
            "promotion_eligible": self.promotion_eligible,
            "row_count": self.row_count,
            "schema_version": _REPORT_SCHEMA,
            "unique_prediction_episode_count": self.unique_prediction_episode_count,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @property
    def generator_identity_status(self) -> str:
        if self.generator_code_digest is None:
            return _GENERATOR_IDENTITY_UNAVAILABLE_LEGACY
        return _GENERATOR_IDENTITY_PRESENT


@dataclass(frozen=True, slots=True)
class CausalAlphaPairedCandidateDelta:
    left_candidate_digest: str
    right_candidate_digest: str
    common_scopes: tuple[tuple[str, int], ...]
    left_only_scope_count: int
    right_only_scope_count: int
    mean_gross_return_delta: float
    mean_net_return_delta: float
    mean_turnover_per_day_delta: float
    mean_execution_cost_delta: float

    def __post_init__(self) -> None:
        _require_digest(self.left_candidate_digest, field="left candidate digest")
        _require_digest(self.right_candidate_digest, field="right candidate digest")
        scopes = tuple(self.common_scopes)
        if not scopes or len(set(scopes)) != len(scopes):
            raise ValueError("paired candidate comparison needs unique common scopes")
        if self.left_only_scope_count < 0 or self.right_only_scope_count < 0:
            raise ValueError("paired candidate exclusive scope count is invalid")
        if not all(
            math.isfinite(value)
            for value in (
                self.mean_gross_return_delta,
                self.mean_net_return_delta,
                self.mean_turnover_per_day_delta,
                self.mean_execution_cost_delta,
            )
        ):
            raise ValueError("paired candidate delta is non-finite")
        object.__setattr__(self, "common_scopes", scopes)

    @property
    def common_scope_count(self) -> int:
        return len(self.common_scopes)


def load_causal_alpha_diagnostic_checkpoint_v2(
    path: Path,
) -> CausalAlphaDiagnosticCheckpointV2:
    """Load one historical checkpoint without asserting current generator identity."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    metrics: list[CausalAlphaCandidateEpisodeMetricsV2] = []
    identities: set[tuple[str, str, int]] = set()
    grid_digest: str | None = None
    generator_digest: str | None = None
    generator_identity_available: bool | None = None
    with source.open("r", encoding="utf-8") as checkpoint:
        for line_number, line in enumerate(checkpoint, start=1):
            if not line.strip():
                raise ValueError(
                    f"causal alpha diagnostic checkpoint has blank row {line_number}"
                )
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(
                    "causal alpha diagnostic checkpoint row is not a mapping"
                )
            if raw.get("schema_version") != _CHECKPOINT_SCHEMA:
                raise ValueError("causal alpha diagnostic checkpoint schema mismatch")
            row_grid = _require_digest(raw.get("grid_digest"), field="grid digest")
            row_generator_available = "generator_code_digest" in raw
            if generator_identity_available is None:
                generator_identity_available = row_generator_available
            elif row_generator_available != generator_identity_available:
                raise ValueError(
                    "causal alpha diagnostic checkpoint generator identity "
                    "availability drifted"
                )
            row_generator = (
                _require_digest(
                    raw["generator_code_digest"],
                    field="generator code digest",
                )
                if row_generator_available
                else None
            )
            if grid_digest is None:
                grid_digest = row_grid
            elif row_grid != grid_digest:
                raise ValueError(
                    "causal alpha diagnostic checkpoint grid identity drifted"
                )
            if generator_digest is None and row_generator is not None:
                generator_digest = row_generator
            elif row_generator is not None and row_generator != generator_digest:
                raise ValueError(
                    "causal alpha diagnostic checkpoint generator identity drifted"
                )
            metric = causal_alpha_candidate_metric_v2_from_payload(raw)
            identity = (metric.candidate_digest, metric.symbol, metric.episode_index)
            if identity in identities:
                raise ValueError("causal alpha diagnostic checkpoint is duplicated")
            identities.add(identity)
            metrics.append(metric)
    if (
        not metrics
        or grid_digest is None
        or generator_identity_available is None
        or (generator_identity_available and generator_digest is None)
    ):
        raise ValueError("causal alpha diagnostic checkpoint is empty")
    return CausalAlphaDiagnosticCheckpointV2(
        grid_digest=grid_digest,
        generator_code_digest=generator_digest,
        metrics=tuple(metrics),
    )


def _candidate_summary(
    candidate_digest: str,
    metrics: tuple[CausalAlphaCandidateEpisodeMetricsV2, ...],
) -> CausalAlphaDiagnosticCandidateSummary:
    return CausalAlphaDiagnosticCandidateSummary(
        candidate_digest=candidate_digest,
        replay_count=len(metrics),
        mean_gross_return=float(
            np.mean([item.gross_return for item in metrics], dtype=np.float64)
        ),
        mean_net_return=float(
            np.mean([item.net_return for item in metrics], dtype=np.float64)
        ),
        worst_net_return=float(min(item.net_return for item in metrics)),
        mean_turnover_per_day=float(
            np.mean([item.turnover_per_day for item in metrics], dtype=np.float64)
        ),
        total_execution_cost=float(
            np.sum([item.total_execution_cost for item in metrics], dtype=np.float64)
        ),
        total_trade_count=sum(item.trade_count for item in metrics),
    )


def build_causal_alpha_research_report(
    snapshot: CausalAlphaDiagnosticCheckpointV2,
) -> CausalAlphaResearchReport:
    if not isinstance(snapshot, CausalAlphaDiagnosticCheckpointV2):
        raise TypeError("causal alpha research report requires a diagnostic snapshot")
    by_candidate: dict[str, list[CausalAlphaCandidateEpisodeMetricsV2]] = {}
    prediction_identities: set[tuple[str, int, str, str]] = set()
    for metric in snapshot.metrics:
        by_candidate.setdefault(metric.candidate_digest, []).append(metric)
        prediction_identities.add(
            (
                metric.symbol,
                metric.episode_index,
                metric.signal_24h.digest,
                metric.signal_72h.digest,
            )
        )
    candidates = tuple(
        _candidate_summary(candidate, tuple(by_candidate[candidate]))
        for candidate in sorted(by_candidate)
    )
    return CausalAlphaResearchReport(
        checkpoint_digest=snapshot.digest,
        grid_digest=snapshot.grid_digest,
        generator_code_digest=snapshot.generator_code_digest,
        row_count=snapshot.row_count,
        unique_prediction_episode_count=len(prediction_identities),
        duplicate_signal_row_count=snapshot.row_count - len(prediction_identities),
        candidates=candidates,
    )


def paired_candidate_delta(
    snapshot: CausalAlphaDiagnosticCheckpointV2,
    left_candidate_digest: str,
    right_candidate_digest: str,
) -> CausalAlphaPairedCandidateDelta:
    left_digest = _require_digest(left_candidate_digest, field="left candidate digest")
    right_digest = _require_digest(
        right_candidate_digest,
        field="right candidate digest",
    )
    left = {
        (item.symbol, item.episode_index): item
        for item in snapshot.metrics
        if item.candidate_digest == left_digest
    }
    right = {
        (item.symbol, item.episode_index): item
        for item in snapshot.metrics
        if item.candidate_digest == right_digest
    }
    if not left or not right:
        raise ValueError("paired candidate comparison candidate is absent")
    common = tuple(sorted(set(left) & set(right)))
    if not common:
        raise ValueError("paired candidate comparison has no common scope")

    def mean_delta(field: str) -> float:
        return float(
            np.mean(
                [
                    float(getattr(left[scope], field))
                    - float(getattr(right[scope], field))
                    for scope in common
                ],
                dtype=np.float64,
            )
        )

    return CausalAlphaPairedCandidateDelta(
        left_candidate_digest=left_digest,
        right_candidate_digest=right_digest,
        common_scopes=common,
        left_only_scope_count=len(set(left) - set(right)),
        right_only_scope_count=len(set(right) - set(left)),
        mean_gross_return_delta=mean_delta("gross_return"),
        mean_net_return_delta=mean_delta("net_return"),
        mean_turnover_per_day_delta=mean_delta("turnover_per_day"),
        mean_execution_cost_delta=mean_delta("total_execution_cost"),
    )


__all__ = [
    "CausalAlphaDiagnosticCandidateSummary",
    "CausalAlphaDiagnosticCheckpointV2",
    "CausalAlphaPairedCandidateDelta",
    "CausalAlphaResearchReport",
    "build_causal_alpha_research_report",
    "load_causal_alpha_diagnostic_checkpoint_v2",
    "paired_candidate_delta",
]
