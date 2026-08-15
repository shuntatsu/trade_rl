"""Immutable, research-only evidence contracts for Causal Alpha V3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_teacher import CausalAlphaTeacherAdmissionEvidence
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)


def _digest_field(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"V3 {name} is invalid")


@dataclass(frozen=True, slots=True)
class CausalAlphaV3CandidateConfig:
    name: str
    fit: CausalAlphaV3FitConfig
    target: CausalAlphaV3TargetConfig
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("V3 candidate name must be non-empty")
        if not isinstance(self.fit, CausalAlphaV3FitConfig) or not isinstance(
            self.target, CausalAlphaV3TargetConfig
        ):
            raise TypeError("V3 candidate configuration is invalid")
        expected = content_digest(
            {
                "fit_digest": self.fit.digest,
                "name": self.name,
                "schema_version": "causal_alpha_v3_candidate_config_v1",
                "target_digest": self.target.digest,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("V3 candidate config digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaV3EpisodeMetric:
    candidate_digest: str
    symbol: str
    episode_index: int
    contract_digest: str
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    trade_count: int
    hard_risk_violation: bool
    unexplained_execution_rejection_count: int
    digest: str = ""

    def __post_init__(self) -> None:
        _digest_field("candidate_digest", self.candidate_digest)
        _digest_field("contract_digest", self.contract_digest)
        if not self.symbol or self.episode_index < 0:
            raise ValueError("V3 episode scope is invalid")
        for field in (
            "gross_return",
            "net_return",
            "turnover_per_day",
            "total_execution_cost",
        ):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"V3 episode {field} is non-finite")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("V3 episode cost metrics are negative")
        for field in ("trade_count", "unexplained_execution_rejection_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V3 episode {field} is invalid")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 episode metric digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_digest": self.candidate_digest,
            "contract_digest": self.contract_digest,
            "episode_index": self.episode_index,
            "gross_return": self.gross_return,
            "hard_risk_violation": self.hard_risk_violation,
            "net_return": self.net_return,
            "schema_version": "causal_alpha_v3_episode_metric_v1",
            "symbol": self.symbol,
            "total_execution_cost": self.total_execution_cost,
            "trade_count": self.trade_count,
            "turnover_per_day": self.turnover_per_day,
            "unexplained_execution_rejection_count": (
                self.unexplained_execution_rejection_count
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV3CandidateEvidence:
    candidate: CausalAlphaV3CandidateConfig
    episode_metrics: tuple[CausalAlphaV3EpisodeMetric, ...]
    lower_tail_net_return: float
    mean_net_return: float
    turnover_per_day: float
    total_execution_cost: float
    negative_gross_episode_count: int
    total_trade_count: int
    unexplained_execution_rejection_count: int
    hard_risk_violation: bool
    admissible: bool
    rejection_reasons: tuple[str, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.episode_metrics)
        if not metrics or any(
            item.candidate_digest != self.candidate.digest for item in metrics
        ):
            raise ValueError("V3 candidate metric identity drifted")
        scopes = tuple((item.symbol, item.episode_index) for item in metrics)
        if len(scopes) != len(set(scopes)):
            raise ValueError("V3 candidate episode metrics are duplicated")
        reasons = tuple(self.rejection_reasons)
        if self.admissible == bool(reasons):
            raise ValueError("V3 candidate admission reasons are inconsistent")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 candidate evidence digest mismatch")
        object.__setattr__(self, "episode_metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "digest", expected)

    @classmethod
    def from_episode_metrics(
        cls,
        *,
        candidate: CausalAlphaV3CandidateConfig,
        episode_metrics: tuple[CausalAlphaV3EpisodeMetric, ...],
        admissible: bool,
        rejection_reasons: tuple[str, ...],
    ) -> CausalAlphaV3CandidateEvidence:
        metrics = tuple(episode_metrics)
        if not metrics:
            raise ValueError("V3 candidate evidence needs episode metrics")
        return cls(
            candidate=candidate,
            episode_metrics=metrics,
            lower_tail_net_return=min(item.net_return for item in metrics),
            mean_net_return=float(np.mean([item.net_return for item in metrics])),
            turnover_per_day=float(
                np.mean([item.turnover_per_day for item in metrics])
            ),
            total_execution_cost=sum(item.total_execution_cost for item in metrics),
            negative_gross_episode_count=sum(
                item.gross_return < 0.0 for item in metrics
            ),
            total_trade_count=sum(item.trade_count for item in metrics),
            unexplained_execution_rejection_count=sum(
                item.unexplained_execution_rejection_count for item in metrics
            ),
            hard_risk_violation=any(item.hard_risk_violation for item in metrics),
            admissible=admissible,
            rejection_reasons=rejection_reasons,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            "admissible": self.admissible,
            "candidate_digest": self.candidate.digest,
            "episode_metric_digests": tuple(
                item.digest for item in self.episode_metrics
            ),
            "hard_risk_violation": self.hard_risk_violation,
            "lower_tail_net_return": self.lower_tail_net_return,
            "mean_net_return": self.mean_net_return,
            "negative_gross_episode_count": self.negative_gross_episode_count,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": "causal_alpha_v3_candidate_evidence_v1",
            "total_execution_cost": self.total_execution_cost,
            "total_trade_count": self.total_trade_count,
            "turnover_per_day": self.turnover_per_day,
            "unexplained_execution_rejection_count": self.unexplained_execution_rejection_count,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SelectionEvidence:
    candidates: tuple[CausalAlphaV3CandidateEvidence, ...]
    selected_candidate_digest: str
    grid_digest: str
    thresholds_digest: str
    generator_code_digest: str
    sample_scope_digest: str
    holdout_episode_digests: Mapping[str, str]
    promotion_eligible: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        selected = [
            item
            for item in candidates
            if item.candidate.digest == self.selected_candidate_digest
        ]
        if not candidates or len(selected) != 1 or not selected[0].admissible:
            raise ValueError("V3 selected candidate is invalid")
        for field in (
            "grid_digest",
            "thresholds_digest",
            "generator_code_digest",
            "sample_scope_digest",
        ):
            _digest_field(field, getattr(self, field))
        holdouts = dict(self.holdout_episode_digests)
        if not holdouts or any(
            not symbol or len(value) != 64 for symbol, value in holdouts.items()
        ):
            raise ValueError("V3 holdout identities are invalid")
        if self.promotion_eligible:
            raise ValueError("V3 research evidence cannot be promotion eligible")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 selection evidence digest mismatch")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "holdout_episode_digests", MappingProxyType(holdouts))
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            "candidate_evidence_digests": tuple(
                item.digest for item in self.candidates
            ),
            "generator_code_digest": self.generator_code_digest,
            "grid_digest": self.grid_digest,
            "holdout_episode_digests": dict(self.holdout_episode_digests),
            "promotion_eligible": self.promotion_eligible,
            "sample_scope_digest": self.sample_scope_digest,
            "schema_version": "causal_alpha_v3_selection_evidence_v1",
            "selected_candidate_digest": self.selected_candidate_digest,
            "thresholds_digest": self.thresholds_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV3TeacherAdmissionEvidence:
    selection_digest: str
    selected_candidate_digest: str
    holdout_episode_digests: Mapping[str, str]
    admission: CausalAlphaTeacherAdmissionEvidence
    digest: str = ""

    def __post_init__(self) -> None:
        _digest_field("selection_digest", self.selection_digest)
        _digest_field("selected_candidate_digest", self.selected_candidate_digest)
        holdouts = dict(self.holdout_episode_digests)
        if tuple(
            self.admission.metrics[i].symbol for i in range(len(self.admission.metrics))
        ) != tuple(holdouts):
            raise ValueError("V3 admission holdout symbol scope drifted")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 teacher admission digest mismatch")
        object.__setattr__(self, "holdout_episode_digests", MappingProxyType(holdouts))
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "admission_digest": self.admission.digest,
            "holdout_episode_digests": dict(self.holdout_episode_digests),
            "schema_version": "causal_alpha_v3_teacher_admission_evidence_v1",
            "selected_candidate_digest": self.selected_candidate_digest,
            "selection_digest": self.selection_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _target_digest(symbol: str, values: np.ndarray) -> str:
    return content_and_arrays_digest(
        {"schema_version": "causal_alpha_v3_target_weights_v1", "symbol": symbol},
        (("target_weights", values),),
    )


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaV3TeacherPackage:
    selection: CausalAlphaV3SelectionEvidence
    teacher_admission: CausalAlphaV3TeacherAdmissionEvidence
    target_weights: Mapping[str, np.ndarray]
    target_digests: Mapping[str, str]
    teacher_config_digest: str
    promotion_eligible: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        if self.teacher_admission.selection_digest != self.selection.digest:
            raise ValueError("V3 package selection identity drifted")
        if (
            self.teacher_admission.selected_candidate_digest
            != self.selection.selected_candidate_digest
        ):
            raise ValueError("V3 package candidate identity drifted")
        if not self.teacher_admission.admission.passed:
            raise ValueError("V3 package requires passed teacher admission")
        if self.promotion_eligible:
            raise ValueError("V3 research package cannot be promotion eligible")
        _digest_field("teacher_config_digest", self.teacher_config_digest)
        weights = {
            symbol: np.asarray(value, dtype=np.float32).reshape(-1).copy()
            for symbol, value in self.target_weights.items()
        }
        digests = dict(self.target_digests)
        if set(weights) != set(digests) or set(weights) != set(
            self.selection.holdout_episode_digests
        ):
            raise ValueError("V3 package target symbol scope drifted")
        for symbol, values in weights.items():
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("V3 package targets must be finite and non-empty")
            if digests[symbol] != _target_digest(symbol, values):
                raise ValueError("V3 package target digest mismatch")
            values.setflags(write=False)
        expected = content_digest(
            {
                "promotion_eligible": self.promotion_eligible,
                "schema_version": "universal_causal_alpha_v3_teacher_package_v1",
                "selection_digest": self.selection.digest,
                "target_digests": digests,
                "teacher_admission_digest": self.teacher_admission.digest,
                "teacher_config_digest": self.teacher_config_digest,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("V3 teacher package digest mismatch")
        object.__setattr__(self, "target_weights", MappingProxyType(weights))
        object.__setattr__(self, "target_digests", MappingProxyType(digests))
        object.__setattr__(self, "digest", expected)


__all__ = [
    "CausalAlphaV3CandidateConfig",
    "CausalAlphaV3CandidateEvidence",
    "CausalAlphaV3EpisodeMetric",
    "CausalAlphaV3SelectionEvidence",
    "CausalAlphaV3TeacherAdmissionEvidence",
    "UniversalCausalAlphaV3TeacherPackage",
]
