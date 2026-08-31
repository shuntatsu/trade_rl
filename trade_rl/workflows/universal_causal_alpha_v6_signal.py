"""V4-fast-bound target-liveness Signal gate for Causal Alpha V6."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import (
    CAUSAL_ALPHA_V6_TARGET_REASONS,
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetPath,
)
from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    CausalAlphaV4LaneSignalEvidence,
    CausalAlphaV4SignalLane,
)

CAUSAL_ALPHA_V6_SIGNAL_SCOPE_SCHEMA: Final = "causal_alpha_v6_signal_scope_v1"
CAUSAL_ALPHA_V6_CANDIDATE_SIGNAL_SCHEMA: Final = (
    "causal_alpha_v6_candidate_signal_evidence_v1"
)
CAUSAL_ALPHA_V6_SIGNAL_EVIDENCE_SCHEMA: Final = "causal_alpha_v6_signal_evidence_v1"
_EXPECTED_SCOPE_COUNT: Final = 72
_EXPECTED_EPISODE_COUNT: Final = 8
_EXPECTED_SYMBOL_COUNT: Final = 9
_EPSILON: Final = 1e-12


def _non_negative_int(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V6 signal {field} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CausalAlphaV6SignalScopeMetric:
    """Compact target-liveness evidence for one candidate/symbol/episode."""

    candidate: CausalAlphaV6Candidate
    run_manifest_digest: str
    config_digest: str
    symbol: str
    episode_index: int
    contract_start: int
    contract_stop: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
    target_digest: str
    initial_weight: float
    decision_count: int
    actionable_count: int
    non_flat_target_count: int
    target_change_count: int
    sign_flip_count: int
    reason_counts: tuple[tuple[str, int], ...]
    slow_direction_sample_count: int
    slow_direction_accuracy: float
    schema_version: str = CAUSAL_ALPHA_V6_SIGNAL_SCOPE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV6Candidate(self.candidate)
        for name in (
            "run_manifest_digest",
            "config_digest",
            "contract_digest",
            "fit_digest",
            "forecast_digest",
            "target_digest",
        ):
            require_sha256(getattr(self, name), field=f"V6 signal {name}")
        if not self.symbol:
            raise ValueError("V6 signal symbol must be non-empty")
        for name in (
            "episode_index",
            "contract_start",
            "decision_count",
            "actionable_count",
            "non_flat_target_count",
            "target_change_count",
            "sign_flip_count",
            "slow_direction_sample_count",
        ):
            _non_negative_int(getattr(self, name), field=name)
        if self.contract_stop <= self.contract_start:
            raise ValueError("V6 signal contract interval is invalid")
        if self.decision_count <= 0 or any(
            value > self.decision_count
            for value in (
                self.actionable_count,
                self.non_flat_target_count,
                self.target_change_count,
                self.sign_flip_count,
                self.slow_direction_sample_count,
            )
        ):
            raise ValueError("V6 signal decision counts are invalid")
        if not math.isfinite(self.initial_weight):
            raise ValueError("V6 signal initial weight must be finite")
        if not 0.0 <= self.slow_direction_accuracy <= 1.0:
            raise ValueError("V6 signal slow accuracy is invalid")
        reasons = tuple(self.reason_counts)
        if (
            tuple(sorted(reasons)) != reasons
            or len({reason for reason, _ in reasons}) != len(reasons)
            or any(
                reason not in CAUSAL_ALPHA_V6_TARGET_REASONS
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                for reason, count in reasons
            )
            or sum(count for _, count in reasons) != self.decision_count
        ):
            raise ValueError("V6 signal reasons do not close over decisions")
        if self.schema_version != CAUSAL_ALPHA_V6_SIGNAL_SCOPE_SCHEMA:
            raise ValueError("unsupported V6 signal scope schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "reason_counts", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 signal scope digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.candidate.value, self.symbol, self.episode_index)

    @property
    def paired_identity(self) -> tuple[str, int]:
        return (self.symbol, self.episode_index)

    @property
    def pairing_payload(self) -> tuple[object, ...]:
        return (
            self.run_manifest_digest,
            self.config_digest,
            self.contract_start,
            self.contract_stop,
            self.contract_digest,
            self.fit_digest,
            self.forecast_digest,
            self.initial_weight,
            self.decision_count,
            self.actionable_count,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "actionable_count": self.actionable_count,
            "candidate": self.candidate.value,
            "config_digest": self.config_digest,
            "contract_digest": self.contract_digest,
            "contract_start": self.contract_start,
            "contract_stop": self.contract_stop,
            "decision_count": self.decision_count,
            "episode_index": self.episode_index,
            "fit_digest": self.fit_digest,
            "forecast_digest": self.forecast_digest,
            "initial_weight": self.initial_weight,
            "non_flat_target_count": self.non_flat_target_count,
            "reason_counts": self.reason_counts,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "sign_flip_count": self.sign_flip_count,
            "slow_direction_accuracy": self.slow_direction_accuracy,
            "slow_direction_sample_count": self.slow_direction_sample_count,
            "symbol": self.symbol,
            "target_change_count": self.target_change_count,
            "target_digest": self.target_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV6CandidateSignalEvidence:
    candidate: CausalAlphaV6Candidate
    metrics: tuple[CausalAlphaV6SignalScopeMetric, ...]
    raw_scope_count: int
    independent_episode_count: int
    symbol_count: int
    decision_count: int
    actionable_count: int
    non_flat_target_count: int
    target_change_count: int
    sign_flip_count: int
    slow_direction_sample_count: int
    slow_direction_accuracy: float
    passed: bool
    rejection_reasons: tuple[str, ...]
    schema_version: str = CAUSAL_ALPHA_V6_CANDIDATE_SIGNAL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV6Candidate(self.candidate)
        metrics = tuple(self.metrics)
        if not metrics or any(metric.candidate is not candidate for metric in metrics):
            raise ValueError("V6 candidate signal metrics are invalid")
        totals = {
            "raw_scope_count": len(metrics),
            "decision_count": sum(metric.decision_count for metric in metrics),
            "actionable_count": sum(metric.actionable_count for metric in metrics),
            "non_flat_target_count": sum(
                metric.non_flat_target_count for metric in metrics
            ),
            "target_change_count": sum(
                metric.target_change_count for metric in metrics
            ),
            "sign_flip_count": sum(metric.sign_flip_count for metric in metrics),
            "slow_direction_sample_count": sum(
                metric.slow_direction_sample_count for metric in metrics
            ),
        }
        if any(getattr(self, name) != value for name, value in totals.items()):
            raise ValueError("V6 candidate signal totals drifted")
        observed_episodes = len({metric.episode_index for metric in metrics})
        observed_symbols = len({metric.symbol for metric in metrics})
        if (
            self.independent_episode_count != observed_episodes
            or self.symbol_count != observed_symbols
        ):
            raise ValueError("V6 candidate signal coverage totals drifted")
        if not 0.0 <= self.slow_direction_accuracy <= 1.0:
            raise ValueError("V6 candidate slow accuracy is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V6 candidate pass state and reasons disagree")
        if self.schema_version != CAUSAL_ALPHA_V6_CANDIDATE_SIGNAL_SCHEMA:
            raise ValueError("unsupported V6 candidate signal schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 candidate signal digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "actionable_count": self.actionable_count,
            "candidate": self.candidate.value,
            "decision_count": self.decision_count,
            "independent_episode_count": self.independent_episode_count,
            "metric_digests": tuple(metric.digest for metric in self.metrics),
            "non_flat_target_count": self.non_flat_target_count,
            "passed": self.passed,
            "raw_scope_count": self.raw_scope_count,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "sign_flip_count": self.sign_flip_count,
            "slow_direction_accuracy": self.slow_direction_accuracy,
            "slow_direction_sample_count": self.slow_direction_sample_count,
            "symbol_count": self.symbol_count,
            "target_change_count": self.target_change_count,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV6SignalEvidence:
    fast_only: CausalAlphaV6CandidateSignalEvidence
    fast_slow_retention: CausalAlphaV6CandidateSignalEvidence
    raw_scope_count_per_candidate: int
    independent_episode_count: int
    symbol_count: int
    v4_fast_lane_digest: str
    v4_fast_lane_passed: bool
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V6_SIGNAL_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.fast_only.candidate is not CausalAlphaV6Candidate.FAST_ONLY:
            raise ValueError("V6 fast-only Signal evidence is invalid")
        if (
            self.fast_slow_retention.candidate
            is not CausalAlphaV6Candidate.FAST_SLOW_RETENTION
        ):
            raise ValueError("V6 retention Signal evidence is invalid")
        require_sha256(self.v4_fast_lane_digest, field="V6 fast lane digest")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V6 Signal pass state and reasons disagree")
        if self.promotion_eligible:
            raise ValueError("V6 Signal cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V6_SIGNAL_EVIDENCE_SCHEMA:
            raise ValueError("unsupported V6 Signal evidence schema")
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 Signal evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "fast_only": self.fast_only.to_payload(),
            "fast_slow_retention": self.fast_slow_retention.to_payload(),
            "independent_episode_count": self.independent_episode_count,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "raw_scope_count_per_candidate": self.raw_scope_count_per_candidate,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "symbol_count": self.symbol_count,
            "v4_fast_lane_digest": self.v4_fast_lane_digest,
            "v4_fast_lane_passed": self.v4_fast_lane_passed,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_causal_alpha_v6_signal_scope_metric(
    *,
    run_manifest_digest: str,
    symbol: str,
    episode_index: int,
    contract_start: int,
    contract_stop: int,
    contract_digest: str,
    fit_digest: str,
    target_path: CausalAlphaV6TargetPath,
    slow_realized_returns: object,
) -> CausalAlphaV6SignalScopeMetric:
    """Reduce one complete target path to compact Signal liveness evidence."""

    if not isinstance(target_path, CausalAlphaV6TargetPath):
        raise TypeError("V6 Signal scope requires a V6 target path")
    decisions = np.asarray(target_path.decision_indices)
    rows = int(decisions.size)
    if np.any(decisions < contract_start) or np.any(decisions >= contract_stop):
        raise ValueError("V6 Signal decisions escape the contract interval")
    realized = np.asarray(slow_realized_returns, dtype=np.float64).reshape(-1)
    if realized.shape != (rows,):
        raise ValueError("V6 Signal slow realized returns must align")
    slow_prediction = 0.5 * (
        np.asarray(target_path.expected_returns_24h)
        + np.asarray(target_path.expected_returns_72h) / 3.0
    )
    eligible = np.isfinite(realized) & (np.sign(realized) != 0.0)
    support = int(np.count_nonzero(eligible))
    accuracy = (
        0.0
        if support == 0
        else float(
            np.mean(np.sign(slow_prediction[eligible]) == np.sign(realized[eligible]))
        )
    )
    previous = np.concatenate(([target_path.initial_weight], target_path.targets[:-1]))
    return CausalAlphaV6SignalScopeMetric(
        candidate=target_path.candidate,
        run_manifest_digest=run_manifest_digest,
        config_digest=target_path.config_digest,
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract_start,
        contract_stop=contract_stop,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=target_path.forecast_digest,
        target_digest=target_path.digest,
        initial_weight=target_path.initial_weight,
        decision_count=rows,
        actionable_count=int(np.count_nonzero(target_path.actionable_mask)),
        non_flat_target_count=int(
            np.count_nonzero(np.abs(target_path.targets) > _EPSILON)
        ),
        target_change_count=int(
            np.count_nonzero(np.abs(target_path.targets - previous) > _EPSILON)
        ),
        sign_flip_count=target_path.sign_flip_count,
        reason_counts=target_path.reason_counts,
        slow_direction_sample_count=support,
        slow_direction_accuracy=accuracy,
    )


def _candidate_evidence(
    candidate: CausalAlphaV6Candidate,
    metrics: tuple[CausalAlphaV6SignalScopeMetric, ...],
) -> CausalAlphaV6CandidateSignalEvidence:
    reasons: list[str] = []
    if len(metrics) != _EXPECTED_SCOPE_COUNT:
        reasons.append("raw_scope_count")
    if len({metric.episode_index for metric in metrics}) != _EXPECTED_EPISODE_COUNT:
        reasons.append("independent_episode_count")
    if len({metric.symbol for metric in metrics}) != _EXPECTED_SYMBOL_COUNT:
        reasons.append("symbol_coverage")
    non_flat = sum(metric.non_flat_target_count for metric in metrics)
    if non_flat == 0:
        reasons.append(f"{candidate.value}_non_flat_target")
    support = sum(metric.slow_direction_sample_count for metric in metrics)
    weighted_accuracy = (
        0.0
        if support == 0
        else sum(
            metric.slow_direction_accuracy * metric.slow_direction_sample_count
            for metric in metrics
        )
        / support
    )
    return CausalAlphaV6CandidateSignalEvidence(
        candidate=candidate,
        metrics=metrics,
        raw_scope_count=len(metrics),
        independent_episode_count=len({metric.episode_index for metric in metrics}),
        symbol_count=len({metric.symbol for metric in metrics}),
        decision_count=sum(metric.decision_count for metric in metrics),
        actionable_count=sum(metric.actionable_count for metric in metrics),
        non_flat_target_count=non_flat,
        target_change_count=sum(metric.target_change_count for metric in metrics),
        sign_flip_count=sum(metric.sign_flip_count for metric in metrics),
        slow_direction_sample_count=support,
        slow_direction_accuracy=float(weighted_accuracy),
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _paired(metrics: tuple[CausalAlphaV6SignalScopeMetric, ...]) -> bool:
    grouped: dict[
        CausalAlphaV6Candidate,
        dict[tuple[str, int], list[CausalAlphaV6SignalScopeMetric]],
    ] = {candidate: {} for candidate in CausalAlphaV6Candidate}
    for metric in metrics:
        grouped[metric.candidate].setdefault(metric.paired_identity, []).append(metric)
    keys = [set(grouped[candidate]) for candidate in CausalAlphaV6Candidate]
    if keys[0] != keys[1]:
        return False
    for identity in keys[0]:
        pairs = [grouped[candidate][identity] for candidate in CausalAlphaV6Candidate]
        if any(len(pair) != 1 for pair in pairs):
            return False
        if pairs[0][0].pairing_payload != pairs[1][0].pairing_payload:
            return False
    return True


def evaluate_causal_alpha_v6_signal_gate(
    metrics: tuple[CausalAlphaV6SignalScopeMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
    v4_fast_lane: CausalAlphaV4LaneSignalEvidence,
) -> CausalAlphaV6SignalEvidence:
    """Require exact target liveness plus the unchanged admitted V4 fast lane."""

    values = tuple(metrics)
    if not values:
        raise ValueError("V6 Signal requires scope metrics")
    expected = tuple(expected_symbols)
    if len(expected) != _EXPECTED_SYMBOL_COUNT or len(set(expected)) != len(expected):
        raise ValueError("V6 Signal requires exactly nine expected symbols")
    if (
        not isinstance(v4_fast_lane, CausalAlphaV4LaneSignalEvidence)
        or v4_fast_lane.lane is not CausalAlphaV4SignalLane.FAST_4H
    ):
        raise TypeError("V6 Signal requires V4 fast-4h lane evidence")
    grouped = {
        candidate: tuple(metric for metric in values if metric.candidate is candidate)
        for candidate in CausalAlphaV6Candidate
    }
    candidate_evidence = {
        candidate: _candidate_evidence(candidate, grouped[candidate])
        for candidate in CausalAlphaV6Candidate
    }
    reasons: list[str] = []
    per_candidate_counts = {len(grouped[candidate]) for candidate in grouped}
    if per_candidate_counts != {_EXPECTED_SCOPE_COUNT}:
        reasons.append("raw_scope_count")
    if len({metric.identity for metric in values}) != len(values):
        reasons.append("duplicate_scope_identity")
    episodes = {metric.episode_index for metric in values}
    if episodes != set(range(_EXPECTED_EPISODE_COUNT)):
        reasons.append("episode_coverage")
    observed_symbols = {metric.symbol for metric in values}
    if observed_symbols != set(expected):
        reasons.append("symbol_coverage")
    if len({metric.config_digest for metric in values}) != 1:
        reasons.append("config_identity")
    if len({metric.run_manifest_digest for metric in values}) != 1:
        reasons.append("run_identity")
    if not _paired(values):
        reasons.append("scope_pairing")
    fast_only = candidate_evidence[CausalAlphaV6Candidate.FAST_ONLY]
    retention = candidate_evidence[CausalAlphaV6Candidate.FAST_SLOW_RETENTION]
    if fast_only.non_flat_target_count == 0:
        reasons.append("fast_only_non_flat_target")
    if retention.non_flat_target_count == 0:
        reasons.append("fast_slow_retention_non_flat_target")
    if not v4_fast_lane.passed:
        reasons.append("v4_fast_4h")
    reasons = list(dict.fromkeys(reasons))
    return CausalAlphaV6SignalEvidence(
        fast_only=fast_only,
        fast_slow_retention=retention,
        raw_scope_count_per_candidate=(
            next(iter(per_candidate_counts)) if len(per_candidate_counts) == 1 else 0
        ),
        independent_episode_count=len(episodes),
        symbol_count=len(observed_symbols),
        v4_fast_lane_digest=v4_fast_lane.digest,
        v4_fast_lane_passed=v4_fast_lane.passed,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "CAUSAL_ALPHA_V6_CANDIDATE_SIGNAL_SCHEMA",
    "CAUSAL_ALPHA_V6_SIGNAL_EVIDENCE_SCHEMA",
    "CAUSAL_ALPHA_V6_SIGNAL_SCOPE_SCHEMA",
    "CausalAlphaV6CandidateSignalEvidence",
    "CausalAlphaV6SignalEvidence",
    "CausalAlphaV6SignalScopeMetric",
    "build_causal_alpha_v6_signal_scope_metric",
    "evaluate_causal_alpha_v6_signal_gate",
]
