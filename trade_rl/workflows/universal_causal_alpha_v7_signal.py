"""Three-way calibration and target liveness Signal gate for V7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate

_SCOPE_SCHEMA: Final = "causal_alpha_v7_signal_scope_v1"
_CANDIDATE_SCHEMA: Final = "causal_alpha_v7_candidate_signal_v1"
_SIGNAL_SCHEMA: Final = "causal_alpha_v7_signal_evidence_v1"
_EXPECTED_SCOPE_COUNT: Final = 72
_EXPECTED_EPISODE_COUNT: Final = 8
_EXPECTED_SYMBOL_COUNT: Final = 9
_MINIMUM_DIRECTION_SUPPORT: Final = 16


def _count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V7 Signal {field} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class CausalAlphaV7SignalScopeMetric:
    candidate: CausalAlphaV7Candidate
    run_manifest_digest: str
    v7_config_digest: str
    symbol: str
    episode_index: int
    contract_start: int
    contract_stop: int
    contract_digest: str
    source_forecast_digest: str
    calibration_fit_digest: str
    calibration_return_model_digest: str
    calibration_direction_model_digest: str
    target_path_digest: str
    decision_count: int
    actionable_count: int
    non_flat_target_count: int
    target_change_count: int
    sign_flip_count: int
    positive_direction_support: int
    negative_direction_support: int
    schema_version: str = _SCOPE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV7Candidate(self.candidate)
        for name in (
            "run_manifest_digest",
            "v7_config_digest",
            "contract_digest",
            "source_forecast_digest",
            "calibration_fit_digest",
            "calibration_return_model_digest",
            "calibration_direction_model_digest",
            "target_path_digest",
        ):
            require_sha256(getattr(self, name), field=f"V7 Signal {name}")
        if not self.symbol:
            raise ValueError("V7 Signal symbol must be non-empty")
        for name in (
            "episode_index",
            "contract_start",
            "decision_count",
            "actionable_count",
            "non_flat_target_count",
            "target_change_count",
            "sign_flip_count",
            "positive_direction_support",
            "negative_direction_support",
        ):
            _count(getattr(self, name), field=name)
        if self.contract_stop <= self.contract_start or self.decision_count <= 0:
            raise ValueError("V7 Signal contract/decision interval is invalid")
        if any(
            value > self.decision_count
            for value in (
                self.actionable_count,
                self.non_flat_target_count,
                self.target_change_count,
                self.sign_flip_count,
            )
        ):
            raise ValueError("V7 Signal path counts exceed decisions")
        if self.schema_version != _SCOPE_SCHEMA:
            raise ValueError("unsupported V7 Signal scope schema")
        object.__setattr__(self, "candidate", candidate)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 Signal scope digest mismatch")
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
            self.v7_config_digest,
            self.contract_start,
            self.contract_stop,
            self.contract_digest,
            self.source_forecast_digest,
            self.calibration_fit_digest,
            self.calibration_return_model_digest,
            self.calibration_direction_model_digest,
            self.decision_count,
            self.actionable_count,
            self.positive_direction_support,
            self.negative_direction_support,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"candidate", "digest"}
        }
        payload["candidate"] = self.candidate.value
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV7CandidateSignalEvidence:
    candidate: CausalAlphaV7Candidate
    metrics: tuple[CausalAlphaV7SignalScopeMetric, ...]
    raw_scope_count: int
    independent_episode_count: int
    symbol_count: int
    decision_count: int
    actionable_count: int
    non_flat_target_count: int
    target_change_count: int
    sign_flip_count: int
    minimum_positive_direction_support: int
    minimum_negative_direction_support: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    schema_version: str = _CANDIDATE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV7Candidate(self.candidate)
        metrics = tuple(self.metrics)
        if any(metric.candidate is not candidate for metric in metrics):
            raise ValueError("V7 candidate Signal metrics are invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V7 candidate Signal pass state is invalid")
        if self.schema_version != _CANDIDATE_SCHEMA:
            raise ValueError("unsupported V7 candidate Signal schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 candidate Signal digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"candidate", "metrics", "digest"}
        }
        payload["candidate"] = self.candidate.value
        payload["metric_digests"] = tuple(metric.digest for metric in self.metrics)
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV7SignalEvidence:
    candidates: tuple[CausalAlphaV7CandidateSignalEvidence, ...]
    raw_scope_count_per_candidate: int
    independent_episode_count: int
    symbol_count: int
    v4_fast_lane_digest: str
    v4_fast_lane_passed: bool
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = _SIGNAL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if tuple(item.candidate for item in candidates) != tuple(CausalAlphaV7Candidate):
            raise ValueError("V7 Signal candidate evidence is not canonical")
        require_sha256(self.v4_fast_lane_digest, field="V7 fast lane digest")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V7 Signal pass state is invalid")
        if self.promotion_eligible:
            raise ValueError("V7 Signal cannot be promotion eligible")
        if self.schema_version != _SIGNAL_SCHEMA:
            raise ValueError("unsupported V7 Signal schema")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 Signal evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidates": tuple(candidate.to_payload() for candidate in self.candidates),
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


def _candidate_evidence(
    candidate: CausalAlphaV7Candidate,
    metrics: tuple[CausalAlphaV7SignalScopeMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV7CandidateSignalEvidence:
    reasons: list[str] = []
    episodes = len({metric.episode_index for metric in metrics})
    symbols = len({metric.symbol for metric in metrics})
    non_flat = sum(metric.non_flat_target_count for metric in metrics)
    minimum_positive = min(
        (metric.positive_direction_support for metric in metrics), default=0
    )
    minimum_negative = min(
        (metric.negative_direction_support for metric in metrics), default=0
    )
    if len(metrics) != _EXPECTED_SCOPE_COUNT:
        reasons.append("raw_scope_count")
    if episodes != _EXPECTED_EPISODE_COUNT:
        reasons.append("independent_episode_count")
    if {metric.symbol for metric in metrics} != set(expected_symbols):
        reasons.append("symbol_coverage")
    if non_flat == 0:
        reasons.append("non_flat_target")
    if minimum_positive < _MINIMUM_DIRECTION_SUPPORT:
        reasons.append("positive_direction_support")
    if minimum_negative < _MINIMUM_DIRECTION_SUPPORT:
        reasons.append("negative_direction_support")
    return CausalAlphaV7CandidateSignalEvidence(
        candidate=candidate,
        metrics=metrics,
        raw_scope_count=len(metrics),
        independent_episode_count=episodes,
        symbol_count=symbols,
        decision_count=sum(metric.decision_count for metric in metrics),
        actionable_count=sum(metric.actionable_count for metric in metrics),
        non_flat_target_count=non_flat,
        target_change_count=sum(metric.target_change_count for metric in metrics),
        sign_flip_count=sum(metric.sign_flip_count for metric in metrics),
        minimum_positive_direction_support=minimum_positive,
        minimum_negative_direction_support=minimum_negative,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _paired(metrics: tuple[CausalAlphaV7SignalScopeMetric, ...]) -> bool:
    grouped: dict[
        CausalAlphaV7Candidate,
        dict[tuple[str, int], list[CausalAlphaV7SignalScopeMetric]],
    ] = {candidate: {} for candidate in CausalAlphaV7Candidate}
    for metric in metrics:
        grouped[metric.candidate].setdefault(metric.paired_identity, []).append(metric)
    identities = [set(grouped[candidate]) for candidate in CausalAlphaV7Candidate]
    if any(values != identities[0] for values in identities[1:]):
        return False
    for identity in identities[0]:
        pairs = [grouped[candidate][identity] for candidate in CausalAlphaV7Candidate]
        if any(len(pair) != 1 for pair in pairs):
            return False
        if any(pair[0].pairing_payload != pairs[0][0].pairing_payload for pair in pairs[1:]):
            return False
    return True


def evaluate_causal_alpha_v7_signal_gate(
    metrics: tuple[CausalAlphaV7SignalScopeMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
    v4_fast_lane_digest: str,
    v4_fast_lane_passed: bool,
) -> CausalAlphaV7SignalEvidence:
    """Require exact three-way liveness, calibration support, and V4 fast lane."""

    values = tuple(metrics)
    expected = tuple(expected_symbols)
    if not values:
        raise ValueError("V7 Signal requires scope metrics")
    if len(expected) != _EXPECTED_SYMBOL_COUNT or len(set(expected)) != len(expected):
        raise ValueError("V7 Signal requires exactly nine expected symbols")
    if not isinstance(v4_fast_lane_passed, bool):
        raise TypeError("V7 fast lane pass state must be boolean")
    grouped = {
        candidate: tuple(metric for metric in values if metric.candidate is candidate)
        for candidate in CausalAlphaV7Candidate
    }
    candidates = tuple(
        _candidate_evidence(
            candidate,
            grouped[candidate],
            expected_symbols=expected,
        )
        for candidate in CausalAlphaV7Candidate
    )
    reasons: list[str] = []
    if not _paired(values):
        reasons.append("scope_pairing")
    if not v4_fast_lane_passed:
        reasons.append("v4_fast_lane")
    if any(not candidate.passed for candidate in candidates):
        reasons.append("candidate_signal")
    return CausalAlphaV7SignalEvidence(
        candidates=candidates,
        raw_scope_count_per_candidate=len(grouped[CausalAlphaV7Candidate.V6_CONTROL]),
        independent_episode_count=len({metric.episode_index for metric in values}),
        symbol_count=len({metric.symbol for metric in values}),
        v4_fast_lane_digest=v4_fast_lane_digest,
        v4_fast_lane_passed=v4_fast_lane_passed,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "CausalAlphaV7CandidateSignalEvidence",
    "CausalAlphaV7SignalEvidence",
    "CausalAlphaV7SignalScopeMetric",
    "evaluate_causal_alpha_v7_signal_gate",
]
