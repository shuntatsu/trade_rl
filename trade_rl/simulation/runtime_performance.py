"""Canonical performance evidence and explicit review-gated approval policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA = "execution_runtime_performance_evidence_v1"
RUNTIME_PERFORMANCE_POLICY_SCHEMA = "execution_runtime_performance_policy_v1"


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    return value


@dataclass(frozen=True, slots=True)
class RuntimePerformanceMeasurement:
    """One isolated training measurement for one runtime and workload size."""

    timesteps: int
    elapsed_seconds: float
    steps_per_second: float
    peak_self_rss_bytes: int
    peak_children_rss_bytes: int
    peak_process_tree_rss_bytes: int
    peak_process_count: int

    def __post_init__(self) -> None:
        timesteps = _positive_int(self.timesteps, field="timesteps")
        elapsed = _positive_float(self.elapsed_seconds, field="elapsed_seconds")
        steps_per_second = _positive_float(
            self.steps_per_second,
            field="steps_per_second",
        )
        expected_steps_per_second = timesteps / elapsed
        if not math.isclose(
            steps_per_second,
            expected_steps_per_second,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "steps_per_second is inconsistent with timesteps and elapsed"
            )
        self_rss = _positive_int(self.peak_self_rss_bytes, field="peak_self_rss_bytes")
        children_rss = _non_negative_int(
            self.peak_children_rss_bytes,
            field="peak_children_rss_bytes",
        )
        tree_rss = _positive_int(
            self.peak_process_tree_rss_bytes,
            field="peak_process_tree_rss_bytes",
        )
        if tree_rss < max(self_rss, children_rss):
            raise ValueError("process-tree RSS must cover self and child RSS evidence")
        _positive_int(self.peak_process_count, field="peak_process_count")

    def to_mapping(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "peak_children_rss_bytes": self.peak_children_rss_bytes,
            "peak_process_count": self.peak_process_count,
            "peak_process_tree_rss_bytes": self.peak_process_tree_rss_bytes,
            "peak_self_rss_bytes": self.peak_self_rss_bytes,
            "steps_per_second": self.steps_per_second,
            "timesteps": self.timesteps,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RuntimePerformanceMeasurement:
        required = {
            "elapsed_seconds",
            "peak_children_rss_bytes",
            "peak_process_count",
            "peak_process_tree_rss_bytes",
            "peak_self_rss_bytes",
            "steps_per_second",
            "timesteps",
        }
        if set(value) != required:
            raise ValueError("runtime performance measurement field closure mismatch")
        return cls(
            timesteps=_positive_int(value["timesteps"], field="timesteps"),
            elapsed_seconds=_positive_float(
                value["elapsed_seconds"], field="elapsed_seconds"
            ),
            steps_per_second=_positive_float(
                value["steps_per_second"], field="steps_per_second"
            ),
            peak_self_rss_bytes=_positive_int(
                value["peak_self_rss_bytes"], field="peak_self_rss_bytes"
            ),
            peak_children_rss_bytes=_non_negative_int(
                value["peak_children_rss_bytes"], field="peak_children_rss_bytes"
            ),
            peak_process_tree_rss_bytes=_positive_int(
                value["peak_process_tree_rss_bytes"],
                field="peak_process_tree_rss_bytes",
            ),
            peak_process_count=_positive_int(
                value["peak_process_count"], field="peak_process_count"
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimePerformanceWorkload:
    """Paired isolated legacy and Nautilus measurements for one workload."""

    timesteps: int
    legacy_authoritative: RuntimePerformanceMeasurement
    nautilus_dual_shadow_streaming: RuntimePerformanceMeasurement

    def __post_init__(self) -> None:
        timesteps = _positive_int(self.timesteps, field="timesteps")
        if self.legacy_authoritative.timesteps != timesteps:
            raise ValueError("legacy performance measurement timestep mismatch")
        if self.nautilus_dual_shadow_streaming.timesteps != timesteps:
            raise ValueError("Nautilus performance measurement timestep mismatch")

    @property
    def elapsed_slowdown_ratio(self) -> float:
        return (
            self.nautilus_dual_shadow_streaming.elapsed_seconds
            / self.legacy_authoritative.elapsed_seconds
        )

    @property
    def peak_process_tree_rss_ratio(self) -> float:
        return (
            self.nautilus_dual_shadow_streaming.peak_process_tree_rss_bytes
            / self.legacy_authoritative.peak_process_tree_rss_bytes
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "elapsed_slowdown_ratio": self.elapsed_slowdown_ratio,
            "legacy_authoritative": self.legacy_authoritative.to_mapping(),
            "nautilus_dual_shadow_streaming": (
                self.nautilus_dual_shadow_streaming.to_mapping()
            ),
            "peak_process_tree_rss_ratio": self.peak_process_tree_rss_ratio,
            "timesteps": self.timesteps,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RuntimePerformanceWorkload:
        required = {
            "elapsed_slowdown_ratio",
            "legacy_authoritative",
            "nautilus_dual_shadow_streaming",
            "peak_process_tree_rss_ratio",
            "timesteps",
        }
        if set(value) != required:
            raise ValueError("runtime performance workload field closure mismatch")
        workload = cls(
            timesteps=_positive_int(value["timesteps"], field="timesteps"),
            legacy_authoritative=RuntimePerformanceMeasurement.from_mapping(
                _mapping(value["legacy_authoritative"], field="legacy_authoritative")
            ),
            nautilus_dual_shadow_streaming=RuntimePerformanceMeasurement.from_mapping(
                _mapping(
                    value["nautilus_dual_shadow_streaming"],
                    field="nautilus_dual_shadow_streaming",
                )
            ),
        )
        for field, actual, expected in (
            (
                "elapsed_slowdown_ratio",
                _positive_float(
                    value["elapsed_slowdown_ratio"],
                    field="elapsed_slowdown_ratio",
                ),
                workload.elapsed_slowdown_ratio,
            ),
            (
                "peak_process_tree_rss_ratio",
                _positive_float(
                    value["peak_process_tree_rss_ratio"],
                    field="peak_process_tree_rss_ratio",
                ),
                workload.peak_process_tree_rss_ratio,
            ),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{field} is inconsistent with paired measurements")
        return workload


@dataclass(frozen=True, slots=True)
class RuntimePerformanceEvidence:
    """Canonical multi-workload evidence; observation never self-authorizes."""

    runtime_version: str
    platform: str
    algorithm: str
    dataset_kind: str
    source_digest: str
    workloads: tuple[RuntimePerformanceWorkload, ...]
    performance_approved: bool
    approval_policy_digest: str | None
    approval_note: str
    schema_version: str = RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        _string(self.runtime_version, field="runtime_version")
        _string(self.platform, field="platform")
        _string(self.algorithm, field="algorithm")
        _string(self.dataset_kind, field="dataset_kind")
        require_sha256(self.source_digest, field="source_digest")
        if not self.workloads:
            raise ValueError("runtime performance evidence requires workloads")
        timesteps = tuple(workload.timesteps for workload in self.workloads)
        if timesteps != tuple(sorted(set(timesteps))):
            raise ValueError("runtime performance workloads must be unique and ordered")
        _boolean(self.performance_approved, field="performance_approved")
        policy_digest = self.approval_policy_digest
        if policy_digest is not None:
            require_sha256(policy_digest, field="approval_policy_digest")
        if self.performance_approved and policy_digest is None:
            raise ValueError("approved performance evidence requires a policy digest")
        _string(self.approval_note, field="approval_note")
        if self.schema_version != RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA:
            raise ValueError("unsupported runtime performance evidence schema")

    @property
    def timesteps(self) -> tuple[int, ...]:
        return tuple(workload.timesteps for workload in self.workloads)

    @property
    def worst_elapsed_slowdown_ratio(self) -> float:
        return max(workload.elapsed_slowdown_ratio for workload in self.workloads)

    @property
    def worst_peak_process_tree_rss_ratio(self) -> float:
        return max(workload.peak_process_tree_rss_ratio for workload in self.workloads)

    @property
    def digest(self) -> str:
        return content_digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "approval_note": self.approval_note,
            "approval_policy_digest": self.approval_policy_digest,
            "dataset_kind": self.dataset_kind,
            "performance_approved": self.performance_approved,
            "platform": self.platform,
            "runtime_version": self.runtime_version,
            "schema_version": self.schema_version,
            "source_digest": self.source_digest,
            "timesteps": self.timesteps,
            "workloads": tuple(workload.to_mapping() for workload in self.workloads),
            "worst_elapsed_slowdown_ratio": self.worst_elapsed_slowdown_ratio,
            "worst_peak_process_tree_rss_ratio": (
                self.worst_peak_process_tree_rss_ratio
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RuntimePerformanceEvidence:
        required = {
            "algorithm",
            "approval_note",
            "approval_policy_digest",
            "dataset_kind",
            "performance_approved",
            "platform",
            "runtime_version",
            "schema_version",
            "source_digest",
            "timesteps",
            "workloads",
            "worst_elapsed_slowdown_ratio",
            "worst_peak_process_tree_rss_ratio",
        }
        if set(value) != required:
            raise ValueError("runtime performance evidence field closure mismatch")
        evidence = cls(
            runtime_version=_string(value["runtime_version"], field="runtime_version"),
            platform=_string(value["platform"], field="platform"),
            algorithm=_string(value["algorithm"], field="algorithm"),
            dataset_kind=_string(value["dataset_kind"], field="dataset_kind"),
            source_digest=_string(value["source_digest"], field="source_digest"),
            workloads=tuple(
                RuntimePerformanceWorkload.from_mapping(
                    _mapping(item, field="workloads[]")
                )
                for item in _sequence(value["workloads"], field="workloads")
            ),
            performance_approved=_boolean(
                value["performance_approved"], field="performance_approved"
            ),
            approval_policy_digest=_optional_string(
                value["approval_policy_digest"], field="approval_policy_digest"
            ),
            approval_note=_string(value["approval_note"], field="approval_note"),
            schema_version=_string(value["schema_version"], field="schema_version"),
        )
        supplied_timesteps = tuple(
            _positive_int(item, field="timesteps[]")
            for item in _sequence(value["timesteps"], field="timesteps")
        )
        if supplied_timesteps != evidence.timesteps:
            raise ValueError("runtime performance timestep summary mismatch")
        for field, actual, expected in (
            (
                "worst_elapsed_slowdown_ratio",
                _positive_float(
                    value["worst_elapsed_slowdown_ratio"],
                    field="worst_elapsed_slowdown_ratio",
                ),
                evidence.worst_elapsed_slowdown_ratio,
            ),
            (
                "worst_peak_process_tree_rss_ratio",
                _positive_float(
                    value["worst_peak_process_tree_rss_ratio"],
                    field="worst_peak_process_tree_rss_ratio",
                ),
                evidence.worst_peak_process_tree_rss_ratio,
            ),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{field} summary mismatch")
        return evidence


@dataclass(frozen=True, slots=True)
class RuntimePerformanceApprovalPolicy:
    """Explicit threshold set which cannot authorize until externally reviewed."""

    max_elapsed_slowdown_ratio: float
    max_peak_process_tree_rss_ratio: float
    minimum_workloads: int
    minimum_max_timesteps: int
    reviewed: bool
    review_reference: str | None
    schema_version: str = RUNTIME_PERFORMANCE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        _positive_float(
            self.max_elapsed_slowdown_ratio,
            field="max_elapsed_slowdown_ratio",
        )
        _positive_float(
            self.max_peak_process_tree_rss_ratio,
            field="max_peak_process_tree_rss_ratio",
        )
        _positive_int(self.minimum_workloads, field="minimum_workloads")
        _positive_int(self.minimum_max_timesteps, field="minimum_max_timesteps")
        _boolean(self.reviewed, field="reviewed")
        if self.reviewed:
            _string(self.review_reference, field="review_reference")
        elif self.review_reference is not None:
            raise ValueError(
                "unreviewed performance policy cannot have review reference"
            )
        if self.schema_version != RUNTIME_PERFORMANCE_POLICY_SCHEMA:
            raise ValueError("unsupported runtime performance policy schema")

    @property
    def digest(self) -> str:
        return content_digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "max_elapsed_slowdown_ratio": self.max_elapsed_slowdown_ratio,
            "max_peak_process_tree_rss_ratio": self.max_peak_process_tree_rss_ratio,
            "minimum_max_timesteps": self.minimum_max_timesteps,
            "minimum_workloads": self.minimum_workloads,
            "review_reference": self.review_reference,
            "reviewed": self.reviewed,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class RuntimePerformanceApprovalDecision:
    """Deterministic evaluation of evidence against one explicit policy."""

    approved: bool
    reasons: tuple[str, ...]
    evidence_digest: str
    policy_digest: str


def assess_runtime_performance(
    *,
    evidence: RuntimePerformanceEvidence,
    policy: RuntimePerformanceApprovalPolicy,
) -> RuntimePerformanceApprovalDecision:
    """Evaluate explicit thresholds without mutating or self-approving evidence."""

    reasons: list[str] = []
    if not policy.reviewed:
        reasons.append("approval_policy_not_reviewed")
    else:
        if len(evidence.workloads) < policy.minimum_workloads:
            reasons.append("insufficient_workload_count")
        if max(evidence.timesteps) < policy.minimum_max_timesteps:
            reasons.append("insufficient_max_timesteps")
        if evidence.worst_elapsed_slowdown_ratio > policy.max_elapsed_slowdown_ratio:
            reasons.append("elapsed_slowdown_ratio_exceeded")
        if (
            evidence.worst_peak_process_tree_rss_ratio
            > policy.max_peak_process_tree_rss_ratio
        ):
            reasons.append("peak_process_tree_rss_ratio_exceeded")
    return RuntimePerformanceApprovalDecision(
        approved=not reasons,
        reasons=tuple(reasons),
        evidence_digest=evidence.digest,
        policy_digest=policy.digest,
    )


__all__ = [
    "RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA",
    "RUNTIME_PERFORMANCE_POLICY_SCHEMA",
    "RuntimePerformanceApprovalDecision",
    "RuntimePerformanceApprovalPolicy",
    "RuntimePerformanceEvidence",
    "RuntimePerformanceMeasurement",
    "RuntimePerformanceWorkload",
    "assess_runtime_performance",
]
