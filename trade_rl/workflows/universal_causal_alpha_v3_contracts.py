"""Immutable evidence contracts for the research-only causal alpha V3 runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3Candidate

_REPLAY_SCHEMA: Final = "causal_alpha_v3_replay_metric_v1"
_RUN_MANIFEST_SCHEMA: Final = "causal_alpha_v3_run_manifest_v1"
_FREEZE_SCHEMA: Final = "causal_alpha_v3_candidate_freeze_v1"
_CANDIDATE_EVIDENCE_SCHEMA: Final = "causal_alpha_v3_candidate_evidence_v1"
_SELECTION_SCHEMA: Final = "causal_alpha_v3_selection_evidence_v1"
_ADMISSION_RECORD_SCHEMA: Final = "causal_alpha_v3_admission_record_v1"
_PACKAGE_SCHEMA: Final = "universal_causal_alpha_v3_teacher_package_v1"


def _finite(value: float, *, field: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


def _non_negative_count(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _reason_counts(
    value: tuple[tuple[str, int], ...], *, field: str
) -> tuple[tuple[str, int], ...]:
    resolved = tuple((str(reason), int(count)) for reason, count in value)
    if any(not reason or count < 0 for reason, count in resolved):
        raise ValueError(f"{field} contains an invalid reason count")
    if len({reason for reason, _ in resolved}) != len(resolved):
        raise ValueError(f"{field} contains duplicate reasons")
    if tuple(sorted(resolved)) != resolved:
        raise ValueError(f"{field} must be sorted by reason")
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV3RunManifest:
    train_symbols: tuple[str, ...]
    config_digest: str
    catalog_digest: str
    partition_digest: str
    split_manifest_digest: str
    feature_schema_digest: str
    statistics_digest: str
    generator_code_digest: str
    nested_partition_digest: str
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = _RUN_MANIFEST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols) or any(not item for item in symbols):
            raise ValueError("V3 run manifest train_symbols must be non-empty and unique")
        for field in (
            "config_digest",
            "catalog_digest",
            "partition_digest",
            "split_manifest_digest",
            "feature_schema_digest",
            "statistics_digest",
            "generator_code_digest",
            "nested_partition_digest",
        ):
            require_sha256(getattr(self, field), field=f"V3 run manifest {field}")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V3 run manifest must remain research-only and non-promotable")
        if self.schema_version != _RUN_MANIFEST_SCHEMA:
            raise ValueError("unsupported V3 run manifest schema")
        object.__setattr__(self, "train_symbols", symbols)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 run manifest digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "catalog_digest": self.catalog_digest,
            "config_digest": self.config_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "generator_code_digest": self.generator_code_digest,
            "nested_partition_digest": self.nested_partition_digest,
            "partition_digest": self.partition_digest,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "schema_version": self.schema_version,
            "split_manifest_digest": self.split_manifest_digest,
            "statistics_digest": self.statistics_digest,
            "train_symbols": self.train_symbols,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV3CandidateFreeze:
    run_manifest_digest: str
    config_digest: str
    generator_code_digest: str
    nested_partition_digest: str
    candidate_digests: tuple[str, ...]
    candidate_semantic_digests: tuple[str, ...]
    fit_config_digests: tuple[str, ...]
    signal_evidence_digests: tuple[str, ...]
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = _FREEZE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field in (
            "run_manifest_digest",
            "config_digest",
            "generator_code_digest",
            "nested_partition_digest",
        ):
            require_sha256(getattr(self, field), field=f"V3 freeze {field}")
        collections = (
            self.candidate_digests,
            self.candidate_semantic_digests,
            self.fit_config_digests,
            self.signal_evidence_digests,
        )
        if not self.candidate_digests:
            raise ValueError("V3 freeze requires at least one candidate")
        if len(self.candidate_digests) != len(self.candidate_semantic_digests):
            raise ValueError("V3 freeze candidate identity arrays must align")
        for values in collections:
            for value in values:
                require_sha256(value, field="V3 freeze digest member")
        if len(set(self.candidate_digests)) != len(self.candidate_digests):
            raise ValueError("V3 freeze candidates are duplicated")
        if len(set(self.candidate_semantic_digests)) != len(self.candidate_semantic_digests):
            raise ValueError("V3 freeze candidate semantics are duplicated")
        if len(set(self.fit_config_digests)) != len(self.fit_config_digests):
            raise ValueError("V3 freeze fit configs are duplicated")
        if len(self.fit_config_digests) != len(self.signal_evidence_digests):
            raise ValueError("V3 freeze signal evidence must cover each fit config")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V3 freeze must remain research-only and non-promotable")
        if self.schema_version != _FREEZE_SCHEMA:
            raise ValueError("unsupported V3 freeze schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 freeze digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_digests": self.candidate_digests,
            "candidate_semantic_digests": self.candidate_semantic_digests,
            "config_digest": self.config_digest,
            "fit_config_digests": self.fit_config_digests,
            "generator_code_digest": self.generator_code_digest,
            "nested_partition_digest": self.nested_partition_digest,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "signal_evidence_digests": self.signal_evidence_digests,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ReplayMetric:
    run_manifest_digest: str
    freeze_digest: str
    candidate_digest: str
    symbol: str
    episode_index: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
    target_path_digest: str
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    trade_count: int
    submitted_change_count: int
    sign_flip_count: int
    liquidity_deleveraging_count: int
    execution_rejection_reason_counts: tuple[tuple[str, int], ...]
    risk_projection_reason_counts: tuple[tuple[str, int], ...]
    target_reason_counts: tuple[tuple[str, int], ...]
    hard_risk_violation: bool
    schema_version: str = _REPLAY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field in (
            "run_manifest_digest",
            "freeze_digest",
            "candidate_digest",
            "contract_digest",
            "fit_digest",
            "forecast_digest",
            "target_path_digest",
        ):
            require_sha256(getattr(self, field), field=f"V3 replay {field}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V3 replay symbol must be non-empty")
        _non_negative_count(self.episode_index, field="V3 replay episode_index")
        for field in (
            "gross_return",
            "net_return",
            "turnover_per_day",
            "total_execution_cost",
        ):
            _finite(getattr(self, field), field=f"V3 replay {field}")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("V3 replay turnover/cost must be non-negative")
        for field in (
            "trade_count",
            "submitted_change_count",
            "sign_flip_count",
            "liquidity_deleveraging_count",
        ):
            _non_negative_count(getattr(self, field), field=f"V3 replay {field}")
        for field in (
            "execution_rejection_reason_counts",
            "risk_projection_reason_counts",
            "target_reason_counts",
        ):
            object.__setattr__(
                self,
                field,
                _reason_counts(getattr(self, field), field=f"V3 replay {field}"),
            )
        if not isinstance(self.hard_risk_violation, bool):
            raise ValueError("V3 replay hard_risk_violation must be boolean")
        if self.schema_version != _REPLAY_SCHEMA:
            raise ValueError("unsupported V3 replay metric schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 replay metric digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.candidate_digest, self.symbol, self.episode_index)

    @property
    def unexplained_execution_rejection_count(self) -> int:
        return sum(count for _, count in self.execution_rejection_reason_counts)

    def irrecoverably_rejected(self, thresholds: object) -> bool:
        return bool(
            self.hard_risk_violation
            or self.net_return
            < float(getattr(thresholds, "minimum_symbol_episode_net_return"))
            or self.unexplained_execution_rejection_count
            > int(getattr(thresholds, "maximum_unexplained_execution_rejections"))
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_digest": self.candidate_digest,
            "contract_digest": self.contract_digest,
            "episode_index": self.episode_index,
            "execution_rejection_reason_counts": self.execution_rejection_reason_counts,
            "fit_digest": self.fit_digest,
            "forecast_digest": self.forecast_digest,
            "freeze_digest": self.freeze_digest,
            "gross_return": self.gross_return,
            "hard_risk_violation": self.hard_risk_violation,
            "liquidity_deleveraging_count": self.liquidity_deleveraging_count,
            "net_return": self.net_return,
            "risk_projection_reason_counts": self.risk_projection_reason_counts,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "sign_flip_count": self.sign_flip_count,
            "submitted_change_count": self.submitted_change_count,
            "symbol": self.symbol,
            "target_path_digest": self.target_path_digest,
            "target_reason_counts": self.target_reason_counts,
            "total_execution_cost": self.total_execution_cost,
            "trade_count": self.trade_count,
            "turnover_per_day": self.turnover_per_day,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> CausalAlphaV3ReplayMetric:
        return cls(
            run_manifest_digest=str(raw["run_manifest_digest"]),
            freeze_digest=str(raw["freeze_digest"]),
            candidate_digest=str(raw["candidate_digest"]),
            symbol=str(raw["symbol"]),
            episode_index=int(raw["episode_index"]),
            contract_digest=str(raw["contract_digest"]),
            fit_digest=str(raw["fit_digest"]),
            forecast_digest=str(raw["forecast_digest"]),
            target_path_digest=str(raw["target_path_digest"]),
            gross_return=float(raw["gross_return"]),
            net_return=float(raw["net_return"]),
            turnover_per_day=float(raw["turnover_per_day"]),
            total_execution_cost=float(raw["total_execution_cost"]),
            trade_count=int(raw["trade_count"]),
            submitted_change_count=int(raw["submitted_change_count"]),
            sign_flip_count=int(raw["sign_flip_count"]),
            liquidity_deleveraging_count=int(raw["liquidity_deleveraging_count"]),
            execution_rejection_reason_counts=tuple(
                (str(reason), int(count))
                for reason, count in raw["execution_rejection_reason_counts"]  # type: ignore[union-attr]
            ),
            risk_projection_reason_counts=tuple(
                (str(reason), int(count))
                for reason, count in raw["risk_projection_reason_counts"]  # type: ignore[union-attr]
            ),
            target_reason_counts=tuple(
                (str(reason), int(count))
                for reason, count in raw["target_reason_counts"]  # type: ignore[union-attr]
            ),
            hard_risk_violation=bool(raw["hard_risk_violation"]),
            schema_version=str(raw["schema_version"]),
            digest=str(raw["artifact_digest"]),
        )


@dataclass(frozen=True, slots=True)
class CausalAlphaV3CandidateEvidence:
    candidate: CausalAlphaV3Candidate
    episode_metrics: tuple[CausalAlphaV3ReplayMetric, ...]
    lower_tail_net_return: float
    mean_gross_return: float
    mean_net_return: float
    turnover_per_day: float
    total_execution_cost: float
    positive_gross_episode_fraction: float
    total_trade_count: int
    unexplained_execution_rejection_count: int
    hard_risk_violation: bool
    admissible: bool
    rejection_reasons: tuple[str, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CausalAlphaV3Candidate):
            raise TypeError("V3 candidate evidence candidate is invalid")
        metrics = tuple(self.episode_metrics)
        if not metrics or any(item.candidate_digest != self.candidate.digest for item in metrics):
            raise ValueError("V3 candidate evidence metrics do not match candidate")
        if len({item.identity for item in metrics}) != len(metrics):
            raise ValueError("V3 candidate evidence metrics are duplicated")
        for field in (
            "lower_tail_net_return",
            "mean_gross_return",
            "mean_net_return",
            "turnover_per_day",
            "total_execution_cost",
            "positive_gross_episode_fraction",
        ):
            _finite(getattr(self, field), field=f"V3 candidate evidence {field}")
        if not 0.0 <= self.positive_gross_episode_fraction <= 1.0:
            raise ValueError("V3 positive gross fraction must be within [0, 1]")
        _non_negative_count(self.total_trade_count, field="V3 candidate total_trade_count")
        _non_negative_count(
            self.unexplained_execution_rejection_count,
            field="V3 candidate unexplained_execution_rejection_count",
        )
        reasons = tuple(self.rejection_reasons)
        if self.admissible == bool(reasons):
            raise ValueError("V3 candidate admissibility and rejection reasons disagree")
        object.__setattr__(self, "episode_metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 candidate evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "admissible": self.admissible,
            "candidate_digest": self.candidate.digest,
            "episode_metric_digests": tuple(item.digest for item in self.episode_metrics),
            "hard_risk_violation": self.hard_risk_violation,
            "lower_tail_net_return": self.lower_tail_net_return,
            "mean_gross_return": self.mean_gross_return,
            "mean_net_return": self.mean_net_return,
            "positive_gross_episode_fraction": self.positive_gross_episode_fraction,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": _CANDIDATE_EVIDENCE_SCHEMA,
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
    freeze_digest: str
    promotion_eligible: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        values = tuple(self.candidates)
        if not values or len({item.candidate.digest for item in values}) != len(values):
            raise ValueError("V3 selection candidate evidence is invalid")
        require_sha256(self.selected_candidate_digest, field="V3 selected candidate")
        require_sha256(self.freeze_digest, field="V3 selection freeze_digest")
        selected = tuple(
            item for item in values if item.candidate.digest == self.selected_candidate_digest
        )
        if len(selected) != 1 or not selected[0].admissible:
            raise ValueError("V3 selected candidate is not uniquely admissible")
        if self.promotion_eligible:
            raise ValueError("V3 selection evidence cannot be promotion eligible")
        object.__setattr__(self, "candidates", values)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 selection evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_evidence_digests": tuple(item.digest for item in self.candidates),
            "freeze_digest": self.freeze_digest,
            "promotion_eligible": self.promotion_eligible,
            "schema_version": _SELECTION_SCHEMA,
            "selected_candidate_digest": self.selected_candidate_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV3AdmissionRecord:
    run_manifest_digest: str
    freeze_digest: str
    selection_digest: str
    selected_candidate_digest: str
    symbol: str
    contract_digest: str
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    trade_count: int
    maximum_drawdown: float
    digest: str = ""

    def __post_init__(self) -> None:
        for field in (
            "run_manifest_digest",
            "freeze_digest",
            "selection_digest",
            "selected_candidate_digest",
            "contract_digest",
        ):
            require_sha256(getattr(self, field), field=f"V3 admission {field}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V3 admission symbol must be non-empty")
        for field in (
            "gross_return",
            "net_return",
            "turnover_per_day",
            "total_execution_cost",
            "maximum_drawdown",
        ):
            _finite(getattr(self, field), field=f"V3 admission {field}")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("V3 admission turnover/cost must be non-negative")
        _non_negative_count(self.trade_count, field="V3 admission trade_count")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 admission record digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_digest": self.contract_digest,
            "freeze_digest": self.freeze_digest,
            "gross_return": self.gross_return,
            "maximum_drawdown": self.maximum_drawdown,
            "net_return": self.net_return,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": _ADMISSION_RECORD_SCHEMA,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selection_digest": self.selection_digest,
            "symbol": self.symbol,
            "total_execution_cost": self.total_execution_cost,
            "trade_count": self.trade_count,
            "turnover_per_day": self.turnover_per_day,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaV3TeacherPackage:
    train_symbols: tuple[str, ...]
    batches: Mapping[str, EpisodeOracleBatch]
    run_manifest_digest: str
    freeze_digest: str
    selection_digest: str
    teacher_admission_digest: str
    selected_candidate_digest: str
    generator_code_digest: str
    teacher_admission_passed: bool = True
    research_only: bool = True
    promotion_eligible: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        batches = dict(self.batches)
        if not symbols or len(set(symbols)) != len(symbols) or set(batches) != set(symbols):
            raise ValueError("V3 teacher package batch scope must match train_symbols")
        if any(not isinstance(batches[symbol], EpisodeOracleBatch) for symbol in symbols):
            raise TypeError("V3 teacher package contains an invalid episode batch")
        for field in (
            "run_manifest_digest",
            "freeze_digest",
            "selection_digest",
            "teacher_admission_digest",
            "selected_candidate_digest",
            "generator_code_digest",
        ):
            require_sha256(getattr(self, field), field=f"V3 teacher package {field}")
        if not self.teacher_admission_passed:
            raise ValueError("V3 teacher package requires passed teacher admission")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V3 teacher package must remain research-only and non-promotable")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "batches", batches)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 teacher package digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "batch_digests": tuple(
                (symbol, self.batches[symbol].digest) for symbol in self.train_symbols
            ),
            "freeze_digest": self.freeze_digest,
            "generator_code_digest": self.generator_code_digest,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": _PACKAGE_SCHEMA,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selection_digest": self.selection_digest,
            "teacher_admission_digest": self.teacher_admission_digest,
            "teacher_admission_passed": self.teacher_admission_passed,
            "train_symbols": self.train_symbols,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = [
    "CausalAlphaV3AdmissionRecord",
    "CausalAlphaV3CandidateEvidence",
    "CausalAlphaV3CandidateFreeze",
    "CausalAlphaV3ReplayMetric",
    "CausalAlphaV3RunManifest",
    "CausalAlphaV3SelectionEvidence",
    "UniversalCausalAlphaV3TeacherPackage",
]
