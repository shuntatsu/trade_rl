"""Immutable contracts and evidence for the Universal causal alpha teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
    CausalAlphaTeacherAdmissionEvidence,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)

_CAUSAL_ALPHA_EPISODE_PARTITION_SCHEMA = "universal_causal_alpha_episode_partition_v1"
_CAUSAL_ALPHA_SYMBOL_SAMPLES_SCHEMA = "universal_causal_alpha_symbol_samples_v1"
_CAUSAL_ALPHA_EXPANDING_FIT_SCHEMA = "universal_causal_alpha_expanding_fit_v1"
_CAUSAL_ALPHA_BATCH_EVIDENCE_SCHEMA = "universal_causal_alpha_batch_evidence_v1"


def _readonly(value: object, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CausalAlphaEpisodePartition:
    """Chronological selection episodes plus one untouched latest holdout."""

    contracts: tuple[OracleEpisodeContract, ...]
    selection_contracts: tuple[OracleEpisodeContract, ...]
    holdout_contract: OracleEpisodeContract
    train_start: int
    train_stop: int
    digest: str = ""

    def __post_init__(self) -> None:
        contracts = tuple(self.contracts)
        selection = tuple(self.selection_contracts)
        if len(contracts) < 2 or selection != contracts[:-1]:
            raise ValueError(
                "causal alpha partition requires selection episodes and one holdout"
            )
        if self.holdout_contract != contracts[-1]:
            raise ValueError("causal alpha holdout must be the latest complete episode")
        if self.train_start < 0 or self.train_stop <= self.train_start:
            raise ValueError("causal alpha partition train range is invalid")
        if tuple(contract.episode_index for contract in contracts) != tuple(
            range(len(contracts))
        ):
            raise ValueError("causal alpha episode indices must be chronological")
        dataset_ids = {contract.dataset_id for contract in contracts}
        if len(dataset_ids) != 1:
            raise ValueError("causal alpha episode dataset identity drifted")
        for previous, current in zip(contracts[:-1], contracts[1:], strict=True):
            if previous.start >= current.start or previous.stop > current.start:
                raise ValueError("causal alpha chronological episodes overlap")
        if selection[-1].stop > self.holdout_contract.start:
            raise ValueError("selection episode support crosses the holdout boundary")
        expected = content_digest(
            {
                "contracts": tuple(contract.digest for contract in contracts),
                "holdout_contract": self.holdout_contract.digest,
                "schema_version": _CAUSAL_ALPHA_EPISODE_PARTITION_SCHEMA,
                "train_start": self.train_start,
                "train_stop": self.train_stop,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha episode partition digest mismatch")
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "selection_contracts", selection)
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaSymbolSamples:
    """One train-symbol causal feature/label table with explicit realization times."""

    symbol: str
    dataset_id: str
    feature_names: tuple[str, ...]
    feature_schema_digest: str
    context_digest: str
    reference_equity_mode: str
    reference_equity: float
    decision_indices: np.ndarray
    features: np.ndarray
    feature_available: np.ndarray
    labels_24h: np.ndarray
    label_end_indices_24h: np.ndarray
    labels_72h: np.ndarray
    label_end_indices_72h: np.ndarray
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("causal alpha sample symbol must be non-empty")
        for field, value in (
            ("dataset_id", self.dataset_id),
            ("feature_schema_digest", self.feature_schema_digest),
            ("context_digest", self.context_digest),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{field} must be a SHA-256 digest")
        if self.reference_equity_mode != "initial_capital":
            raise ValueError(
                "causal alpha reference_equity_mode must be initial_capital"
            )
        if not np.isfinite(self.reference_equity) or self.reference_equity <= 0.0:
            raise ValueError("causal alpha reference_equity must be positive")
        names = tuple(self.feature_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError("causal alpha sample feature names must be unique")
        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        features = _readonly(self.features, dtype=np.float64)
        available = _readonly(self.feature_available, dtype=np.bool_)
        labels_24h = _readonly(self.labels_24h, dtype=np.float64).reshape(-1)
        labels_72h = _readonly(self.labels_72h, dtype=np.float64).reshape(-1)
        ends_24h = _readonly(self.label_end_indices_24h, dtype=np.int64).reshape(-1)
        ends_72h = _readonly(self.label_end_indices_72h, dtype=np.int64).reshape(-1)
        rows = decisions.size
        if rows == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError(
                "causal alpha decision indices must be strictly increasing"
            )
        if features.shape != (rows, len(names)) or available.shape != features.shape:
            raise ValueError("causal alpha feature arrays are not schema aligned")
        if not np.isfinite(features).all():
            raise ValueError("causal alpha features must be finite")
        for field, labels, ends in (
            ("24h", labels_24h, ends_24h),
            ("72h", labels_72h, ends_72h),
        ):
            if labels.shape != (rows,) or ends.shape != (rows,):
                raise ValueError(f"causal alpha {field} labels are not sample aligned")
            valid = ends >= 0
            if np.any(valid & ~np.isfinite(labels)):
                raise ValueError(f"causal alpha {field} realized labels must be finite")
            if np.any(~valid & np.isfinite(labels)):
                raise ValueError(
                    f"causal alpha {field} unavailable labels require non-finite values"
                )
        expected = content_and_arrays_digest(
            {
                "context_digest": self.context_digest,
                "dataset_id": self.dataset_id,
                "feature_names": names,
                "feature_schema_digest": self.feature_schema_digest,
                "reference_equity": float(self.reference_equity),
                "reference_equity_mode": self.reference_equity_mode,
                "schema_version": _CAUSAL_ALPHA_SYMBOL_SAMPLES_SCHEMA,
                "symbol": self.symbol,
            },
            (
                ("decision_indices", decisions),
                ("features", features),
                ("feature_available", available),
                ("labels_24h", labels_24h),
                ("label_end_indices_24h", ends_24h),
                ("labels_72h", labels_72h),
                ("label_end_indices_72h", ends_72h),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha symbol sample digest mismatch")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_available", available)
        object.__setattr__(self, "labels_24h", labels_24h)
        object.__setattr__(self, "labels_72h", labels_72h)
        object.__setattr__(self, "label_end_indices_24h", ends_24h)
        object.__setattr__(self, "label_end_indices_72h", ends_72h)
        object.__setattr__(self, "digest", expected)

    def prediction_inputs_for_decisions(
        self, decision_indices: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        requested = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
        if (
            requested.size == 0
            or np.any(requested < 0)
            or np.any(np.diff(requested) <= 0)
        ):
            raise ValueError(
                "causal alpha prediction decisions must be strictly increasing"
            )
        positions = np.searchsorted(self.decision_indices, requested)
        present = positions < self.decision_indices.size
        matched = np.zeros(requested.shape, dtype=np.bool_)
        if np.any(present):
            matched[present] = (
                self.decision_indices[positions[present]] == requested[present]
            )
        width = len(self.feature_names)
        features = np.zeros((requested.size, width), dtype=np.float64)
        availability = np.zeros((requested.size, width), dtype=np.bool_)
        if np.any(matched):
            source = positions[matched]
            features[matched] = self.features[source]
            availability[matched] = self.feature_available[source]
        return features, availability, matched

    def features_for_decisions(self, decision_indices: object) -> np.ndarray:
        features, availability, present = self.prediction_inputs_for_decisions(
            decision_indices
        )
        if not np.all(present):
            raise ValueError(
                "causal alpha prediction decisions are absent from samples"
            )
        if not np.all(availability):
            raise ValueError("causal alpha prediction features are unavailable")
        return features


@dataclass(frozen=True, slots=True)
class CausalAlphaExpandingFit:
    train_symbols: tuple[str, ...]
    knowledge_cutoff: int
    model_24h: CausalAlphaRidgeModel
    model_72h: CausalAlphaRidgeModel
    sample_count_24h: int
    sample_count_72h: int
    max_label_end_24h: int
    max_label_end_72h: int
    sample_scope_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.train_symbols or len(set(self.train_symbols)) != len(
            self.train_symbols
        ):
            raise ValueError("causal alpha fit train_symbols must be unique")
        if self.knowledge_cutoff <= 0:
            raise ValueError("causal alpha knowledge cutoff must be positive")
        for field, count in (
            ("sample_count_24h", self.sample_count_24h),
            ("sample_count_72h", self.sample_count_72h),
        ):
            if count < 2:
                raise ValueError(f"{field} must contain fitted samples")
        if self.max_label_end_24h >= self.knowledge_cutoff:
            raise ValueError("24h fit crosses the causal knowledge cutoff")
        if self.max_label_end_72h >= self.knowledge_cutoff:
            raise ValueError("72h fit crosses the causal knowledge cutoff")
        expected = content_digest(
            {
                "knowledge_cutoff": self.knowledge_cutoff,
                "max_label_end_24h": self.max_label_end_24h,
                "max_label_end_72h": self.max_label_end_72h,
                "model_24h_digest": self.model_24h.digest,
                "model_72h_digest": self.model_72h.digest,
                "sample_count_24h": self.sample_count_24h,
                "sample_count_72h": self.sample_count_72h,
                "sample_scope_digest": self.sample_scope_digest,
                "schema_version": _CAUSAL_ALPHA_EXPANDING_FIT_SCHEMA,
                "train_symbols": self.train_symbols,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha expanding fit digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        def compact_model(model: CausalAlphaRidgeModel) -> dict[str, object]:
            payload = dict(model.to_payload())
            payload.pop("eligible_indices", None)
            payload["artifact_digest"] = model.digest
            return payload

        return {
            "artifact_digest": self.digest,
            "knowledge_cutoff": self.knowledge_cutoff,
            "max_label_end_24h": self.max_label_end_24h,
            "max_label_end_72h": self.max_label_end_72h,
            "model_24h": compact_model(self.model_24h),
            "model_72h": compact_model(self.model_72h),
            "sample_count_24h": self.sample_count_24h,
            "sample_count_72h": self.sample_count_72h,
            "sample_scope_digest": self.sample_scope_digest,
            "schema_version": _CAUSAL_ALPHA_EXPANDING_FIT_SCHEMA,
            "train_symbols": list(self.train_symbols),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaPredictionDiagnostics:
    sample_count: int
    pearson_correlation: float | None
    directional_accuracy: float
    prediction_mean: float
    prediction_std: float
    prediction_min: float
    prediction_max: float
    prediction_quantiles: tuple[float, float, float, float, float, float, float]
    digest: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
        ):
            raise ValueError(
                "causal alpha prediction diagnostics need positive samples"
            )
        if self.pearson_correlation is not None and (
            not math.isfinite(self.pearson_correlation)
            or not -1.0 <= self.pearson_correlation <= 1.0
        ):
            raise ValueError("causal alpha prediction correlation is invalid")
        if (
            not math.isfinite(self.directional_accuracy)
            or not 0.0 <= self.directional_accuracy <= 1.0
        ):
            raise ValueError("causal alpha directional accuracy is invalid")
        scalars = (
            self.prediction_mean,
            self.prediction_std,
            self.prediction_min,
            self.prediction_max,
            *self.prediction_quantiles,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("causal alpha prediction distribution is non-finite")
        if self.prediction_std < 0.0 or self.prediction_min > self.prediction_max:
            raise ValueError("causal alpha prediction distribution is invalid")
        if len(self.prediction_quantiles) != 7 or any(
            left > right
            for left, right in zip(
                self.prediction_quantiles, self.prediction_quantiles[1:], strict=False
            )
        ):
            raise ValueError("causal alpha prediction quantiles are invalid")
        payload = self._payload_without_digest()
        expected = content_digest(payload)
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha prediction diagnostics digest mismatch")
        object.__setattr__(self, "digest", expected)

    def _payload_without_digest(self) -> dict[str, object]:
        quantiles = self.prediction_quantiles
        return {
            "directional_accuracy": self.directional_accuracy,
            "pearson_correlation": self.pearson_correlation,
            "prediction_max": self.prediction_max,
            "prediction_mean": self.prediction_mean,
            "prediction_min": self.prediction_min,
            "prediction_quantiles": {
                "p00": quantiles[0],
                "p10": quantiles[1],
                "p25": quantiles[2],
                "p50": quantiles[3],
                "p75": quantiles[4],
                "p90": quantiles[5],
                "p100": quantiles[6],
            },
            "prediction_std": self.prediction_std,
            "sample_count": self.sample_count,
            "schema_version": "causal_alpha_prediction_diagnostics_v1",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_digest(), "artifact_digest": self.digest}


@dataclass(frozen=True, slots=True)
class CausalAlphaEpisodeEvidence:
    episode_index: int
    scope: str
    knowledge_cutoff: int
    initial_weight: float
    fit_digest: str
    fit_sample_count_24h: int
    fit_sample_count_72h: int
    max_label_end_24h: int
    max_label_end_72h: int
    prediction_24h: CausalAlphaPredictionDiagnostics
    prediction_72h: CausalAlphaPredictionDiagnostics
    decision_count: int
    actionable_decision_count: int
    submitted_change_count: int
    suppressed_change_count: int
    sign_flip_count: int
    target_path_digest: str
    prediction_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        if self.scope not in {"selection", "holdout"}:
            raise ValueError("causal alpha episode evidence scope is invalid")
        for field in (
            "fit_sample_count_24h",
            "fit_sample_count_72h",
            "decision_count",
            "actionable_decision_count",
            "submitted_change_count",
            "suppressed_change_count",
            "sign_flip_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"causal alpha episode {field} is invalid")
        if self.fit_sample_count_24h < 2 or self.fit_sample_count_72h < 2:
            raise ValueError("causal alpha episode fit sample count is insufficient")
        if (
            self.decision_count <= 0
            or not 0 < self.actionable_decision_count <= self.decision_count
        ):
            raise ValueError("causal alpha episode decision support is invalid")
        if (
            self.max_label_end_24h >= self.knowledge_cutoff
            or self.max_label_end_72h >= self.knowledge_cutoff
        ):
            raise ValueError("causal alpha episode fit crossed knowledge cutoff")
        for field in ("fit_digest", "target_path_digest", "prediction_digest"):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"causal alpha episode {field} is invalid")
        expected = content_digest(self._payload_without_digest())
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha episode evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def _payload_without_digest(self) -> dict[str, object]:
        return {
            "actionable_decision_count": self.actionable_decision_count,
            "decision_count": self.decision_count,
            "episode_index": self.episode_index,
            "fit_digest": self.fit_digest,
            "fit_sample_count_24h": self.fit_sample_count_24h,
            "fit_sample_count_72h": self.fit_sample_count_72h,
            "initial_weight": self.initial_weight,
            "knowledge_cutoff": self.knowledge_cutoff,
            "max_label_end_24h": self.max_label_end_24h,
            "max_label_end_72h": self.max_label_end_72h,
            "prediction_24h": self.prediction_24h.to_payload(),
            "prediction_72h": self.prediction_72h.to_payload(),
            "prediction_digest": self.prediction_digest,
            "schema_version": "causal_alpha_episode_evidence_v2",
            "scope": self.scope,
            "sign_flip_count": self.sign_flip_count,
            "submitted_change_count": self.submitted_change_count,
            "suppressed_change_count": self.suppressed_change_count,
            "target_path_digest": self.target_path_digest,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_digest(), "artifact_digest": self.digest}


@dataclass(frozen=True, slots=True)
class CausalAlphaBatchEvidence:
    symbol: str
    train_symbols: tuple[str, ...]
    partition_digest: str
    sample_scope_digest: str
    ridge_config_digest: str
    controller_config_digest: str
    fits: Mapping[str, CausalAlphaExpandingFit]
    episodes: tuple[CausalAlphaEpisodeEvidence, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        fits = dict(self.fits)
        if not episodes:
            raise ValueError("causal alpha batch evidence must contain episodes")
        referenced = {item.fit_digest for item in episodes}
        if set(fits) != referenced or any(
            digest != fit.digest for digest, fit in fits.items()
        ):
            raise ValueError(
                "causal alpha batch fit evidence does not close over episodes"
            )
        expected = content_digest(
            {
                "controller_config_digest": self.controller_config_digest,
                "episode_evidence_digests": tuple(item.digest for item in episodes),
                "fit_digests": tuple(sorted(fits)),
                "partition_digest": self.partition_digest,
                "ridge_config_digest": self.ridge_config_digest,
                "sample_scope_digest": self.sample_scope_digest,
                "schema_version": _CAUSAL_ALPHA_BATCH_EVIDENCE_SCHEMA,
                "symbol": self.symbol,
                "train_symbols": self.train_symbols,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha batch evidence digest mismatch")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "fits", fits)
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "controller_config_digest": self.controller_config_digest,
            "episodes": [item.to_payload() for item in self.episodes],
            "fits": {
                digest: self.fits[digest].to_payload() for digest in sorted(self.fits)
            },
            "partition_digest": self.partition_digest,
            "ridge_config_digest": self.ridge_config_digest,
            "sample_scope_digest": self.sample_scope_digest,
            "schema_version": _CAUSAL_ALPHA_BATCH_EVIDENCE_SCHEMA,
            "symbol": self.symbol,
            "train_symbols": list(self.train_symbols),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaCandidateConfig:
    """One member of the bounded, predeclared train-only selection grid."""

    name: str
    ridge: CausalAlphaRidgeConfig
    controller: CausalAlphaControllerConfig
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("causal alpha candidate name must be non-empty")
        if not isinstance(self.ridge, CausalAlphaRidgeConfig):
            raise TypeError("causal alpha candidate ridge config is invalid")
        if not isinstance(self.controller, CausalAlphaControllerConfig):
            raise TypeError("causal alpha candidate controller config is invalid")
        expected = content_digest(
            {
                "controller_digest": self.controller.digest,
                "name": self.name,
                "ridge_digest": self.ridge.digest,
                "schema_version": "causal_alpha_candidate_v1",
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha candidate digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaCandidateEpisodeMetrics:
    candidate_digest: str
    symbol: str
    episode_index: int
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    trade_count: int
    risk_violation: bool
    digest: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_digest, str)
            or len(self.candidate_digest) != 64
        ):
            raise ValueError("causal alpha candidate metric digest is invalid")
        if not self.symbol:
            raise ValueError("causal alpha candidate metric symbol is empty")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise ValueError("causal alpha candidate episode index is invalid")
        for field, value in (
            ("gross_return", self.gross_return),
            ("net_return", self.net_return),
            ("turnover_per_day", self.turnover_per_day),
            ("total_execution_cost", self.total_execution_cost),
        ):
            if not np.isfinite(value):
                raise ValueError(f"causal alpha {field} must be finite")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("causal alpha turnover and cost must be non-negative")
        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count < 0
        ):
            raise ValueError("causal alpha trade_count must be non-negative")
        if not isinstance(self.risk_violation, bool):
            raise TypeError("causal alpha risk_violation must be boolean")
        expected = content_digest(
            {
                "candidate_digest": self.candidate_digest,
                "episode_index": self.episode_index,
                "gross_return": self.gross_return,
                "net_return": self.net_return,
                "risk_violation": self.risk_violation,
                "schema_version": "causal_alpha_candidate_episode_metrics_v1",
                "symbol": self.symbol,
                "total_execution_cost": self.total_execution_cost,
                "trade_count": self.trade_count,
                "turnover_per_day": self.turnover_per_day,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha candidate episode metric digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaCandidateEvidence:
    candidate: CausalAlphaCandidateConfig
    episode_metrics: tuple[CausalAlphaCandidateEpisodeMetrics, ...]
    lower_tail_net_return: float
    mean_net_return: float
    turnover_per_day: float
    total_execution_cost: float
    negative_gross_episode_count: int
    total_trade_count: int
    risk_violation: bool
    admissible: bool
    rejection_reasons: tuple[str, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.episode_metrics)
        if not metrics:
            raise ValueError("causal alpha candidate evidence needs episode metrics")
        if any(item.candidate_digest != self.candidate.digest for item in metrics):
            raise ValueError("causal alpha candidate metric identity drifted")
        scopes = tuple((item.symbol, item.episode_index) for item in metrics)
        if len(set(scopes)) != len(scopes):
            raise ValueError("causal alpha candidate episode metrics are duplicated")
        for value in (
            self.lower_tail_net_return,
            self.mean_net_return,
            self.turnover_per_day,
            self.total_execution_cost,
        ):
            if not np.isfinite(value):
                raise ValueError(
                    "causal alpha candidate aggregate metric is non-finite"
                )
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError(
                "causal alpha candidate aggregate cost metrics are invalid"
            )
        if self.negative_gross_episode_count < 0 or self.total_trade_count < 0:
            raise ValueError("causal alpha candidate aggregate counts are invalid")
        if not isinstance(self.risk_violation, bool) or not isinstance(
            self.admissible, bool
        ):
            raise TypeError("causal alpha candidate gate flags must be boolean")
        reasons = tuple(self.rejection_reasons)
        if self.admissible == bool(reasons):
            raise ValueError(
                "causal alpha candidate admission reasons are inconsistent"
            )
        expected = content_digest(
            {
                "admissible": self.admissible,
                "candidate_digest": self.candidate.digest,
                "episode_metric_digests": tuple(item.digest for item in metrics),
                "lower_tail_net_return": self.lower_tail_net_return,
                "mean_net_return": self.mean_net_return,
                "negative_gross_episode_count": self.negative_gross_episode_count,
                "rejection_reasons": reasons,
                "risk_violation": self.risk_violation,
                "schema_version": "causal_alpha_candidate_evidence_v1",
                "total_execution_cost": self.total_execution_cost,
                "total_trade_count": self.total_trade_count,
                "turnover_per_day": self.turnover_per_day,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha candidate evidence digest mismatch")
        object.__setattr__(self, "episode_metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaSelectionEvidence:
    candidates: tuple[CausalAlphaCandidateEvidence, ...]
    selected_candidate_digest: str
    grid_digest: str
    holdout_episode_digests: Mapping[str, str]
    lower_tail_definition: str = "minimum_symbol_episode_net_return"
    digest: str = ""

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("causal alpha selection evidence needs candidates")
        selected = tuple(
            item
            for item in candidates
            if item.candidate.digest == self.selected_candidate_digest
        )
        if len(selected) != 1 or not selected[0].admissible:
            raise ValueError(
                "causal alpha selected candidate is not uniquely admissible"
            )
        if self.lower_tail_definition != "minimum_symbol_episode_net_return":
            raise ValueError("causal alpha lower-tail definition is unsupported")
        holdouts = dict(self.holdout_episode_digests)
        if any(not symbol or len(digest) != 64 for symbol, digest in holdouts.items()):
            raise ValueError("causal alpha holdout episode identities are invalid")
        expected = content_digest(
            {
                "candidate_evidence_digests": tuple(item.digest for item in candidates),
                "grid_digest": self.grid_digest,
                "holdout_episode_digests": holdouts,
                "lower_tail_definition": self.lower_tail_definition,
                "schema_version": "causal_alpha_selection_evidence_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha selection evidence digest mismatch")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "holdout_episode_digests", holdouts)
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "candidates": [
                {
                    "admissible": item.admissible,
                    "candidate": {
                        "controller": {
                            "digest": item.candidate.controller.digest,
                            "entry_threshold": item.candidate.controller.entry_threshold,
                            "exit_threshold": item.candidate.controller.exit_threshold,
                            "horizon_mix": CausalAlphaHorizonMix(
                                item.candidate.controller.horizon_mix
                            ).value,
                            "max_target_delta": item.candidate.controller.max_target_delta,
                            "no_trade_band": item.candidate.controller.no_trade_band,
                            "score_scale": item.candidate.controller.score_scale,
                        },
                        "digest": item.candidate.digest,
                        "name": item.candidate.name,
                        "ridge": {
                            "digest": item.candidate.ridge.digest,
                            "ridge_strength": item.candidate.ridge.ridge_strength,
                        },
                    },
                    "episode_metrics": [
                        {
                            "artifact_digest": metric.digest,
                            "episode_index": metric.episode_index,
                            "gross_return": metric.gross_return,
                            "net_return": metric.net_return,
                            "risk_violation": metric.risk_violation,
                            "symbol": metric.symbol,
                            "total_execution_cost": metric.total_execution_cost,
                            "trade_count": metric.trade_count,
                            "turnover_per_day": metric.turnover_per_day,
                        }
                        for metric in item.episode_metrics
                    ],
                    "lower_tail_net_return": item.lower_tail_net_return,
                    "mean_net_return": item.mean_net_return,
                    "negative_gross_episode_count": item.negative_gross_episode_count,
                    "rejection_reasons": list(item.rejection_reasons),
                    "risk_violation": item.risk_violation,
                    "total_execution_cost": item.total_execution_cost,
                    "total_trade_count": item.total_trade_count,
                    "turnover_per_day": item.turnover_per_day,
                }
                for item in self.candidates
            ],
            "grid_digest": self.grid_digest,
            "holdout_episode_digests": dict(self.holdout_episode_digests),
            "lower_tail_definition": self.lower_tail_definition,
            "schema_version": "causal_alpha_selection_evidence_v1",
            "selected_candidate_digest": self.selected_candidate_digest,
        }


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaTeacherPackage:
    """One immutable train-only teacher identity shared across Universal consumers."""

    train_symbols: tuple[str, ...]
    batches: Mapping[str, EpisodeOracleBatch]
    partitions: Mapping[str, CausalAlphaEpisodePartition]
    samples: Mapping[str, CausalAlphaSymbolSamples]
    selection: CausalAlphaSelectionEvidence
    teacher_admission: CausalAlphaTeacherAdmissionEvidence
    selected_candidate_digest: str
    teacher_config_digest: str
    generator_code_digest: str
    episode_hours: float
    batch_evidence: Mapping[str, CausalAlphaBatchEvidence]
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("causal alpha package train_symbols must be unique")
        batches = dict(self.batches)
        partitions = dict(self.partitions)
        samples = dict(self.samples)
        batch_evidence = dict(self.batch_evidence)
        for field, values in (
            ("batches", batches),
            ("partitions", partitions),
            ("samples", samples),
            ("batch_evidence", batch_evidence),
        ):
            if set(values) != set(symbols):
                raise ValueError(
                    f"causal alpha package {field} must exactly match train_symbols"
                )
        if self.selection.selected_candidate_digest != self.selected_candidate_digest:
            raise ValueError("causal alpha package selected candidate identity drifted")
        if not isinstance(self.teacher_admission, CausalAlphaTeacherAdmissionEvidence):
            raise TypeError("causal alpha package teacher admission is invalid")
        if tuple(metric.symbol for metric in self.teacher_admission.metrics) != symbols:
            raise ValueError(
                "causal alpha package teacher admission symbol scope drifted"
            )
        if not np.isfinite(self.episode_hours) or self.episode_hours <= 0.0:
            raise ValueError("causal alpha package episode_hours must be positive")
        for field, value in (
            ("selected_candidate_digest", self.selected_candidate_digest),
            ("teacher_config_digest", self.teacher_config_digest),
            ("generator_code_digest", self.generator_code_digest),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"causal alpha package {field} is invalid")
        for symbol in symbols:
            batch = batches[symbol]
            if batch.teacher_config_digest != self.teacher_config_digest:
                raise ValueError("causal alpha package batch teacher identity drifted")
            if len(partitions[symbol].digest) != 64:
                raise ValueError("causal alpha package partition digest is unavailable")
            if len(samples[symbol].digest) != 64:
                raise ValueError("causal alpha package sample digest is unavailable")
            if len(batch_evidence[symbol].digest) != 64:
                raise ValueError(
                    "causal alpha package batch evidence digest is unavailable"
                )
        expected = content_digest(
            {
                "batch_digests": {symbol: batches[symbol].digest for symbol in symbols},
                "batch_evidence_digests": {
                    symbol: batch_evidence[symbol].digest for symbol in symbols
                },
                "partition_digests": {
                    symbol: partitions[symbol].digest for symbol in symbols
                },
                "sample_digests": {
                    symbol: samples[symbol].digest for symbol in symbols
                },
                "episode_hours": self.episode_hours,
                "generator_code_digest": self.generator_code_digest,
                "schema_version": "universal_causal_alpha_teacher_package_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
                "selection_digest": self.selection.digest,
                "teacher_admission_digest": self.teacher_admission.digest,
                "teacher_config_digest": self.teacher_config_digest,
                "train_symbols": symbols,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher package digest mismatch")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "batches", batches)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "batch_evidence", batch_evidence)
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "batch_evidence": {
                symbol: self.batch_evidence[symbol].to_payload()
                for symbol in self.train_symbols
            },
            "episode_hours": self.episode_hours,
            "generator_code_digest": self.generator_code_digest,
            "partitions": {
                symbol: {
                    "artifact_digest": self.partitions[symbol].digest,
                    "holdout_episode_digest": self.partitions[
                        symbol
                    ].holdout_contract.digest,
                    "selection_episode_digests": [
                        contract.digest
                        for contract in self.partitions[symbol].selection_contracts
                    ],
                }
                for symbol in self.train_symbols
            },
            "samples": {
                symbol: {
                    "artifact_digest": self.samples[symbol].digest,
                    "context_digest": self.samples[symbol].context_digest,
                    "feature_schema_digest": self.samples[symbol].feature_schema_digest,
                    "reference_equity": self.samples[symbol].reference_equity,
                    "reference_equity_mode": self.samples[symbol].reference_equity_mode,
                }
                for symbol in self.train_symbols
            },
            "schema_version": "universal_causal_alpha_teacher_package_evidence_v1",
            "selected_candidate_digest": self.selected_candidate_digest,
            "selection": self.selection.to_payload(),
            "teacher_admission": self.teacher_admission.to_payload(),
            "teacher_config_digest": self.teacher_config_digest,
            "train_symbols": list(self.train_symbols),
        }


__all__ = [
    "CausalAlphaBatchEvidence",
    "CausalAlphaCandidateConfig",
    "CausalAlphaCandidateEpisodeMetrics",
    "CausalAlphaCandidateEvidence",
    "CausalAlphaEpisodeEvidence",
    "CausalAlphaEpisodePartition",
    "CausalAlphaExpandingFit",
    "CausalAlphaPredictionDiagnostics",
    "CausalAlphaSelectionEvidence",
    "CausalAlphaSymbolSamples",
    "UniversalCausalAlphaTeacherPackage",
]
