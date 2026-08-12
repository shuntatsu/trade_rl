"""Train-only chronological fitting contracts for the Universal causal alpha teacher."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.universal_features import (
    UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
    universal_feature_schema_digest_from_names,
)
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
    causal_alpha_target_path,
    combine_causal_alpha_predictions,
    fit_causal_alpha_ridge,
    forward_log_return_label,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path,
    resolve_episode_initial_weights,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding

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

    def features_for_decisions(self, decision_indices: object) -> np.ndarray:
        requested = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
        positions = np.searchsorted(self.decision_indices, requested)
        if np.any(positions >= self.decision_indices.size) or not np.array_equal(
            self.decision_indices[positions], requested
        ):
            raise ValueError(
                "causal alpha prediction decisions are absent from samples"
            )
        if not np.all(self.feature_available[positions]):
            raise ValueError("causal alpha prediction features are unavailable")
        return np.asarray(self.features[positions], dtype=np.float64)


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


@dataclass(frozen=True, slots=True)
class CausalAlphaEpisodeEvidence:
    episode_index: int
    knowledge_cutoff: int
    initial_weight: float
    fit_digest: str
    max_label_end_24h: int
    max_label_end_72h: int
    target_path_digest: str
    prediction_digest: str


@dataclass(frozen=True, slots=True)
class CausalAlphaBatchEvidence:
    symbol: str
    train_symbols: tuple[str, ...]
    partition_digest: str
    sample_scope_digest: str
    ridge_config_digest: str
    controller_config_digest: str
    episodes: tuple[CausalAlphaEpisodeEvidence, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("causal alpha batch evidence must contain episodes")
        expected = content_digest(
            {
                "controller_config_digest": self.controller_config_digest,
                "episodes": tuple(
                    {
                        "episode_index": item.episode_index,
                        "fit_digest": item.fit_digest,
                        "initial_weight": item.initial_weight,
                        "knowledge_cutoff": item.knowledge_cutoff,
                        "max_label_end_24h": item.max_label_end_24h,
                        "max_label_end_72h": item.max_label_end_72h,
                        "prediction_digest": item.prediction_digest,
                        "target_path_digest": item.target_path_digest,
                    }
                    for item in self.episodes
                ),
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
        object.__setattr__(self, "digest", expected)


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


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaTeacherPackage:
    """One immutable train-only teacher identity shared across Universal consumers."""

    train_symbols: tuple[str, ...]
    batches: Mapping[str, EpisodeOracleBatch]
    partitions: Mapping[str, CausalAlphaEpisodePartition]
    samples: Mapping[str, CausalAlphaSymbolSamples]
    selection: CausalAlphaSelectionEvidence
    selected_candidate_digest: str
    teacher_config_digest: str
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
        for field, value in (
            ("selected_candidate_digest", self.selected_candidate_digest),
            ("teacher_config_digest", self.teacher_config_digest),
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
                "schema_version": "universal_causal_alpha_teacher_package_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
                "selection_digest": self.selection.digest,
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


def _candidate(
    *,
    name: str,
    ridge_strength: float,
    horizon_mix: CausalAlphaHorizonMix,
    score_scale: float,
    entry_threshold: float,
    exit_threshold: float,
    no_trade_band: float,
    max_target_delta: float,
) -> CausalAlphaCandidateConfig:
    return CausalAlphaCandidateConfig(
        name=name,
        ridge=CausalAlphaRidgeConfig(ridge_strength=ridge_strength),
        controller=CausalAlphaControllerConfig(
            horizon_mix=horizon_mix,
            score_scale=score_scale,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            no_trade_band=no_trade_band,
            max_target_delta=max_target_delta,
        ),
    )


def default_causal_alpha_candidate_grid(
    risk_config: PreTradeRiskConfig,
) -> tuple[CausalAlphaCandidateConfig, ...]:
    """Return the maintained bounded one-factor-at-a-time causal teacher grid."""

    if not isinstance(risk_config, PreTradeRiskConfig):
        raise TypeError("causal alpha default grid requires PreTradeRiskConfig")
    no_trade = float(risk_config.no_trade_band)
    max_delta = min(0.25, float(risk_config.max_abs_weight))
    if max_delta <= 0.0:
        raise ValueError("causal alpha max target delta cannot be resolved")
    base = dict(
        ridge_strength=0.01,
        horizon_mix=CausalAlphaHorizonMix.EQUAL,
        score_scale=50.0,
        entry_threshold=0.003,
        exit_threshold=0.001,
        no_trade_band=no_trade,
        max_target_delta=max_delta,
    )
    variants: tuple[tuple[str, dict[str, object]], ...] = (
        ("baseline", {}),
        ("ridge-strong", {"ridge_strength": 0.1}),
        ("horizon-24h", {"horizon_mix": CausalAlphaHorizonMix.H24}),
        ("horizon-72h", {"horizon_mix": CausalAlphaHorizonMix.H72}),
        ("scale-low", {"score_scale": 25.0}),
        ("scale-high", {"score_scale": 100.0}),
        (
            "threshold-low",
            {"entry_threshold": 0.0015, "exit_threshold": 0.0005},
        ),
        (
            "threshold-high",
            {"entry_threshold": 0.006, "exit_threshold": 0.002},
        ),
        ("no-trade-low", {"no_trade_band": no_trade * 0.5}),
        (
            "no-trade-high",
            {"no_trade_band": min(float(risk_config.max_abs_weight), no_trade * 2.0)},
        ),
        ("delta-low", {"max_target_delta": max_delta * 0.5}),
        (
            "delta-high",
            {
                "max_target_delta": min(
                    float(risk_config.max_abs_weight), max_delta * 2.0
                )
            },
        ),
    )
    result: list[CausalAlphaCandidateConfig] = []
    observed: set[str] = set()
    for name, overrides in variants:
        kwargs = {**base, **overrides, "name": name}
        candidate = _candidate(**kwargs)  # type: ignore[arg-type]
        if candidate.digest not in observed:
            observed.add(candidate.digest)
            result.append(candidate)
    if len(result) < 8:
        raise ValueError("causal alpha default grid collapsed unexpectedly")
    return tuple(result)


def _train_range(
    environment: Any,
    train_range: tuple[int, int],
) -> tuple[int, int, int]:
    start, stop = train_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise ValueError("causal alpha train range is invalid")
    dataset = getattr(environment, "dataset", None)
    n_bars = getattr(dataset, "n_bars", None)
    if isinstance(n_bars, bool) or not isinstance(n_bars, int) or n_bars <= 0:
        raise ValueError("causal alpha environment dataset size is unavailable")
    minimum_start = getattr(environment, "minimum_start_index", None)
    if (
        isinstance(minimum_start, bool)
        or not isinstance(minimum_start, int)
        or minimum_start < 0
    ):
        raise ValueError("causal alpha environment minimum start is unavailable")
    effective_start = max(start, minimum_start)
    effective_stop = min(stop, n_bars)
    if effective_stop <= effective_start:
        raise ValueError("causal alpha effective train range is empty")
    return effective_start, effective_stop, n_bars


def build_chronological_episode_partition(
    environment: Any,
    *,
    train_range: tuple[int, int],
) -> CausalAlphaEpisodePartition:
    """Reserve the latest complete episode and use only earlier complete episodes."""

    if getattr(environment, "decision_bars", None) != 1:
        raise ValueError("causal alpha teacher currently requires one bar per decision")
    episode_bars = getattr(environment, "episode_bars", None)
    if (
        isinstance(episode_bars, bool)
        or not isinstance(episode_bars, int)
        or episode_bars <= 0
    ):
        raise ValueError("causal alpha episode horizon must be positive")
    dataset = getattr(environment, "dataset", None)
    dataset_id = getattr(dataset, "dataset_id", None)
    n_symbols = getattr(dataset, "n_symbols", None)
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("causal alpha dataset identity is unavailable")
    if isinstance(n_symbols, bool) or not isinstance(n_symbols, int) or n_symbols <= 0:
        raise ValueError("causal alpha dataset symbol count is unavailable")
    effective_start, effective_stop, _ = _train_range(environment, train_range)

    stride = episode_bars + 1
    latest_start = effective_stop - stride
    if latest_start < effective_start:
        raise ValueError(
            "causal alpha train range contains no complete holdout episode"
        )
    starts: list[int] = []
    cursor = latest_start
    while cursor >= effective_start:
        starts.append(cursor)
        cursor -= stride
    starts.reverse()
    if len(starts) < 2:
        raise ValueError(
            "causal alpha train range requires at least one selection episode "
            "before the holdout"
        )

    config = getattr(environment, "config", None)
    modes = tuple(getattr(config, "initial_state_modes", ()))
    if not modes or any(mode not in {"cash", "baseline"} for mode in modes):
        raise ValueError(
            "causal alpha episodes support only declared cash and baseline reset modes"
        )

    contracts: list[OracleEpisodeContract] = []
    for episode_index, contract_start in enumerate(starts):
        mode = modes[episode_index % len(modes)]
        initial_weights = resolve_episode_initial_weights(
            environment,
            mode,
            contract_start,
        )
        if initial_weights.shape != (n_symbols,):
            raise ValueError(
                "causal alpha initial weights do not match dataset symbols"
            )
        contracts.append(
            OracleEpisodeContract(
                dataset_id=dataset_id,
                episode_index=episode_index,
                start=contract_start,
                stop=contract_start + stride,
                initial_state_mode=mode,
                initial_weights=initial_weights,
            )
        )
    resolved = tuple(contracts)
    return CausalAlphaEpisodePartition(
        contracts=resolved,
        selection_contracts=resolved[:-1],
        holdout_contract=resolved[-1],
        train_start=effective_start,
        train_stop=effective_stop,
    )


def _sample_int_vector(dataset: Any, field: str, sample_count: int) -> np.ndarray:
    raw = np.asarray(getattr(dataset, field, None))
    if (
        raw.ndim != 1
        or len(raw) != sample_count
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError(f"{field} must be a sample-aligned integer vector")
    values = np.asarray(raw, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError(f"{field} must be non-negative")
    return values


def latest_complete_episode_split(
    dataset: Any,
    *,
    holdout_episode_id: int,
) -> BehaviorCloningSplit:
    """Return an explicit split whose validation set is exactly one latest episode."""

    sample_count = int(getattr(dataset, "sample_count", 0))
    if sample_count <= 0:
        raise ValueError("causal alpha teacher dataset must contain samples")
    if (
        isinstance(holdout_episode_id, bool)
        or not isinstance(holdout_episode_id, int)
        or holdout_episode_id < 0
    ):
        raise ValueError("holdout_episode_id must be non-negative")
    episode_ids = _sample_int_vector(dataset, "episode_ids", sample_count)
    decision_indices = _sample_int_vector(dataset, "decision_indices", sample_count)
    holdout_mask = episode_ids == holdout_episode_id
    if not np.any(holdout_mask):
        raise ValueError("causal alpha holdout episode is absent from the dataset")
    holdout_start = int(np.min(decision_indices[holdout_mask]))

    train_episode_ids: list[int] = []
    purged_episode_ids: list[int] = []
    for raw_episode_id in np.unique(episode_ids):
        episode_id = int(raw_episode_id)
        if episode_id == holdout_episode_id:
            continue
        mask = episode_ids == episode_id
        episode_start = int(np.min(decision_indices[mask]))
        support_stop = int(np.max(decision_indices[mask])) + 2
        if episode_start >= holdout_start:
            raise ValueError("causal alpha holdout episode must be latest")
        if support_stop <= holdout_start:
            train_episode_ids.append(episode_id)
        else:
            purged_episode_ids.append(episode_id)
    if not train_episode_ids:
        raise ValueError("causal alpha holdout leaves no BC training episodes")

    train_ids = np.asarray(sorted(train_episode_ids), dtype=np.int64)
    purged_ids = np.asarray(sorted(purged_episode_ids), dtype=np.int64)
    validation_ids = np.asarray([holdout_episode_id], dtype=np.int64)
    return BehaviorCloningSplit(
        train_indices=np.flatnonzero(np.isin(episode_ids, train_ids)),
        validation_indices=np.flatnonzero(holdout_mask),
        train_episode_ids=train_ids,
        validation_episode_ids=validation_ids,
        purged_indices=np.flatnonzero(np.isin(episode_ids, purged_ids)),
        purged_episode_ids=purged_ids,
    )


def validate_universal_causal_alpha_partitions(
    *,
    train_symbols: tuple[str, ...],
    partitions: Mapping[str, CausalAlphaEpisodePartition],
) -> dict[str, CausalAlphaEpisodePartition]:
    """Close the causal teacher episode scope over exactly the train symbols."""

    symbols = tuple(train_symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not symbol for symbol in symbols)
    ):
        raise ValueError("causal alpha train_symbols must be non-empty and unique")
    if set(partitions) != set(symbols):
        raise ValueError("causal alpha partitions must exactly match train_symbols")
    ordered: dict[str, CausalAlphaEpisodePartition] = {}
    for symbol in symbols:
        partition = partitions[symbol]
        if not isinstance(partition, CausalAlphaEpisodePartition):
            raise TypeError("causal alpha partition has an invalid type")
        if not partition.selection_contracts:
            raise ValueError("causal alpha partition has no selection episode")
        ordered[symbol] = partition
    return ordered


def _prefix_forward_label(
    dataset: Any,
    *,
    decision_index: int,
    horizon_hours: float,
    signal_delay_decisions: int,
    decision_bars: int,
    train_stop: int,
) -> tuple[float, int]:
    bars_for_hours = getattr(dataset, "bars_for_hours", None)
    if not callable(bars_for_hours):
        raise TypeError("causal alpha dataset cannot resolve label horizons")
    horizon_bars = int(bars_for_hours(horizon_hours))
    execution_start = decision_index + signal_delay_decisions * decision_bars + 1
    label_end = execution_start + horizon_bars - 1
    if execution_start >= train_stop or label_end >= train_stop:
        return float("nan"), -1
    label = forward_log_return_label(
        dataset,
        decision_index=decision_index,
        horizon_hours=horizon_hours,
        signal_delay_decisions=signal_delay_decisions,
        decision_bars=decision_bars,
    )
    if label.label_end_index != label_end:
        raise RuntimeError("causal alpha label timing drifted")
    return label.value, label.label_end_index


def build_causal_alpha_symbol_samples(
    *,
    environment: Any,
    binding: InstrumentDatasetBinding,
    instrument_context_provider: Any,
    train_range: tuple[int, int],
    feature_schema_digest: str,
) -> CausalAlphaSymbolSamples:
    """Extract one train-symbol causal table without action-dependent context."""

    if not isinstance(binding, InstrumentDatasetBinding):
        raise TypeError("causal alpha binding must be InstrumentDatasetBinding")
    if binding.split != "train":
        raise ValueError("causal alpha sample extraction requires a train binding")
    if not callable(instrument_context_provider):
        raise TypeError("causal alpha instrument context provider must be callable")
    dataset = getattr(environment, "dataset", None)
    if dataset is None:
        raise TypeError("causal alpha environment must expose its dataset")
    if tuple(getattr(dataset, "symbols", ())) != (binding.concrete_symbol,):
        raise ValueError("causal alpha dataset symbol does not match train binding")
    if getattr(dataset, "dataset_id", None) != binding.source_dataset_id:
        raise ValueError("causal alpha dataset identity does not match train binding")
    if getattr(dataset, "n_symbols", None) != 1:
        raise ValueError("causal alpha sample extraction requires one symbol")
    market_feature_names = tuple(getattr(dataset, "feature_names", ()))
    expected_schema = universal_feature_schema_digest_from_names(market_feature_names)
    if feature_schema_digest != expected_schema:
        raise ValueError("causal alpha feature schema digest does not match dataset")
    provider_schema_digest = getattr(instrument_context_provider, "schema_digest", None)
    provider_digest = getattr(instrument_context_provider, "digest", None)
    for field, value in (
        ("instrument context schema digest", provider_schema_digest),
        ("instrument context provider digest", provider_digest),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"causal alpha {field} is unavailable")
    initial_capital = float(getattr(environment, "initial_capital", np.nan))
    if not np.isfinite(initial_capital) or initial_capital <= 0.0:
        raise ValueError("causal alpha environment initial_capital must be positive")
    decision_bars = getattr(environment, "decision_bars", None)
    if (
        isinstance(decision_bars, bool)
        or not isinstance(decision_bars, int)
        or decision_bars <= 0
    ):
        raise ValueError("causal alpha decision_bars must be positive")
    config = getattr(environment, "config", None)
    signal_delay_decisions = getattr(config, "signal_delay_decisions", None)
    if signal_delay_decisions not in {0, 1}:
        raise ValueError("causal alpha signal delay must be zero or one decision")
    start, stop, _ = _train_range(environment, train_range)

    market_features = np.asarray(getattr(dataset, "features", None), dtype=np.float64)
    market_available = np.asarray(
        getattr(dataset, "feature_available", None), dtype=np.bool_
    )
    expected_market_shape = (
        int(getattr(dataset, "n_bars", 0)),
        1,
        len(market_feature_names),
    )
    if market_features.shape != expected_market_shape:
        raise ValueError("causal alpha market feature shape is invalid")
    if market_available.shape != expected_market_shape:
        raise ValueError("causal alpha market availability shape is invalid")
    if not np.isfinite(market_features).all():
        raise ValueError("causal alpha market features must be finite")
    active = np.asarray(getattr(dataset, "asset_active", None), dtype=np.bool_)
    tradable = np.asarray(getattr(dataset, "tradable", None), dtype=np.bool_)
    if active.shape != expected_market_shape[:2] or tradable.shape != active.shape:
        raise ValueError("causal alpha active/tradable masks are invalid")

    decision_values: list[int] = []
    feature_rows: list[np.ndarray] = []
    availability_rows: list[np.ndarray] = []
    labels_24h: list[float] = []
    ends_24h: list[int] = []
    labels_72h: list[float] = []
    ends_72h: list[int] = []
    for index in range(start, stop):
        if not bool(active[index, 0] and tradable[index, 0]):
            continue
        proxy = SimpleNamespace(
            dataset=dataset,
            current_index=index,
            config=config,
            hybrid=SimpleNamespace(portfolio_value=initial_capital),
        )
        context = np.asarray(
            instrument_context_provider(proxy, binding), dtype=np.float64
        )
        expected_context_shape = (1, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES))
        if context.shape != expected_context_shape or not np.isfinite(context).all():
            raise ValueError("causal alpha instrument context shape is invalid")
        decision_values.append(index)
        feature_rows.append(
            np.concatenate((market_features[index, 0], context[0]), axis=0)
        )
        availability_rows.append(
            np.concatenate(
                (
                    market_available[index, 0],
                    np.ones(context.shape[1], dtype=np.bool_),
                ),
                axis=0,
            )
        )
        label_24h, end_24h = _prefix_forward_label(
            dataset,
            decision_index=index,
            horizon_hours=24.0,
            signal_delay_decisions=int(signal_delay_decisions),
            decision_bars=decision_bars,
            train_stop=stop,
        )
        label_72h, end_72h = _prefix_forward_label(
            dataset,
            decision_index=index,
            horizon_hours=72.0,
            signal_delay_decisions=int(signal_delay_decisions),
            decision_bars=decision_bars,
            train_stop=stop,
        )
        labels_24h.append(label_24h)
        ends_24h.append(end_24h)
        labels_72h.append(label_72h)
        ends_72h.append(end_72h)
    if not decision_values:
        raise ValueError("causal alpha train range contains no active tradable samples")
    context_digest = content_digest(
        {
            "binding_instrument_descriptor_digest": binding.instrument_descriptor_digest,
            "provider_digest": provider_digest,
            "provider_schema_digest": provider_schema_digest,
            "reference_equity": initial_capital,
            "reference_equity_mode": "initial_capital",
            "schema_version": "causal_alpha_signal_context_v1",
        }
    )
    feature_names = (
        *market_feature_names,
        *UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
    )
    return CausalAlphaSymbolSamples(
        symbol=binding.concrete_symbol,
        dataset_id=binding.source_dataset_id,
        feature_names=feature_names,
        feature_schema_digest=feature_schema_digest,
        context_digest=context_digest,
        reference_equity_mode="initial_capital",
        reference_equity=initial_capital,
        decision_indices=np.asarray(decision_values, dtype=np.int64),
        features=np.asarray(feature_rows, dtype=np.float64),
        feature_available=np.asarray(availability_rows, dtype=np.bool_),
        labels_24h=np.asarray(labels_24h, dtype=np.float64),
        label_end_indices_24h=np.asarray(ends_24h, dtype=np.int64),
        labels_72h=np.asarray(labels_72h, dtype=np.float64),
        label_end_indices_72h=np.asarray(ends_72h, dtype=np.int64),
    )


def _validated_sample_scope(
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
) -> tuple[tuple[str, ...], tuple[CausalAlphaSymbolSamples, ...], str]:
    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("causal alpha train_symbols must be non-empty and unique")
    if set(samples) != set(symbols):
        raise ValueError("causal alpha samples must exactly match train_symbols")
    blocks = tuple(samples[symbol] for symbol in symbols)
    for symbol, block in zip(symbols, blocks, strict=True):
        if not isinstance(block, CausalAlphaSymbolSamples) or block.symbol != symbol:
            raise ValueError("causal alpha sample symbol identity drifted")
    names = {block.feature_names for block in blocks}
    schemas = {block.feature_schema_digest for block in blocks}
    if len(names) != 1 or len(schemas) != 1:
        raise ValueError("causal alpha sample feature schema drifted across symbols")
    scope_digest = content_digest(
        {
            "sample_digests": tuple(block.digest for block in blocks),
            "schema_version": "universal_causal_alpha_sample_scope_v1",
            "train_symbols": symbols,
        }
    )
    return symbols, blocks, scope_digest


def fit_expanding_causal_alpha_models(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    knowledge_cutoff: int,
    ridge_config: CausalAlphaRidgeConfig,
) -> CausalAlphaExpandingFit:
    """Fit both horizons on pooled train-symbol labels realized before a cutoff."""

    symbols, blocks, scope_digest = _validated_sample_scope(train_symbols, samples)
    features = np.concatenate(tuple(block.features for block in blocks), axis=0)
    available = np.concatenate(
        tuple(block.feature_available for block in blocks), axis=0
    )
    labels_24h = np.concatenate(tuple(block.labels_24h for block in blocks), axis=0)
    labels_72h = np.concatenate(tuple(block.labels_72h for block in blocks), axis=0)
    ends_24h = np.concatenate(tuple(block.label_end_indices_24h for block in blocks))
    ends_72h = np.concatenate(tuple(block.label_end_indices_72h for block in blocks))
    feature_names = blocks[0].feature_names
    model_24h = fit_causal_alpha_ridge(
        features=features,
        labels=labels_24h,
        feature_available=available,
        label_end_indices=ends_24h,
        knowledge_cutoff=knowledge_cutoff,
        feature_names=feature_names,
        config=ridge_config,
    )
    model_72h = fit_causal_alpha_ridge(
        features=features,
        labels=labels_72h,
        feature_available=available,
        label_end_indices=ends_72h,
        knowledge_cutoff=knowledge_cutoff,
        feature_names=feature_names,
        config=ridge_config,
    )
    fitted_ends_24h = ends_24h[model_24h.eligible_indices]
    fitted_ends_72h = ends_72h[model_72h.eligible_indices]
    return CausalAlphaExpandingFit(
        train_symbols=symbols,
        knowledge_cutoff=knowledge_cutoff,
        model_24h=model_24h,
        model_72h=model_72h,
        sample_count_24h=model_24h.sample_count,
        sample_count_72h=model_72h.sample_count,
        max_label_end_24h=int(np.max(fitted_ends_24h)),
        max_label_end_72h=int(np.max(fitted_ends_72h)),
        sample_scope_digest=scope_digest,
    )


def _candidate_evidence(
    candidate: CausalAlphaCandidateConfig,
    metrics: tuple[CausalAlphaCandidateEpisodeMetrics, ...],
) -> CausalAlphaCandidateEvidence:
    if not metrics:
        raise ValueError("causal alpha candidate has no selection metrics")
    net_returns = np.asarray([item.net_return for item in metrics], dtype=np.float64)
    negative_gross = sum(item.gross_return < 0.0 for item in metrics)
    total_trades = sum(item.trade_count for item in metrics)
    risk_violation = any(item.risk_violation for item in metrics)
    reasons: list[str] = []
    if negative_gross > len(metrics) / 2.0:
        reasons.append("majority_negative_gross_return")
    if total_trades == 0:
        reasons.append("no_meaningful_trades")
    if risk_violation:
        reasons.append("risk_contract_violation")
    return CausalAlphaCandidateEvidence(
        candidate=candidate,
        episode_metrics=metrics,
        lower_tail_net_return=float(np.min(net_returns)),
        mean_net_return=float(np.mean(net_returns, dtype=np.float64)),
        turnover_per_day=float(
            np.mean([item.turnover_per_day for item in metrics], dtype=np.float64)
        ),
        total_execution_cost=float(
            np.sum([item.total_execution_cost for item in metrics], dtype=np.float64)
        ),
        negative_gross_episode_count=negative_gross,
        total_trade_count=total_trades,
        risk_violation=risk_violation,
        admissible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def rank_causal_alpha_candidates(
    *,
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    metrics: Mapping[str, tuple[CausalAlphaCandidateEpisodeMetrics, ...]],
    holdout_episode_digests: Mapping[str, str] | None = None,
) -> CausalAlphaSelectionEvidence:
    """Rank a complete candidate grid without consulting causal holdout metrics."""

    candidate_values = tuple(candidates)
    if not candidate_values:
        raise ValueError("causal alpha candidate grid must be non-empty")
    digests = tuple(candidate.digest for candidate in candidate_values)
    if len(set(digests)) != len(digests):
        raise ValueError("causal alpha candidate grid contains duplicate configs")
    if set(metrics) != set(digests):
        raise ValueError("causal alpha candidate metrics must cover the complete grid")
    evidence = tuple(
        _candidate_evidence(candidate, tuple(metrics[candidate.digest]))
        for candidate in candidate_values
    )
    admissible = tuple(item for item in evidence if item.admissible)
    if not admissible:
        raise RuntimeError("no admissible causal alpha candidate")
    selected = max(
        admissible,
        key=lambda item: (
            item.lower_tail_net_return,
            item.mean_net_return,
            -item.turnover_per_day,
            -item.total_execution_cost,
        ),
    )
    grid_digest = content_digest(
        {
            "candidate_digests": digests,
            "lower_tail_definition": "minimum_symbol_episode_net_return",
            "schema_version": "causal_alpha_selection_grid_v1",
        }
    )
    return CausalAlphaSelectionEvidence(
        candidates=evidence,
        selected_candidate_digest=selected.candidate.digest,
        grid_digest=grid_digest,
        holdout_episode_digests=(
            {} if holdout_episode_digests is None else dict(holdout_episode_digests)
        ),
    )


def _causal_alpha_target_for_contract(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    contract: OracleEpisodeContract,
    candidate: CausalAlphaCandidateConfig,
) -> np.ndarray:
    fitted = fit_expanding_causal_alpha_models(
        train_symbols=train_symbols,
        samples=samples,
        knowledge_cutoff=contract.start,
        ridge_config=candidate.ridge,
    )
    block = samples[symbol]
    if contract.dataset_id != block.dataset_id:
        raise ValueError("causal alpha selection contract dataset identity drifted")
    decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
    prediction_features = block.features_for_decisions(decisions)
    prediction_24h = fitted.model_24h.predict(prediction_features)
    prediction_72h = fitted.model_72h.predict(prediction_features)
    scores = combine_causal_alpha_predictions(
        prediction_24h,
        prediction_72h,
        candidate.controller.horizon_mix,
    )
    target_path = causal_alpha_target_path(
        scores,
        config=candidate.controller,
        initial_weight=float(contract.initial_weights[0]),
    )
    return np.asarray(target_path.targets, dtype=np.float32).reshape(-1, 1)


def evaluate_causal_alpha_selection(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partitions: Mapping[str, CausalAlphaEpisodePartition],
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    environment_factories: Mapping[str, Any],
    episode_hours: float,
) -> CausalAlphaSelectionEvidence:
    """Replay only earlier selection episodes through the production environment."""

    symbols, _, _ = _validated_sample_scope(train_symbols, samples)
    partition_values = validate_universal_causal_alpha_partitions(
        train_symbols=symbols,
        partitions=partitions,
    )
    if set(environment_factories) != set(symbols):
        raise ValueError(
            "causal alpha environment factories must exactly match train_symbols"
        )
    if not np.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("causal alpha episode_hours must be finite and positive")
    episode_days = float(episode_hours) / 24.0
    by_candidate: dict[str, tuple[CausalAlphaCandidateEpisodeMetrics, ...]] = {}
    for candidate in candidates:
        records: list[CausalAlphaCandidateEpisodeMetrics] = []
        for symbol in symbols:
            factory = environment_factories[symbol]
            if not callable(factory):
                raise TypeError(
                    "causal alpha selection environment factory is not callable"
                )
            partition = partition_values[symbol]
            for contract in partition.selection_contracts:
                actions = _causal_alpha_target_for_contract(
                    symbol=symbol,
                    train_symbols=symbols,
                    samples=samples,
                    contract=contract,
                    candidate=candidate,
                )
                evaluation = evaluate_episode_action_path(
                    factory,
                    contract,
                    actions=actions,
                )
                performance = evaluation.performance
                collapse = evaluation.collapse_evidence
                records.append(
                    CausalAlphaCandidateEpisodeMetrics(
                        candidate_digest=candidate.digest,
                        symbol=symbol,
                        episode_index=contract.episode_index,
                        gross_return=float(performance.gross_return),
                        net_return=float(performance.net_return),
                        turnover_per_day=(
                            float(performance.turnover_total) / episode_days
                        ),
                        total_execution_cost=float(performance.cost_total),
                        trade_count=int(performance.trade_count),
                        risk_violation=(int(collapse.execution_rejection_count) > 0),
                    )
                )
        by_candidate[candidate.digest] = tuple(records)
    return rank_causal_alpha_candidates(
        candidates=tuple(candidates),
        metrics=by_candidate,
        holdout_episode_digests={
            symbol: partition_values[symbol].holdout_contract.digest
            for symbol in symbols
        },
    )


def build_universal_causal_alpha_teacher_package(
    *,
    train_symbols: tuple[str, ...],
    bindings: tuple[InstrumentDatasetBinding, ...],
    concrete_environment_factory: Any,
    instrument_context_provider: Any,
    fold_train_range: tuple[int, int],
    feature_schema_digest: str,
    episode_hours: float | None = None,
    candidates: tuple[CausalAlphaCandidateConfig, ...] | None = None,
) -> UniversalCausalAlphaTeacherPackage:
    """Build the causal teacher exactly once for all Universal consumers."""

    symbols = tuple(train_symbols)
    binding_values = tuple(bindings)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("causal alpha package train_symbols must be unique")
    if tuple(binding.concrete_symbol for binding in binding_values) != symbols:
        raise ValueError("causal alpha package bindings must follow train_symbols")
    if any(binding.split != "train" for binding in binding_values):
        raise ValueError("causal alpha package accepts train bindings only")
    if not callable(concrete_environment_factory):
        raise TypeError("causal alpha concrete environment factory must be callable")

    partitions: dict[str, CausalAlphaEpisodePartition] = {}
    samples: dict[str, CausalAlphaSymbolSamples] = {}
    risk_configs: list[PreTradeRiskConfig] = []
    observed_episode_hours: list[float] = []
    for symbol, binding in zip(symbols, binding_values, strict=True):
        environment = concrete_environment_factory(binding)
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("causal alpha concrete environment must be closable")
        try:
            partitions[symbol] = build_chronological_episode_partition(
                environment,
                train_range=fold_train_range,
            )
            samples[symbol] = build_causal_alpha_symbol_samples(
                environment=environment,
                binding=binding,
                instrument_context_provider=instrument_context_provider,
                train_range=fold_train_range,
                feature_schema_digest=feature_schema_digest,
            )
            risk_config = getattr(
                getattr(environment, "pre_trade_risk", None), "config", None
            )
            if not isinstance(risk_config, PreTradeRiskConfig):
                raise TypeError("causal alpha environment risk config is unavailable")
            risk_configs.append(risk_config)
            environment_episode_hours = getattr(
                getattr(environment, "config", None), "episode_hours", None
            )
            if isinstance(environment_episode_hours, bool) or not isinstance(
                environment_episode_hours, int | float
            ):
                raise ValueError(
                    "causal alpha environment episode_hours is unavailable"
                )
            observed_episode_hours.append(float(environment_episode_hours))
        finally:
            close()
    validate_universal_causal_alpha_partitions(
        train_symbols=symbols,
        partitions=partitions,
    )
    if len({content_digest(config) for config in risk_configs}) != 1:
        raise ValueError("causal alpha train-symbol risk configs differ")
    if len(set(observed_episode_hours)) != 1:
        raise ValueError("causal alpha train-symbol episode horizons differ")
    resolved_episode_hours = (
        observed_episode_hours[0] if episode_hours is None else float(episode_hours)
    )
    if not np.isfinite(resolved_episode_hours) or resolved_episode_hours <= 0.0:
        raise ValueError("causal alpha package episode_hours must be positive")
    if any(
        abs(value - resolved_episode_hours) > 1e-12 for value in observed_episode_hours
    ):
        raise ValueError(
            "causal alpha requested episode_hours differs from environment"
        )

    candidate_values = (
        default_causal_alpha_candidate_grid(risk_configs[0])
        if candidates is None
        else tuple(candidates)
    )
    if not candidate_values:
        raise ValueError("causal alpha candidate grid must be non-empty")
    binding_by_symbol = {binding.concrete_symbol: binding for binding in binding_values}
    selection = evaluate_causal_alpha_selection(
        train_symbols=symbols,
        samples=samples,
        partitions=partitions,
        candidates=candidate_values,
        environment_factories={
            symbol: partial(concrete_environment_factory, binding_by_symbol[symbol])
            for symbol in symbols
        },
        episode_hours=resolved_episode_hours,
    )
    selected_evidence = tuple(
        item
        for item in selection.candidates
        if item.candidate.digest == selection.selected_candidate_digest
    )
    if len(selected_evidence) != 1:
        raise RuntimeError("causal alpha selected candidate cannot be resolved")
    selected = selected_evidence[0].candidate
    teacher_config_digest = content_digest(
        {
            "feature_schema_digest": feature_schema_digest,
            "schema_version": "universal_causal_alpha_teacher_config_v1",
            "selected_candidate_digest": selected.digest,
            "selection_digest": selection.digest,
        }
    )
    batches: dict[str, EpisodeOracleBatch] = {}
    batch_evidence: dict[str, CausalAlphaBatchEvidence] = {}
    for symbol in symbols:
        batch, evidence = build_causal_alpha_episode_batch(
            symbol=symbol,
            train_symbols=symbols,
            samples=samples,
            partition=partitions[symbol],
            ridge_config=selected.ridge,
            controller_config=selected.controller,
            teacher_config_digest=teacher_config_digest,
        )
        batches[symbol] = batch
        batch_evidence[symbol] = evidence
    return UniversalCausalAlphaTeacherPackage(
        train_symbols=symbols,
        batches=batches,
        partitions=partitions,
        samples=samples,
        selection=selection,
        selected_candidate_digest=selected.digest,
        teacher_config_digest=teacher_config_digest,
        batch_evidence=batch_evidence,
    )


def build_causal_alpha_episode_batch(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partition: CausalAlphaEpisodePartition,
    ridge_config: CausalAlphaRidgeConfig,
    controller_config: CausalAlphaControllerConfig,
    teacher_config_digest: str | None = None,
) -> tuple[EpisodeOracleBatch, CausalAlphaBatchEvidence]:
    """Fit at each episode start and generate one causal target path per contract."""

    symbols, _, scope_digest = _validated_sample_scope(train_symbols, samples)
    if symbol not in samples or symbol not in symbols:
        raise ValueError("causal alpha batch symbol must be inside train_symbols")
    block = samples[symbol]
    if any(contract.dataset_id != block.dataset_id for contract in partition.contracts):
        raise ValueError("causal alpha partition dataset identity drifted")
    targets: list[np.ndarray] = []
    episode_evidence: list[CausalAlphaEpisodeEvidence] = []
    for contract in partition.contracts:
        fitted = fit_expanding_causal_alpha_models(
            train_symbols=symbols,
            samples=samples,
            knowledge_cutoff=contract.start,
            ridge_config=ridge_config,
        )
        decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
        prediction_features = block.features_for_decisions(decisions)
        prediction_24h = fitted.model_24h.predict(prediction_features)
        prediction_72h = fitted.model_72h.predict(prediction_features)
        scores = combine_causal_alpha_predictions(
            prediction_24h,
            prediction_72h,
            controller_config.horizon_mix,
        )
        initial_weight = float(contract.initial_weights[0])
        target_path = causal_alpha_target_path(
            scores,
            config=controller_config,
            initial_weight=initial_weight,
        )
        target_matrix = np.asarray(target_path.targets, dtype=np.float32).reshape(-1, 1)
        prediction_digest = content_and_arrays_digest(
            {
                "episode_index": contract.episode_index,
                "fit_digest": fitted.digest,
                "knowledge_cutoff": contract.start,
                "schema_version": "causal_alpha_episode_predictions_v1",
                "symbol": symbol,
            },
            (
                ("prediction_24h", prediction_24h),
                ("prediction_72h", prediction_72h),
                ("scores", scores),
                ("targets", target_matrix),
            ),
        )
        targets.append(target_matrix)
        episode_evidence.append(
            CausalAlphaEpisodeEvidence(
                episode_index=contract.episode_index,
                knowledge_cutoff=contract.start,
                initial_weight=initial_weight,
                fit_digest=fitted.digest,
                max_label_end_24h=fitted.max_label_end_24h,
                max_label_end_72h=fitted.max_label_end_72h,
                target_path_digest=target_path.digest,
                prediction_digest=prediction_digest,
            )
        )
    evidence = CausalAlphaBatchEvidence(
        symbol=symbol,
        train_symbols=symbols,
        partition_digest=partition.digest,
        sample_scope_digest=scope_digest,
        ridge_config_digest=ridge_config.digest,
        controller_config_digest=controller_config.digest,
        episodes=tuple(episode_evidence),
    )
    resolved_teacher_config_digest = (
        evidence.digest if teacher_config_digest is None else teacher_config_digest
    )
    if (
        not isinstance(resolved_teacher_config_digest, str)
        or len(resolved_teacher_config_digest) != 64
    ):
        raise ValueError("causal alpha teacher_config_digest must be SHA-256")
    batch = EpisodeOracleBatch(
        dataset_id=block.dataset_id,
        teacher_config_digest=resolved_teacher_config_digest,
        sampling_config_digest=content_digest(
            {
                "partition_digest": partition.digest,
                "sample_scope_digest": scope_digest,
                "schema_version": "causal_alpha_episode_sampling_v1",
            }
        ),
        contracts=partition.contracts,
        targets=tuple(targets),
    )
    return batch, evidence


__all__ = [
    "CausalAlphaBatchEvidence",
    "CausalAlphaCandidateConfig",
    "CausalAlphaCandidateEpisodeMetrics",
    "CausalAlphaCandidateEvidence",
    "CausalAlphaEpisodeEvidence",
    "CausalAlphaEpisodePartition",
    "CausalAlphaExpandingFit",
    "CausalAlphaSelectionEvidence",
    "CausalAlphaSymbolSamples",
    "UniversalCausalAlphaTeacherPackage",
    "build_causal_alpha_episode_batch",
    "build_causal_alpha_symbol_samples",
    "build_chronological_episode_partition",
    "build_universal_causal_alpha_teacher_package",
    "default_causal_alpha_candidate_grid",
    "evaluate_causal_alpha_selection",
    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
    "rank_causal_alpha_candidates",
    "validate_universal_causal_alpha_partitions",
]
