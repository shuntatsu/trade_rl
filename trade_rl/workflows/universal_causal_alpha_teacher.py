"""Train-only chronological fitting contracts for the Universal causal alpha teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
    causal_alpha_target_path,
    combine_causal_alpha_predictions,
    fit_causal_alpha_ridge,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_bc import resolve_episode_initial_weights
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
        names = tuple(self.feature_names)
        if not names or len(set(names)) != len(names) or any(not name for name in names):
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
            raise ValueError("causal alpha decision indices must be strictly increasing")
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
        if (
            np.any(positions >= self.decision_indices.size)
            or not np.array_equal(self.decision_indices[positions], requested)
        ):
            raise ValueError("causal alpha prediction decisions are absent from samples")
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
        if not self.train_symbols or len(set(self.train_symbols)) != len(self.train_symbols):
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
        raise ValueError("causal alpha train range contains no complete holdout episode")
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
            raise ValueError("causal alpha initial weights do not match dataset symbols")
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
    if not symbols or len(set(symbols)) != len(symbols) or any(not symbol for symbol in symbols):
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
    available = np.concatenate(tuple(block.feature_available for block in blocks), axis=0)
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


def build_causal_alpha_episode_batch(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partition: CausalAlphaEpisodePartition,
    ridge_config: CausalAlphaRidgeConfig,
    controller_config: CausalAlphaControllerConfig,
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
    batch = EpisodeOracleBatch(
        dataset_id=block.dataset_id,
        teacher_config_digest=evidence.digest,
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
    "CausalAlphaEpisodeEvidence",
    "CausalAlphaEpisodePartition",
    "CausalAlphaExpandingFit",
    "CausalAlphaSymbolSamples",
    "build_causal_alpha_episode_batch",
    "build_chronological_episode_partition",
    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
    "validate_universal_causal_alpha_partitions",
]
