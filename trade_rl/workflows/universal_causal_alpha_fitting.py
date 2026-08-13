"""Chronological data preparation and expanding fits for causal alpha."""

from __future__ import annotations

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
    CausalAlphaCostAwareConfig,
    CausalAlphaRidgeConfig,
    causal_alpha_cost_aware_target_path,
    causal_alpha_target_path,
    combine_causal_alpha_predictions,
    fit_causal_alpha_ridge,
    forward_log_return_label,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_bc import resolve_episode_initial_weights
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaBatchEvidence,
    CausalAlphaEpisodeEvidence,
    CausalAlphaEpisodePartition,
    CausalAlphaExpandingFit,
    CausalAlphaPredictionDiagnostics,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_costs import (
    causal_alpha_one_way_cost_rates,
)


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


class CausalAlphaExpandingFitCache:
    """Reuse pooled train-symbol arrays and expanding fits within one package build."""

    __slots__ = (
        "_available",
        "_blocks",
        "_cache",
        "_ends_24h",
        "_ends_72h",
        "_features",
        "_labels_24h",
        "_labels_72h",
        "_scope_digest",
        "_symbols",
        "fit_count",
        "hit_count",
    )

    def __init__(
        self,
        *,
        train_symbols: tuple[str, ...],
        samples: Mapping[str, CausalAlphaSymbolSamples],
    ) -> None:
        symbols, blocks, scope_digest = _validated_sample_scope(
            train_symbols,
            samples,
        )
        self._symbols = symbols
        self._blocks = blocks
        self._scope_digest = scope_digest
        self._features = np.concatenate(
            tuple(block.features for block in blocks), axis=0
        )
        self._available = np.concatenate(
            tuple(block.feature_available for block in blocks), axis=0
        )
        self._labels_24h = np.concatenate(
            tuple(block.labels_24h for block in blocks), axis=0
        )
        self._labels_72h = np.concatenate(
            tuple(block.labels_72h for block in blocks), axis=0
        )
        self._ends_24h = np.concatenate(
            tuple(block.label_end_indices_24h for block in blocks)
        )
        self._ends_72h = np.concatenate(
            tuple(block.label_end_indices_72h for block in blocks)
        )
        self._cache: dict[tuple[int, str], CausalAlphaExpandingFit] = {}
        self.fit_count = 0
        self.hit_count = 0

    @property
    def entry_count(self) -> int:
        return len(self._cache)

    @property
    def sample_scope_digest(self) -> str:
        return self._scope_digest

    def resolve(
        self,
        *,
        knowledge_cutoff: int,
        ridge_config: CausalAlphaRidgeConfig,
    ) -> CausalAlphaExpandingFit:
        key = (knowledge_cutoff, ridge_config.digest)
        cached = self._cache.get(key)
        if cached is not None:
            self.hit_count += 1
            return cached
        feature_names = self._blocks[0].feature_names
        model_24h = fit_causal_alpha_ridge(
            features=self._features,
            labels=self._labels_24h,
            feature_available=self._available,
            label_end_indices=self._ends_24h,
            knowledge_cutoff=knowledge_cutoff,
            feature_names=feature_names,
            config=ridge_config,
        )
        model_72h = fit_causal_alpha_ridge(
            features=self._features,
            labels=self._labels_72h,
            feature_available=self._available,
            label_end_indices=self._ends_72h,
            knowledge_cutoff=knowledge_cutoff,
            feature_names=feature_names,
            config=ridge_config,
        )
        fitted_ends_24h = self._ends_24h[model_24h.eligible_indices]
        fitted_ends_72h = self._ends_72h[model_72h.eligible_indices]
        fitted = CausalAlphaExpandingFit(
            train_symbols=self._symbols,
            knowledge_cutoff=knowledge_cutoff,
            model_24h=model_24h,
            model_72h=model_72h,
            sample_count_24h=model_24h.sample_count,
            sample_count_72h=model_72h.sample_count,
            max_label_end_24h=int(np.max(fitted_ends_24h)),
            max_label_end_72h=int(np.max(fitted_ends_72h)),
            sample_scope_digest=self._scope_digest,
        )
        self._cache[key] = fitted
        self.fit_count += 1
        return fitted


def fit_expanding_causal_alpha_models(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    knowledge_cutoff: int,
    ridge_config: CausalAlphaRidgeConfig,
) -> CausalAlphaExpandingFit:
    """Fit both horizons on pooled train-symbol labels realized before a cutoff."""

    return CausalAlphaExpandingFitCache(
        train_symbols=train_symbols,
        samples=samples,
    ).resolve(
        knowledge_cutoff=knowledge_cutoff,
        ridge_config=ridge_config,
    )


def causal_alpha_prediction_diagnostics(
    predictions: object, labels: object
) -> CausalAlphaPredictionDiagnostics:
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    realized = np.asarray(labels, dtype=np.float64).reshape(-1)
    if predicted.shape != realized.shape or predicted.size == 0:
        raise ValueError("causal alpha prediction diagnostics require aligned samples")
    if not np.isfinite(predicted).all():
        raise ValueError("causal alpha predictions must be finite")
    realized_mask = np.isfinite(realized)
    if not np.any(realized_mask):
        raise ValueError("causal alpha prediction diagnostics have no realized labels")
    predicted = predicted[realized_mask]
    realized = realized[realized_mask]
    predicted_std = float(predicted.std())
    realized_std = float(realized.std())
    correlation = (
        None
        if predicted.size < 2 or predicted_std <= 1e-12 or realized_std <= 1e-12
        else float(np.corrcoef(predicted, realized)[0, 1])
    )
    directional = float(np.mean(np.sign(predicted) == np.sign(realized)))
    quantile_values = np.quantile(
        predicted, np.asarray((0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
    )
    quantiles = (
        float(quantile_values[0]),
        float(quantile_values[1]),
        float(quantile_values[2]),
        float(quantile_values[3]),
        float(quantile_values[4]),
        float(quantile_values[5]),
        float(quantile_values[6]),
    )
    return CausalAlphaPredictionDiagnostics(
        sample_count=int(predicted.size),
        pearson_correlation=correlation,
        directional_accuracy=directional,
        prediction_mean=float(predicted.mean()),
        prediction_std=predicted_std,
        prediction_min=float(predicted.min()),
        prediction_max=float(predicted.max()),
        prediction_quantiles=quantiles,
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
    fit_cache: CausalAlphaExpandingFitCache | None = None,
    economic_controller_config: CausalAlphaCostAwareConfig | None = None,
    dataset: Any | None = None,
    execution_cost: ExecutionCostConfig | None = None,
    signal_delay_decisions: int | None = None,
    decision_bars: int | None = None,
) -> tuple[EpisodeOracleBatch, CausalAlphaBatchEvidence]:
    """Fit at each episode start and generate one causal target path per contract."""

    symbols, _, scope_digest = _validated_sample_scope(train_symbols, samples)
    if symbol not in samples or symbol not in symbols:
        raise ValueError("causal alpha batch symbol must be inside train_symbols")
    block = samples[symbol]
    if any(contract.dataset_id != block.dataset_id for contract in partition.contracts):
        raise ValueError("causal alpha partition dataset identity drifted")
    cost_inputs = (dataset, execution_cost, signal_delay_decisions, decision_bars)
    if economic_controller_config is None:
        if any(value is not None for value in cost_inputs):
            raise ValueError("causal alpha cost-aware batch inputs are incomplete")
    elif (
        dataset is None
        or not isinstance(execution_cost, ExecutionCostConfig)
        or isinstance(signal_delay_decisions, bool)
        or not isinstance(signal_delay_decisions, int)
        or isinstance(decision_bars, bool)
        or not isinstance(decision_bars, int)
    ):
        raise ValueError("causal alpha cost-aware batch inputs are incomplete")
    targets: list[np.ndarray] = []
    episode_evidence: list[CausalAlphaEpisodeEvidence] = []
    fits: dict[str, CausalAlphaExpandingFit] = {}
    for contract in partition.contracts:
        fitted = (
            fit_expanding_causal_alpha_models(
                train_symbols=symbols,
                samples=samples,
                knowledge_cutoff=contract.start,
                ridge_config=ridge_config,
            )
            if fit_cache is None
            else fit_cache.resolve(
                knowledge_cutoff=contract.start,
                ridge_config=ridge_config,
            )
        )
        decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
        prediction_features, prediction_available, actionable = (
            block.prediction_inputs_for_decisions(decisions)
        )
        prediction_24h = fitted.model_24h.predict(
            prediction_features, feature_available=prediction_available
        )
        prediction_72h = fitted.model_72h.predict(
            prediction_features, feature_available=prediction_available
        )
        scores = combine_causal_alpha_predictions(
            prediction_24h,
            prediction_72h,
            controller_config.horizon_mix,
        )
        positions = np.searchsorted(block.decision_indices, decisions)
        diagnostic_positions = positions[actionable]
        diagnostics_24h = causal_alpha_prediction_diagnostics(
            prediction_24h[actionable], block.labels_24h[diagnostic_positions]
        )
        diagnostics_72h = causal_alpha_prediction_diagnostics(
            prediction_72h[actionable], block.labels_72h[diagnostic_positions]
        )
        initial_weight = float(contract.initial_weights[0])
        if economic_controller_config is None:
            legacy_path = causal_alpha_target_path(
                scores,
                config=controller_config,
                initial_weight=initial_weight,
                actionable_mask=actionable,
            )
            cost_rates: np.ndarray | None = None
            cost_aware_digest: str | None = None
            cost_suppressed_count: int | None = None
            strong_reversal_count: int | None = None
            submitted_change_count = legacy_path.submitted_change_count
            suppressed_change_count = legacy_path.suppressed_change_count
            sign_flip_count = legacy_path.sign_flip_count
            target_values = legacy_path.targets
            target_path_digest = legacy_path.digest
        else:
            assert execution_cost is not None
            assert signal_delay_decisions is not None
            assert decision_bars is not None
            cost_rates = causal_alpha_one_way_cost_rates(
                dataset,
                execution_cost,
                decision_indices=decisions,
                signal_delay_decisions=signal_delay_decisions,
                decision_bars=decision_bars,
            )
            economic_path = causal_alpha_cost_aware_target_path(
                scores,
                one_way_cost_rates=cost_rates,
                controller=controller_config,
                economic=economic_controller_config,
                initial_weight=initial_weight,
                actionable_mask=actionable,
            )
            cost_aware_digest = economic_path.digest
            cost_suppressed_count = economic_path.cost_suppressed_change_count
            strong_reversal_count = economic_path.strong_reversal_count
            submitted_change_count = economic_path.submitted_change_count
            suppressed_change_count = economic_path.cost_suppressed_change_count
            sign_flip_count = economic_path.sign_flip_count
            target_values = economic_path.targets
            target_path_digest = economic_path.digest
        target_matrix = np.asarray(target_values, dtype=np.float32).reshape(-1, 1)
        prediction_digest = content_and_arrays_digest(
            {
                "episode_index": contract.episode_index,
                "fit_digest": fitted.digest,
                "knowledge_cutoff": contract.start,
                "economic_controller_digest": (
                    None
                    if economic_controller_config is None
                    else economic_controller_config.digest
                ),
                "schema_version": (
                    "causal_alpha_episode_predictions_v1"
                    if economic_controller_config is None
                    else "causal_alpha_episode_predictions_v2"
                ),
                "symbol": symbol,
            },
            (
                ("prediction_24h", prediction_24h),
                ("prediction_72h", prediction_72h),
                ("prediction_feature_available", prediction_available),
                ("actionable_mask", actionable),
                ("scores", scores),
                *(() if cost_rates is None else (("one_way_cost_rates", cost_rates),)),
                ("targets", target_matrix),
            ),
        )
        targets.append(target_matrix)
        fits[fitted.digest] = fitted
        episode_evidence.append(
            CausalAlphaEpisodeEvidence(
                episode_index=contract.episode_index,
                scope=(
                    "holdout"
                    if contract.episode_index
                    == partition.holdout_contract.episode_index
                    else "selection"
                ),
                knowledge_cutoff=contract.start,
                initial_weight=initial_weight,
                fit_digest=fitted.digest,
                fit_sample_count_24h=fitted.sample_count_24h,
                fit_sample_count_72h=fitted.sample_count_72h,
                max_label_end_24h=fitted.max_label_end_24h,
                max_label_end_72h=fitted.max_label_end_72h,
                prediction_24h=diagnostics_24h,
                prediction_72h=diagnostics_72h,
                decision_count=int(decisions.size),
                actionable_decision_count=int(np.count_nonzero(actionable)),
                submitted_change_count=submitted_change_count,
                suppressed_change_count=suppressed_change_count,
                sign_flip_count=sign_flip_count,
                target_path_digest=target_path_digest,
                prediction_digest=prediction_digest,
                cost_aware_target_path_digest=cost_aware_digest,
                cost_suppressed_change_count=cost_suppressed_count,
                strong_reversal_count=strong_reversal_count,
            )
        )
    evidence = CausalAlphaBatchEvidence(
        symbol=symbol,
        train_symbols=symbols,
        partition_digest=partition.digest,
        sample_scope_digest=scope_digest,
        ridge_config_digest=ridge_config.digest,
        controller_config_digest=controller_config.digest,
        fits=fits,
        episodes=tuple(episode_evidence),
        economic_controller_config_digest=(
            None
            if economic_controller_config is None
            else economic_controller_config.digest
        ),
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
    "CausalAlphaExpandingFitCache",
    "build_causal_alpha_episode_batch",
    "build_causal_alpha_symbol_samples",
    "build_chronological_episode_partition",
    "causal_alpha_prediction_diagnostics",
    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
    "validate_universal_causal_alpha_partitions",
]
