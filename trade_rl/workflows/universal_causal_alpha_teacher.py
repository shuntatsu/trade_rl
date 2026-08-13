"""Public orchestration facade for the Universal causal alpha teacher."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import trade_rl.learning.causal_alpha_teacher as _causal_learning_module
import trade_rl.workflows.universal_causal_alpha_contracts as _causal_contracts_module
import trade_rl.workflows.universal_causal_alpha_fitting as _causal_fitting_module
import trade_rl.workflows.universal_causal_alpha_selection as _causal_selection_module
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherAdmissionEvidence,
    CausalAlphaTeacherHoldoutMetric,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path,
    evaluate_episode_action_path_on_environment,
)
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaBatchEvidence,
    CausalAlphaCandidateConfig,
    CausalAlphaCandidateEpisodeMetrics,
    CausalAlphaCandidateEvidence,
    CausalAlphaEpisodeEvidence,
    CausalAlphaEpisodePartition,
    CausalAlphaExpandingFit,
    CausalAlphaPredictionDiagnostics,
    CausalAlphaSelectionEvidence,
    CausalAlphaSymbolSamples,
    UniversalCausalAlphaTeacherPackage,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    CausalAlphaExpandingFitCache,
    _validated_sample_scope,
    build_causal_alpha_episode_batch,
    build_causal_alpha_symbol_samples,
    build_chronological_episode_partition,
    fit_expanding_causal_alpha_models,
    latest_complete_episode_split,
    validate_universal_causal_alpha_partitions,
)
from trade_rl.workflows.universal_causal_alpha_selection import (
    _causal_alpha_target_for_contract,
    _CausalAlphaPredictionCache,
    default_causal_alpha_candidate_grid,
    rank_causal_alpha_candidates,
)


def evaluate_causal_alpha_selection(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partitions: Mapping[str, CausalAlphaEpisodePartition],
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    environment_factories: Mapping[str, Any],
    episode_hours: float,
    fit_cache: CausalAlphaExpandingFitCache | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
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
    total_replays = len(candidates) * sum(
        len(partition_values[symbol].selection_contracts) for symbol in symbols
    )
    completed_replays = 0
    prediction_cache = _CausalAlphaPredictionCache()
    records_by_candidate: dict[str, list[CausalAlphaCandidateEpisodeMetrics]] = {
        candidate.digest: [] for candidate in candidates
    }
    for symbol in symbols:
        factory = environment_factories[symbol]
        if not callable(factory):
            raise TypeError(
                "causal alpha selection environment factory is not callable"
            )
        environment = factory()
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("causal alpha selection environment is not closable")
        try:
            partition = partition_values[symbol]
            for candidate in candidates:
                for contract in partition.selection_contracts:
                    actions = _causal_alpha_target_for_contract(
                        symbol=symbol,
                        train_symbols=symbols,
                        samples=samples,
                        contract=contract,
                        candidate=candidate,
                        fit_cache=fit_cache,
                        prediction_cache=prediction_cache,
                    )
                    evaluation = evaluate_episode_action_path_on_environment(
                        environment,
                        contract,
                        actions=actions,
                    )
                    performance = evaluation.performance
                    collapse = evaluation.collapse_evidence
                    records_by_candidate[candidate.digest].append(
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
                            risk_violation=(
                                int(collapse.execution_rejection_count) > 0
                            ),
                        )
                    )
                    completed_replays += 1
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "candidate_digest": candidate.digest,
                                "completed_replays": completed_replays,
                                "episode_index": contract.episode_index,
                                "fit_cache_entries": (
                                    fit_cache.entry_count
                                    if fit_cache is not None
                                    else 0
                                ),
                                "fit_cache_hits": (
                                    fit_cache.hit_count if fit_cache is not None else 0
                                ),
                                "fit_count": (
                                    fit_cache.fit_count if fit_cache is not None else 0
                                ),
                                "phase": "causal_teacher_selection",
                                "prediction_cache_hits": prediction_cache.hit_count,
                                "prediction_count": prediction_cache.prediction_count,
                                "symbol": symbol,
                                "total_replays": total_replays,
                            }
                        )
        finally:
            close()
    by_candidate = {
        digest: tuple(records) for digest, records in records_by_candidate.items()
    }
    return rank_causal_alpha_candidates(
        candidates=tuple(candidates),
        metrics=by_candidate,
        holdout_episode_digests={
            symbol: partition_values[symbol].holdout_contract.digest
            for symbol in symbols
        },
    )


def causal_alpha_generator_code_digest() -> str:
    """Bind teacher identity to the exact causal-generator source files."""

    modules = (
        _causal_learning_module,
        _causal_contracts_module,
        _causal_fitting_module,
        _causal_selection_module,
    )
    files: dict[str, str] = {}
    for module in modules:
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise RuntimeError("causal alpha generator source path is unavailable")
        path = Path(raw_path)
        files[module.__name__] = hashlib.sha256(path.read_bytes()).hexdigest()
    files[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return content_digest(
        {
            "files": files,
            "schema_version": "universal_causal_alpha_generator_code_v1",
        }
    )


def evaluate_causal_alpha_teacher_holdouts(
    *,
    train_symbols: tuple[str, ...],
    batches: Mapping[str, EpisodeOracleBatch],
    environment_factories: Mapping[str, Any],
    episode_hours: float,
) -> CausalAlphaTeacherAdmissionEvidence:
    """Replay each untouched teacher holdout exactly once and freeze admission."""

    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("causal alpha teacher holdout symbols must be unique")
    if set(batches) != set(symbols) or set(environment_factories) != set(symbols):
        raise ValueError("causal alpha teacher holdout scope must match train_symbols")
    if not np.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("causal alpha teacher holdout episode_hours must be positive")
    episode_days = float(episode_hours) / 24.0
    metrics: list[CausalAlphaTeacherHoldoutMetric] = []
    for symbol in symbols:
        batch = batches[symbol]
        if not isinstance(batch, EpisodeOracleBatch):
            raise TypeError("causal alpha teacher holdout batch type is invalid")
        if not batch.contracts or len(batch.targets) != len(batch.contracts):
            raise ValueError(
                f"causal alpha teacher holdout batch is invalid for {symbol}"
            )
        factory = environment_factories[symbol]
        if not callable(factory):
            raise TypeError("causal alpha teacher holdout factory must be callable")
        evaluation = evaluate_episode_action_path(
            factory,
            batch.contracts[-1],
            actions=batch.targets[-1],
        )
        performance = evaluation.performance
        metrics.append(
            CausalAlphaTeacherHoldoutMetric(
                symbol=symbol,
                gross_return=float(performance.gross_return),
                net_return=float(performance.net_return),
                turnover_per_day=float(performance.turnover_total) / episode_days,
                total_execution_cost=float(performance.cost_total),
                trade_count=int(performance.trade_count),
                maximum_drawdown=float(performance.maximum_drawdown),
            )
        )
    return evaluate_causal_alpha_teacher_admission(tuple(metrics))


def build_universal_causal_alpha_teacher_package(
    *,
    train_symbols: tuple[str, ...],
    bindings: tuple[InstrumentDatasetBinding, ...],
    concrete_environment_factory: Any,
    instrument_context_provider: Any,
    fold_train_range: tuple[int, int],
    feature_schema_digest: str,
    selection_evidence_path: Path,
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
    environment_factories = {
        symbol: partial(concrete_environment_factory, binding_by_symbol[symbol])
        for symbol in symbols
    }
    fit_cache = CausalAlphaExpandingFitCache(
        train_symbols=symbols,
        samples=samples,
    )
    selection_path = Path(selection_evidence_path)
    progress_path = selection_path.parent / "causal-teacher-progress.json"
    latest_progress: dict[str, object] = {}

    def persist_selection_progress(payload: Mapping[str, object]) -> None:
        latest_progress.clear()
        latest_progress.update(payload)
        atomic_write_bytes(
            progress_path,
            canonical_json_bytes(dict(payload)) + b"\n",
        )

    selection = evaluate_causal_alpha_selection(
        train_symbols=symbols,
        samples=samples,
        partitions=partitions,
        candidates=candidate_values,
        environment_factories=environment_factories,
        episode_hours=resolved_episode_hours,
        fit_cache=fit_cache,
        progress_callback=persist_selection_progress,
    )
    atomic_write_bytes(
        selection_path,
        canonical_json_bytes(selection.to_payload()) + b"\n",
    )
    selected_evidence = tuple(
        item
        for item in selection.candidates
        if item.candidate.digest == selection.selected_candidate_digest
    )
    if len(selected_evidence) != 1:
        raise RuntimeError("causal alpha selected candidate cannot be resolved")
    selected = selected_evidence[0].candidate
    generator_code_digest = causal_alpha_generator_code_digest()
    teacher_config_digest = content_digest(
        {
            "feature_schema_digest": feature_schema_digest,
            "generator_code_digest": generator_code_digest,
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
            fit_cache=fit_cache,
        )
        batches[symbol] = batch
        batch_evidence[symbol] = evidence
    teacher_admission = evaluate_causal_alpha_teacher_holdouts(
        train_symbols=symbols,
        batches=batches,
        environment_factories=environment_factories,
        episode_hours=resolved_episode_hours,
    )
    evidence_root = selection_path.parent
    atomic_write_bytes(
        evidence_root / "causal-teacher-admission.json",
        canonical_json_bytes(teacher_admission.to_payload()) + b"\n",
    )
    package = UniversalCausalAlphaTeacherPackage(
        train_symbols=symbols,
        batches=batches,
        partitions=partitions,
        samples=samples,
        selection=selection,
        teacher_admission=teacher_admission,
        selected_candidate_digest=selected.digest,
        teacher_config_digest=teacher_config_digest,
        generator_code_digest=generator_code_digest,
        episode_hours=resolved_episode_hours,
        batch_evidence=batch_evidence,
    )
    atomic_write_bytes(
        evidence_root / "causal-teacher-package.json",
        canonical_json_bytes(package.to_payload()) + b"\n",
    )
    persist_selection_progress(
        {
            **latest_progress,
            "package_digest": package.digest,
            "phase": "causal_teacher_package_completed",
            "teacher_admission_digest": teacher_admission.digest,
            "teacher_admission_passed": teacher_admission.passed,
        }
    )
    return package


__all__ = [
    "CausalAlphaExpandingFitCache",
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
    "build_causal_alpha_episode_batch",
    "build_causal_alpha_symbol_samples",
    "build_chronological_episode_partition",
    "build_universal_causal_alpha_teacher_package",
    "causal_alpha_generator_code_digest",
    "default_causal_alpha_candidate_grid",
    "evaluate_causal_alpha_selection",
    "evaluate_causal_alpha_teacher_holdouts",
    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
    "rank_causal_alpha_candidates",
    "validate_universal_causal_alpha_partitions",
]
