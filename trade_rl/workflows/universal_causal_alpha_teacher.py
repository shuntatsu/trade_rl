"""Public orchestration facade for the Universal causal alpha teacher."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_oracle_bc import evaluate_episode_action_path
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
    CausalAlphaSelectionEvidence,
    CausalAlphaSymbolSamples,
    UniversalCausalAlphaTeacherPackage,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
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
        episode_hours=resolved_episode_hours,
        batch_evidence=batch_evidence,
    )


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
