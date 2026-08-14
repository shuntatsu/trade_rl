"""Public orchestration facade for the Universal causal alpha teacher."""

from __future__ import annotations

import hashlib
import json
import os
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
from trade_rl.learning.causal_alpha_diagnostics import (
    causal_alpha_signal_diagnostics_from_payload,
)
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
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaBatchEvidence,
    CausalAlphaCandidateConfig,
    CausalAlphaCandidateEpisodeMetrics,
    CausalAlphaCandidateEpisodeMetricsV2,
    CausalAlphaCandidateEvidence,
    CausalAlphaEpisodeEvidence,
    CausalAlphaEpisodePartition,
    CausalAlphaExpandingFit,
    CausalAlphaPredictionDiagnostics,
    CausalAlphaSelectionEvidence,
    CausalAlphaSelectionEvidenceV2,
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
    CausalAlphaSelectionRejected,
    CausalAlphaSelectionRejectedV2,
    CausalAlphaSelectionThresholds,
    _causal_alpha_target_for_contract,
    _CausalAlphaLiquidityCapCache,
    _CausalAlphaPredictionCache,
    _cost_aware_causal_alpha_target_for_contract,
    cost_aware_causal_alpha_grid_digest,
    default_causal_alpha_candidate_grid,
    default_cost_aware_causal_alpha_candidate_grid,
    rank_causal_alpha_candidates,
    rank_cost_aware_causal_alpha_candidates,
)


def persist_causal_alpha_selection_rejection(
    path: Path,
    rejection: CausalAlphaSelectionRejected | CausalAlphaSelectionRejectedV2,
) -> None:
    """Durably preserve complete candidate economics before failing closed."""

    atomic_write_bytes(
        Path(path),
        canonical_json_bytes(rejection.to_payload()) + b"\n",
    )


def _selection_checkpoint_payload(
    metric: CausalAlphaCandidateEpisodeMetrics,
) -> dict[str, object]:
    return {
        "artifact_digest": metric.digest,
        "candidate_digest": metric.candidate_digest,
        "episode_index": metric.episode_index,
        "gross_return": metric.gross_return,
        "net_return": metric.net_return,
        "risk_violation": metric.risk_violation,
        "schema_version": "causal_alpha_selection_checkpoint_metric_v1",
        "symbol": metric.symbol,
        "total_execution_cost": metric.total_execution_cost,
        "trade_count": metric.trade_count,
        "turnover_per_day": metric.turnover_per_day,
    }


def write_causal_alpha_selection_checkpoint_metric(
    path: Path,
    metric: CausalAlphaCandidateEpisodeMetrics,
) -> None:
    """Append and fsync one completed production replay metric."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as checkpoint:
        checkpoint.write(canonical_json_bytes(_selection_checkpoint_payload(metric)))
        checkpoint.write(b"\n")
        checkpoint.flush()
        os.fsync(checkpoint.fileno())


def load_causal_alpha_selection_checkpoint(
    path: Path,
) -> dict[str, tuple[CausalAlphaCandidateEpisodeMetrics, ...]]:
    """Load a durable replay checkpoint and reject malformed or duplicate rows."""

    source = Path(path)
    if not source.is_file():
        return {}
    by_candidate: dict[str, list[CausalAlphaCandidateEpisodeMetrics]] = {}
    identities: set[tuple[str, str, int]] = set()
    with source.open("r", encoding="utf-8") as checkpoint:
        for line in checkpoint:
            raw = json.loads(line)
            if raw.get("schema_version") != (
                "causal_alpha_selection_checkpoint_metric_v1"
            ):
                raise ValueError("causal alpha selection checkpoint schema mismatch")
            metric = CausalAlphaCandidateEpisodeMetrics(
                candidate_digest=str(raw["candidate_digest"]),
                symbol=str(raw["symbol"]),
                episode_index=int(raw["episode_index"]),
                gross_return=float(raw["gross_return"]),
                net_return=float(raw["net_return"]),
                turnover_per_day=float(raw["turnover_per_day"]),
                total_execution_cost=float(raw["total_execution_cost"]),
                trade_count=int(raw["trade_count"]),
                risk_violation=bool(raw["risk_violation"]),
                digest=str(raw["artifact_digest"]),
            )
            identity = (
                metric.candidate_digest,
                metric.symbol,
                metric.episode_index,
            )
            if identity in identities:
                raise ValueError("causal alpha selection checkpoint is duplicated")
            identities.add(identity)
            by_candidate.setdefault(metric.candidate_digest, []).append(metric)
    return {digest: tuple(metrics) for digest, metrics in by_candidate.items()}


def _resolved_generator_code_digest(value: str | None) -> str:
    resolved = causal_alpha_generator_code_digest() if value is None else value
    if not isinstance(resolved, str) or len(resolved) != 64:
        raise ValueError("causal alpha v2 generator code digest is invalid")
    return resolved


def write_causal_alpha_selection_checkpoint_metric_v2(
    path: Path,
    metric: CausalAlphaCandidateEpisodeMetricsV2,
    *,
    grid_digest: str,
    generator_code_digest: str | None = None,
) -> None:
    if not isinstance(grid_digest, str) or len(grid_digest) != 64:
        raise ValueError("causal alpha v2 grid digest is invalid")
    resolved_generator_code_digest = _resolved_generator_code_digest(
        generator_code_digest
    )
    payload = {
        **metric.to_payload(),
        "generator_code_digest": resolved_generator_code_digest,
        "grid_digest": grid_digest,
        "schema_version": "causal_alpha_selection_checkpoint_metric_v2",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as checkpoint:
        checkpoint.write(canonical_json_bytes(payload))
        checkpoint.write(b"\n")
        checkpoint.flush()
        os.fsync(checkpoint.fileno())


def causal_alpha_candidate_metric_v2_from_payload(
    raw: Mapping[str, Any],
) -> CausalAlphaCandidateEpisodeMetricsV2:
    return CausalAlphaCandidateEpisodeMetricsV2(
        candidate_digest=str(raw["candidate_digest"]),
        symbol=str(raw["symbol"]),
        episode_index=int(raw["episode_index"]),
        gross_return=float(raw["gross_return"]),
        net_return=float(raw["net_return"]),
        turnover_per_day=float(raw["turnover_per_day"]),
        total_execution_cost=float(raw["total_execution_cost"]),
        trade_count=int(raw["trade_count"]),
        signal_24h=causal_alpha_signal_diagnostics_from_payload(raw["signal_24h"]),
        signal_72h=causal_alpha_signal_diagnostics_from_payload(raw["signal_72h"]),
        cost_suppressed_change_count=int(raw["cost_suppressed_change_count"]),
        submitted_change_count=int(raw["submitted_change_count"]),
        strong_reversal_count=int(raw["strong_reversal_count"]),
        command_sign_flip_count=int(raw["command_sign_flip_count"]),
        execution_rejection_count=int(raw["execution_rejection_count"]),
        execution_rejection_reason_counts=tuple(
            (str(reason), int(count))
            for reason, count in raw["execution_rejection_reason_counts"]
        ),
        risk_projection_reason_counts=tuple(
            (str(reason), int(count))
            for reason, count in raw["risk_projection_reason_counts"]
        ),
        hard_risk_violation=bool(raw["hard_risk_violation"]),
        liquidity_deleveraging_count=int(raw.get("liquidity_deleveraging_count", 0)),
        liquidity_weight_cap_min=float(raw.get("liquidity_weight_cap_min", 0.0)),
        liquidity_weight_cap_median=float(raw.get("liquidity_weight_cap_median", 0.0)),
        liquidity_weight_cap_max=float(raw.get("liquidity_weight_cap_max", 0.0)),
        digest=str(raw["artifact_digest"]),
    )


def load_causal_alpha_selection_checkpoint_v2(
    path: Path,
    *,
    expected_grid_digest: str,
    expected_generator_code_digest: str | None = None,
) -> dict[str, tuple[CausalAlphaCandidateEpisodeMetricsV2, ...]]:
    source = Path(path)
    if not source.is_file():
        return {}
    resolved_generator_code_digest = _resolved_generator_code_digest(
        expected_generator_code_digest
    )
    by_candidate: dict[str, list[CausalAlphaCandidateEpisodeMetricsV2]] = {}
    identities: set[tuple[str, str, int]] = set()
    with source.open("r", encoding="utf-8") as checkpoint:
        for line in checkpoint:
            raw = json.loads(line)
            if (
                raw.get("schema_version")
                != "causal_alpha_selection_checkpoint_metric_v2"
            ):
                raise ValueError("causal alpha v2 selection checkpoint schema mismatch")
            if raw.get("grid_digest") != expected_grid_digest:
                raise ValueError(
                    "causal alpha v2 selection checkpoint grid digest mismatch"
                )
            if raw.get("generator_code_digest") != resolved_generator_code_digest:
                raise ValueError(
                    "causal alpha v2 selection checkpoint generator code digest mismatch"
                )
            metric = causal_alpha_candidate_metric_v2_from_payload(raw)
            identity = (metric.candidate_digest, metric.symbol, metric.episode_index)
            if identity in identities:
                raise ValueError("causal alpha v2 selection checkpoint is duplicated")
            identities.add(identity)
            by_candidate.setdefault(metric.candidate_digest, []).append(metric)
    return {digest: tuple(metrics) for digest, metrics in by_candidate.items()}


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
    initial_metrics: Mapping[str, tuple[CausalAlphaCandidateEpisodeMetrics, ...]]
    | None = None,
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
    candidate_digests = {candidate.digest for candidate in candidates}
    resumed = {} if initial_metrics is None else dict(initial_metrics)
    if not set(resumed).issubset(candidate_digests):
        raise ValueError("causal alpha resumed metrics contain an unknown candidate")
    prediction_cache = _CausalAlphaPredictionCache()
    records_by_candidate: dict[str, list[CausalAlphaCandidateEpisodeMetrics]] = {
        candidate.digest: list(resumed.get(candidate.digest, ()))
        for candidate in candidates
    }
    completed_identities: set[tuple[str, str, int]] = set()
    selection_indices = {
        symbol: {
            contract.episode_index
            for contract in partition_values[symbol].selection_contracts
        }
        for symbol in symbols
    }
    for candidate_digest, records in records_by_candidate.items():
        for record in records:
            identity = (candidate_digest, record.symbol, record.episode_index)
            if (
                record.candidate_digest != candidate_digest
                or record.symbol not in selection_indices
                or record.episode_index not in selection_indices[record.symbol]
                or identity in completed_identities
            ):
                raise ValueError("causal alpha resumed metric identity is invalid")
            completed_identities.add(identity)
    completed_replays = len(completed_identities)
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
                    identity = (candidate.digest, symbol, contract.episode_index)
                    if identity in completed_identities:
                        continue
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
                        metric := CausalAlphaCandidateEpisodeMetrics(
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
                    completed_identities.add(identity)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "candidate_digest": candidate.digest,
                                "completed_replays": completed_replays,
                                "episode_index": contract.episode_index,
                                "episode_metric": {
                                    "artifact_digest": metric.digest,
                                    "candidate_digest": metric.candidate_digest,
                                    "episode_index": metric.episode_index,
                                    "gross_return": metric.gross_return,
                                    "net_return": metric.net_return,
                                    "risk_violation": metric.risk_violation,
                                    "symbol": metric.symbol,
                                    "total_execution_cost": (
                                        metric.total_execution_cost
                                    ),
                                    "trade_count": metric.trade_count,
                                    "turnover_per_day": metric.turnover_per_day,
                                },
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


def evaluate_cost_aware_causal_alpha_selection(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partitions: Mapping[str, CausalAlphaEpisodePartition],
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    environment_factories: Mapping[str, Any],
    episode_hours: float,
    thresholds: CausalAlphaSelectionThresholds,
    fit_cache: CausalAlphaExpandingFitCache | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    initial_metrics: Mapping[str, tuple[CausalAlphaCandidateEpisodeMetricsV2, ...]]
    | None = None,
) -> CausalAlphaSelectionEvidenceV2:
    symbols, _, _ = _validated_sample_scope(train_symbols, samples)
    partition_values = validate_universal_causal_alpha_partitions(
        train_symbols=symbols, partitions=partitions
    )
    if set(environment_factories) != set(symbols):
        raise ValueError("cost-aware environment factories must match train_symbols")
    if not np.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("cost-aware episode_hours must be finite and positive")
    episode_days = float(episode_hours) / 24.0
    total_replays = len(candidates) * sum(
        len(partition_values[symbol].selection_contracts) for symbol in symbols
    )
    candidate_digests = {candidate.digest for candidate in candidates}
    resumed = {} if initial_metrics is None else dict(initial_metrics)
    if not set(resumed).issubset(candidate_digests):
        raise ValueError("cost-aware resumed metrics contain an unknown candidate")
    records: dict[str, list[CausalAlphaCandidateEpisodeMetricsV2]] = {
        candidate.digest: list(resumed.get(candidate.digest, ()))
        for candidate in candidates
    }
    completed = {
        (digest, item.symbol, item.episode_index)
        for digest, values in records.items()
        for item in values
    }
    prediction_cache = _CausalAlphaPredictionCache()
    liquidity_cache = _CausalAlphaLiquidityCapCache()
    completed_replays = len(completed)
    for symbol in symbols:
        environment = environment_factories[symbol]()
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("cost-aware selection environment is not closable")
        try:
            config = getattr(environment, "config", None)
            execution_cost = getattr(config, "execution_cost", None)
            signal_delay = getattr(config, "signal_delay_decisions", None)
            decision_bars = getattr(environment, "decision_bars", None)
            if not isinstance(execution_cost, ExecutionCostConfig):
                raise TypeError("cost-aware execution cost config is unavailable")
            if (
                isinstance(signal_delay, bool)
                or not isinstance(signal_delay, int)
                or signal_delay not in {0, 1}
            ):
                raise ValueError("cost-aware signal delay is unavailable")
            if (
                isinstance(decision_bars, bool)
                or not isinstance(decision_bars, int)
                or decision_bars <= 0
            ):
                raise ValueError("cost-aware decision_bars is unavailable")
            for candidate in candidates:
                for contract in partition_values[symbol].selection_contracts:
                    identity = (candidate.digest, symbol, contract.episode_index)
                    if identity in completed:
                        continue
                    targets = _cost_aware_causal_alpha_target_for_contract(
                        symbol=symbol,
                        train_symbols=symbols,
                        samples=samples,
                        contract=contract,
                        candidate=candidate,
                        dataset=environment.dataset,
                        execution_cost=execution_cost,
                        signal_delay_decisions=signal_delay,
                        decision_bars=decision_bars,
                        fit_cache=fit_cache,
                        prediction_cache=prediction_cache,
                        liquidity_cache=liquidity_cache,
                    )
                    evaluation = evaluate_episode_action_path_on_environment(
                        environment, contract, actions=targets.actions
                    )
                    performance = evaluation.performance
                    collapse = evaluation.collapse_evidence
                    metric = CausalAlphaCandidateEpisodeMetricsV2(
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
                        signal_24h=targets.signal_24h,
                        signal_72h=targets.signal_72h,
                        cost_suppressed_change_count=(
                            targets.target_path.cost_suppressed_change_count
                        ),
                        submitted_change_count=(
                            targets.target_path.submitted_change_count
                        ),
                        strong_reversal_count=(
                            targets.target_path.strong_reversal_count
                        ),
                        command_sign_flip_count=targets.target_path.sign_flip_count,
                        execution_rejection_count=(collapse.execution_rejection_count),
                        execution_rejection_reason_counts=(
                            collapse.execution_rejection_reason_counts
                        ),
                        risk_projection_reason_counts=(
                            collapse.risk_projection_reason_counts
                        ),
                        hard_risk_violation=collapse.hard_risk_violation,
                        liquidity_deleveraging_count=(
                            targets.target_path.liquidity_deleveraging_count
                        ),
                        liquidity_weight_cap_min=float(
                            np.min(targets.target_path.liquidity_weight_caps)
                        ),
                        liquidity_weight_cap_median=float(
                            np.median(targets.target_path.liquidity_weight_caps)
                        ),
                        liquidity_weight_cap_max=float(
                            np.max(targets.target_path.liquidity_weight_caps)
                        ),
                    )
                    records[candidate.digest].append(metric)
                    completed.add(identity)
                    completed_replays += 1
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "candidate_digest": candidate.digest,
                                "completed_replays": completed_replays,
                                "episode_index": contract.episode_index,
                                "episode_metric": metric.to_payload(),
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
                                "phase": "causal_teacher_selection_v2",
                                "liquidity_cap_cache_hits": liquidity_cache.hit_count,
                                "liquidity_cap_calculation_count": (
                                    liquidity_cache.calculation_count
                                ),
                                "prediction_cache_hits": prediction_cache.hit_count,
                                "prediction_count": prediction_cache.prediction_count,
                                "symbol": symbol,
                                "total_replays": total_replays,
                            }
                        )
        finally:
            close()
    return rank_cost_aware_causal_alpha_candidates(
        candidates=candidates,
        metrics={digest: tuple(values) for digest, values in records.items()},
        thresholds=thresholds,
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
    portfolio_risk_configs: list[PortfolioRiskConfig] = []
    observed_episode_hours: list[float] = []
    datasets: dict[str, Any] = {}
    execution_costs: dict[str, Any] = {}
    signal_delays: dict[str, int] = {}
    decision_bars_by_symbol: dict[str, int] = {}
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
            portfolio_risk_config = getattr(
                getattr(environment, "portfolio_risk", None), "config", None
            )
            if not isinstance(portfolio_risk_config, PortfolioRiskConfig):
                raise TypeError(
                    "causal alpha environment portfolio risk config is unavailable"
                )
            portfolio_risk_configs.append(portfolio_risk_config)
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
            datasets[symbol] = environment.dataset
            execution_costs[symbol] = getattr(
                environment.config, "execution_cost", None
            )
            signal_delay = getattr(environment.config, "signal_delay_decisions", None)
            decision_bars = getattr(environment, "decision_bars", None)
            if signal_delay not in {0, 1}:
                raise ValueError("causal alpha signal delay is unavailable")
            if (
                isinstance(decision_bars, bool)
                or not isinstance(decision_bars, int)
                or decision_bars <= 0
            ):
                raise ValueError("causal alpha decision_bars is unavailable")
            signal_delays[symbol] = int(signal_delay)
            decision_bars_by_symbol[symbol] = decision_bars
        finally:
            close()
    validate_universal_causal_alpha_partitions(
        train_symbols=symbols,
        partitions=partitions,
    )
    if len({content_digest(config) for config in risk_configs}) != 1:
        raise ValueError("causal alpha train-symbol risk configs differ")
    if len({content_digest(config) for config in portfolio_risk_configs}) != 1:
        raise ValueError("causal alpha train-symbol portfolio risk configs differ")
    max_position_to_market_notional = portfolio_risk_configs[
        0
    ].max_position_to_market_notional
    if max_position_to_market_notional is None:
        raise ValueError(
            "causal alpha canonical package requires a hard market-notional cap"
        )
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
        default_cost_aware_causal_alpha_candidate_grid(
            risk_config=risk_configs[0],
            max_position_to_market_notional=max_position_to_market_notional,
        )
        if candidates is None
        else tuple(candidates)
    )
    if not candidate_values or any(
        candidate.economic_controller is None for candidate in candidate_values
    ):
        raise ValueError("causal alpha canonical package requires v2 candidates")
    if any(
        candidate.economic_controller is None
        or candidate.economic_controller.max_position_to_market_notional
        != max_position_to_market_notional
        for candidate in candidate_values
    ):
        raise ValueError(
            "causal alpha candidate liquidity cap differs from hard portfolio risk"
        )
    thresholds = CausalAlphaSelectionThresholds()
    grid_digest = cost_aware_causal_alpha_grid_digest(candidate_values, thresholds)
    generator_code_digest = causal_alpha_generator_code_digest()
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
    checkpoint_path = (
        selection_path.parent / "causal-teacher-selection-checkpoint-v2.jsonl"
    )
    initial_selection_metrics = load_causal_alpha_selection_checkpoint_v2(
        checkpoint_path,
        expected_grid_digest=grid_digest,
        expected_generator_code_digest=generator_code_digest,
    )
    latest_progress: dict[str, object] = {}

    def persist_selection_progress(payload: Mapping[str, object]) -> None:
        raw_metric = payload.get("episode_metric")
        if isinstance(raw_metric, Mapping):
            metric = causal_alpha_candidate_metric_v2_from_payload(raw_metric)
            write_causal_alpha_selection_checkpoint_metric_v2(
                checkpoint_path,
                metric,
                grid_digest=grid_digest,
                generator_code_digest=generator_code_digest,
            )
        latest_progress.clear()
        latest_progress.update(payload)
        atomic_write_bytes(
            progress_path,
            canonical_json_bytes(dict(payload)) + b"\n",
        )

    try:
        selection = evaluate_cost_aware_causal_alpha_selection(
            train_symbols=symbols,
            samples=samples,
            partitions=partitions,
            candidates=candidate_values,
            environment_factories=environment_factories,
            episode_hours=resolved_episode_hours,
            thresholds=thresholds,
            fit_cache=fit_cache,
            progress_callback=persist_selection_progress,
            initial_metrics=initial_selection_metrics,
        )
    except CausalAlphaSelectionRejectedV2 as rejection:
        rejection_path = (
            selection_path.parent / "causal-teacher-selection-rejected.json"
        )
        persist_causal_alpha_selection_rejection(rejection_path, rejection)
        persist_selection_progress(
            {
                **latest_progress,
                "phase": "causal_teacher_selection_rejected",
                "selection_rejection_digest": rejection.digest,
            }
        )
        raise
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
    selected_economic = selected.economic_controller
    if selected_economic is None:
        raise RuntimeError("causal alpha selected economic controller is unavailable")
    teacher_config_digest = content_digest(
        {
            "feature_schema_digest": feature_schema_digest,
            "generator_code_digest": generator_code_digest,
            "economic_controller_digest": selected_economic.digest,
            "schema_version": "universal_causal_alpha_teacher_config_v2",
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
            economic_controller_config=selected_economic,
            dataset=datasets[symbol],
            execution_cost=execution_costs[symbol],
            signal_delay_decisions=signal_delays[symbol],
            decision_bars=decision_bars_by_symbol[symbol],
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
    "CausalAlphaCandidateEpisodeMetricsV2",
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
    "causal_alpha_candidate_metric_v2_from_payload",
    "default_causal_alpha_candidate_grid",
    "evaluate_causal_alpha_selection",
    "evaluate_cost_aware_causal_alpha_selection",
    "evaluate_causal_alpha_teacher_holdouts",
    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
    "load_causal_alpha_selection_checkpoint",
    "load_causal_alpha_selection_checkpoint_v2",
    "persist_causal_alpha_selection_rejection",
    "rank_causal_alpha_candidates",
    "validate_universal_causal_alpha_partitions",
    "write_causal_alpha_selection_checkpoint_metric",
    "write_causal_alpha_selection_checkpoint_metric_v2",
]
