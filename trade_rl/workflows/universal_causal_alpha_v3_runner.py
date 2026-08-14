"""Artifact-bound orchestration for causal alpha V3 research evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherAdmissionEvidence,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path,
    evaluate_episode_action_path_on_environment,
)
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3SelectionGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3AdmissionRecord,
    CausalAlphaV3ReplayMetric,
    CausalAlphaV3SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    rank_causal_alpha_v3_candidates,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3NestedPartition,
)
from trade_rl.workflows.universal_causal_alpha_v3_store import CausalAlphaV3RecordStore
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    CausalAlphaV3FitCache,
    build_causal_alpha_v3_contract_targets,
)


def _expected_selection_contracts(
    *,
    train_symbols: tuple[str, ...],
    nested_partitions: Mapping[str, CausalAlphaV3NestedPartition],
    candidates: tuple[CausalAlphaV3Candidate, ...],
) -> dict[tuple[str, str, int], str]:
    expected: dict[tuple[str, str, int], str] = {}
    for candidate in candidates:
        for symbol in train_symbols:
            for contract in nested_partitions[symbol].economic_contracts:
                identity = (candidate.digest, symbol, contract.episode_index)
                if identity in expected:
                    raise ValueError("V3 expected selection scope is duplicated")
                expected[identity] = contract.digest
    return expected


def _environment_target_kwargs(environment: Any) -> dict[str, object]:
    config = getattr(environment, "config", None)
    return {
        "dataset": getattr(environment, "dataset", None),
        "decision_bars": getattr(environment, "decision_bars", None),
        "execution_cost": getattr(config, "execution_cost", None),
        "signal_delay_decisions": getattr(config, "signal_delay_decisions", None),
    }


def evaluate_causal_alpha_v3_selection(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    nested_partitions: Mapping[str, CausalAlphaV3NestedPartition],
    candidates: tuple[CausalAlphaV3Candidate, ...],
    environment_factories: Mapping[str, Callable[[], Any]],
    episode_hours: float,
    thresholds: CausalAlphaV3SelectionGate,
    run_manifest_digest: str,
    freeze_digest: str,
    store: CausalAlphaV3RecordStore,
    max_position_to_market_notional: float = 0.02,
) -> CausalAlphaV3SelectionEvidence:
    """Resume or execute V3 economic scopes through the production environment."""

    symbols = tuple(train_symbols)
    candidate_values = tuple(candidates)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("V3 selection train_symbols must be unique")
    if set(samples) != set(symbols) or set(nested_partitions) != set(symbols):
        raise ValueError("V3 selection sample/partition scope must match train_symbols")
    if set(environment_factories) != set(symbols):
        raise ValueError("V3 selection environment factories must match train_symbols")
    if not np.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("V3 selection episode_hours must be positive")
    if not candidate_values:
        raise ValueError("V3 selection requires frozen candidates")
    if (
        store.run_manifest_digest != run_manifest_digest
        or store.freeze_digest != freeze_digest
    ):
        raise ValueError("V3 selection store identity does not match run/freeze")

    expected = _expected_selection_contracts(
        train_symbols=symbols,
        nested_partitions=nested_partitions,
        candidates=candidate_values,
    )
    completed = store.load_replay_metrics(expected_contract_digests=expected)
    records: dict[str, list[CausalAlphaV3ReplayMetric]] = {
        candidate.digest: [] for candidate in candidate_values
    }
    for metric in completed.values():
        records[metric.candidate_digest].append(metric)
    rejected = {
        candidate.digest
        for candidate in candidate_values
        if any(
            item.irrecoverably_rejected(thresholds)
            for item in records[candidate.digest]
        )
    }
    fit_cache = CausalAlphaV3FitCache(train_symbols=symbols, samples=samples)
    episode_days = float(episode_hours) / 24.0

    for symbol in symbols:
        missing_for_symbol = tuple(
            (candidate, contract)
            for candidate in candidate_values
            if candidate.digest not in rejected
            for contract in nested_partitions[symbol].economic_contracts
            if (candidate.digest, symbol, contract.episode_index) not in completed
        )
        if not missing_for_symbol:
            continue
        factory = environment_factories[symbol]
        if not callable(factory):
            raise TypeError("V3 selection environment factory must be callable")
        environment = factory()
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("V3 selection environment must be closable")
        try:
            for candidate, contract in missing_for_symbol:
                if candidate.digest in rejected:
                    continue
                identity = (candidate.digest, symbol, contract.episode_index)
                if identity in completed:
                    continue
                targets = build_causal_alpha_v3_contract_targets(
                    symbol=symbol,
                    train_symbols=symbols,
                    samples=samples,
                    contract=contract,
                    candidate=candidate,
                    max_position_to_market_notional=max_position_to_market_notional,
                    fit_cache=fit_cache,
                    **_environment_target_kwargs(environment),
                )
                evaluation = evaluate_episode_action_path_on_environment(
                    environment,
                    contract,
                    actions=targets.actions,
                )
                performance = evaluation.performance
                collapse = evaluation.collapse_evidence
                metric = CausalAlphaV3ReplayMetric(
                    run_manifest_digest=run_manifest_digest,
                    freeze_digest=freeze_digest,
                    candidate_digest=candidate.digest,
                    symbol=symbol,
                    episode_index=contract.episode_index,
                    contract_digest=contract.digest,
                    fit_digest=targets.fit_digest,
                    forecast_digest=targets.forecast_digest,
                    target_path_digest=targets.target_path.digest,
                    gross_return=float(performance.gross_return),
                    net_return=float(performance.net_return),
                    turnover_per_day=float(performance.turnover_total) / episode_days,
                    total_execution_cost=float(performance.cost_total),
                    trade_count=int(performance.trade_count),
                    submitted_change_count=int(
                        targets.target_path.submitted_change_count
                    ),
                    sign_flip_count=int(targets.target_path.sign_flip_count),
                    liquidity_deleveraging_count=int(
                        targets.target_path.liquidity_deleveraging_count
                    ),
                    execution_rejection_reason_counts=tuple(
                        sorted(
                            tuple(
                                getattr(
                                    collapse,
                                    "execution_rejection_reason_counts",
                                    (),
                                )
                            )
                        )
                    ),
                    risk_projection_reason_counts=tuple(
                        sorted(
                            tuple(
                                getattr(collapse, "risk_projection_reason_counts", ())
                            )
                        )
                    ),
                    target_reason_counts=tuple(
                        sorted(Counter(targets.target_path.reasons).items())
                    ),
                    hard_risk_violation=bool(
                        getattr(collapse, "hard_risk_violation", False)
                    ),
                )
                store.write_replay_metric(metric)
                completed[identity] = metric
                records[candidate.digest].append(metric)
                if metric.irrecoverably_rejected(thresholds):
                    rejected.add(candidate.digest)
        finally:
            close()

    ordered_metrics = {
        candidate.digest: tuple(
            sorted(
                records[candidate.digest],
                key=lambda item: (symbols.index(item.symbol), item.episode_index),
            )
        )
        for candidate in candidate_values
    }
    return rank_causal_alpha_v3_candidates(
        candidates=candidate_values,
        metrics=ordered_metrics,
        thresholds=thresholds,
        freeze_digest=freeze_digest,
    )


def evaluate_causal_alpha_v3_admission(
    *,
    train_symbols: tuple[str, ...],
    batches: Mapping[str, EpisodeOracleBatch],
    environment_factories: Mapping[str, Callable[[], Any]],
    episode_hours: float,
    run_manifest_digest: str,
    freeze_digest: str,
    selection: CausalAlphaV3SelectionEvidence,
    store: CausalAlphaV3RecordStore,
) -> CausalAlphaTeacherAdmissionEvidence:
    """Evaluate the selected untouched holdout once per durably persisted symbol."""

    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("V3 admission train_symbols must be unique")
    if set(batches) != set(symbols) or set(environment_factories) != set(symbols):
        raise ValueError("V3 admission batch/factory scope must match train_symbols")
    if not np.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("V3 admission episode_hours must be positive")
    if selection.freeze_digest != freeze_digest:
        raise ValueError("V3 admission selection/freeze identity drifted")
    if (
        store.run_manifest_digest != run_manifest_digest
        or store.freeze_digest != freeze_digest
    ):
        raise ValueError("V3 admission store identity does not match run/freeze")
    expected = {symbol: batches[symbol].contracts[-1].digest for symbol in symbols}
    records = store.load_admission_records(
        expected_contract_digests=expected,
        selection_digest=selection.digest,
        selected_candidate_digest=selection.selected_candidate_digest,
    )
    episode_days = float(episode_hours) / 24.0
    for symbol in symbols:
        if symbol in records:
            continue
        batch = batches[symbol]
        if not batch.contracts or len(batch.targets) != len(batch.contracts):
            raise ValueError("V3 admission batch is invalid")
        factory = environment_factories[symbol]
        if not callable(factory):
            raise TypeError("V3 admission environment factory must be callable")
        contract = batch.contracts[-1]
        evaluation = evaluate_episode_action_path(
            factory,
            contract,
            actions=batch.targets[-1],
        )
        performance = evaluation.performance
        record = CausalAlphaV3AdmissionRecord(
            run_manifest_digest=run_manifest_digest,
            freeze_digest=freeze_digest,
            selection_digest=selection.digest,
            selected_candidate_digest=selection.selected_candidate_digest,
            symbol=symbol,
            contract_digest=contract.digest,
            gross_return=float(performance.gross_return),
            net_return=float(performance.net_return),
            turnover_per_day=float(performance.turnover_total) / episode_days,
            total_execution_cost=float(performance.cost_total),
            trade_count=int(performance.trade_count),
            maximum_drawdown=float(performance.maximum_drawdown),
        )
        store.write_admission_record(record)
        records[symbol] = record
    return evaluate_causal_alpha_teacher_admission(
        tuple(records[symbol].to_holdout_metric() for symbol in symbols)
    )


__all__ = [
    "evaluate_causal_alpha_v3_admission",
    "evaluate_causal_alpha_v3_selection",
]
