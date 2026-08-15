"""Production replay stages for the hardened causal alpha V3 workflow."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path,
    evaluate_episode_action_path_on_environment,
    resolve_episode_initial_weights,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3_admission import (
    CausalAlphaV3AdmissionEvidenceV2,
    CausalAlphaV3AdmissionRecordV2,
    evaluate_causal_alpha_v3_admission_gate,
)
from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3SelectionGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3ReplayMetric,
    CausalAlphaV3SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    rank_causal_alpha_v3_candidates,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3NestedPartition,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    CausalAlphaV3ContractTargets,
    CausalAlphaV3FitCache,
    build_causal_alpha_v3_contract_targets,
)


def assert_causal_alpha_v3_contract_initial_state(
    environment: Any, contract: OracleEpisodeContract
) -> None:
    """Require the replay environment to resolve the frozen contract initial state."""

    resolved = resolve_episode_initial_weights(
        environment,
        contract.initial_state_mode,
        contract.start,
    )
    expected = np.asarray(contract.initial_weights, dtype=np.float64)
    if resolved.shape != expected.shape or not np.array_equal(resolved, expected):
        raise ValueError("V3 replay initial state drifted from frozen contract")


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
    store: CausalAlphaV3ArtifactStore,
    max_position_to_market_notional: float = 0.02,
    build_targets: Callable[..., CausalAlphaV3ContractTargets] = (
        build_causal_alpha_v3_contract_targets
    ),
    evaluate_path: Callable[..., Any] = evaluate_episode_action_path_on_environment,
) -> CausalAlphaV3SelectionEvidence:
    """Resume or execute economic scopes through the production environment."""

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
        missing = tuple(
            (candidate, contract)
            for candidate in candidate_values
            if candidate.digest not in rejected
            for contract in nested_partitions[symbol].economic_contracts
            if (candidate.digest, symbol, contract.episode_index) not in completed
        )
        if not missing:
            continue
        factory = environment_factories[symbol]
        if not callable(factory):
            raise TypeError("V3 selection environment factory must be callable")
        environment = factory()
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("V3 selection environment must be closable")
        try:
            config = getattr(environment, "config", None)
            execution_cost = getattr(config, "execution_cost", None)
            if not isinstance(execution_cost, ExecutionCostConfig):
                raise TypeError("V3 selection execution cost config is unavailable")
            signal_delay = getattr(config, "signal_delay_decisions", None)
            if (
                isinstance(signal_delay, bool)
                or not isinstance(signal_delay, int)
                or signal_delay not in {0, 1}
            ):
                raise ValueError("V3 selection signal delay is unavailable")
            decision_bars = getattr(environment, "decision_bars", None)
            if (
                isinstance(decision_bars, bool)
                or not isinstance(decision_bars, int)
                or decision_bars <= 0
            ):
                raise ValueError("V3 selection decision_bars is unavailable")
            for candidate, contract in missing:
                if candidate.digest in rejected:
                    continue
                identity = (candidate.digest, symbol, contract.episode_index)
                if identity in completed:
                    continue
                assert_causal_alpha_v3_contract_initial_state(environment, contract)
                targets = build_targets(
                    symbol=symbol,
                    train_symbols=symbols,
                    samples=samples,
                    contract=contract,
                    candidate=candidate,
                    dataset=environment.dataset,
                    execution_cost=execution_cost,
                    signal_delay_decisions=signal_delay,
                    decision_bars=decision_bars,
                    max_position_to_market_notional=max_position_to_market_notional,
                    fit_cache=fit_cache,
                )
                evaluation = evaluate_path(
                    environment, contract, actions=targets.actions
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
                        sorted(collapse.execution_rejection_reason_counts)
                    ),
                    risk_projection_reason_counts=tuple(
                        sorted(collapse.risk_projection_reason_counts)
                    ),
                    target_reason_counts=tuple(
                        sorted(Counter(targets.target_path.reasons).items())
                    ),
                    hard_risk_violation=bool(collapse.hard_risk_violation),
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
    store: CausalAlphaV3ArtifactStore,
    evaluate_path: Callable[..., Any] = evaluate_episode_action_path,
) -> CausalAlphaV3AdmissionEvidenceV2:
    """Replay untouched holdouts and retain hard-risk/rejection evidence."""

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
    expected = {
        symbol: batches[symbol].contracts[-1].digest for symbol in symbols
    }
    records = store.load_admission_records_v2(
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
        probe = factory()
        close = getattr(probe, "close", None)
        if not callable(close):
            raise TypeError("V3 admission environment must be closable")
        try:
            assert_causal_alpha_v3_contract_initial_state(probe, contract)
        finally:
            close()
        evaluation = evaluate_path(factory, contract, actions=batch.targets[-1])
        performance = evaluation.performance
        collapse = evaluation.collapse_evidence
        record = CausalAlphaV3AdmissionRecordV2(
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
            execution_rejection_reason_counts=tuple(
                sorted(collapse.execution_rejection_reason_counts)
            ),
            risk_projection_reason_counts=tuple(
                sorted(collapse.risk_projection_reason_counts)
            ),
            hard_risk_violation=bool(collapse.hard_risk_violation),
        )
        store.write_admission_record_v2(record)
        records[symbol] = record
    return evaluate_causal_alpha_v3_admission_gate(
        tuple(records[symbol] for symbol in symbols)
    )


__all__ = [
    "assert_causal_alpha_v3_contract_initial_state",
    "evaluate_causal_alpha_v3_admission",
    "evaluate_causal_alpha_v3_selection",
]
