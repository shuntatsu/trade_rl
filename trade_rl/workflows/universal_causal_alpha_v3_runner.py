"""Artifact-bound orchestration for causal alpha V3 research evidence."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

import trade_rl.learning.causal_alpha_teacher as _causal_teacher_module
import trade_rl.learning.causal_alpha_v3 as _causal_v3_module
import trade_rl.learning.episode_oracle_bc as _episode_evaluation_module
import trade_rl.workflows.universal_causal_alpha_costs as _causal_costs_module
import trade_rl.workflows.universal_causal_alpha_v3 as _universal_v3_module
import trade_rl.workflows.universal_causal_alpha_v3_config as _v3_config_module
import trade_rl.workflows.universal_causal_alpha_v3_contracts as _v3_contracts_module
import trade_rl.workflows.universal_causal_alpha_v3_selection as _v3_selection_module
import trade_rl.workflows.universal_causal_alpha_v3_signal as _v3_signal_module
import trade_rl.workflows.universal_causal_alpha_v3_store as _v3_store_module
import trade_rl.workflows.universal_causal_alpha_v3_teacher as _v3_teacher_module
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherAdmissionEvidence,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path,
    evaluate_episode_action_path_on_environment,
)
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    build_causal_alpha_symbol_samples,
    build_chronological_episode_partition,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3ResearchConfig,
    CausalAlphaV3SelectionGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3AdmissionRecord,
    CausalAlphaV3CandidateFreeze,
    CausalAlphaV3ReplayMetric,
    CausalAlphaV3RunManifest,
    CausalAlphaV3SelectionEvidence,
    UniversalCausalAlphaV3TeacherPackage,
)
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    CausalAlphaV3SelectionRejected,
    rank_causal_alpha_v3_candidates,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3NestedPartition,
    CausalAlphaV3SignalGateEvidence,
    evaluate_causal_alpha_v3_signal_gate,
    split_causal_alpha_v3_partitions,
)
from trade_rl.workflows.universal_causal_alpha_v3_store import CausalAlphaV3RecordStore
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    CausalAlphaV3FitCache,
    CausalAlphaV3SignalScopeUnavailable,
    build_causal_alpha_v3_contract_targets,
    build_causal_alpha_v3_episode_batch,
    build_causal_alpha_v3_signal_scope_metric,
)
from trade_rl.workflows.universal_training_runner import UniversalTrainingRuntime


@dataclass(frozen=True, slots=True)
class CausalAlphaV3PreparedResearchData:
    """Verified real-data inputs required by the deterministic V3 runner."""

    train_symbols: tuple[str, ...]
    partitions: Mapping[str, CausalAlphaEpisodePartition]
    samples: Mapping[str, Any]
    environment_factories: Mapping[str, Callable[[], Any]]
    episode_hours: float
    datasets: Mapping[str, Any]
    execution_costs: Mapping[str, Any]
    signal_delays: Mapping[str, int]
    decision_bars: Mapping[str, int]
    max_position_to_market_notional: float
    catalog_digest: str
    partition_digest: str
    split_manifest_digest: str
    feature_schema_digest: str
    statistics_digest: str

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols) or any(not item for item in symbols):
            raise ValueError("V3 prepared train_symbols must be non-empty and unique")
        mappings = {
            "partitions": dict(self.partitions),
            "samples": dict(self.samples),
            "environment_factories": dict(self.environment_factories),
            "datasets": dict(self.datasets),
            "execution_costs": dict(self.execution_costs),
            "signal_delays": dict(self.signal_delays),
            "decision_bars": dict(self.decision_bars),
        }
        if any(set(value) != set(symbols) for value in mappings.values()):
            raise ValueError("V3 prepared mappings must exactly match train_symbols")
        for symbol in symbols:
            partition = mappings["partitions"][symbol]
            if not isinstance(partition, CausalAlphaEpisodePartition):
                raise TypeError("V3 prepared partition type is invalid")
            sample_dataset_id = getattr(mappings["samples"][symbol], "dataset_id", None)
            if sample_dataset_id != partition.holdout_contract.dataset_id:
                raise ValueError("V3 prepared sample/partition dataset identity drifted")
            if not callable(mappings["environment_factories"][symbol]):
                raise TypeError("V3 prepared environment factory must be callable")
            delay = mappings["signal_delays"][symbol]
            if isinstance(delay, bool) or not isinstance(delay, int) or delay not in {0, 1}:
                raise ValueError("V3 prepared signal delay must be 0 or 1")
            bars = mappings["decision_bars"][symbol]
            if isinstance(bars, bool) or not isinstance(bars, int) or bars <= 0:
                raise ValueError("V3 prepared decision_bars must be positive")
        if not math.isfinite(self.episode_hours) or self.episode_hours <= 0.0:
            raise ValueError("V3 prepared episode_hours must be positive")
        if (
            not math.isfinite(self.max_position_to_market_notional)
            or abs(self.max_position_to_market_notional - 0.02) > 1e-12
        ):
            raise ValueError("V3 prepared hard market-notional cap must remain 0.02")
        for field in (
            "catalog_digest",
            "partition_digest",
            "split_manifest_digest",
            "feature_schema_digest",
            "statistics_digest",
        ):
            require_sha256(getattr(self, field), field=f"V3 prepared {field}")
        object.__setattr__(self, "train_symbols", symbols)
        for field, value in mappings.items():
            object.__setattr__(self, field, value)


class CausalAlphaV3SignalRejected(RuntimeError):
    """Terminal research outcome when no authored V3 fit clears the signal gate."""

    def __init__(self, fit_results: tuple[Mapping[str, object], ...]) -> None:
        self.fit_results = tuple(dict(item) for item in fit_results)
        self.digest = content_digest(
            {
                "fit_results": self.fit_results,
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_signal_rejection_v1",
            }
        )
        super().__init__("no causal alpha V3 fit cleared the signal gate")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "fit_results": self.fit_results,
            "promotion_eligible": False,
            "schema_version": "causal_alpha_v3_signal_rejection_v1",
        }


class CausalAlphaV3AdmissionRejected(RuntimeError):
    """Terminal research outcome when the selected teacher fails untouched holdouts."""

    def __init__(self, *, admission_digest: str, selected_candidate_digest: str) -> None:
        require_sha256(admission_digest, field="V3 rejected admission digest")
        require_sha256(selected_candidate_digest, field="V3 rejected selected candidate")
        self.admission_digest = admission_digest
        self.selected_candidate_digest = selected_candidate_digest
        self.digest = content_digest(
            {
                "admission_digest": admission_digest,
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_admission_rejection_v1",
                "selected_candidate_digest": selected_candidate_digest,
            }
        )
        super().__init__("causal alpha V3 teacher admission failed")

    def to_payload(self) -> dict[str, object]:
        return {
            "admission_digest": self.admission_digest,
            "artifact_digest": self.digest,
            "promotion_eligible": False,
            "schema_version": "causal_alpha_v3_admission_rejection_v1",
            "selected_candidate_digest": self.selected_candidate_digest,
        }


def _authored_config_payload(config: CausalAlphaV3ResearchConfig) -> dict[str, object]:
    return {
        "candidates": tuple(
            {
                "fit": {"ridge_strength": candidate.fit.ridge_strength},
                "name": candidate.name,
                "target": {
                    "alpha_rebalance_decisions": candidate.target.alpha_rebalance_decisions,
                    "edge_margin": candidate.target.edge_margin,
                    "execution_cost_multiplier": candidate.target.execution_cost_multiplier,
                    "max_target_delta": candidate.target.max_target_delta,
                    "strong_reversal_threshold": candidate.target.strong_reversal_threshold,
                    "target_magnitudes": candidate.target.target_magnitudes,
                    "uncertainty_multiplier": candidate.target.uncertainty_multiplier,
                },
            }
            for candidate in config.candidates
        ),
        "nested_selection": {
            "minimum_economic_contract_count": (
                config.nested_selection.minimum_economic_contract_count
            ),
            "signal_contract_count": config.nested_selection.signal_contract_count,
        },
        "schema_version": config.schema_version,
        "selection_gate": {
            "maximum_mean_turnover_per_day": (
                config.selection_gate.maximum_mean_turnover_per_day
            ),
            "maximum_unexplained_execution_rejections": (
                config.selection_gate.maximum_unexplained_execution_rejections
            ),
            "minimum_mean_gross_return": config.selection_gate.minimum_mean_gross_return,
            "minimum_mean_net_return": config.selection_gate.minimum_mean_net_return,
            "minimum_positive_gross_episode_fraction": (
                config.selection_gate.minimum_positive_gross_episode_fraction
            ),
            "minimum_symbol_episode_net_return": (
                config.selection_gate.minimum_symbol_episode_net_return
            ),
        },
        "signal_gate": {
            "bootstrap_block_size": config.signal_gate.bootstrap_block_size,
            "bootstrap_resamples": config.signal_gate.bootstrap_resamples,
            "bootstrap_seed": config.signal_gate.bootstrap_seed,
            "minimum_direction_accuracy_excess_lower_ci": (
                config.signal_gate.minimum_direction_accuracy_excess_lower_ci
            ),
            "minimum_rank_ic_lower_ci": config.signal_gate.minimum_rank_ic_lower_ci,
            "minimum_scope_count": config.signal_gate.minimum_scope_count,
            "minimum_scope_coverage": config.signal_gate.minimum_scope_coverage,
            "minimum_top_bottom_spread_lower_ci": (
                config.signal_gate.minimum_top_bottom_spread_lower_ci
            ),
        },
    }


def causal_alpha_v3_generator_code_digest() -> str:
    """Bind V3 evidence to all source modules that define fit/replay semantics."""

    modules = (
        _causal_teacher_module,
        _causal_v3_module,
        _episode_evaluation_module,
        _causal_costs_module,
        _universal_v3_module,
        _v3_config_module,
        _v3_contracts_module,
        _v3_selection_module,
        _v3_signal_module,
        _v3_store_module,
        _v3_teacher_module,
    )
    files: dict[str, str] = {}
    for module in modules:
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise RuntimeError("V3 generator source path is unavailable")
        files[module.__name__] = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    files[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return content_digest(
        {
            "files": files,
            "schema_version": "universal_causal_alpha_v3_generator_code_v1",
        }
    )


def prepare_causal_alpha_v3_research_data(
    *,
    runtime: UniversalTrainingRuntime,
    fold_train_range: tuple[int, int],
) -> CausalAlphaV3PreparedResearchData:
    """Resolve real train artifacts once and close all maintained runtime identities."""

    if not isinstance(runtime, UniversalTrainingRuntime):
        raise TypeError("V3 research preparation requires UniversalTrainingRuntime")
    start, stop = fold_train_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop
    ):
        raise ValueError("V3 fold_train_range must be a valid half-open range")
    routed = runtime.routed_environment_factory
    symbols = tuple(runtime.train_symbols)
    bindings = tuple(routed.bindings)
    if tuple(binding.concrete_symbol for binding in bindings) != symbols:
        raise ValueError("V3 runtime bindings drifted from train_symbols")
    concrete = routed.concrete_environment_factory
    provider = routed.instrument_context_provider
    if not callable(concrete) or not callable(provider):
        raise ValueError("V3 runtime concrete factory/context provider is unavailable")

    partitions: dict[str, CausalAlphaEpisodePartition] = {}
    samples: dict[str, CausalAlphaSymbolSamples] = {}
    datasets: dict[str, Any] = {}
    execution_costs: dict[str, ExecutionCostConfig] = {}
    signal_delays: dict[str, int] = {}
    decision_bars: dict[str, int] = {}
    environment_factories: dict[str, Callable[[], Any]] = {}
    episode_hours: list[float] = []
    market_caps: list[float] = []

    for symbol, binding in zip(symbols, bindings, strict=True):
        environment = concrete(binding)
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("V3 concrete environment must be closable")
        try:
            partition = build_chronological_episode_partition(
                environment,
                train_range=fold_train_range,
            )
            sample = build_causal_alpha_symbol_samples(
                environment=environment,
                binding=binding,
                instrument_context_provider=provider,
                train_range=fold_train_range,
                feature_schema_digest=runtime.feature_schema_digest,
            )
            if partition.holdout_contract.dataset_id != sample.dataset_id:
                raise ValueError("V3 prepared partition/sample dataset identity drifted")
            config = getattr(environment, "config", None)
            execution = getattr(config, "execution_cost", None)
            if not isinstance(execution, ExecutionCostConfig):
                raise TypeError("V3 execution cost config is unavailable")
            delay = getattr(config, "signal_delay_decisions", None)
            if isinstance(delay, bool) or not isinstance(delay, int) or delay not in {0, 1}:
                raise ValueError("V3 signal delay is unavailable")
            bars = getattr(environment, "decision_bars", None)
            if isinstance(bars, bool) or not isinstance(bars, int) or bars <= 0:
                raise ValueError("V3 decision_bars is unavailable")
            hours = getattr(config, "episode_hours", None)
            if isinstance(hours, bool) or not isinstance(hours, int | float):
                raise ValueError("V3 episode_hours is unavailable")
            risk = getattr(getattr(environment, "portfolio_risk", None), "config", None)
            if not isinstance(risk, PortfolioRiskConfig):
                raise TypeError("V3 portfolio risk config is unavailable")
            cap = risk.max_position_to_market_notional
            if cap is None or not math.isfinite(cap) or cap <= 0.0:
                raise ValueError("V3 hard market-notional cap is unavailable")
            partitions[symbol] = partition
            samples[symbol] = sample
            datasets[symbol] = environment.dataset
            execution_costs[symbol] = execution
            signal_delays[symbol] = delay
            decision_bars[symbol] = bars
            episode_hours.append(float(hours))
            market_caps.append(float(cap))
            environment_factories[symbol] = partial(concrete, binding)
        finally:
            close()

    if len(set(episode_hours)) != 1:
        raise ValueError("V3 episode_hours differs across train symbols")
    if len(set(market_caps)) != 1 or abs(market_caps[0] - 0.02) > 1e-12:
        raise ValueError("V3 hard market-notional cap must remain exactly 0.02")
    return CausalAlphaV3PreparedResearchData(
        train_symbols=symbols,
        partitions=partitions,
        samples=samples,
        environment_factories=environment_factories,
        episode_hours=episode_hours[0],
        datasets=datasets,
        execution_costs=execution_costs,
        signal_delays=signal_delays,
        decision_bars=decision_bars,
        max_position_to_market_notional=market_caps[0],
        catalog_digest=runtime.catalog_digest,
        partition_digest=runtime.partition_digest,
        split_manifest_digest=runtime.split_manifest_digest,
        feature_schema_digest=runtime.feature_schema_digest,
        statistics_digest=runtime.statistics_digest,
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
            bars = getattr(environment, "decision_bars", None)
            if isinstance(bars, bool) or not isinstance(bars, int) or bars <= 0:
                raise ValueError("V3 selection decision_bars is unavailable")
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
                    dataset=environment.dataset,
                    execution_cost=execution_cost,
                    signal_delay_decisions=signal_delay,
                    decision_bars=bars,
                    max_position_to_market_notional=max_position_to_market_notional,
                    fit_cache=fit_cache,
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
    """Evaluate each selected holdout at most once after a durable result exists."""

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


def run_universal_causal_alpha_v3_research(
    *,
    config: CausalAlphaV3ResearchConfig,
    prepared: CausalAlphaV3PreparedResearchData,
    output_root: Path,
) -> UniversalCausalAlphaV3TeacherPackage:
    """Run signal gate -> freeze -> production selection -> untouched admission."""

    if not isinstance(config, CausalAlphaV3ResearchConfig):
        raise TypeError("V3 research runner requires CausalAlphaV3ResearchConfig")
    if not isinstance(prepared, CausalAlphaV3PreparedResearchData):
        raise TypeError("V3 research runner requires prepared research data")
    symbols = prepared.train_symbols
    nested = split_causal_alpha_v3_partitions(
        prepared.partitions,
        train_symbols=symbols,
        signal_contract_count=config.nested_selection.signal_contract_count,
        minimum_economic_contract_count=(
            config.nested_selection.minimum_economic_contract_count
        ),
    )
    nested_digest = content_digest(
        {
            "partitions": tuple((symbol, nested[symbol].digest) for symbol in symbols),
            "schema_version": "causal_alpha_v3_nested_scope_v1",
        }
    )
    generator_digest = causal_alpha_v3_generator_code_digest()
    manifest = CausalAlphaV3RunManifest(
        train_symbols=symbols,
        config_digest=config.digest,
        catalog_digest=prepared.catalog_digest,
        partition_digest=prepared.partition_digest,
        split_manifest_digest=prepared.split_manifest_digest,
        feature_schema_digest=prepared.feature_schema_digest,
        statistics_digest=prepared.statistics_digest,
        generator_code_digest=generator_digest,
        nested_partition_digest=nested_digest,
    )
    root = Path(output_root)
    base_store = CausalAlphaV3RecordStore(
        root,
        run_manifest_digest=manifest.digest,
    )
    base_store.write_exact_artifact("run-manifest.json", manifest.to_payload())
    base_store.write_exact_artifact("authored-config.json", _authored_config_payload(config))

    representatives: dict[str, CausalAlphaV3Candidate] = {}
    for candidate in config.candidates:
        representatives.setdefault(candidate.fit.digest, candidate)
    fit_cache = CausalAlphaV3FitCache(
        train_symbols=symbols,
        samples=prepared.samples,
    )
    expected_signal_scopes = sum(
        len(nested[symbol].signal_contracts) for symbol in symbols
    )
    passed_signal: dict[str, CausalAlphaV3SignalGateEvidence] = {}
    fit_results: list[dict[str, object]] = []
    for fit_digest, candidate in representatives.items():
        metrics = []
        unavailable: list[str] = []
        for symbol in symbols:
            for contract in nested[symbol].signal_contracts:
                try:
                    metric = build_causal_alpha_v3_signal_scope_metric(
                        symbol=symbol,
                        train_symbols=symbols,
                        samples=prepared.samples,
                        contract=contract,
                        candidate=candidate,
                        fit_cache=fit_cache,
                    )
                except CausalAlphaV3SignalScopeUnavailable:
                    unavailable.append(contract.digest)
                    continue
                if (
                    metric.fit_config_digest != fit_digest
                    or metric.symbol != symbol
                    or metric.episode_index != contract.episode_index
                    or metric.contract_digest != contract.digest
                ):
                    raise ValueError("V3 signal scope evidence identity drifted")
                metrics.append(metric)
        evidence = (
            None
            if not metrics
            else evaluate_causal_alpha_v3_signal_gate(
                tuple(metrics),
                expected_scope_count=expected_signal_scopes,
                gate=config.signal_gate,
            )
        )
        result_payload: dict[str, object] = {
            "evidence": None if evidence is None else evidence.to_payload(),
            "fit_config_digest": fit_digest,
            "passed": False if evidence is None else evidence.passed,
            "promotion_eligible": False,
            "schema_version": "causal_alpha_v3_fit_signal_result_v1",
            "unavailable_scope_contract_digests": tuple(unavailable),
        }
        base_store.write_exact_artifact(
            Path("signal") / f"{fit_digest}.json",
            result_payload,
        )
        fit_results.append(result_payload)
        if evidence is not None and evidence.passed:
            passed_signal[fit_digest] = evidence

    if not passed_signal:
        rejection = CausalAlphaV3SignalRejected(tuple(fit_results))
        base_store.write_exact_artifact("signal/rejection.json", rejection.to_payload())
        raise rejection

    frozen_candidates = tuple(
        candidate for candidate in config.candidates if candidate.fit.digest in passed_signal
    )
    frozen_fit_digests = tuple(
        dict.fromkeys(candidate.fit.digest for candidate in frozen_candidates)
    )
    freeze = CausalAlphaV3CandidateFreeze(
        run_manifest_digest=manifest.digest,
        config_digest=config.digest,
        generator_code_digest=generator_digest,
        nested_partition_digest=nested_digest,
        candidate_digests=tuple(candidate.digest for candidate in frozen_candidates),
        candidate_semantic_digests=tuple(
            candidate.semantic_digest for candidate in frozen_candidates
        ),
        fit_config_digests=frozen_fit_digests,
        signal_evidence_digests=tuple(
            passed_signal[fit_digest].digest for fit_digest in frozen_fit_digests
        ),
    )
    base_store.write_exact_artifact("freeze/candidates.json", freeze.to_payload())
    store = CausalAlphaV3RecordStore(
        root,
        run_manifest_digest=manifest.digest,
        freeze_digest=freeze.digest,
    )
    try:
        selection = evaluate_causal_alpha_v3_selection(
            train_symbols=symbols,
            samples=prepared.samples,
            nested_partitions=nested,
            candidates=frozen_candidates,
            environment_factories=prepared.environment_factories,
            episode_hours=prepared.episode_hours,
            thresholds=config.selection_gate,
            run_manifest_digest=manifest.digest,
            freeze_digest=freeze.digest,
            store=store,
            max_position_to_market_notional=(
                prepared.max_position_to_market_notional
            ),
        )
    except CausalAlphaV3SelectionRejected as rejection:
        store.write_exact_artifact("selection/rejection.json", rejection.to_payload())
        raise
    store.write_exact_artifact("selection/evidence.json", selection.to_payload())

    selected_matches = tuple(
        candidate
        for candidate in frozen_candidates
        if candidate.digest == selection.selected_candidate_digest
    )
    if len(selected_matches) != 1:
        raise RuntimeError("V3 selected candidate cannot be resolved")
    selected = selected_matches[0]
    teacher_config_digest = content_digest(
        {
            "freeze_digest": freeze.digest,
            "generator_code_digest": generator_digest,
            "run_manifest_digest": manifest.digest,
            "schema_version": "causal_alpha_v3_teacher_config_v1",
            "selected_candidate_digest": selected.digest,
            "selection_digest": selection.digest,
        }
    )
    sampling_config_digest = content_digest(
        {
            "nested_partition_digest": nested_digest,
            "schema_version": "causal_alpha_v3_sampling_config_v1",
        }
    )
    batches: dict[str, EpisodeOracleBatch] = {}
    for symbol in symbols:
        execution = prepared.execution_costs[symbol]
        if not isinstance(execution, ExecutionCostConfig):
            raise TypeError("V3 prepared execution cost must be ExecutionCostConfig")
        batches[symbol] = build_causal_alpha_v3_episode_batch(
            symbol=symbol,
            train_symbols=symbols,
            samples=prepared.samples,
            contracts=prepared.partitions[symbol].contracts,
            candidate=selected,
            dataset=prepared.datasets[symbol],
            execution_cost=execution,
            signal_delay_decisions=prepared.signal_delays[symbol],
            decision_bars=prepared.decision_bars[symbol],
            max_position_to_market_notional=(
                prepared.max_position_to_market_notional
            ),
            teacher_config_digest=teacher_config_digest,
            sampling_config_digest=sampling_config_digest,
            fit_cache=fit_cache,
        )
    admission = evaluate_causal_alpha_v3_admission(
        train_symbols=symbols,
        batches=batches,
        environment_factories=prepared.environment_factories,
        episode_hours=prepared.episode_hours,
        run_manifest_digest=manifest.digest,
        freeze_digest=freeze.digest,
        selection=selection,
        store=store,
    )
    store.write_exact_artifact("admission/evidence.json", admission.to_payload())
    if not admission.passed:
        rejection = CausalAlphaV3AdmissionRejected(
            admission_digest=admission.digest,
            selected_candidate_digest=selected.digest,
        )
        store.write_exact_artifact("admission/rejection.json", rejection.to_payload())
        raise rejection

    package = UniversalCausalAlphaV3TeacherPackage(
        train_symbols=symbols,
        batches=batches,
        run_manifest_digest=manifest.digest,
        freeze_digest=freeze.digest,
        selection_digest=selection.digest,
        teacher_admission_digest=admission.digest,
        selected_candidate_digest=selected.digest,
        generator_code_digest=generator_digest,
    )
    store.write_exact_artifact("teacher/package.json", package.to_payload())
    return package


__all__ = [
    "CausalAlphaV3AdmissionRejected",
    "CausalAlphaV3PreparedResearchData",
    "CausalAlphaV3SignalRejected",
    "causal_alpha_v3_generator_code_digest",
    "evaluate_causal_alpha_v3_admission",
    "evaluate_causal_alpha_v3_selection",
    "prepare_causal_alpha_v3_research_data",
    "run_universal_causal_alpha_v3_research",
]
