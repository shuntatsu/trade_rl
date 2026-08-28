"""Artifact-bound DB-backed concrete execution for Causal Alpha V6."""

from __future__ import annotations

import gc
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import causal_alpha_overlap_uniqueness_weights
from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4FitConfig,
    CausalAlphaV4UncertaintyModel,
    fit_causal_alpha_v4_uncertainty,
)
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetConfig,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v6_target import causal_alpha_v6_target_path
from trade_rl.learning.rollout_evaluation import evaluate_action_path
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_costs import (
    causal_alpha_liquidity_weight_caps,
    causal_alpha_one_way_cost_rates,
)
from trade_rl.workflows.universal_causal_alpha_v4_fitting import (
    CausalAlphaV4Fit,
    fit_causal_alpha_v4,
)
from trade_rl.workflows.universal_causal_alpha_v4_liveness_inputs import (
    build_causal_alpha_v4_liveness_inputs,
)
from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    CausalAlphaV4LivenessEvidence,
    CausalAlphaV4SignalGateConfig,
    CausalAlphaV4SignalScopeMetric,
    build_causal_alpha_v4_liveness_evidence,
    build_causal_alpha_v4_signal_scope_metrics,
    evaluate_causal_alpha_v4_signal_gate,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    prepare_causal_alpha_v4_stage_data,
    slice_causal_alpha_v4_forecast,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_science import (
    resolve_causal_alpha_v4_contract_rows,
    resolve_causal_alpha_v4_stage_state_inputs,
)
from trade_rl.workflows.universal_causal_alpha_v6_admission import (
    evaluate_causal_alpha_v6_admission,
)
from trade_rl.workflows.universal_causal_alpha_v6_artifact_store import (
    CausalAlphaV6ArtifactStore,
    CausalAlphaV6RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v6_pipeline import (
    CausalAlphaV6ResearchPackage,
    run_universal_causal_alpha_v6_research_pipeline,
)
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
    build_causal_alpha_v6_replay_metric,
)
from trade_rl.workflows.universal_causal_alpha_v6_selection import (
    CausalAlphaV6SelectionEvidence,
    evaluate_causal_alpha_v6_selection,
)
from trade_rl.workflows.universal_causal_alpha_v6_signal import (
    CausalAlphaV6SignalEvidence,
    CausalAlphaV6SignalScopeMetric,
    build_causal_alpha_v6_signal_scope_metric,
    evaluate_causal_alpha_v6_signal_gate,
)


def _fit_one(prepared: Any, cutoff: int) -> CausalAlphaV4Fit:
    return fit_causal_alpha_v4(
        train_symbols=prepared.train_symbols,
        samples=prepared.samples,
        knowledge_cutoff=cutoff,
        config=CausalAlphaV4FitConfig(),
    )


def _horizon_weights(sample: Any, *, cutoff: int, state: Any) -> dict[str, np.ndarray]:
    decisions = np.asarray(sample.decision_indices, dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        labels = np.asarray(getattr(sample, f"labels_{horizon}"), dtype=np.float64)
        ends = np.asarray(
            getattr(sample, f"label_end_indices_{horizon}"), dtype=np.int64
        )
        raw = causal_alpha_overlap_uniqueness_weights(
            decisions,
            ends,
            knowledge_cutoff=cutoff,
        )
        eligible = (
            state.state_eligible & np.isfinite(labels) & (ends >= 0) & (ends < cutoff)
        )
        result[horizon] = np.where(eligible, raw, 0.0)
    return result


def _uncertainties(
    fit: CausalAlphaV4Fit,
    sample: Any,
    *,
    forecast: Any | None = None,
) -> dict[str, np.ndarray]:
    full_forecast = fit.predict(sample) if forecast is None else forecast
    state = resolve_causal_alpha_v4_stage_state_inputs(sample)
    labels = {
        horizon: np.asarray(getattr(sample, f"labels_{horizon}"), dtype=np.float64)
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    model: CausalAlphaV4UncertaintyModel = fit_causal_alpha_v4_uncertainty(
        final_predictions=full_forecast.final_predictions,
        labels=labels,
        weights=_horizon_weights(sample, cutoff=fit.knowledge_cutoff, state=state),
        state_eligible=state.state_eligible,
        realized_volatility=state.realized_volatility,
        liquidity=state.liquidity,
        basis_positioning_stress=state.basis_positioning_stress,
    )
    return {
        horizon: model.resolve_uncertainty(
            horizon=horizon,
            realized_volatility=state.realized_volatility,
            liquidity=state.liquidity,
            basis_positioning_stress=state.basis_positioning_stress,
        )
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }


def _contract_column(prepared: Any, field: str, index: int) -> tuple[Any, ...]:
    contracts = tuple(
        getattr(prepared.nested_partitions[symbol], field)[index]
        for symbol in prepared.train_symbols
    )
    if len({contract.start for contract in contracts}) != 1:
        raise ValueError(f"V6 {field} cutoff scope drifted")
    return contracts


def _environment(prepared: Any, symbol: str) -> Any:
    factories = getattr(prepared.prepared_v3, "environment_factories", None)
    if not isinstance(factories, Mapping) or not callable(factories.get(symbol)):
        raise ValueError("V6 environment factory scope drifted")
    return factories[symbol]()


def _costs_and_caps(
    prepared: Any,
    symbol: str,
    environment: Any,
    decisions: np.ndarray,
    config: CausalAlphaV6TargetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    execution = getattr(getattr(environment, "config", None), "execution_cost", None)
    if not isinstance(execution, ExecutionCostConfig):
        raise TypeError("V6 environment execution cost config is invalid")
    prepared_costs = getattr(prepared.prepared_v3, "execution_costs", None)
    if not isinstance(prepared_costs, Mapping) or execution != prepared_costs.get(symbol):
        raise ValueError("V6 execution cost identity drifted")
    delay = getattr(prepared.prepared_v3, "signal_delays")[symbol]
    decision_bars = getattr(prepared.prepared_v3, "decision_bars")[symbol]
    costs = causal_alpha_one_way_cost_rates(
        environment.dataset,
        execution,
        decision_indices=decisions,
        signal_delay_decisions=delay,
        decision_bars=decision_bars,
    )
    caps = causal_alpha_liquidity_weight_caps(
        environment.dataset,
        decision_indices=decisions,
        reference_portfolio_value=float(environment.initial_capital),
        max_position_to_market_notional=float(
            prepared.prepared_v3.max_position_to_market_notional
        ),
        lookback_decisions=config.liquidity_lookback_decisions,
        lower_quantile=config.liquidity_lower_quantile,
        safety_multiplier=config.liquidity_safety_multiplier,
    )
    return costs, caps


class _InitialStateEnvironment:
    def __init__(self, environment: Any, mode: str) -> None:
        self._environment = environment
        self._mode = mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def reset(self, *, options: dict[str, object]) -> tuple[object, dict[str, object]]:
        values = dict(options)
        values["initial_state_mode"] = self._mode
        return self._environment.reset(options=values)


def _initial_weight(environment: Any, contract: Any) -> float:
    resolver = getattr(environment, "initial_weights_for_reset", None)
    if not callable(resolver):
        resolver = getattr(environment, "_initial_weights", None)
    if not callable(resolver):
        raise TypeError("V6 environment cannot resolve initial weights")
    weights = np.asarray(
        resolver(contract.initial_state_mode, contract.start),
        dtype=np.float64,
    )
    expected = np.asarray(contract.initial_weights, dtype=np.float64)
    if weights.shape != (1,) or not np.array_equal(weights, expected):
        raise ValueError("V6 initial state drifted from frozen contract")
    return float(weights[0])


def _paired_target_paths(
    *,
    forecast: Any,
    uncertainty: Mapping[str, np.ndarray],
    costs: np.ndarray,
    caps: np.ndarray,
    actionable: np.ndarray,
    config: CausalAlphaV6TargetConfig,
    initial_weight: float,
) -> dict[CausalAlphaV6Candidate, CausalAlphaV6TargetPath]:
    return {
        candidate: causal_alpha_v6_target_path(
            forecast,
            uncertainty=uncertainty,
            one_way_cost_rates=costs,
            liquidity_weight_caps=caps,
            actionable_mask=actionable,
            candidate=candidate,
            config=config,
            initial_weight=initial_weight,
        )
        for candidate in CausalAlphaV6Candidate
    }


def _target_bundle(
    prepared: Any,
    fit: CausalAlphaV4Fit,
    symbol: str,
    contract: Any,
    config: CausalAlphaV6TargetConfig,
) -> tuple[
    np.ndarray,
    Any,
    Mapping[str, np.ndarray],
    dict[CausalAlphaV6Candidate, CausalAlphaV6TargetPath],
]:
    sample = prepared.samples[symbol]
    rows = resolve_causal_alpha_v4_contract_rows(
        sample,
        start=contract.start,
        stop=contract.stop,
    )
    full_forecast = fit.predict(sample)
    forecast = slice_causal_alpha_v4_forecast(full_forecast, rows)
    uncertainty_full = _uncertainties(fit, sample, forecast=full_forecast)
    uncertainty = {
        horizon: uncertainty_full[horizon][rows]
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    state = resolve_causal_alpha_v4_stage_state_inputs(sample)
    environment = _environment(prepared, symbol)
    try:
        costs, caps = _costs_and_caps(
            prepared,
            symbol,
            environment,
            forecast.decision_indices,
            config,
        )
        initial_weight = _initial_weight(environment, contract)
    finally:
        environment.close()
    targets = _paired_target_paths(
        forecast=forecast,
        uncertainty=uncertainty,
        costs=costs,
        caps=caps,
        actionable=np.asarray(state.actionable)[rows],
        config=config,
        initial_weight=initial_weight,
    )
    return rows, forecast, uncertainty, targets


def _reward_scale(environment: Any) -> float:
    config = getattr(environment, "config", None)
    resolver = getattr(config, "resolved_reward_config", None)
    if not callable(resolver):
        raise TypeError("V6 environment cannot resolve the reward contract")
    reward = resolver()
    pure = getattr(reward, "is_pure_net_log_growth", None)
    if not callable(pure) or not pure():
        raise ValueError("V6 replay requires the pure net-log reward contract")
    return float(reward.scale)


def _replay_target(
    prepared: Any,
    symbol: str,
    contract: Any,
    fit: CausalAlphaV4Fit,
    forecast: Any,
    target: CausalAlphaV6TargetPath,
) -> CausalAlphaV6ReplayMetric:
    environment = _environment(prepared, symbol)
    try:
        reward_scale = _reward_scale(environment)
        evaluation = evaluate_action_path(
            _InitialStateEnvironment(environment, contract.initial_state_mode),
            evaluation_range=(contract.start, contract.stop),
            actions=target.targets[:, None].astype(np.float32),
        )
        return build_causal_alpha_v6_replay_metric(
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            symbol=symbol,
            episode_index=contract.episode_index,
            contract_digest=contract.digest,
            fit_digest=fit.digest,
            forecast_digest=forecast.digest,
            target_path=target,
            evaluation=evaluation,
            episode_hours=float(prepared.prepared_v3.episode_hours),
            reward_scale=reward_scale,
        )
    finally:
        environment.close()


def _replay_pair(
    prepared: Any,
    fit: CausalAlphaV4Fit,
    symbol: str,
    contract: Any,
    config: CausalAlphaV6TargetConfig,
) -> dict[CausalAlphaV6Candidate, CausalAlphaV6ReplayMetric]:
    _rows, forecast, _uncertainty, targets = _target_bundle(
        prepared,
        fit,
        symbol,
        contract,
        config,
    )
    return {
        candidate: _replay_target(
            prepared,
            symbol,
            contract,
            fit,
            forecast,
            targets[candidate],
        )
        for candidate in CausalAlphaV6Candidate
    }


def _v4_liveness_digests(
    *,
    fit: CausalAlphaV4Fit,
    sample: Any,
    rows: np.ndarray,
    forecast: Any,
    symbol: str,
) -> dict[str, str]:
    full_forecast = fit.predict(sample)
    state = resolve_causal_alpha_v4_stage_state_inputs(sample)
    uncertainty_model = fit_causal_alpha_v4_uncertainty(
        final_predictions=full_forecast.final_predictions,
        labels={
            horizon: np.asarray(getattr(sample, f"labels_{horizon}"), dtype=np.float64)
            for horizon in CAUSAL_ALPHA_V4_HORIZONS
        },
        weights=_horizon_weights(sample, cutoff=fit.knowledge_cutoff, state=state),
        state_eligible=state.state_eligible,
        realized_volatility=state.realized_volatility,
        liquidity=state.liquidity,
        basis_positioning_stress=state.basis_positioning_stress,
    )
    evidence: dict[str, CausalAlphaV4LivenessEvidence] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        inputs = build_causal_alpha_v4_liveness_inputs(
            fit=fit,
            sample=sample,
            forecast=forecast,
            horizon=horizon,
            row_indices=rows,
        )
        evidence[horizon] = build_causal_alpha_v4_liveness_evidence(
            fit_digest=fit.digest,
            forecast_digest=forecast.digest,
            symbol=symbol,
            horizon=horizon,
            prediction=forecast.final_predictions[horizon],
            direction_score=forecast.direction_scores[horizon],
            intercept=inputs.intercept,
            weighted_final_rmse=uncertainty_model.global_rmse[horizon],
            feature_available=inputs.feature_available,
            constant_feature_mask=inputs.constant_feature_mask,
            contribution_series=inputs.contribution_series,
        )
    return {
        "fast_4h": evidence["4h"].digest,
        "slow_fused": content_digest(
            {
                "liveness_24h_digest": evidence["24h"].digest,
                "liveness_72h_digest": evidence["72h"].digest,
                "schema_version": "causal_alpha_v4_slow_fused_liveness_v1",
            }
        ),
    }


def _progress(*, stage: str, cutoff: int, candidate: str | None = None) -> None:
    payload: dict[str, object] = {
        "cutoff": cutoff,
        "stage": stage,
        "status": "evaluated",
    }
    if candidate is not None:
        payload["candidate"] = candidate
    print(json.dumps(payload, sort_keys=True), flush=True)


def _signal_scope_metrics(
    prepared: Any,
    fit: CausalAlphaV4Fit,
    symbol: str,
    contract: Any,
    config: CausalAlphaV6TargetConfig,
) -> tuple[
    tuple[CausalAlphaV6SignalScopeMetric, ...],
    tuple[CausalAlphaV4SignalScopeMetric, ...],
]:
    sample = prepared.samples[symbol]
    rows, forecast, _uncertainty, targets = _target_bundle(
        prepared,
        fit,
        symbol,
        contract,
        config,
    )
    slow_realized = 0.5 * (
        np.asarray(sample.labels_24h)[rows]
        + np.asarray(sample.labels_72h)[rows] / 3.0
    )
    v6_metrics = tuple(
        build_causal_alpha_v6_signal_scope_metric(
            run_manifest_digest=prepared.run_manifest_digest,
            symbol=symbol,
            episode_index=contract.episode_index,
            contract_start=contract.start,
            contract_stop=contract.stop,
            contract_digest=contract.digest,
            fit_digest=fit.digest,
            target_path=targets[candidate],
            slow_realized_returns=slow_realized,
        )
        for candidate in CausalAlphaV6Candidate
    )
    liveness = _v4_liveness_digests(
        fit=fit,
        sample=sample,
        rows=rows,
        forecast=forecast,
        symbol=symbol,
    )
    state = resolve_causal_alpha_v4_stage_state_inputs(sample)
    v4 = build_causal_alpha_v4_signal_scope_metrics(
        run_manifest_digest=prepared.run_manifest_digest,
        fit_config_digest=CausalAlphaV4FitConfig().digest,
        symbol=symbol,
        episode_index=contract.episode_index,
        contract_start=contract.start,
        contract_stop=contract.stop,
        contract_digest=contract.digest,
        fit_digest=fit.digest,
        forecast=forecast,
        liveness_digests=liveness,
        actionable_mask=state.actionable[rows],
        labels_4h=np.asarray(sample.labels_4h)[rows],
        label_end_indices_4h=np.asarray(sample.label_end_indices_4h)[rows],
        labels_24h=np.asarray(sample.labels_24h)[rows],
        label_end_indices_24h=np.asarray(sample.label_end_indices_24h)[rows],
        labels_72h=np.asarray(sample.labels_72h)[rows],
        label_end_indices_72h=np.asarray(sample.label_end_indices_72h)[rows],
    )
    return v6_metrics, tuple(v4.values())


def _signal_stage(
    prepared: Any,
    config: CausalAlphaV6TargetConfig,
) -> CausalAlphaV6SignalEvidence:
    v6_metrics: list[CausalAlphaV6SignalScopeMetric] = []
    v4_metrics: list[CausalAlphaV4SignalScopeMetric] = []
    count = len(prepared.nested_partitions[prepared.train_symbols[0]].signal_contracts)
    for index in range(count):
        contracts = _contract_column(prepared, "signal_contracts", index)
        cutoff = int(contracts[0].start)
        fit = _fit_one(prepared, cutoff)
        try:
            for symbol, contract in zip(
                prepared.train_symbols,
                contracts,
                strict=True,
            ):
                v6, v4 = _signal_scope_metrics(
                    prepared,
                    fit,
                    symbol,
                    contract,
                    config,
                )
                v6_metrics.extend(v6)
                v4_metrics.extend(v4)
            _progress(stage="signal", cutoff=cutoff)
        finally:
            del fit
            gc.collect()
    expected = len(prepared.train_symbols) * 8
    v4_evidence = evaluate_causal_alpha_v4_signal_gate(
        tuple(v4_metrics),
        expected_raw_scope_count_per_lane=expected,
        gate=CausalAlphaV4SignalGateConfig(),
    )
    return evaluate_causal_alpha_v6_signal_gate(
        tuple(v6_metrics),
        expected_symbols=prepared.train_symbols,
        v4_fast_lane=v4_evidence.fast_4h,
    )


def _artifact(body: dict[str, object]) -> dict[str, object]:
    return {**body, "artifact_digest": content_digest(body)}


def _signal_stage_with_diagnostics(
    prepared: Any,
    config: CausalAlphaV6TargetConfig,
    store: CausalAlphaV6ArtifactStore,
) -> CausalAlphaV6SignalEvidence:
    evidence = _signal_stage(prepared, config)
    store.write_leaf(
        Path("signal") / "diagnostics.json",
        _artifact(
            {
                "schema_version": "causal_alpha_v6_signal_diagnostics_v1",
                "run_manifest_digest": prepared.run_manifest_digest,
                "v4_context_manifest_digest": prepared.v4_context_manifest_digest,
                "config_digest": config.digest,
                "generator_code_digest": prepared.generator_code_digest,
                "signal": evidence.to_payload(),
                "research_only": True,
                "promotion_eligible": False,
            }
        ),
    )
    return evidence


def _selection_stage(
    prepared: Any,
    signal: CausalAlphaV6SignalEvidence,
    config: CausalAlphaV6TargetConfig,
) -> CausalAlphaV6SelectionEvidence:
    if not signal.passed:
        raise ValueError("V6 Selection cannot bypass Signal")
    records: list[CausalAlphaV6ReplayMetric] = []
    count = len(
        prepared.nested_partitions[prepared.train_symbols[0]].economic_contracts
    )
    for index in range(count):
        contracts = _contract_column(prepared, "economic_contracts", index)
        cutoff = int(contracts[0].start)
        fit = _fit_one(prepared, cutoff)
        try:
            for symbol, contract in zip(
                prepared.train_symbols,
                contracts,
                strict=True,
            ):
                pair = _replay_pair(prepared, fit, symbol, contract, config)
                records.extend(pair.values())
            for candidate in CausalAlphaV6Candidate:
                _progress(
                    stage="selection",
                    cutoff=cutoff,
                    candidate=candidate.value,
                )
        finally:
            del fit
            gc.collect()
    return evaluate_causal_alpha_v6_selection(
        tuple(records),
        expected_symbols=prepared.train_symbols,
    )


def _admission_stage(
    prepared: Any,
    signal: CausalAlphaV6SignalEvidence,
    selection: CausalAlphaV6SelectionEvidence,
    config: CausalAlphaV6TargetConfig,
) -> Any:
    if not signal.passed or not selection.passed:
        raise ValueError("V6 Admission cannot bypass upstream gates")
    holdouts = tuple(
        prepared.nested_partitions[symbol].holdout_contract
        for symbol in prepared.train_symbols
    )
    starts = {contract.start for contract in holdouts}
    if len(starts) != 1:
        raise ValueError("V6 holdout starts drifted")
    holdout_start = int(next(iter(starts)))
    fit = _fit_one(prepared, holdout_start)
    try:
        paired = tuple(
            _replay_pair(prepared, fit, symbol, contract, config)
            for symbol, contract in zip(
                prepared.train_symbols,
                holdouts,
                strict=True,
            )
        )
        fast = tuple(pair[CausalAlphaV6Candidate.FAST_ONLY] for pair in paired)
        retention = tuple(
            pair[CausalAlphaV6Candidate.FAST_SLOW_RETENTION] for pair in paired
        )
        selected = (
            fast
            if selection.selected_candidate is CausalAlphaV6Candidate.FAST_ONLY
            else retention
        )
        _progress(stage="admission", cutoff=holdout_start)
        return evaluate_causal_alpha_v6_admission(
            selected,
            fast,
            signal_evidence=signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=holdout_start,
            holdout_start=holdout_start,
        )
    finally:
        del fit
        gc.collect()


def run_causal_alpha_v6_concrete_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV6ResearchPackage:
    """Resolve immutable V4 inputs and run all fixed V6 gates on real data."""

    from trade_rl.workflows.universal_causal_alpha_v4_runtime_adapter import (
        prepare_causal_alpha_v4_runtime_adapter,
    )
    config_path = Path(config_path)
    runner = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v6_runner"
    )
    config = runner.CausalAlphaV6ResearchConfig.from_json(config_path)
    context, runtime, prepared_v3 = prepare_causal_alpha_v4_runtime_adapter(
        run_config_path=Path(run_config_path),
        runtime_manifest_path=Path(runtime_manifest_path),
        v4_context_manifest_path=Path(v4_context_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
    )
    generator_digest = content_digest(
        {
            "schema_version": "causal_alpha_v6_generator_code_v1",
            "source_tree_digest": prepared_v3.execution_identity.source_tree_digest,
        }
    )
    prepared = prepare_causal_alpha_v4_stage_data(
        config_digest=config.target.digest,
        generator_code_digest=generator_digest,
        runtime_context=context,
        runtime=runtime,
        prepared_v3=prepared_v3,
    )
    del context, runtime, prepared_v3
    gc.collect()
    root = Path(output_root)
    with CausalAlphaV6RunLock(root):
        store = CausalAlphaV6ArtifactStore(
            root,
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=config.target.digest,
            generator_code_digest=prepared.generator_code_digest,
        )
        source = json.loads(config_path.read_text(encoding="utf-8"))
        store.write_leaf(
            "authored-config.json",
            _artifact(
                {
                    "schema_version": "causal_alpha_v6_authored_config_record_v1",
                    "run_manifest_digest": prepared.run_manifest_digest,
                    "v4_context_manifest_digest": prepared.v4_context_manifest_digest,
                    "config_digest": config.target.digest,
                    "generator_code_digest": prepared.generator_code_digest,
                    "research_config_digest": config.digest,
                    "source_config": source,
                    "research_only": True,
                    "promotion_eligible": False,
                }
            ),
        )
        return run_universal_causal_alpha_v6_research_pipeline(
            store=store,
            prepare_stage=lambda: prepared,
            signal_stage=lambda value: _signal_stage_with_diagnostics(
                value,
                config.target,
                store,
            ),
            selection_stage=lambda value, signal: _selection_stage(
                value,
                cast(CausalAlphaV6SignalEvidence, signal),
                config.target,
            ),
            admission_stage=lambda value, signal, selection: _admission_stage(
                value,
                cast(CausalAlphaV6SignalEvidence, signal),
                cast(CausalAlphaV6SelectionEvidence, selection),
                config.target,
            ),
        )


__all__ = ["run_causal_alpha_v6_concrete_entry"]
