"""Restart-safe DB-backed Signal and Selection execution for Causal Alpha V10."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.learning.causal_alpha_v8 import (
    CausalAlphaV8Candidate,
    CausalAlphaV8TargetConfig,
    CausalAlphaV8TargetPath,
)
from trade_rl.learning.causal_alpha_v8_target import (
    causal_alpha_v8_target_paths_from_v7,
)
from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Config
from trade_rl.learning.causal_alpha_v9_wave import causal_alpha_v9_wave_target_path
from trade_rl.learning.causal_alpha_v10 import (
    CausalAlphaV10Candidate,
    CausalAlphaV10Config,
    CausalAlphaV10TargetPath,
)
from trade_rl.learning.causal_alpha_v10_fit import (
    CausalAlphaV10DualFit,
    CausalAlphaV10TrainingRows,
    fit_causal_alpha_v10,
)
from trade_rl.learning.causal_alpha_v10_hierarchy import (
    CausalAlphaV10ExecutionContract,
    CausalAlphaV10HierarchyPolicy,
    prepare_causal_alpha_v10_hierarchy_policy,
)
from trade_rl.learning.rollout_evaluation import evaluate_action_path
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
    CausalAlphaV4RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    prepare_causal_alpha_v4_stage_data,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_science import (
    resolve_causal_alpha_v4_contract_rows,
)
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    build_causal_alpha_v6_replay_metric,
)
from trade_rl.workflows.universal_causal_alpha_v6_stage_entry import (
    _contract_column,
    _environment,
    _InitialStateEnvironment,
    _progress,
    _reward_scale,
)
from trade_rl.workflows.universal_causal_alpha_v7_runner import (
    CausalAlphaV7ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v7_stage_entry import (
    _fit_one_calibration,
    _target_bundle,
)
from trade_rl.workflows.universal_causal_alpha_v8_attribution import (
    CausalAlphaV8AttributionEvidence,
    build_causal_alpha_v8_attribution,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v9_stage_entry import (
    _fit_wave as _fit_v9_wave,
)
from trade_rl.workflows.universal_causal_alpha_v9_stage_entry import (
    _signal_evidence as _v9_signal_evidence,
)
from trade_rl.workflows.universal_causal_alpha_v9_stage_entry import (
    _wave_rows as _v9_wave_rows,
)
from trade_rl.workflows.universal_causal_alpha_v10_diagnostics import (
    execution_diagnostics as _execution_diagnostics,
)
from trade_rl.workflows.universal_causal_alpha_v10_diagnostics import (
    execution_trace_payload as _execution_trace_payload,
)
from trade_rl.workflows.universal_causal_alpha_v10_diagnostics import (
    validate_execution_diagnostics as _validate_execution_diagnostics,
)
from trade_rl.workflows.universal_causal_alpha_v10_diagnostics import (
    validate_execution_trace_payload as _validate_execution_trace_payload,
)
from trade_rl.workflows.universal_causal_alpha_v10_gates import (
    V8_CANDIDATE_BY_V10,
    CausalAlphaV10SelectionEvidence,
    CausalAlphaV10SignalEvidence,
    evaluate_causal_alpha_v10_selection,
)

_REPLAY_LEAF_SCHEMA: Final = "causal_alpha_v10_replay_leaf_v3"
_RESULT_SCHEMA: Final = "causal_alpha_v10_terminal_result_v1"


class CausalAlphaV10RunLock(CausalAlphaV4RunLock):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.path = self.root / ".causal-alpha-v10.lock"


def causal_alpha_v10_stage_config_digest(
    source: CausalAlphaV7ResearchConfig,
    v8_target: CausalAlphaV8TargetConfig,
    v9_wave: CausalAlphaV9Config,
    v10: CausalAlphaV10Config,
) -> str:
    return content_digest(
        {
            "calibration_config_digest": source.calibration.digest,
            "schema_version": "causal_alpha_v10_stage_config_v1",
            "v8_target_config_digest": v8_target.digest,
            "v9_wave_config_digest": v9_wave.digest,
            "v10_config_digest": v10.digest,
        }
    )


def _artifact(body: dict[str, object]) -> dict[str, object]:
    return {**body, "artifact_digest": content_digest(body)}


def _feature_surface(sample: Any) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    names = (
        *(f"local:{name}" for name in sample.local_context.feature_names),
        *(f"global:{name}" for name in sample.global_context.feature_names),
    )
    features = np.column_stack(
        (sample.local_context.values, sample.global_context.values)
    )
    available = np.column_stack(
        (sample.local_context.available, sample.global_context.available)
    )
    return names, features, available


def _training_rows(prepared: Any) -> dict[str, CausalAlphaV10TrainingRows]:
    result: dict[str, CausalAlphaV10TrainingRows] = {}
    for symbol in prepared.train_symbols:
        sample = prepared.samples[symbol]
        names, features, available = _feature_surface(sample)
        result[symbol] = CausalAlphaV10TrainingRows(
            symbol=symbol,
            decision_indices=sample.decision_indices,
            fast_label_end_indices=sample.label_end_indices_4h,
            slow_label_end_indices=sample.label_end_indices_72h,
            features=features,
            feature_available=available,
            fast_labels=sample.labels_4h,
            slow_labels=sample.labels_72h,
            feature_names=names,
        )
    return result


def _write_evidence(
    store: CausalAlphaV4ArtifactStore,
    stage: str,
    evidence: object,
) -> None:
    store.write_leaf(
        Path(stage) / "evidence.json",
        store.envelope(
            schema_version=f"causal_alpha_v10_{stage}_envelope_v1",
            evidence_digest=getattr(evidence, "digest"),
            payload=getattr(evidence, "to_payload")(),
        ),
    )


def _signal_evidence(
    prepared: Any,
    *,
    source_config: CausalAlphaV7ResearchConfig,
    v9_config: CausalAlphaV9Config,
    v10_config: CausalAlphaV10Config,
    config_digest: str,
) -> CausalAlphaV10SignalEvidence:
    source = _v9_signal_evidence(
        prepared,
        source_config=source_config,
        wave_config=v9_config,
        config_digest=config_digest,
    )
    training = _training_rows(prepared)
    fit_digests: list[str] = []
    scope_count = 0
    qualified_count = 0
    count = len(prepared.nested_partitions[prepared.train_symbols[0]].signal_contracts)
    for index in range(count):
        contracts = _contract_column(prepared, "signal_contracts", index)
        cutoff = int(contracts[0].start)
        fit = fit_causal_alpha_v10(
            training,
            knowledge_cutoff=cutoff,
            config=v10_config,
        )
        fit_digests.append(fit.digest)
        for symbol, contract in zip(prepared.train_symbols, contracts, strict=True):
            sample = prepared.samples[symbol]
            rows = resolve_causal_alpha_v4_contract_rows(
                sample,
                start=contract.start,
                stop=contract.stop,
            )
            _names, features, _available = _feature_surface(sample)
            heads = fit.slow.predict_heads(features[rows])
            mean = np.mean(heads, axis=0)
            std = np.std(heads, axis=0)
            qualified = np.all(np.sign(heads) == np.sign(heads[0]), axis=0) & (
                np.abs(mean) > std + v10_config.edge_margin
            )
            scope_count += 1
            qualified_count += int(np.any(qualified))
    return CausalAlphaV10SignalEvidence(
        source_v9=source,
        slow_scope_count=scope_count,
        qualified_slow_scope_count=qualified_count,
        dual_fit_digests=tuple(fit_digests),
    )


def _environment_rebalance_contract(
    environment: Any,
) -> CausalAlphaV10ExecutionContract:
    risk = getattr(environment, "pre_trade_risk", None)
    risk_config = getattr(risk, "config", None)
    if not isinstance(risk_config, PreTradeRiskConfig):
        raise TypeError("V10 environment PreTradeRiskConfig is invalid")
    return CausalAlphaV10ExecutionContract(
        max_gross=float(risk_config.max_gross),
        max_abs_weight=float(risk_config.max_abs_weight),
        max_turnover=(
            None
            if risk_config.max_turnover is None
            else float(risk_config.max_turnover)
        ),
        entry_threshold=float(risk_config.entry_threshold),
        exit_threshold=float(risk_config.exit_threshold),
        no_trade_band=float(risk_config.no_trade_band),
        drawdown_start=float(risk_config.drawdown_start),
        drawdown_stop=float(risk_config.drawdown_stop),
        emergency_turnover_override=risk_config.emergency_turnover_override,
        fail_closed_tolerance=float(risk_config.fail_closed_tolerance),
    )


def _execution_rebalance_contract(
    prepared: Any,
    symbol: str,
) -> CausalAlphaV10ExecutionContract:
    environment = _environment(prepared, symbol)
    try:
        return _environment_rebalance_contract(environment)
    finally:
        environment.close()


def _require_execution_rebalance_contract(
    environment: Any,
    expected: CausalAlphaV10ExecutionContract,
) -> None:
    observed = _environment_rebalance_contract(environment)
    if observed.digest != expected.digest:
        raise ValueError("V10 replay execution rebalance contract drifted")


def _target_paths(
    *,
    sample: Any,
    rows: np.ndarray,
    forecast: Any,
    state: Any,
    boundaries: Any,
    v7_targets: Any,
    v9_fit: Any,
    dual_fit: CausalAlphaV10DualFit,
    v8_config: CausalAlphaV8TargetConfig,
    v9_config: CausalAlphaV9Config,
    v10_config: CausalAlphaV10Config,
    execution_rebalance_contract: CausalAlphaV10ExecutionContract,
) -> tuple[
    dict[CausalAlphaV10Candidate, CausalAlphaV10TargetPath],
    CausalAlphaV10HierarchyPolicy,
]:
    v8_targets = causal_alpha_v8_target_paths_from_v7(
        forecast=forecast,
        v7_paths=v7_targets,
        config=v8_config,
    )
    control = v7_targets[CausalAlphaV7Candidate.V6_CONTROL].v6_target_path
    robust = v8_targets[CausalAlphaV8Candidate.ROBUST_CALIBRATED].v6_target_path
    _names, features, _available = _feature_surface(sample)
    scoped_features = features[rows]
    v9 = causal_alpha_v9_wave_target_path(
        decision_indices=forecast.decision_indices,
        head_predictions=v9_fit.predict_heads(scoped_features),
        one_way_cost_rates=control.one_way_cost_rates,
        liquidity_weight_caps=control.liquidity_weight_caps,
        risk_weight_caps=control.risk_weight_caps,
        actionable_mask=control.actionable_mask,
        source_forecast_digest=forecast.digest,
        config=v9_config,
        initial_weight=control.initial_weight,
    )
    controls = {
        CausalAlphaV10Candidate.V8_ROBUST_CONTROL: CausalAlphaV10TargetPath(
            candidate=CausalAlphaV10Candidate.V8_ROBUST_CONTROL,
            v6_target_path=robust,
            source_forecast_digest=forecast.digest,
            fast_fit_digest=v9_fit.digest,
            slow_fit_digest=v9_fit.digest,
            v10_config_digest=v10_config.digest,
        ),
        CausalAlphaV10Candidate.V9_NONLINEAR_CONTROL: CausalAlphaV10TargetPath(
            candidate=CausalAlphaV10Candidate.V9_NONLINEAR_CONTROL,
            v6_target_path=v9,
            source_forecast_digest=forecast.digest,
            fast_fit_digest=v9_fit.digest,
            slow_fit_digest=v9_fit.digest,
            v10_config_digest=v10_config.digest,
        ),
    }
    hierarchy_policy = prepare_causal_alpha_v10_hierarchy_policy(
        decision_indices=forecast.decision_indices,
        fast_head_predictions=dual_fit.fast.predict_heads(scoped_features),
        slow_head_predictions=dual_fit.slow.predict_heads(scoped_features),
        one_way_cost_rates=control.one_way_cost_rates,
        liquidity_weight_caps=control.liquidity_weight_caps,
        risk_weight_caps=control.risk_weight_caps,
        realized_volatility=np.asarray(state.realized_volatility)[rows],
        liquidity=np.asarray(state.liquidity)[rows],
        attribution_boundaries=boundaries,
        actionable_mask=control.actionable_mask,
        source_forecast_digest=forecast.digest,
        dual_fit_digest=dual_fit.digest,
        config=v10_config,
        initial_weight=control.initial_weight,
        execution_contract=execution_rebalance_contract,
    )
    return controls, hierarchy_policy


def _metric_from_evaluation(
    *,
    prepared: Any,
    resolved: Any,
    symbol: str,
    contract: Any,
    forecast: Any,
    state: Any,
    rows: np.ndarray,
    target: CausalAlphaV10TargetPath,
    evaluation: Any,
    config_digest: str,
    boundaries: Any,
    environment: Any,
) -> CausalAlphaV8ReplayMetric:
    base = build_causal_alpha_v6_replay_metric(
        run_manifest_digest=prepared.run_manifest_digest,
        v4_context_manifest_digest=prepared.v4_context_manifest_digest,
        symbol=symbol,
        episode_index=contract.episode_index,
        contract_digest=contract.digest,
        fit_digest=resolved.base_fit.digest,
        forecast_digest=target.v6_target_path.forecast_digest,
        target_path=target.v6_target_path,
        evaluation=evaluation,
        episode_hours=float(prepared.prepared_v3.episode_hours),
        reward_scale=_reward_scale(environment),
    )
    mapped = V8_CANDIDATE_BY_V10[target.candidate]
    compatibility = CausalAlphaV8TargetPath(
        candidate=mapped,
        v6_target_path=target.v6_target_path,
        source_forecast_digest=forecast.digest,
        calibration_fit_digest=target.fast_fit_digest,
        v8_config_digest=target.v10_config_digest,
    )
    source = build_causal_alpha_v8_attribution(
        target_path=compatibility,
        evaluation=evaluation,
        confidence=np.abs(target.v6_target_path.direction_scores_4h),
        realized_volatility=np.asarray(state.realized_volatility)[rows],
        liquidity=np.asarray(state.liquidity)[rows],
        boundaries=boundaries,
        step_hours=float(prepared.prepared_v3.episode_hours) / len(rows),
    )
    attribution = CausalAlphaV8AttributionEvidence(
        candidate=mapped,
        target_path_digest=target.digest,
        boundaries_digest=source.boundaries_digest,
        step_economics_digest=source.step_economics_digest,
        decision_count=source.decision_count,
        gross_log_return=source.gross_log_return,
        net_log_return=source.net_log_return,
        total_execution_cost=source.total_execution_cost,
        total_exposure_hours=source.total_exposure_hours,
        cells=source.cells,
    )
    return CausalAlphaV8ReplayMetric(
        candidate=mapped,
        v6_metric=base,
        attribution=attribution,
        v8_target_path_digest=target.digest,
        source_forecast_digest=forecast.digest,
        calibration_fit_digest=target.fast_fit_digest,
        v8_config_digest=config_digest,
    )


def _build_replay(
    *,
    prepared: Any,
    resolved: Any,
    symbol: str,
    contract: Any,
    forecast: Any,
    state: Any,
    rows: np.ndarray,
    target: CausalAlphaV10TargetPath,
    config_digest: str,
    boundaries: Any,
    execution_rebalance_contract: CausalAlphaV10ExecutionContract,
) -> tuple[CausalAlphaV8ReplayMetric, dict[str, object], dict[str, object]]:
    environment = _environment(prepared, symbol)
    try:
        _require_execution_rebalance_contract(
            environment,
            execution_rebalance_contract,
        )
        evaluation = evaluate_action_path(
            _InitialStateEnvironment(environment, contract.initial_state_mode),
            evaluation_range=(contract.start, contract.stop),
            actions=target.v6_target_path.targets[:, None].astype(np.float32),
        )
        metric = _metric_from_evaluation(
            prepared=prepared,
            resolved=resolved,
            symbol=symbol,
            contract=contract,
            forecast=forecast,
            state=state,
            rows=rows,
            target=target,
            evaluation=evaluation,
            config_digest=config_digest,
            boundaries=boundaries,
            environment=environment,
        )
        execution_trace = _execution_trace_payload(evaluation)
        execution_diagnostics = _execution_diagnostics(evaluation, execution_trace)
        return metric, execution_trace, execution_diagnostics
    finally:
        environment.close()


def _build_hierarchical_replay(
    *,
    prepared: Any,
    resolved: Any,
    symbol: str,
    contract: Any,
    forecast: Any,
    state: Any,
    rows: np.ndarray,
    policy: CausalAlphaV10HierarchyPolicy,
    dual_fit: CausalAlphaV10DualFit,
    v10_config: CausalAlphaV10Config,
    config_digest: str,
    boundaries: Any,
    execution_rebalance_contract: CausalAlphaV10ExecutionContract,
) -> tuple[
    CausalAlphaV10TargetPath,
    CausalAlphaV8ReplayMetric,
    dict[str, object],
    dict[str, object],
]:
    environment = _environment(prepared, symbol)
    try:
        _require_execution_rebalance_contract(
            environment,
            execution_rebalance_contract,
        )
        evaluation = evaluate_action_path(
            _InitialStateEnvironment(environment, contract.initial_state_mode),
            evaluation_range=(contract.start, contract.stop),
            model=policy,
            deterministic=True,
        )
        result = policy.result()
        target = CausalAlphaV10TargetPath(
            candidate=CausalAlphaV10Candidate.HIERARCHICAL_WAVE,
            v6_target_path=result.v6_target_path,
            source_forecast_digest=forecast.digest,
            fast_fit_digest=dual_fit.fast.digest,
            slow_fit_digest=dual_fit.slow.digest,
            v10_config_digest=v10_config.digest,
            hierarchy_input_digest=result.input_digest,
            hierarchy_reasons=result.hierarchy_reasons,
            hierarchy_reason_counts=result.hierarchy_reason_counts,
        )
        metric = _metric_from_evaluation(
            prepared=prepared,
            resolved=resolved,
            symbol=symbol,
            contract=contract,
            forecast=forecast,
            state=state,
            rows=rows,
            target=target,
            evaluation=evaluation,
            config_digest=config_digest,
            boundaries=boundaries,
            environment=environment,
        )
        execution_trace = _execution_trace_payload(evaluation)
        execution_diagnostics = _execution_diagnostics(evaluation, execution_trace)
        return target, metric, execution_trace, execution_diagnostics
    finally:
        environment.close()


def _leaf(
    store: CausalAlphaV4ArtifactStore,
    candidate: CausalAlphaV10Candidate,
    metric: CausalAlphaV8ReplayMetric,
    *,
    target: CausalAlphaV10TargetPath,
    candidate_input_digest: str,
    execution_trace: dict[str, object],
    execution_diagnostics: dict[str, object],
) -> dict[str, object]:
    base = metric.v6_metric
    body: dict[str, object] = {
        "candidate": candidate.value,
        "candidate_input_digest": candidate_input_digest,
        "config_digest": store.config_digest,
        "contract_digest": base.contract_digest,
        "episode_index": base.episode_index,
        "execution_diagnostics": execution_diagnostics,
        "execution_trace": execution_trace,
        "generator_code_digest": store.generator_code_digest,
        "replay": metric.to_payload(),
        "replay_digest": metric.digest,
        "run_manifest_digest": store.run_manifest_digest,
        "schema_version": _REPLAY_LEAF_SCHEMA,
        "symbol": base.symbol,
        "target": target.to_payload(),
        "target_path_digest": metric.v8_target_path_digest,
        "v4_context_manifest_digest": store.v4_context_manifest_digest,
    }
    return _artifact(body)


def _path(candidate: CausalAlphaV10Candidate, symbol: str, episode: int) -> Path:
    return (
        Path("selection")
        / "replays"
        / f"{episode:02d}"
        / symbol
        / f"{candidate.value}.json"
    )


def _load(
    store: CausalAlphaV4ArtifactStore,
    *,
    path: Path,
    candidate: CausalAlphaV10Candidate,
    candidate_input_digest: str,
    expected_fast_fit_digest: str,
    expected_target_digest: str | None,
    symbol: str,
    episode: int,
    contract_digest: str,
) -> CausalAlphaV8ReplayMetric | None:
    leaf = store.load_leaf(path, expected_schema=_REPLAY_LEAF_SCHEMA)
    if leaf is None:
        return None
    metric = CausalAlphaV8ReplayMetric.from_payload(leaf["replay"])
    target_payload = leaf.get("target")
    if not isinstance(target_payload, dict):
        raise ValueError("V10 resumed replay target evidence is invalid")
    execution_trace = _validate_execution_trace_payload(leaf.get("execution_trace"))
    execution_diagnostics = _validate_execution_diagnostics(
        leaf.get("execution_diagnostics"), execution_trace
    )
    hierarchy_input_digest = target_payload.get("hierarchy_input_digest")
    hierarchical = candidate is CausalAlphaV10Candidate.HIERARCHICAL_WAVE
    if (
        leaf["candidate"] != candidate.value
        or leaf.get("candidate_input_digest") != candidate_input_digest
        or metric.candidate is not V8_CANDIDATE_BY_V10[candidate]
        or metric.v8_config_digest != store.config_digest
        or metric.v6_metric.symbol != symbol
        or metric.v6_metric.episode_index != episode
        or metric.v6_metric.contract_digest != contract_digest
        or metric.calibration_fit_digest != expected_fast_fit_digest
        or leaf["replay_digest"] != metric.digest
        or leaf.get("target_path_digest") != metric.v8_target_path_digest
        or target_payload.get("artifact_digest") != metric.v8_target_path_digest
        or target_payload.get("candidate") != candidate.value
        or metric.v6_metric.hard_risk_violation
        != execution_diagnostics["hard_risk_violation"]
        or (
            expected_target_digest is not None
            and metric.v8_target_path_digest != expected_target_digest
        )
        or (hierarchical and hierarchy_input_digest != candidate_input_digest)
        or (not hierarchical and hierarchy_input_digest is not None)
    ):
        raise ValueError("V10 resumed replay identity drifted")
    return metric


def selection_stage(
    prepared: Any,
    signal: CausalAlphaV10SignalEvidence,
    *,
    source_config: CausalAlphaV7ResearchConfig,
    v8_config: CausalAlphaV8TargetConfig,
    v9_config: CausalAlphaV9Config,
    v10_config: CausalAlphaV10Config,
    config_digest: str,
    store: CausalAlphaV4ArtifactStore,
) -> CausalAlphaV10SelectionEvidence:
    if not signal.passed:
        raise ValueError("V10 Selection cannot bypass Signal")
    records: list[CausalAlphaV8ReplayMetric] = []
    v9_rows = _v9_wave_rows(prepared)
    training = _training_rows(prepared)
    execution_contracts = {
        symbol: _execution_rebalance_contract(prepared, symbol)
        for symbol in prepared.train_symbols
    }
    count = len(
        prepared.nested_partitions[prepared.train_symbols[0]].economic_contracts
    )
    for index in range(count):
        contracts = _contract_column(prepared, "economic_contracts", index)
        cutoff = int(contracts[0].start)
        resolved = _fit_one_calibration(
            prepared,
            train_stop=cutoff,
            config=source_config.calibration,
        )
        v9_fit = _fit_v9_wave(v9_rows, cutoff=cutoff, config=v9_config)
        dual_fit = fit_causal_alpha_v10(
            training,
            knowledge_cutoff=cutoff,
            config=v10_config,
        )
        try:
            for symbol, contract in zip(prepared.train_symbols, contracts, strict=True):
                rows, forecast, state, _features, v7_targets = _target_bundle(
                    prepared,
                    resolved,
                    symbol,
                    contract,
                    source_config.target,
                )
                controls, hierarchy_policy = _target_paths(
                    sample=prepared.samples[symbol],
                    rows=rows,
                    forecast=forecast,
                    state=state,
                    boundaries=resolved.boundaries,
                    v7_targets=v7_targets,
                    v9_fit=v9_fit,
                    dual_fit=dual_fit,
                    v8_config=v8_config,
                    v9_config=v9_config,
                    v10_config=v10_config,
                    execution_rebalance_contract=execution_contracts[symbol],
                )
                for candidate in CausalAlphaV10Candidate:
                    relative = _path(candidate, symbol, contract.episode_index)
                    target: CausalAlphaV10TargetPath | None
                    if candidate is CausalAlphaV10Candidate.HIERARCHICAL_WAVE:
                        target = None
                        candidate_input_digest = hierarchy_policy.input_digest
                        expected_fast_fit_digest = dual_fit.fast.digest
                        expected_target_digest = None
                    else:
                        target = controls[candidate]
                        candidate_input_digest = target.digest
                        expected_fast_fit_digest = target.fast_fit_digest
                        expected_target_digest = target.digest
                    metric = _load(
                        store,
                        path=relative,
                        candidate=candidate,
                        candidate_input_digest=candidate_input_digest,
                        expected_fast_fit_digest=expected_fast_fit_digest,
                        expected_target_digest=expected_target_digest,
                        symbol=symbol,
                        episode=contract.episode_index,
                        contract_digest=contract.digest,
                    )
                    if metric is None:
                        if candidate is CausalAlphaV10Candidate.HIERARCHICAL_WAVE:
                            (
                                target,
                                metric,
                                execution_trace,
                                execution_diagnostics,
                            ) = _build_hierarchical_replay(
                                prepared=prepared,
                                resolved=resolved,
                                symbol=symbol,
                                contract=contract,
                                forecast=forecast,
                                state=state,
                                rows=rows,
                                policy=hierarchy_policy,
                                dual_fit=dual_fit,
                                v10_config=v10_config,
                                config_digest=config_digest,
                                boundaries=resolved.boundaries,
                                execution_rebalance_contract=execution_contracts[
                                    symbol
                                ],
                            )
                        else:
                            assert target is not None
                            (
                                metric,
                                execution_trace,
                                execution_diagnostics,
                            ) = _build_replay(
                                prepared=prepared,
                                resolved=resolved,
                                symbol=symbol,
                                contract=contract,
                                forecast=forecast,
                                state=state,
                                rows=rows,
                                target=target,
                                config_digest=config_digest,
                                boundaries=resolved.boundaries,
                                execution_rebalance_contract=execution_contracts[
                                    symbol
                                ],
                            )
                        assert target is not None
                        store.write_leaf(
                            relative,
                            _leaf(
                                store,
                                candidate,
                                metric,
                                target=target,
                                candidate_input_digest=candidate_input_digest,
                                execution_trace=execution_trace,
                                execution_diagnostics=execution_diagnostics,
                            ),
                        )
                    records.append(metric)
            for candidate in CausalAlphaV10Candidate:
                _progress(stage="selection", cutoff=cutoff, candidate=candidate.value)
        finally:
            del resolved, v9_fit, dual_fit
            gc.collect()
    return evaluate_causal_alpha_v10_selection(
        tuple(records),
        expected_symbols=prepared.train_symbols,
    )


def run_causal_alpha_v10_selection(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV10SelectionEvidence:
    from trade_rl.workflows.universal_causal_alpha_v4_runtime_adapter import (
        prepare_causal_alpha_v4_runtime_adapter,
    )

    source_config = CausalAlphaV7ResearchConfig.from_json(config_path)
    v8_config = CausalAlphaV8TargetConfig(base=source_config.target)
    v9_config = CausalAlphaV9Config()
    v10_config = CausalAlphaV10Config()
    config_digest = causal_alpha_v10_stage_config_digest(
        source_config,
        v8_config,
        v9_config,
        v10_config,
    )
    context, runtime, prepared_v3 = prepare_causal_alpha_v4_runtime_adapter(
        run_config_path=run_config_path,
        runtime_manifest_path=runtime_manifest_path,
        v4_context_manifest_path=v4_context_manifest_path,
        frozen_metadata_root=frozen_metadata_root,
    )
    generator_digest = content_digest(
        {
            "schema_version": "causal_alpha_v10_generator_code_v1",
            "source_tree_digest": prepared_v3.execution_identity.source_tree_digest,
        }
    )
    prepared = prepare_causal_alpha_v4_stage_data(
        config_digest=config_digest,
        generator_code_digest=generator_digest,
        runtime_context=context,
        runtime=runtime,
        prepared_v3=prepared_v3,
    )
    del context, runtime, prepared_v3
    gc.collect()
    root = Path(output_root)
    with CausalAlphaV10RunLock(root):
        store = CausalAlphaV4ArtifactStore(
            root,
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=config_digest,
            generator_code_digest=prepared.generator_code_digest,
        )
        signal = _signal_evidence(
            prepared,
            source_config=source_config,
            v9_config=v9_config,
            v10_config=v10_config,
            config_digest=config_digest,
        )
        _write_evidence(store, "signal", signal)
        selection = selection_stage(
            prepared,
            signal,
            source_config=source_config,
            v8_config=v8_config,
            v9_config=v9_config,
            v10_config=v10_config,
            config_digest=config_digest,
            store=store,
        )
        _write_evidence(store, "selection", selection)
        status = "selection_passed" if selection.passed else "selection_rejected"
        store.write_leaf(
            "result.json",
            _artifact(
                {
                    "evidence_digest": selection.digest,
                    "promotion_eligible": False,
                    "schema_version": _RESULT_SCHEMA,
                    "status": status,
                }
            ),
        )
        return selection


__all__ = [
    "causal_alpha_v10_stage_config_digest",
    "run_causal_alpha_v10_selection",
    "selection_stage",
]
