"""DB-backed Signal and Selection stage assembly for Causal Alpha V7."""

from __future__ import annotations

import gc
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4FitConfig,
    fit_causal_alpha_v4_uncertainty,
)
from trade_rl.learning.causal_alpha_v5 import CausalAlphaV5CalibrationConfig
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6TargetConfig
from trade_rl.learning.causal_alpha_v7 import (
    CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV7CalibrationConfig,
    CausalAlphaV7CalibrationRange,
    CausalAlphaV7Candidate,
    CausalAlphaV7TargetPath,
)
from trade_rl.learning.causal_alpha_v7_calibration import (
    CausalAlphaV7CalibrationFit,
    CausalAlphaV7CalibrationRows,
    fit_causal_alpha_v7_calibration,
)
from trade_rl.learning.causal_alpha_v7_target import causal_alpha_v7_target_paths
from trade_rl.learning.rollout_evaluation import evaluate_action_path
from trade_rl.workflows.universal_causal_alpha_v4_fitting import CausalAlphaV4Fit
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
from trade_rl.workflows.universal_causal_alpha_v5_calibration import (
    CausalAlphaV5CalibrationSplit,
)
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    build_causal_alpha_v6_replay_metric,
)
from trade_rl.workflows.universal_causal_alpha_v6_stage_entry import (
    _contract_column,
    _costs_and_caps,
    _environment,
    _fit_one,
    _horizon_weights,
    _initial_weight,
    _InitialStateEnvironment,
    _progress,
    _reward_scale,
    _uncertainties,
)
from trade_rl.workflows.universal_causal_alpha_v7_admission import (
    CausalAlphaV7AdmissionEvidence,
    evaluate_causal_alpha_v7_admission,
)
from trade_rl.workflows.universal_causal_alpha_v7_artifact_store import (
    CausalAlphaV7ArtifactStore,
    CausalAlphaV7RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionBoundaries,
    build_causal_alpha_v7_attribution,
)
from trade_rl.workflows.universal_causal_alpha_v7_pipeline import (
    CausalAlphaV7ResearchPackage,
    run_universal_causal_alpha_v7_research_pipeline,
)
from trade_rl.workflows.universal_causal_alpha_v7_replay import (
    CausalAlphaV7ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_selection import (
    CausalAlphaV7SelectionEvidence,
    evaluate_causal_alpha_v7_selection,
)
from trade_rl.workflows.universal_causal_alpha_v7_signal import (
    CausalAlphaV7SignalEvidence,
    CausalAlphaV7SignalScopeMetric,
    evaluate_causal_alpha_v7_signal_gate,
)
from trade_rl.workflows.universal_causal_alpha_v7_stage_science import (
    build_causal_alpha_v7_attribution_boundaries,
    build_causal_alpha_v7_calibration_rows,
    build_causal_alpha_v7_feature_matrix,
)


@dataclass(frozen=True, slots=True)
class _CalibrationResolved:
    base_fit: CausalAlphaV4Fit
    calibration_fit: CausalAlphaV7CalibrationFit
    boundaries: CausalAlphaV7AttributionBoundaries


def causal_alpha_v7_stage_config_digest(
    *,
    calibration: CausalAlphaV7CalibrationConfig,
    target: CausalAlphaV6TargetConfig,
) -> str:
    """Bind the only authored V7 calibration and target contracts."""

    if not isinstance(calibration, CausalAlphaV7CalibrationConfig):
        raise TypeError("V7 stage calibration config is invalid")
    if not isinstance(target, CausalAlphaV6TargetConfig):
        raise TypeError("V7 stage target config is invalid")
    return content_digest(
        {
            "calibration_config_digest": calibration.digest,
            "schema_version": "causal_alpha_v7_stage_config_v1",
            "target_config_digest": target.digest,
        }
    )


def _v5_split_config(
    config: CausalAlphaV7CalibrationConfig,
) -> CausalAlphaV5CalibrationConfig:
    return CausalAlphaV5CalibrationConfig(
        calibration_fraction=config.calibration_fraction,
        forward_block_count=config.forward_block_count,
        ridge_strength=config.ridge_strength,
        minimum_pooled_support=config.minimum_pooled_support,
        minimum_symbol_support=config.minimum_symbol_support,
    )


def _calibration_range(
    split: CausalAlphaV5CalibrationSplit,
) -> CausalAlphaV7CalibrationRange:
    return CausalAlphaV7CalibrationRange(
        base_fit_cutoff=split.calibration_start,
        calibration_start=split.calibration_start,
        train_stop=split.train_stop,
        block_boundaries=split.block_boundaries,
        split_digest=split.digest,
        feature_names=CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    )


def _fit_one_calibration(
    prepared: Any,
    *,
    train_stop: int,
    config: CausalAlphaV7CalibrationConfig,
) -> _CalibrationResolved:
    split = CausalAlphaV5CalibrationSplit.from_samples(
        train_symbols=prepared.train_symbols,
        samples=prepared.samples,
        train_stop=train_stop,
        config=_v5_split_config(config),
    )
    calibration_range = _calibration_range(split)
    base_fit = _fit_one(prepared, calibration_range.base_fit_cutoff)
    records: dict[str, CausalAlphaV7CalibrationRows] = {}
    for symbol in prepared.train_symbols:
        sample = prepared.samples[symbol]
        forecast = base_fit.predict(sample)
        uncertainty = _uncertainties(base_fit, sample, forecast=forecast)
        state = resolve_causal_alpha_v4_stage_state_inputs(sample)
        records[symbol] = build_causal_alpha_v7_calibration_rows(
            sample=sample,
            forecast=forecast,
            uncertainty=uncertainty,
            state=state,
            calibration_range=calibration_range,
        )
    calibration_fit = fit_causal_alpha_v7_calibration(
        rows=records,
        calibration_range=calibration_range,
        config=config,
    )
    boundaries = build_causal_alpha_v7_attribution_boundaries(
        rows=records,
        fit=calibration_fit,
    )
    return _CalibrationResolved(
        base_fit=base_fit,
        calibration_fit=calibration_fit,
        boundaries=boundaries,
    )


def _target_bundle(
    prepared: Any,
    resolved: _CalibrationResolved,
    symbol: str,
    contract: Any,
    target_config: CausalAlphaV6TargetConfig,
) -> tuple[
    np.ndarray,
    Any,
    Any,
    np.ndarray,
    Mapping[CausalAlphaV7Candidate, CausalAlphaV7TargetPath],
]:
    sample = prepared.samples[symbol]
    rows = resolve_causal_alpha_v4_contract_rows(
        sample,
        start=contract.start,
        stop=contract.stop,
    )
    full_forecast = resolved.base_fit.predict(sample)
    forecast = slice_causal_alpha_v4_forecast(full_forecast, rows)
    uncertainty_full = _uncertainties(
        resolved.base_fit,
        sample,
        forecast=full_forecast,
    )
    uncertainty = {
        horizon: uncertainty_full[horizon][rows] for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    state = resolve_causal_alpha_v4_stage_state_inputs(sample)
    features_full, available_full = build_causal_alpha_v7_feature_matrix(
        forecast=full_forecast,
        uncertainty=uncertainty_full,
        state=state,
    )
    environment = _environment(prepared, symbol)
    try:
        costs, caps = _costs_and_caps(
            prepared,
            symbol,
            environment,
            forecast.decision_indices,
            target_config,
        )
        initial_weight = _initial_weight(environment, contract)
    finally:
        environment.close()
    targets = causal_alpha_v7_target_paths(
        forecast=forecast,
        calibration_fit=resolved.calibration_fit,
        calibration_features=features_full[rows],
        calibration_feature_available=available_full[rows],
        uncertainty=uncertainty,
        one_way_cost_rates=costs,
        liquidity_weight_caps=caps,
        actionable_mask=np.asarray(state.actionable)[rows],
        config=target_config,
        initial_weight=initial_weight,
    )
    return rows, forecast, state, features_full[rows], targets


def _v7_signal_scope_metrics(
    *,
    prepared: Any,
    symbol: str,
    contract: Any,
    forecast: Any,
    calibration_fit: Any,
    targets: Mapping[CausalAlphaV7Candidate, Any],
    v7_config_digest: str,
) -> tuple[CausalAlphaV7SignalScopeMetric, ...]:
    result: list[CausalAlphaV7SignalScopeMetric] = []
    for candidate in CausalAlphaV7Candidate:
        target = targets[candidate]
        path = target.v6_target_path
        values = np.asarray(path.targets, dtype=np.float64)
        previous = np.concatenate(([path.initial_weight], values[:-1]))
        result.append(
            CausalAlphaV7SignalScopeMetric(
                candidate=candidate,
                run_manifest_digest=prepared.run_manifest_digest,
                v7_config_digest=v7_config_digest,
                symbol=symbol,
                episode_index=contract.episode_index,
                contract_start=contract.start,
                contract_stop=contract.stop,
                contract_digest=contract.digest,
                source_forecast_digest=forecast.digest,
                calibration_fit_digest=calibration_fit.digest,
                calibration_return_model_digest=calibration_fit.return_model.digest,
                calibration_direction_model_digest=calibration_fit.direction_model.digest,
                target_path_digest=target.digest,
                decision_count=len(values),
                actionable_count=int(np.count_nonzero(path.actionable_mask)),
                non_flat_target_count=int(np.count_nonzero(np.abs(values) > 1e-12)),
                target_change_count=int(
                    np.count_nonzero(np.abs(values - previous) > 1e-12)
                ),
                sign_flip_count=int(np.count_nonzero(values * previous < 0.0)),
                positive_direction_support=calibration_fit.positive_direction_support,
                negative_direction_support=calibration_fit.negative_direction_support,
            )
        )
    return tuple(result)


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


def _v4_signal_scope_metrics(
    *,
    prepared: Any,
    resolved: _CalibrationResolved,
    symbol: str,
    contract: Any,
    rows: np.ndarray,
    forecast: Any,
    state: Any,
) -> tuple[CausalAlphaV4SignalScopeMetric, ...]:
    sample = prepared.samples[symbol]
    liveness = _v4_liveness_digests(
        fit=resolved.base_fit,
        sample=sample,
        rows=rows,
        forecast=forecast,
        symbol=symbol,
    )
    metrics = build_causal_alpha_v4_signal_scope_metrics(
        run_manifest_digest=prepared.run_manifest_digest,
        fit_config_digest=CausalAlphaV4FitConfig().digest,
        symbol=symbol,
        episode_index=contract.episode_index,
        contract_start=contract.start,
        contract_stop=contract.stop,
        contract_digest=contract.digest,
        fit_digest=resolved.base_fit.digest,
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
    return tuple(metrics.values())


def signal_stage(
    prepared: Any,
    *,
    calibration_config: CausalAlphaV7CalibrationConfig,
    target_config: CausalAlphaV6TargetConfig,
    v7_config_digest: str,
) -> CausalAlphaV7SignalEvidence:
    """Fit every Signal cutoff causally and require all three candidate paths."""

    v7_metrics: list[CausalAlphaV7SignalScopeMetric] = []
    v4_metrics: list[CausalAlphaV4SignalScopeMetric] = []
    count = len(prepared.nested_partitions[prepared.train_symbols[0]].signal_contracts)
    for index in range(count):
        contracts = _contract_column(prepared, "signal_contracts", index)
        cutoff = int(contracts[0].start)
        resolved = _fit_one_calibration(
            prepared,
            train_stop=cutoff,
            config=calibration_config,
        )
        try:
            for symbol, contract in zip(prepared.train_symbols, contracts, strict=True):
                rows, forecast, state, _features, targets = _target_bundle(
                    prepared,
                    resolved,
                    symbol,
                    contract,
                    target_config,
                )
                v7_metrics.extend(
                    _v7_signal_scope_metrics(
                        prepared=prepared,
                        symbol=symbol,
                        contract=contract,
                        forecast=forecast,
                        calibration_fit=resolved.calibration_fit,
                        targets=targets,
                        v7_config_digest=v7_config_digest,
                    )
                )
                v4_metrics.extend(
                    _v4_signal_scope_metrics(
                        prepared=prepared,
                        resolved=resolved,
                        symbol=symbol,
                        contract=contract,
                        rows=rows,
                        forecast=forecast,
                        state=state,
                    )
                )
            _progress(stage="signal", cutoff=cutoff)
        finally:
            del resolved
            gc.collect()
    expected = len(prepared.train_symbols) * 8
    v4_evidence = evaluate_causal_alpha_v4_signal_gate(
        tuple(v4_metrics),
        expected_raw_scope_count_per_lane=expected,
        gate=CausalAlphaV4SignalGateConfig(),
    )
    return evaluate_causal_alpha_v7_signal_gate(
        tuple(v7_metrics),
        expected_symbols=prepared.train_symbols,
        v4_fast_lane_digest=v4_evidence.fast_4h.digest,
        v4_fast_lane_passed=v4_evidence.fast_4h.passed,
    )


def _replay_target(
    *,
    prepared: Any,
    resolved: _CalibrationResolved,
    symbol: str,
    contract: Any,
    forecast: Any,
    state: Any,
    rows: np.ndarray,
    target: CausalAlphaV7TargetPath,
    v7_config_digest: str,
) -> CausalAlphaV7ReplayMetric:
    environment = _environment(prepared, symbol)
    try:
        reward_scale = _reward_scale(environment)
        evaluation = evaluate_action_path(
            _InitialStateEnvironment(environment, contract.initial_state_mode),
            evaluation_range=(contract.start, contract.stop),
            actions=target.v6_target_path.targets[:, None].astype(np.float32),
        )
        v6_metric = build_causal_alpha_v6_replay_metric(
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
            reward_scale=reward_scale,
        )
        attribution = build_causal_alpha_v7_attribution(
            target_path=target,
            evaluation=evaluation,
            confidence=np.abs(target.v6_target_path.direction_scores_4h),
            realized_volatility=np.asarray(state.realized_volatility)[rows],
            liquidity=np.asarray(state.liquidity)[rows],
            boundaries=resolved.boundaries,
            step_hours=float(prepared.prepared_v3.episode_hours) / len(rows),
        )
        return CausalAlphaV7ReplayMetric(
            candidate=target.candidate,
            v6_metric=v6_metric,
            attribution=attribution,
            v7_target_path_digest=target.digest,
            source_forecast_digest=forecast.digest,
            calibration_fit_digest=resolved.calibration_fit.digest,
            v7_config_digest=v7_config_digest,
        )
    finally:
        environment.close()


def selection_stage(
    prepared: Any,
    signal: CausalAlphaV7SignalEvidence,
    *,
    calibration_config: CausalAlphaV7CalibrationConfig,
    target_config: CausalAlphaV6TargetConfig,
    v7_config_digest: str,
) -> CausalAlphaV7SelectionEvidence:
    """Replay the fixed three candidates and apply unchanged universal gates."""

    if not signal.passed:
        raise ValueError("V7 Selection cannot bypass Signal")
    records: list[CausalAlphaV7ReplayMetric] = []
    count = len(
        prepared.nested_partitions[prepared.train_symbols[0]].economic_contracts
    )
    for index in range(count):
        contracts = _contract_column(prepared, "economic_contracts", index)
        cutoff = int(contracts[0].start)
        resolved = _fit_one_calibration(
            prepared,
            train_stop=cutoff,
            config=calibration_config,
        )
        try:
            for symbol, contract in zip(prepared.train_symbols, contracts, strict=True):
                rows, forecast, state, _features, targets = _target_bundle(
                    prepared,
                    resolved,
                    symbol,
                    contract,
                    target_config,
                )
                records.extend(
                    _replay_target(
                        prepared=prepared,
                        resolved=resolved,
                        symbol=symbol,
                        contract=contract,
                        forecast=forecast,
                        state=state,
                        rows=rows,
                        target=targets[candidate],
                        v7_config_digest=v7_config_digest,
                    )
                    for candidate in CausalAlphaV7Candidate
                )
            for candidate in CausalAlphaV7Candidate:
                _progress(stage="selection", cutoff=cutoff, candidate=candidate.value)
        finally:
            del resolved
            gc.collect()
    return evaluate_causal_alpha_v7_selection(
        tuple(records),
        expected_symbols=prepared.train_symbols,
    )


def admission_stage(
    prepared: Any,
    signal: CausalAlphaV7SignalEvidence,
    selection: CausalAlphaV7SelectionEvidence,
    *,
    calibration_config: CausalAlphaV7CalibrationConfig,
    target_config: CausalAlphaV6TargetConfig,
    v7_config_digest: str,
) -> CausalAlphaV7AdmissionEvidence:
    """Open the untouched holdout only after both upstream gates pass."""

    if (
        not signal.passed
        or not selection.passed
        or selection.selected_candidate is None
    ):
        raise ValueError("V7 Admission cannot bypass upstream gates")
    holdouts = tuple(
        prepared.nested_partitions[symbol].holdout_contract
        for symbol in prepared.train_symbols
    )
    starts = {contract.start for contract in holdouts}
    if len(starts) != 1:
        raise ValueError("V7 holdout starts drifted")
    holdout_start = int(next(iter(starts)))
    resolved = _fit_one_calibration(
        prepared,
        train_stop=holdout_start,
        config=calibration_config,
    )
    selected_records: list[CausalAlphaV7ReplayMetric] = []
    control_records: list[CausalAlphaV7ReplayMetric] = []
    try:
        for symbol, contract in zip(
            prepared.train_symbols,
            holdouts,
            strict=True,
        ):
            rows, forecast, state, _features, targets = _target_bundle(
                prepared,
                resolved,
                symbol,
                contract,
                target_config,
            )
            replays = {
                candidate: _replay_target(
                    prepared=prepared,
                    resolved=resolved,
                    symbol=symbol,
                    contract=contract,
                    forecast=forecast,
                    state=state,
                    rows=rows,
                    target=targets[candidate],
                    v7_config_digest=v7_config_digest,
                )
                for candidate in CausalAlphaV7Candidate
                if candidate
                in {
                    CausalAlphaV7Candidate.V6_CONTROL,
                    selection.selected_candidate,
                }
            }
            selected_records.append(replays[selection.selected_candidate])
            control_records.append(replays[CausalAlphaV7Candidate.V6_CONTROL])
        _progress(stage="admission", cutoff=holdout_start)
        return evaluate_causal_alpha_v7_admission(
            tuple(selected_records),
            tuple(control_records),
            signal_evidence=signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=holdout_start,
            holdout_start=holdout_start,
        )
    finally:
        del resolved
        gc.collect()


def progress_payload(*, stage: str, cutoff: int) -> str:
    """Stable helper retained for diagnostic consumers."""

    return json.dumps(
        {"cutoff": cutoff, "stage": stage, "status": "evaluated"},
        sort_keys=True,
    )


def _artifact(body: dict[str, object]) -> dict[str, object]:
    return {**body, "artifact_digest": content_digest(body)}


def run_causal_alpha_v7_concrete_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV7ResearchPackage:
    """Resolve immutable V4 inputs and run all fixed V7 gates on real data."""

    from trade_rl.workflows.universal_causal_alpha_v4_runtime_adapter import (
        prepare_causal_alpha_v4_runtime_adapter,
    )

    config_path = Path(config_path)
    runner = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v7_runner"
    )
    config = runner.CausalAlphaV7ResearchConfig.from_json(config_path)
    context, runtime, prepared_v3 = prepare_causal_alpha_v4_runtime_adapter(
        run_config_path=Path(run_config_path),
        runtime_manifest_path=Path(runtime_manifest_path),
        v4_context_manifest_path=Path(v4_context_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
    )
    generator_digest = content_digest(
        {
            "schema_version": "causal_alpha_v7_generator_code_v1",
            "source_tree_digest": prepared_v3.execution_identity.source_tree_digest,
        }
    )
    prepared = prepare_causal_alpha_v4_stage_data(
        config_digest=config.digest,
        generator_code_digest=generator_digest,
        runtime_context=context,
        runtime=runtime,
        prepared_v3=prepared_v3,
    )
    del context, runtime, prepared_v3
    gc.collect()
    root = Path(output_root)
    with CausalAlphaV7RunLock(root):
        store = CausalAlphaV7ArtifactStore(
            root,
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=config.digest,
            generator_code_digest=prepared.generator_code_digest,
        )
        source = json.loads(config_path.read_text(encoding="utf-8"))
        store.write_leaf(
            "authored-config.json",
            _artifact(
                {
                    "schema_version": "causal_alpha_v7_authored_config_record_v1",
                    "run_manifest_digest": prepared.run_manifest_digest,
                    "v4_context_manifest_digest": prepared.v4_context_manifest_digest,
                    "config_digest": config.digest,
                    "generator_code_digest": prepared.generator_code_digest,
                    "source_config": source,
                    "research_only": True,
                    "promotion_eligible": False,
                }
            ),
        )
        return run_universal_causal_alpha_v7_research_pipeline(
            store=store,
            prepare_stage=lambda: prepared,
            signal_stage=lambda value: signal_stage(
                value,
                calibration_config=config.calibration,
                target_config=config.target,
                v7_config_digest=config.digest,
            ),
            selection_stage=lambda value, signal: selection_stage(
                value,
                cast(CausalAlphaV7SignalEvidence, signal),
                calibration_config=config.calibration,
                target_config=config.target,
                v7_config_digest=config.digest,
            ),
            admission_stage=lambda value, signal, selection: admission_stage(
                value,
                cast(CausalAlphaV7SignalEvidence, signal),
                cast(CausalAlphaV7SelectionEvidence, selection),
                calibration_config=config.calibration,
                target_config=config.target,
                v7_config_digest=config.digest,
            ),
        )


__all__ = [
    "CausalAlphaV6TargetConfig",
    "admission_stage",
    "causal_alpha_v7_stage_config_digest",
    "run_causal_alpha_v7_concrete_entry",
    "selection_stage",
    "signal_stage",
]
