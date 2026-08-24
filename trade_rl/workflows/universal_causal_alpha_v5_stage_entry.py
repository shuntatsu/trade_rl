"""Artifact-bound DB-backed concrete execution for Causal Alpha V5."""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v3 import causal_alpha_overlap_uniqueness_weights
from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4FitConfig,
    CausalAlphaV4SymbolSamples,
    CausalAlphaV4TargetConfig,
    CausalAlphaV4UncertaintyModel,
    fit_causal_alpha_v4_uncertainty,
)
from trade_rl.learning.causal_alpha_v5 import (
    CausalAlphaV5CalibrationFit,
    causal_alpha_v5_target_path,
)
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
from trade_rl.workflows.universal_causal_alpha_v5_admission import (
    evaluate_causal_alpha_v5_admission,
)
from trade_rl.workflows.universal_causal_alpha_v5_artifact_store import (
    CausalAlphaV5ArtifactStore,
    CausalAlphaV5RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v5_calibration import (
    CausalAlphaV5CalibrationSplit,
    calibrate_causal_alpha_v5_forecast,
    fit_causal_alpha_v5_calibration,
)
from trade_rl.workflows.universal_causal_alpha_v5_pipeline import (
    CausalAlphaV5ResearchPackage,
    run_universal_causal_alpha_v5_research_pipeline,
)
from trade_rl.workflows.universal_causal_alpha_v5_replay import (
    CausalAlphaV5ReplayMetric,
    build_causal_alpha_v5_replay_metric,
)
from trade_rl.workflows.universal_causal_alpha_v5_runner import (
    CausalAlphaV5ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v5_selection import (
    evaluate_causal_alpha_v5_selection,
)
from trade_rl.workflows.universal_causal_alpha_v5_signal import (
    CausalAlphaV5SignalEvidence,
    CausalAlphaV5SignalScopeMetric,
    build_causal_alpha_v5_signal_scope_metric,
    evaluate_causal_alpha_v5_signal_gate,
)

_LIQUIDITY_LOOKBACK_DECISIONS: Final = 96
_LIQUIDITY_LOWER_QUANTILE: Final = 0.10
_LIQUIDITY_SAFETY_MULTIPLIER: Final = 0.80


@dataclass(frozen=True, slots=True)
class _CalibrationResolved:
    base_fit: CausalAlphaV4Fit
    calibration_fit: CausalAlphaV5CalibrationFit


@dataclass(frozen=True, slots=True)
class CausalAlphaV5CalibrationStageEvidence:
    fits: Mapping[int, _CalibrationResolved]
    config_digest: str
    passed: bool
    rejection_reasons: tuple[str, ...] = ()
    schema_version: str = "causal_alpha_v5_calibration_stage_evidence_v1"
    digest: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.config_digest, field="V5 calibration config_digest")
        fits = dict(self.fits)
        if self.passed and (not fits or self.rejection_reasons):
            raise ValueError("V5 calibration pass evidence is incomplete")
        if not self.passed and (fits or not self.rejection_reasons):
            raise ValueError("V5 calibration rejection evidence is incomplete")
        object.__setattr__(self, "fits", MappingProxyType(fits))
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 calibration stage digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_digest": self.config_digest,
            "fit_digests": tuple(
                (cutoff, value.calibration_fit.digest)
                for cutoff, value in sorted(self.fits.items())
            ),
            "passed": self.passed,
            "promotion_eligible": False,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _horizon_weights(sample: Any, *, cutoff: int, state: Any) -> dict[str, np.ndarray]:
    decisions = np.asarray(sample.decision_indices, dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        labels = np.asarray(getattr(sample, f"labels_{horizon}"), dtype=np.float64)
        ends = np.asarray(
            getattr(sample, f"label_end_indices_{horizon}"), dtype=np.int64
        )
        raw = causal_alpha_overlap_uniqueness_weights(
            decisions, ends, knowledge_cutoff=cutoff
        )
        eligible = (
            state.state_eligible & np.isfinite(labels) & (ends >= 0) & (ends < cutoff)
        )
        result[horizon] = np.where(eligible, raw, 0.0)
    return result


def _uncertainties(
    fit: CausalAlphaV4Fit, sample: Any, *, forecast: Any | None = None
) -> dict[str, np.ndarray]:
    resolved_forecast = fit.predict(sample) if forecast is None else forecast
    state = resolve_causal_alpha_v4_stage_state_inputs(sample)
    labels = {
        horizon: np.asarray(getattr(sample, f"labels_{horizon}"), dtype=np.float64)
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    model: CausalAlphaV4UncertaintyModel = fit_causal_alpha_v4_uncertainty(
        final_predictions=resolved_forecast.final_predictions,
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


def _slow_uncertainty(
    forecast: Any,
    uncertainty: Mapping[str, np.ndarray],
) -> np.ndarray:
    p24 = np.asarray(forecast.final_predictions["24h"])
    p72 = np.asarray(forecast.final_predictions["72h"]) / 3.0
    disagreement = 0.5 * np.abs(p24 - p72)
    return np.sqrt(
        0.25 * (np.square(uncertainty["24h"]) + np.square(uncertainty["72h"] / 3.0))
        + np.square(disagreement)
    )


def _cutoffs(prepared: Any) -> tuple[int, ...]:
    values: set[int] = set()
    for symbol in prepared.train_symbols:
        partition = prepared.nested_partitions[symbol]
        values.update(contract.start for contract in partition.signal_contracts)
        values.update(contract.start for contract in partition.economic_contracts)
        values.add(partition.holdout_contract.start)
    return tuple(sorted(values))


def _fit_calibrations(
    prepared: Any, config: CausalAlphaV5ResearchConfig
) -> CausalAlphaV5CalibrationStageEvidence:
    try:
        resolved: dict[int, _CalibrationResolved] = {}
        fit_config = CausalAlphaV4FitConfig()
        for cutoff in _cutoffs(prepared):
            split = CausalAlphaV5CalibrationSplit.from_samples(
                train_symbols=prepared.train_symbols,
                samples=prepared.samples,
                train_stop=cutoff,
                config=config.calibration,
            )
            base_fit = fit_causal_alpha_v4(
                train_symbols=prepared.train_symbols,
                samples=prepared.samples,
                knowledge_cutoff=split.calibration_start,
                config=fit_config,
            )
            uncertainty: dict[str, dict[str, np.ndarray]] = {}
            slow: dict[str, np.ndarray] = {}
            for symbol in prepared.train_symbols:
                sample = prepared.samples[symbol]
                forecast = base_fit.predict(sample)
                uncertainty[symbol] = _uncertainties(
                    base_fit, sample, forecast=forecast
                )
                slow[symbol] = _slow_uncertainty(forecast, uncertainty[symbol])
            calibration_fit = fit_causal_alpha_v5_calibration(
                train_symbols=prepared.train_symbols,
                samples=prepared.samples,
                v4_fit=base_fit,
                slow_uncertainty=slow,
                train_stop=cutoff,
                config=config.calibration,
            )
            resolved[cutoff] = _CalibrationResolved(
                base_fit=base_fit,
                calibration_fit=calibration_fit,
            )
            del uncertainty, slow
            gc.collect()
        return CausalAlphaV5CalibrationStageEvidence(
            fits=resolved, config_digest=config.calibration.digest, passed=True
        )
    except (TypeError, ValueError, RuntimeError) as error:
        return CausalAlphaV5CalibrationStageEvidence(
            fits={},
            config_digest=config.calibration.digest,
            passed=False,
            rejection_reasons=(f"{type(error).__name__}:{error}",),
        )


def _environment(prepared: Any, symbol: str) -> Any:
    factories = getattr(prepared.prepared_v3, "environment_factories", None)
    if not isinstance(factories, Mapping) or not callable(factories.get(symbol)):
        raise ValueError("V5 environment factory scope drifted")
    return factories[symbol]()


def _costs_and_caps(
    prepared: Any,
    symbol: str,
    environment: Any,
    decisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    execution = getattr(getattr(environment, "config", None), "execution_cost", None)
    if not isinstance(execution, ExecutionCostConfig):
        raise TypeError("V5 environment execution cost config is invalid")
    prepared_costs = getattr(prepared.prepared_v3, "execution_costs", None)
    if not isinstance(prepared_costs, Mapping) or execution != prepared_costs.get(
        symbol
    ):
        raise ValueError("V5 execution cost identity drifted")
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
        lookback_decisions=_LIQUIDITY_LOOKBACK_DECISIONS,
        lower_quantile=_LIQUIDITY_LOWER_QUANTILE,
        safety_multiplier=_LIQUIDITY_SAFETY_MULTIPLIER,
    )
    return costs, caps


def _slice_context(context: Any, rows: np.ndarray) -> Any:
    return type(context)(
        feature_names=context.feature_names,
        decision_indices=np.asarray(context.decision_indices)[rows],
        values=np.asarray(context.values)[rows],
        available=np.asarray(context.available)[rows],
        staleness_hours=np.asarray(context.staleness_hours)[rows],
        source_digest=context.source_digest,
    )


def _slice_sample(sample: Any, rows: np.ndarray) -> CausalAlphaV4SymbolSamples:
    return CausalAlphaV4SymbolSamples(
        symbol=sample.symbol,
        dataset_id=sample.dataset_id,
        target_local_feature_schema_digest=sample.target_local_feature_schema_digest,
        source_sample_digest=sample.source_sample_digest,
        source_context_digest=sample.source_context_digest,
        decision_indices=np.asarray(sample.decision_indices)[rows],
        label_end_indices_4h=np.asarray(sample.label_end_indices_4h)[rows],
        label_end_indices_24h=np.asarray(sample.label_end_indices_24h)[rows],
        label_end_indices_72h=np.asarray(sample.label_end_indices_72h)[rows],
        labels_4h=np.asarray(sample.labels_4h)[rows],
        labels_24h=np.asarray(sample.labels_24h)[rows],
        labels_72h=np.asarray(sample.labels_72h)[rows],
        target_local_feature_names=sample.target_local_feature_names,
        target_local_features=np.asarray(sample.target_local_features)[rows],
        target_local_available=np.asarray(sample.target_local_available)[rows],
        local_context=_slice_context(sample.local_context, rows),
        global_context=_slice_context(sample.global_context, rows),
        instrument_descriptor_names=sample.instrument_descriptor_names,
        instrument_descriptors=np.asarray(sample.instrument_descriptors)[rows],
        instrument_descriptor_available=np.asarray(
            sample.instrument_descriptor_available
        )[rows],
        beta=np.asarray(sample.beta)[rows],
        beta_available=np.asarray(sample.beta_available)[rows],
    )


def _selective_bundle(
    prepared: Any,
    calibration: CausalAlphaV5CalibrationStageEvidence,
    symbol: str,
    contract: Any,
) -> tuple[
    np.ndarray,
    Any,
    Any,
    Any,
    np.ndarray,
    np.ndarray,
    Mapping[str, np.ndarray],
]:
    fitted = calibration.fits[contract.start]
    sample = prepared.samples[symbol]
    rows = resolve_causal_alpha_v4_contract_rows(
        sample, start=contract.start, stop=contract.stop
    )
    full_forecast = fitted.base_fit.predict(sample)
    forecast = slice_causal_alpha_v4_forecast(full_forecast, rows)
    uncertainty = _uncertainties(fitted.base_fit, sample, forecast=full_forecast)
    state = resolve_causal_alpha_v4_stage_state_inputs(sample)
    environment = _environment(prepared, symbol)
    costs, caps = _costs_and_caps(
        prepared, symbol, environment, forecast.decision_indices
    )
    slow_full = _slow_uncertainty(full_forecast, uncertainty)
    selective = calibrate_causal_alpha_v5_forecast(
        v4_forecast=forecast,
        sample=_slice_sample(sample, rows),
        slow_uncertainty=slow_full[rows],
        one_way_cost_rates=costs,
        actionable_mask=state.actionable[rows],
        calibration_fit=fitted.calibration_fit,
    )
    return rows, forecast, selective, environment, costs, caps, uncertainty


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
        raise TypeError("V5 environment cannot resolve initial weights")
    weights = np.asarray(
        resolver(contract.initial_state_mode, contract.start), dtype=np.float64
    )
    expected = np.asarray(contract.initial_weights, dtype=np.float64)
    if weights.shape != (1,) or not np.array_equal(weights, expected):
        raise ValueError("V5 initial state drifted from frozen contract")
    return float(weights[0])


def _replay(
    prepared: Any,
    calibration: CausalAlphaV5CalibrationStageEvidence,
    config: CausalAlphaV5ResearchConfig,
    symbol: str,
    contract: Any,
) -> CausalAlphaV5ReplayMetric:
    rows, forecast, selective, environment, costs, caps, uncertainty = (
        _selective_bundle(prepared, calibration, symbol, contract)
    )
    try:
        fitted = calibration.fits[contract.start]
        target = causal_alpha_v5_target_path(
            selective,
            forecast.final_predictions["4h"],
            direction_score_4h=forecast.direction_scores["4h"],
            uncertainty_4h=uncertainty["4h"][rows],
            one_way_cost_rates=costs,
            liquidity_weight_caps=caps,
            config=CausalAlphaV4TargetConfig(),
            initial_weight=_initial_weight(environment, contract),
        )
        evaluation = evaluate_action_path(
            _InitialStateEnvironment(environment, contract.initial_state_mode),
            evaluation_range=(contract.start, contract.stop),
            actions=target.targets[:, None].astype(np.float32),
        )
        return build_causal_alpha_v5_replay_metric(
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=config.calibration.digest,
            symbol=symbol,
            episode_index=contract.episode_index,
            contract_digest=contract.digest,
            fit_digest=fitted.base_fit.digest,
            forecast_digest=forecast.digest,
            calibration_fit_digest=fitted.calibration_fit.digest,
            target_path=target,
            evaluation=evaluation,
            episode_hours=float(prepared.prepared_v3.episode_hours),
        )
    finally:
        environment.close()


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


def _signal_stage(
    prepared: Any,
    calibration: CausalAlphaV5CalibrationStageEvidence,
    config: CausalAlphaV5ResearchConfig,
) -> CausalAlphaV5SignalEvidence:
    v5_metrics: list[CausalAlphaV5SignalScopeMetric] = []
    v4_metrics: list[CausalAlphaV4SignalScopeMetric] = []
    full_fit_cache: dict[int, CausalAlphaV4Fit] = {}
    fit_config = CausalAlphaV4FitConfig()
    for symbol in prepared.train_symbols:
        sample = prepared.samples[symbol]
        for contract in prepared.nested_partitions[symbol].signal_contracts:
            (
                rows,
                _forecast,
                selective,
                environment,
                _costs,
                _caps,
                _uncertainty,
            ) = _selective_bundle(prepared, calibration, symbol, contract)
            environment.close()
            v5_metrics.append(
                build_causal_alpha_v5_signal_scope_metric(
                    run_manifest_digest=prepared.run_manifest_digest,
                    calibration_config_digest=config.calibration.digest,
                    symbol=symbol,
                    episode_index=contract.episode_index,
                    contract_start=contract.start,
                    contract_stop=contract.stop,
                    contract_digest=contract.digest,
                    selective_forecast=selective,
                    labels_24h=np.asarray(sample.labels_24h)[rows],
                    label_end_indices_24h=np.asarray(sample.label_end_indices_24h)[
                        rows
                    ],
                    labels_72h=np.asarray(sample.labels_72h)[rows],
                    label_end_indices_72h=np.asarray(sample.label_end_indices_72h)[
                        rows
                    ],
                )
            )
            full_fit = full_fit_cache.get(contract.start)
            if full_fit is None:
                full_fit = fit_causal_alpha_v4(
                    train_symbols=prepared.train_symbols,
                    samples=prepared.samples,
                    knowledge_cutoff=contract.start,
                    config=fit_config,
                )
                full_fit_cache[contract.start] = full_fit
            v4_forecast = slice_causal_alpha_v4_forecast(full_fit.predict(sample), rows)
            liveness = _v4_liveness_digests(
                fit=full_fit,
                sample=sample,
                rows=rows,
                forecast=v4_forecast,
                symbol=symbol,
            )
            state = resolve_causal_alpha_v4_stage_state_inputs(sample)
            built = build_causal_alpha_v4_signal_scope_metrics(
                run_manifest_digest=prepared.run_manifest_digest,
                fit_config_digest=fit_config.digest,
                symbol=symbol,
                episode_index=contract.episode_index,
                contract_start=contract.start,
                contract_stop=contract.stop,
                contract_digest=contract.digest,
                fit_digest=full_fit.digest,
                forecast=v4_forecast,
                liveness_digests=liveness,
                actionable_mask=state.actionable[rows],
                labels_4h=np.asarray(sample.labels_4h)[rows],
                label_end_indices_4h=np.asarray(sample.label_end_indices_4h)[rows],
                labels_24h=np.asarray(sample.labels_24h)[rows],
                label_end_indices_24h=np.asarray(sample.label_end_indices_24h)[rows],
                labels_72h=np.asarray(sample.labels_72h)[rows],
                label_end_indices_72h=np.asarray(sample.label_end_indices_72h)[rows],
            )
            v4_metrics.extend(built.values())
    expected = len(prepared.train_symbols) * 8
    v4 = evaluate_causal_alpha_v4_signal_gate(
        tuple(v4_metrics),
        expected_raw_scope_count_per_lane=expected,
        gate=CausalAlphaV4SignalGateConfig(),
    )
    return evaluate_causal_alpha_v5_signal_gate(
        tuple(v5_metrics),
        expected_symbols=prepared.train_symbols,
        v4_fast_lane_digest=v4.fast_4h.digest,
        v4_fast_lane_passed=v4.fast_4h.passed,
        config=config.calibration,
    )


def _selection_stage(
    prepared: Any,
    calibration: CausalAlphaV5CalibrationStageEvidence,
    signal: CausalAlphaV5SignalEvidence,
    config: CausalAlphaV5ResearchConfig,
) -> Any:
    if not signal.passed:
        raise ValueError("V5 Selection cannot bypass Signal")
    records = tuple(
        _replay(prepared, calibration, config, symbol, contract)
        for symbol in prepared.train_symbols
        for contract in prepared.nested_partitions[symbol].economic_contracts
    )
    return evaluate_causal_alpha_v5_selection(
        records, expected_symbols=prepared.train_symbols
    )


def _admission_stage(
    prepared: Any,
    calibration: CausalAlphaV5CalibrationStageEvidence,
    signal: Any,
    selection: Any,
    config: CausalAlphaV5ResearchConfig,
) -> Any:
    if not signal.passed or not selection.passed:
        raise ValueError("V5 Admission cannot bypass upstream gates")
    holdouts = tuple(
        prepared.nested_partitions[symbol].holdout_contract
        for symbol in prepared.train_symbols
    )
    starts = {contract.start for contract in holdouts}
    if len(starts) != 1:
        raise ValueError("V5 holdout starts drifted")
    holdout_start = next(iter(starts))
    records = tuple(
        _replay(prepared, calibration, config, symbol, contract)
        for symbol, contract in zip(prepared.train_symbols, holdouts, strict=True)
    )
    return evaluate_causal_alpha_v5_admission(
        records,
        signal_evidence=signal,
        selection_evidence=selection,
        fit_knowledge_cutoff=holdout_start,
        holdout_start=holdout_start,
    )


def _artifact(body: dict[str, object]) -> dict[str, object]:
    return {**body, "artifact_digest": content_digest(body)}


def run_causal_alpha_v5_concrete_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV5ResearchPackage:
    """Resolve immutable V4 inputs and run all fixed V5 gates on real data."""

    from trade_rl.workflows.universal_causal_alpha_v4_runtime_adapter import (
        prepare_causal_alpha_v4_runtime_adapter,
    )

    config_path = Path(config_path)
    config = CausalAlphaV5ResearchConfig.from_json(config_path)
    context, runtime, prepared_v3 = prepare_causal_alpha_v4_runtime_adapter(
        run_config_path=Path(run_config_path),
        runtime_manifest_path=Path(runtime_manifest_path),
        v4_context_manifest_path=Path(v4_context_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
    )
    generator_digest = content_digest(
        {
            "schema_version": "causal_alpha_v5_generator_code_v1",
            "source_tree_digest": prepared_v3.execution_identity.source_tree_digest,
        }
    )
    prepared = prepare_causal_alpha_v4_stage_data(
        config_digest=config.calibration.digest,
        generator_code_digest=generator_digest,
        runtime_context=context,
        runtime=runtime,
        prepared_v3=prepared_v3,
    )
    del runtime
    gc.collect()
    root = Path(output_root)
    with CausalAlphaV5RunLock(root):
        store = CausalAlphaV5ArtifactStore(
            root,
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=config.calibration.digest,
            generator_code_digest=prepared.generator_code_digest,
        )
        source = json.loads(config_path.read_text(encoding="utf-8"))
        store.write_leaf(
            "authored-config.json",
            _artifact(
                {
                    "schema_version": "causal_alpha_v5_authored_config_record_v1",
                    "run_manifest_digest": prepared.run_manifest_digest,
                    "v4_context_manifest_digest": prepared.v4_context_manifest_digest,
                    "config_digest": config.calibration.digest,
                    "generator_code_digest": prepared.generator_code_digest,
                    "research_config_digest": config.digest,
                    "source_config": source,
                    "research_only": True,
                    "promotion_eligible": False,
                }
            ),
        )
        return run_universal_causal_alpha_v5_research_pipeline(
            store=store,
            prepare_stage=lambda: prepared,
            calibration_stage=lambda value: _fit_calibrations(value, config),
            signal_stage=lambda value, calibration: _signal_stage(
                value,
                cast(CausalAlphaV5CalibrationStageEvidence, calibration),
                config,
            ),
            selection_stage=lambda value, calibration, signal: _selection_stage(
                value,
                cast(CausalAlphaV5CalibrationStageEvidence, calibration),
                cast(CausalAlphaV5SignalEvidence, signal),
                config,
            ),
            admission_stage=lambda value, calibration, signal, selection: (
                _admission_stage(
                    value,
                    cast(CausalAlphaV5CalibrationStageEvidence, calibration),
                    signal,
                    selection,
                    config,
                )
            ),
        )


__all__ = [
    "CausalAlphaV5CalibrationStageEvidence",
    "run_causal_alpha_v5_concrete_entry",
]
