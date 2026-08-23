"""Concrete Signal/economic/admission stages for research-only Causal Alpha V4."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import causal_alpha_overlap_uniqueness_weights
from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4Forecast,
    CausalAlphaV4UncertaintyModel,
    causal_alpha_v4_target_path,
    fit_causal_alpha_v4_uncertainty,
)
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path_on_environment,
    resolve_episode_initial_weights,
)
from trade_rl.workflows.universal_causal_alpha_costs import (
    causal_alpha_liquidity_weight_caps,
    causal_alpha_one_way_cost_rates,
)
from trade_rl.workflows.universal_causal_alpha_v4_admission import (
    CausalAlphaV4AdmissionEvidence,
    evaluate_causal_alpha_v4_admission,
)
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v4_fitting import (
    CausalAlphaV4Fit,
    fit_causal_alpha_v4,
)
from trade_rl.workflows.universal_causal_alpha_v4_liveness_inputs import (
    build_causal_alpha_v4_liveness_inputs,
)
from trade_rl.workflows.universal_causal_alpha_v4_replay import (
    CausalAlphaV4ReplayMetric,
    build_causal_alpha_v4_replay_metric,
)
from trade_rl.workflows.universal_causal_alpha_v4_selection import (
    CausalAlphaV4SelectionEvidence,
    evaluate_causal_alpha_v4_selection,
)
from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    CausalAlphaV4LivenessEvidence,
    CausalAlphaV4SignalEvidence,
    CausalAlphaV4SignalLane,
    CausalAlphaV4SignalScopeMetric,
    build_causal_alpha_v4_liveness_evidence,
    build_causal_alpha_v4_signal_scope_metrics,
    evaluate_causal_alpha_v4_signal_gate,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_science import (
    CausalAlphaV4StageStateInputs,
    resolve_causal_alpha_v4_contract_rows,
    resolve_causal_alpha_v4_stage_state_inputs,
)

_SIGNAL_LIVENESS_SCOPE_SCHEMA: Final = "causal_alpha_v4_signal_liveness_scope_v1"
_SIGNAL_LANE_PATH = {
    CausalAlphaV4SignalLane.FAST_4H: "fast",
    CausalAlphaV4SignalLane.SLOW_FUSED: "slow",
}
_LIQUIDITY_LOOKBACK_DECISIONS: Final = 96
_LIQUIDITY_LOWER_QUANTILE: Final = 0.10
_LIQUIDITY_SAFETY_MULTIPLIER: Final = 0.80


class CausalAlphaV4FitCache:
    """Reuse the single authored V4 fit once per chronological knowledge cutoff."""

    def __init__(
        self, *, train_symbols: tuple[str, ...], samples: Mapping[str, Any], config: Any
    ) -> None:
        self.train_symbols = tuple(train_symbols)
        self.samples = dict(samples)
        self.config = config
        self._cache: dict[int, CausalAlphaV4Fit] = {}

    def resolve(self, knowledge_cutoff: int) -> CausalAlphaV4Fit:
        cached = self._cache.get(knowledge_cutoff)
        if cached is not None:
            return cached
        fitted = fit_causal_alpha_v4(
            train_symbols=self.train_symbols,
            samples=self.samples,
            knowledge_cutoff=knowledge_cutoff,
            config=self.config,
        )
        self._cache[knowledge_cutoff] = fitted
        return fitted


def _signal_metric_from_payload(
    raw: Mapping[str, object],
) -> CausalAlphaV4SignalScopeMetric:
    return CausalAlphaV4SignalScopeMetric(
        run_manifest_digest=str(raw["run_manifest_digest"]),
        fit_config_digest=str(raw["fit_config_digest"]),
        lane=CausalAlphaV4SignalLane(str(raw["lane"])),
        symbol=str(raw["symbol"]),
        episode_index=int(raw["episode_index"]),
        contract_start=int(raw["contract_start"]),
        contract_stop=int(raw["contract_stop"]),
        contract_digest=str(raw["contract_digest"]),
        fit_digest=str(raw["fit_digest"]),
        forecast_digest=str(raw["forecast_digest"]),
        liveness_digest=str(raw["liveness_digest"]),
        sample_count=int(raw["sample_count"]),
        direction_sample_count=int(raw["direction_sample_count"]),
        rank_correlation=float(raw["rank_correlation"]),
        direction_accuracy=float(raw["direction_accuracy"]),
        top_bottom_realized_spread=float(raw["top_bottom_realized_spread"]),
        cohort_indices=tuple(int(value) for value in raw["cohort_indices"]),
        schema_version=str(raw["schema_version"]),
        digest=str(raw["artifact_digest"]),
    )


def _reason_counts(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("V4 replay reason counts are invalid")
    return tuple((str(reason), int(count)) for reason, count in value)


def _replay_metric_from_payload(raw: Mapping[str, object]) -> CausalAlphaV4ReplayMetric:
    return CausalAlphaV4ReplayMetric(
        run_manifest_digest=str(raw["run_manifest_digest"]),
        v4_context_manifest_digest=str(raw["v4_context_manifest_digest"]),
        config_digest=str(raw["config_digest"]),
        symbol=str(raw["symbol"]),
        episode_index=int(raw["episode_index"]),
        contract_digest=str(raw["contract_digest"]),
        fit_digest=str(raw["fit_digest"]),
        forecast_digest=str(raw["forecast_digest"]),
        target_path_digest=str(raw["target_path_digest"]),
        gross_return=float(raw["gross_return"]),
        net_return=float(raw["net_return"]),
        turnover_per_day=float(raw["turnover_per_day"]),
        total_execution_cost=float(raw["total_execution_cost"]),
        submitted_change_count=int(raw["submitted_change_count"]),
        downstream_no_trade_suppression_count=int(
            raw["downstream_no_trade_suppression_count"]
        ),
        executed_change_count=int(raw["executed_change_count"]),
        closed_trade_count=int(raw["closed_trade_count"]),
        sign_flip_count=int(raw["sign_flip_count"]),
        maximum_drawdown=float(raw["maximum_drawdown"]),
        execution_rejection_reason_counts=_reason_counts(
            raw["execution_rejection_reason_counts"]
        ),
        risk_projection_reason_counts=_reason_counts(
            raw["risk_projection_reason_counts"]
        ),
        target_reason_counts=_reason_counts(raw["target_reason_counts"]),
        hard_risk_violation=bool(raw["hard_risk_violation"]),
        has_meaningful_execution=bool(raw["has_meaningful_execution"]),
        schema_version=str(raw["schema_version"]),
        digest=str(raw["artifact_digest"]),
    )


def _validate_contract_metric(
    metric: CausalAlphaV4SignalScopeMetric,
    *,
    prepared: Any,
    config: Any,
    symbol: str,
    contract: Any,
    lane: CausalAlphaV4SignalLane,
) -> None:
    if (
        metric.run_manifest_digest != prepared.run_manifest_digest
        or metric.fit_config_digest != config.fit.digest
        or metric.lane is not lane
        or metric.symbol != symbol
        or metric.episode_index != contract.episode_index
        or metric.contract_start != contract.start
        or metric.contract_stop != contract.stop
        or metric.contract_digest != contract.digest
    ):
        raise ValueError("V4 persisted signal scope identity drifted")


def _validate_replay_metric(
    metric: CausalAlphaV4ReplayMetric,
    *,
    prepared: Any,
    symbol: str,
    contract: Any,
) -> None:
    if (
        metric.run_manifest_digest != prepared.run_manifest_digest
        or metric.v4_context_manifest_digest != prepared.v4_context_manifest_digest
        or metric.config_digest != prepared.config_digest
        or metric.symbol != symbol
        or metric.episode_index != contract.episode_index
        or metric.contract_digest != contract.digest
    ):
        raise ValueError("V4 persisted replay scope identity drifted")


def _horizon_labels(sample: Any) -> dict[str, np.ndarray]:
    return {
        horizon: np.asarray(getattr(sample, f"labels_{horizon}"), dtype=np.float64)
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }


def _horizon_weights(
    *,
    sample: Any,
    cutoff: int,
    state: CausalAlphaV4StageStateInputs,
) -> dict[str, np.ndarray]:
    decisions = np.asarray(sample.decision_indices, dtype=np.int64)
    labels = _horizon_labels(sample)
    result: dict[str, np.ndarray] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        ends = np.asarray(
            getattr(sample, f"label_end_indices_{horizon}"), dtype=np.int64
        )
        weights = causal_alpha_overlap_uniqueness_weights(
            decisions,
            ends,
            knowledge_cutoff=cutoff,
        )
        eligible = (
            state.state_eligible
            & np.isfinite(labels[horizon])
            & (ends >= 0)
            & (ends < cutoff)
        )
        result[horizon] = np.where(eligible, weights, 0.0).astype(
            np.float64, copy=False
        )
    return result


def _uncertainty_model(
    *,
    fit: CausalAlphaV4Fit,
    sample: Any,
    full_forecast: CausalAlphaV4Forecast,
    state: CausalAlphaV4StageStateInputs,
) -> CausalAlphaV4UncertaintyModel:
    return fit_causal_alpha_v4_uncertainty(
        final_predictions=full_forecast.final_predictions,
        labels=_horizon_labels(sample),
        weights=_horizon_weights(
            sample=sample, cutoff=fit.knowledge_cutoff, state=state
        ),
        state_eligible=state.state_eligible,
        realized_volatility=state.realized_volatility,
        liquidity=state.liquidity,
        basis_positioning_stress=state.basis_positioning_stress,
    )


def _contract_bundle(
    *,
    fit: CausalAlphaV4Fit,
    sample: Any,
    contract: Any,
    slice_forecast: Any,
) -> tuple[
    np.ndarray,
    CausalAlphaV4Forecast,
    CausalAlphaV4StageStateInputs,
    CausalAlphaV4UncertaintyModel,
    dict[str, np.ndarray],
]:
    rows = resolve_causal_alpha_v4_contract_rows(
        sample, start=contract.start, stop=contract.stop
    )
    full_forecast = fit.predict(sample)
    forecast = slice_forecast(full_forecast, rows)
    state = resolve_causal_alpha_v4_stage_state_inputs(sample)
    uncertainty_model = _uncertainty_model(
        fit=fit,
        sample=sample,
        full_forecast=full_forecast,
        state=state,
    )
    uncertainties = {
        horizon: uncertainty_model.resolve_uncertainty(
            horizon=horizon,
            realized_volatility=state.realized_volatility[rows],
            liquidity=state.liquidity[rows],
            basis_positioning_stress=state.basis_positioning_stress[rows],
        )
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    return rows, forecast, state, uncertainty_model, uncertainties


def _liveness_payload(
    *,
    prepared: Any,
    config: Any,
    symbol: str,
    contract: Any,
    fit: CausalAlphaV4Fit,
    sample: Any,
    rows: np.ndarray,
    forecast: CausalAlphaV4Forecast,
    uncertainty_model: CausalAlphaV4UncertaintyModel,
) -> tuple[dict[str, object], dict[str, str]]:
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
    slow_digest = content_digest(
        {
            "liveness_24h_digest": evidence["24h"].digest,
            "liveness_72h_digest": evidence["72h"].digest,
            "schema_version": "causal_alpha_v4_slow_fused_liveness_v1",
        }
    )
    body: dict[str, object] = {
        "schema_version": _SIGNAL_LIVENESS_SCOPE_SCHEMA,
        "run_manifest_digest": prepared.run_manifest_digest,
        "v4_context_manifest_digest": prepared.v4_context_manifest_digest,
        "config_digest": prepared.config_digest,
        "generator_code_digest": prepared.generator_code_digest,
        "fit_config_digest": config.fit.digest,
        "symbol": symbol,
        "episode_index": int(contract.episode_index),
        "contract_digest": contract.digest,
        "fit_digest": fit.digest,
        "forecast_digest": forecast.digest,
        "fast_4h_liveness_digest": evidence["4h"].digest,
        "slow_fused_liveness_digest": slow_digest,
        "horizon_evidence": {
            horizon: evidence[horizon].to_payload()
            for horizon in CAUSAL_ALPHA_V4_HORIZONS
        },
    }
    return {**body, "artifact_digest": content_digest(body)}, {
        "fast_4h": evidence["4h"].digest,
        "slow_fused": slow_digest,
    }


def _validate_liveness_payload(
    payload: Mapping[str, object],
    *,
    prepared: Any,
    config: Any,
    symbol: str,
    contract: Any,
) -> dict[str, str]:
    if (
        payload.get("run_manifest_digest") != prepared.run_manifest_digest
        or payload.get("v4_context_manifest_digest")
        != prepared.v4_context_manifest_digest
        or payload.get("config_digest") != prepared.config_digest
        or payload.get("generator_code_digest") != prepared.generator_code_digest
        or payload.get("fit_config_digest") != config.fit.digest
        or payload.get("symbol") != symbol
        or payload.get("episode_index") != contract.episode_index
        or payload.get("contract_digest") != contract.digest
    ):
        raise ValueError("V4 persisted liveness scope identity drifted")
    fast = str(payload["fast_4h_liveness_digest"])
    slow = str(payload["slow_fused_liveness_digest"])
    if len(fast) != 64 or len(slow) != 64:
        raise ValueError("V4 persisted liveness lane digests are invalid")
    return {"fast_4h": fast, "slow_fused": slow}


def run_causal_alpha_v4_signal_stage(
    prepared: Any,
    *,
    config: Any,
    store: CausalAlphaV4ArtifactStore,
    slice_forecast: Any,
) -> CausalAlphaV4SignalEvidence:
    fit_cache = CausalAlphaV4FitCache(
        train_symbols=prepared.train_symbols,
        samples=prepared.samples,
        config=config.fit,
    )
    metrics: list[CausalAlphaV4SignalScopeMetric] = []
    for symbol in prepared.train_symbols:
        sample = prepared.samples[symbol]
        for contract in prepared.nested_partitions[symbol].signal_contracts:
            liveness_path = (
                Path("signal/liveness") / symbol / f"{contract.episode_index}.json"
            )
            fast_path = Path("signal/fast") / symbol / f"{contract.episode_index}.json"
            slow_path = Path("signal/slow") / symbol / f"{contract.episode_index}.json"
            liveness_raw = store.load_leaf(
                liveness_path, expected_schema=_SIGNAL_LIVENESS_SCOPE_SCHEMA
            )
            fast_raw = store.load_leaf(
                fast_path, expected_schema="causal_alpha_v4_signal_scope_v1"
            )
            slow_raw = store.load_leaf(
                slow_path, expected_schema="causal_alpha_v4_signal_scope_v1"
            )
            if (
                liveness_raw is not None
                and fast_raw is not None
                and slow_raw is not None
            ):
                lane_digests = _validate_liveness_payload(
                    liveness_raw,
                    prepared=prepared,
                    config=config,
                    symbol=symbol,
                    contract=contract,
                )
                fast = _signal_metric_from_payload(fast_raw)
                slow = _signal_metric_from_payload(slow_raw)
                _validate_contract_metric(
                    fast,
                    prepared=prepared,
                    config=config,
                    symbol=symbol,
                    contract=contract,
                    lane=CausalAlphaV4SignalLane.FAST_4H,
                )
                _validate_contract_metric(
                    slow,
                    prepared=prepared,
                    config=config,
                    symbol=symbol,
                    contract=contract,
                    lane=CausalAlphaV4SignalLane.SLOW_FUSED,
                )
                if (
                    fast.liveness_digest != lane_digests["fast_4h"]
                    or slow.liveness_digest != lane_digests["slow_fused"]
                ):
                    raise ValueError("V4 persisted signal/liveness identity drifted")
            else:
                fit = fit_cache.resolve(contract.start)
                rows, forecast, state, uncertainty_model, _ = _contract_bundle(
                    fit=fit,
                    sample=sample,
                    contract=contract,
                    slice_forecast=slice_forecast,
                )
                liveness_payload, lane_digests = _liveness_payload(
                    prepared=prepared,
                    config=config,
                    symbol=symbol,
                    contract=contract,
                    fit=fit,
                    sample=sample,
                    rows=rows,
                    forecast=forecast,
                    uncertainty_model=uncertainty_model,
                )
                built = build_causal_alpha_v4_signal_scope_metrics(
                    run_manifest_digest=prepared.run_manifest_digest,
                    fit_config_digest=config.fit.digest,
                    symbol=symbol,
                    episode_index=contract.episode_index,
                    contract_start=contract.start,
                    contract_stop=contract.stop,
                    contract_digest=contract.digest,
                    fit_digest=fit.digest,
                    forecast=forecast,
                    liveness_digests=lane_digests,
                    actionable_mask=state.actionable[rows],
                    labels_4h=np.asarray(sample.labels_4h)[rows],
                    label_end_indices_4h=np.asarray(sample.label_end_indices_4h)[rows],
                    labels_24h=np.asarray(sample.labels_24h)[rows],
                    label_end_indices_24h=np.asarray(sample.label_end_indices_24h)[
                        rows
                    ],
                    labels_72h=np.asarray(sample.labels_72h)[rows],
                    label_end_indices_72h=np.asarray(sample.label_end_indices_72h)[
                        rows
                    ],
                )
                fast = built[CausalAlphaV4SignalLane.FAST_4H]
                slow = built[CausalAlphaV4SignalLane.SLOW_FUSED]
                store.write_leaf(liveness_path, liveness_payload)
                store.write_leaf(fast_path, fast.to_payload())
                store.write_leaf(slow_path, slow.to_payload())
            metrics.extend((fast, slow))
    expected_per_lane = len(prepared.train_symbols) * 8
    return evaluate_causal_alpha_v4_signal_gate(
        tuple(metrics),
        expected_raw_scope_count_per_lane=expected_per_lane,
        gate=config.signal_gate,
    )


def _environment_for(prepared: Any, symbol: str) -> Any:
    factories = getattr(prepared.prepared_v3, "environment_factories", None)
    if not isinstance(factories, Mapping) or symbol not in factories:
        raise ValueError("V4 stage environment factory scope drifted")
    factory = factories[symbol]
    if not callable(factory):
        raise TypeError("V4 stage environment factory is not callable")
    environment = factory()
    if not callable(getattr(environment, "close", None)):
        raise TypeError("V4 stage environment must be closable")
    return environment


def _target_and_replay(
    *,
    prepared: Any,
    config: Any,
    symbol: str,
    contract: Any,
    fit: CausalAlphaV4Fit,
    slice_forecast: Any,
) -> CausalAlphaV4ReplayMetric:
    sample = prepared.samples[symbol]
    rows, forecast, state, _uncertainty_model_value, uncertainties = _contract_bundle(
        fit=fit,
        sample=sample,
        contract=contract,
        slice_forecast=slice_forecast,
    )
    environment = _environment_for(prepared, symbol)
    try:
        expected_initial = np.asarray(contract.initial_weights, dtype=np.float64)
        resolved_initial = resolve_episode_initial_weights(
            environment, contract.initial_state_mode, contract.start
        )
        if expected_initial.shape != (1,) or not np.array_equal(
            expected_initial, resolved_initial
        ):
            raise ValueError("V4 replay initial state drifted from frozen contract")
        execution = getattr(
            getattr(environment, "config", None), "execution_cost", None
        )
        prepared_costs = getattr(prepared.prepared_v3, "execution_costs", None)
        if (
            not isinstance(prepared_costs, Mapping)
            or execution != prepared_costs[symbol]
        ):
            raise ValueError("V4 replay execution cost identity drifted")
        delay = getattr(prepared.prepared_v3, "signal_delays", {})[symbol]
        decision_bars = getattr(prepared.prepared_v3, "decision_bars", {})[symbol]
        costs = causal_alpha_one_way_cost_rates(
            environment.dataset,
            execution,
            decision_indices=forecast.decision_indices,
            signal_delay_decisions=delay,
            decision_bars=decision_bars,
        )
        reference_equity = float(getattr(environment, "initial_capital", np.nan))
        market_cap = float(
            getattr(prepared.prepared_v3, "max_position_to_market_notional", np.nan)
        )
        caps = causal_alpha_liquidity_weight_caps(
            environment.dataset,
            decision_indices=forecast.decision_indices,
            reference_portfolio_value=reference_equity,
            max_position_to_market_notional=market_cap,
            lookback_decisions=_LIQUIDITY_LOOKBACK_DECISIONS,
            lower_quantile=_LIQUIDITY_LOWER_QUANTILE,
            safety_multiplier=_LIQUIDITY_SAFETY_MULTIPLIER,
        )
        target = causal_alpha_v4_target_path(
            forecast.final_predictions["4h"],
            forecast.final_predictions["24h"],
            forecast.final_predictions["72h"],
            direction_score_4h=forecast.direction_scores["4h"],
            uncertainty_4h=uncertainties["4h"],
            uncertainty_24h=uncertainties["24h"],
            uncertainty_72h=uncertainties["72h"],
            one_way_cost_rates=costs,
            liquidity_weight_caps=caps,
            config=config.target,
            initial_weight=float(expected_initial[0]),
            actionable_mask=state.actionable[rows],
        )
        evaluation = evaluate_episode_action_path_on_environment(
            environment,
            contract,
            actions=target.targets[:, None].astype(np.float32),
        )
        return build_causal_alpha_v4_replay_metric(
            run_manifest_digest=prepared.run_manifest_digest,
            v4_context_manifest_digest=prepared.v4_context_manifest_digest,
            config_digest=prepared.config_digest,
            symbol=symbol,
            episode_index=contract.episode_index,
            contract_digest=contract.digest,
            fit_digest=fit.digest,
            forecast_digest=forecast.digest,
            target_path=target,
            evaluation=evaluation,
            episode_hours=float(getattr(prepared.prepared_v3, "episode_hours")),
        )
    finally:
        environment.close()


def run_causal_alpha_v4_selection_stage(
    prepared: Any,
    signal_evidence: CausalAlphaV4SignalEvidence,
    *,
    config: Any,
    store: CausalAlphaV4ArtifactStore,
    slice_forecast: Any,
) -> CausalAlphaV4SelectionEvidence:
    if not signal_evidence.passed:
        raise ValueError("V4 selection cannot bypass failed Signal evidence")
    fit_cache = CausalAlphaV4FitCache(
        train_symbols=prepared.train_symbols,
        samples=prepared.samples,
        config=config.fit,
    )
    records: list[CausalAlphaV4ReplayMetric] = []
    for symbol in prepared.train_symbols:
        for contract in prepared.nested_partitions[symbol].economic_contracts:
            path = Path("selection") / symbol / f"{contract.episode_index}.json"
            raw = store.load_leaf(
                path, expected_schema="causal_alpha_v4_replay_metric_v1"
            )
            if raw is None:
                metric = _target_and_replay(
                    prepared=prepared,
                    config=config,
                    symbol=symbol,
                    contract=contract,
                    fit=fit_cache.resolve(contract.start),
                    slice_forecast=slice_forecast,
                )
                store.write_leaf(path, metric.to_payload())
            else:
                metric = _replay_metric_from_payload(raw)
                _validate_replay_metric(
                    metric, prepared=prepared, symbol=symbol, contract=contract
                )
            records.append(metric)
    return evaluate_causal_alpha_v4_selection(tuple(records))


def run_causal_alpha_v4_admission_stage(
    prepared: Any,
    signal_evidence: CausalAlphaV4SignalEvidence,
    selection_evidence: CausalAlphaV4SelectionEvidence,
    *,
    config: Any,
    store: CausalAlphaV4ArtifactStore,
    slice_forecast: Any,
) -> CausalAlphaV4AdmissionEvidence:
    if not signal_evidence.passed or not selection_evidence.passed:
        raise ValueError("V4 admission cannot bypass upstream gates")
    holdouts = tuple(
        prepared.nested_partitions[symbol].holdout_contract
        for symbol in prepared.train_symbols
    )
    starts = {contract.start for contract in holdouts}
    if len(starts) != 1:
        raise ValueError("V4 holdout starts drifted across train symbols")
    holdout_start = next(iter(starts))
    fit = fit_causal_alpha_v4(
        train_symbols=prepared.train_symbols,
        samples=prepared.samples,
        knowledge_cutoff=holdout_start,
        config=config.fit,
    )
    records: list[CausalAlphaV4ReplayMetric] = []
    for symbol, contract in zip(prepared.train_symbols, holdouts, strict=True):
        path = Path("admission") / f"{symbol}.json"
        raw = store.load_leaf(path, expected_schema="causal_alpha_v4_replay_metric_v1")
        if raw is None:
            metric = _target_and_replay(
                prepared=prepared,
                config=config,
                symbol=symbol,
                contract=contract,
                fit=fit,
                slice_forecast=slice_forecast,
            )
            store.write_leaf(path, metric.to_payload())
        else:
            metric = _replay_metric_from_payload(raw)
            _validate_replay_metric(
                metric, prepared=prepared, symbol=symbol, contract=contract
            )
            if metric.fit_digest != fit.digest:
                raise ValueError("V4 persisted admission fit identity drifted")
        records.append(metric)
    return evaluate_causal_alpha_v4_admission(
        tuple(records),
        signal_evidence=signal_evidence,
        selection_evidence=selection_evidence,
        fit_knowledge_cutoff=holdout_start,
        holdout_start=holdout_start,
    )


__all__ = [
    "CausalAlphaV4FitCache",
    "run_causal_alpha_v4_admission_stage",
    "run_causal_alpha_v4_selection_stage",
    "run_causal_alpha_v4_signal_stage",
]
