"""Build research-only causal alpha V3 Signal diagnostic sidecars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v3 import CausalAlphaV3Forecast
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3 import (
    CausalAlphaV3Fit,
    build_causal_alpha_v3_symbol_balanced_weights,
    causal_alpha_v3_weight_digest,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticModel,
    CausalAlphaV3SignalDiagnosticPredictionRow,
    CausalAlphaV3SignalDiagnosticRealizedRow,
    CausalAlphaV3SignalDiagnosticScope,
    weighted_effective_sample_size,
)


def _diagnostic_model(
    *,
    fitted: CausalAlphaV3Fit,
    samples: Mapping[str, CausalAlphaSymbolSamples],
    horizon: Literal["24h", "72h"],
) -> CausalAlphaV3SignalDiagnosticModel:
    weights = build_causal_alpha_v3_symbol_balanced_weights(
        train_symbols=fitted.train_symbols,
        samples=samples,
        knowledge_cutoff=fitted.knowledge_cutoff,
        horizon=horizon,
    )
    weight_digest = causal_alpha_v3_weight_digest(
        fitted.train_symbols,
        weights,
        horizon=horizon,
        knowledge_cutoff=fitted.knowledge_cutoff,
    )
    expected_weight_digest = (
        fitted.weight_digest_24h if horizon == "24h" else fitted.weight_digest_72h
    )
    if weight_digest != expected_weight_digest:
        raise ValueError(
            "V3 diagnostic weight digest does not reproduce fitted evidence"
        )
    pooled_weights = np.concatenate(
        tuple(weights[symbol] for symbol in fitted.train_symbols)
    )
    model = fitted.model_24h if horizon == "24h" else fitted.model_72h
    residual_rmse = (
        fitted.residual_rmse_24h if horizon == "24h" else fitted.residual_rmse_72h
    )
    per_symbol_ess = tuple(
        sorted(
            (
                symbol,
                weighted_effective_sample_size(weights[symbol]),
            )
            for symbol in fitted.train_symbols
        )
    )
    return CausalAlphaV3SignalDiagnosticModel(
        model_digest=model.digest,
        feature_names=model.feature_names,
        intercept=float(model.intercept),
        coefficients=tuple(float(value) for value in model.coefficients),
        location=tuple(float(value) for value in model.location),
        scale=tuple(float(value) for value in model.scale),
        constant_mask=tuple(bool(value) for value in model.constant_mask),
        fitted_row_count=model.sample_count,
        weighted_residual_rmse=residual_rmse,
        pooled_weighted_ess=weighted_effective_sample_size(pooled_weights),
        per_symbol_weighted_ess=per_symbol_ess,
        overlap_weight_digest=weight_digest,
    )


def _aligned_vector(
    value: object,
    *,
    size: int,
    dtype: Any,
    field: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"V3 diagnostic {field} must be decision aligned")
    return array


def build_causal_alpha_v3_signal_diagnostic_scope(
    *,
    run_manifest_digest: str,
    symbol: str,
    samples: Mapping[str, CausalAlphaSymbolSamples],
    fitted: CausalAlphaV3Fit,
    forecast: CausalAlphaV3Forecast,
    block: CausalAlphaSymbolSamples,
    contract: OracleEpisodeContract,
    decisions: object,
    actionable: object,
    feature_available: object,
    matched: object,
    labels_24h: object,
    labels_72h: object,
    ends_24h: object,
    ends_72h: object,
    signal_metric_digest: str,
    canonical_cohort_indices: tuple[int, ...],
) -> CausalAlphaV3SignalDiagnosticScope:
    """Persist ephemeral Signal inputs without changing canonical Gate evidence."""

    require_sha256(run_manifest_digest, field="V3 diagnostic run manifest digest")
    require_sha256(signal_metric_digest, field="V3 diagnostic signal metric digest")
    if symbol != block.symbol or symbol not in samples:
        raise ValueError("V3 diagnostic symbol identity drifted")
    if block is not samples[symbol] and block.digest != samples[symbol].digest:
        raise ValueError("V3 diagnostic sample block identity drifted")
    if set(fitted.train_symbols) != set(samples):
        raise ValueError("V3 diagnostic fitted sample scope drifted")
    if fitted.knowledge_cutoff != contract.start:
        raise ValueError("V3 diagnostic fit cutoff does not match contract start")
    if contract.dataset_id != block.dataset_id:
        raise ValueError("V3 diagnostic contract dataset identity drifted")
    if fitted.model_24h.feature_names != block.feature_names:
        raise ValueError("V3 diagnostic fitted feature order drifted")

    decision_values = np.asarray(decisions, dtype=np.int64).reshape(-1)
    if decision_values.size == 0:
        raise ValueError("V3 diagnostic decisions must not be empty")
    size = int(decision_values.size)
    action_mask = _aligned_vector(
        actionable,
        size=size,
        dtype=np.bool_,
        field="actionable mask",
    )
    matched_mask = _aligned_vector(
        matched,
        size=size,
        dtype=np.bool_,
        field="matched mask",
    )
    first_labels = _aligned_vector(
        labels_24h,
        size=size,
        dtype=np.float64,
        field="24h labels",
    )
    second_labels = _aligned_vector(
        labels_72h,
        size=size,
        dtype=np.float64,
        field="72h labels",
    )
    first_ends = _aligned_vector(
        ends_24h,
        size=size,
        dtype=np.int64,
        field="24h label ends",
    )
    second_ends = _aligned_vector(
        ends_72h,
        size=size,
        dtype=np.int64,
        field="72h label ends",
    )
    availability = np.asarray(feature_available, dtype=np.bool_)
    feature_width = len(block.feature_names)
    if availability.shape != (size, feature_width):
        raise ValueError("V3 diagnostic feature availability must be decision aligned")
    forecast_arrays = (
        forecast.prediction_24h,
        forecast.prediction_72h,
        forecast.expected_return_24h_equivalent,
        forecast.uncertainty_24h_equivalent,
        forecast.signal_to_uncertainty,
    )
    if any(np.asarray(value).shape != (size,) for value in forecast_arrays):
        raise ValueError("V3 diagnostic forecast must be decision aligned")

    available_counts = availability.sum(axis=1, dtype=np.int64)
    available_fractions = available_counts.astype(np.float64) / float(feature_width)
    prediction_rows = tuple(
        CausalAlphaV3SignalDiagnosticPredictionRow(
            decision_index=int(decision_values[row]),
            actionable=bool(action_mask[row]),
            available_feature_count=int(available_counts[row]),
            available_feature_fraction=float(available_fractions[row]),
            prediction_24h=float(forecast.prediction_24h[row]),
            prediction_72h=float(forecast.prediction_72h[row]),
            prediction_72h_24h_equivalent=float(forecast.prediction_72h[row] / 3.0),
            expected_return_24h_equivalent=float(
                forecast.expected_return_24h_equivalent[row]
            ),
            uncertainty_24h_equivalent=float(forecast.uncertainty_24h_equivalent[row]),
            signal_to_uncertainty=float(forecast.signal_to_uncertainty[row]),
        )
        for row in range(size)
    )

    eligible_24h = (
        action_mask
        & matched_mask
        & np.isfinite(first_labels)
        & (first_ends >= decision_values)
        & (first_ends < contract.stop)
    )
    eligible_72h = (
        action_mask
        & matched_mask
        & np.isfinite(second_labels)
        & (second_ends >= decision_values)
        & (second_ends < contract.stop)
    )
    eligible_fused = eligible_24h & eligible_72h

    realized_24h_rows = tuple(
        CausalAlphaV3SignalDiagnosticRealizedRow(
            decision_index=int(decision_values[row]),
            label_end_index=int(first_ends[row]),
            available_feature_count=int(available_counts[row]),
            available_feature_fraction=float(available_fractions[row]),
            prediction=float(forecast.prediction_24h[row]),
            realized_return=float(first_labels[row]),
        )
        for row in np.flatnonzero(eligible_24h)
    )
    realized_72h_rows = tuple(
        CausalAlphaV3SignalDiagnosticRealizedRow(
            decision_index=int(decision_values[row]),
            label_end_index=int(second_ends[row]),
            available_feature_count=int(available_counts[row]),
            available_feature_fraction=float(available_fractions[row]),
            prediction=float(forecast.prediction_72h[row] / 3.0),
            realized_return=float(second_labels[row] / 3.0),
            raw_prediction=float(forecast.prediction_72h[row]),
            raw_realized_return=float(second_labels[row]),
        )
        for row in np.flatnonzero(eligible_72h)
    )
    realized_fused_rows = tuple(
        CausalAlphaV3SignalDiagnosticRealizedRow(
            decision_index=int(decision_values[row]),
            label_end_index=int(max(first_ends[row], second_ends[row])),
            available_feature_count=int(available_counts[row]),
            available_feature_fraction=float(available_fractions[row]),
            prediction=float(forecast.expected_return_24h_equivalent[row]),
            realized_return=float(0.5 * (first_labels[row] + second_labels[row] / 3.0)),
        )
        for row in np.flatnonzero(eligible_fused)
    )
    complete_count = int(np.count_nonzero(available_counts == feature_width))
    per_feature_fraction = tuple(
        float(value) for value in np.mean(availability, axis=0, dtype=np.float64)
    )

    return CausalAlphaV3SignalDiagnosticScope(
        run_manifest_digest=run_manifest_digest,
        fit_config_digest=fitted.config.digest,
        symbol=symbol,
        episode_index=contract.episode_index,
        contract_start=contract.start,
        contract_stop=contract.stop,
        contract_digest=contract.digest,
        signal_metric_digest=signal_metric_digest,
        fit_digest=fitted.digest,
        forecast_digest=forecast.digest,
        feature_schema_digest=block.feature_schema_digest,
        model_24h=_diagnostic_model(fitted=fitted, samples=samples, horizon="24h"),
        model_72h=_diagnostic_model(fitted=fitted, samples=samples, horizon="72h"),
        prediction_rows=prediction_rows,
        realized_24h_rows=realized_24h_rows,
        realized_72h_rows=realized_72h_rows,
        realized_fused_rows=realized_fused_rows,
        canonical_cohort_indices=canonical_cohort_indices,
        per_feature_available_fraction=per_feature_fraction,
        complete_feature_row_count=complete_count,
        incomplete_feature_row_count=size - complete_count,
        available_feature_fraction_minimum=float(np.min(available_fractions)),
        available_feature_fraction_mean=float(
            np.mean(available_fractions, dtype=np.float64)
        ),
        available_feature_fraction_maximum=float(np.max(available_fractions)),
    )


__all__ = ["build_causal_alpha_v3_signal_diagnostic_scope"]
