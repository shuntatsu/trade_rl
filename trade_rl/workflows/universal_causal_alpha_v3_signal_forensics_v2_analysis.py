"""Pure builders for read-only Causal Alpha V3 Signal Forensics V2 summaries."""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np

from trade_rl.learning.causal_alpha_diagnostics import (
    CAUSAL_ALPHA_SIGNAL_QUANTILES,
    CausalAlphaSignalDiagnostics,
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticModel,
    CausalAlphaV3SignalDiagnosticRealizedRow,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_analysis_contracts import (
    CausalAlphaV3AvailabilityHorizon,
    CausalAlphaV3AvailabilityPartition,
    CausalAlphaV3AvailabilitySummary,
    CausalAlphaV3ChronologicalHorizonSeries,
    CausalAlphaV3ChronologicalMetric,
    CausalAlphaV3EpisodePredictionDistributions,
    CausalAlphaV3FitPredictionDistributions,
    CausalAlphaV3ForensicsQuantile,
    CausalAlphaV3ModelSeries,
    CausalAlphaV3ModelSnapshot,
    CausalAlphaV3ModelTransition,
    CausalAlphaV3PairedHorizonDiagnostics,
    CausalAlphaV3PerSymbolEssSeries,
    CausalAlphaV3PredictionDistribution,
    CausalAlphaV3ScopeSidecarSummary,
    CausalAlphaV3SignalForensicsV2Analysis,
    Horizon,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_loader import (
    CausalAlphaV3SignalForensicsV2BoundScope,
)

_HORIZONS: Final[tuple[Horizon, ...]] = ("24h", "72h", "fused")
_MODEL_HORIZONS: Final[tuple[Literal["24h", "72h"], ...]] = ("24h", "72h")
_PREDICTION_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("prediction_24h", "prediction_24h"),
    ("prediction_72h_24h_equivalent", "prediction_72h_24h_equivalent"),
    ("expected_return_24h_equivalent", "expected_return_24h_equivalent"),
    ("uncertainty_24h_equivalent", "uncertainty_24h_equivalent"),
    ("signal_to_uncertainty", "signal_to_uncertainty"),
)
_EPSILON: Final = 1e-15


def _finite_values(values: object, *, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"V3 signal forensics {field} must be non-empty and finite")
    return array


def _prediction_distribution(
    name: str, values: object
) -> CausalAlphaV3PredictionDistribution:
    array = _finite_values(values, field=name)
    quantile_values = np.quantile(
        array,
        np.asarray(CAUSAL_ALPHA_SIGNAL_QUANTILES, dtype=np.float64),
        method="linear",
    )
    return CausalAlphaV3PredictionDistribution(
        name=name,
        count=int(array.size),
        mean=float(np.mean(array, dtype=np.float64)),
        standard_deviation=float(np.std(array, dtype=np.float64)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        quantiles=tuple(
            CausalAlphaV3ForensicsQuantile(
                quantile=float(quantile),
                value=float(value),
            )
            for quantile, value in zip(
                CAUSAL_ALPHA_SIGNAL_QUANTILES, quantile_values, strict=True
            )
        ),
    )


def _diagnostics_from_rows(
    rows: tuple[CausalAlphaV3SignalDiagnosticRealizedRow, ...],
) -> CausalAlphaSignalDiagnostics:
    return evaluate_causal_alpha_signal_diagnostics(
        np.asarray([row.prediction for row in rows], dtype=np.float64),
        np.asarray([row.realized_return for row in rows], dtype=np.float64),
    )


def _rows_for_horizon(
    scope: CausalAlphaV3SignalForensicsV2BoundScope, horizon: Horizon
) -> tuple[CausalAlphaV3SignalDiagnosticRealizedRow, ...]:
    if horizon == "24h":
        return scope.diagnostic.realized_24h_rows
    if horizon == "72h":
        return scope.diagnostic.realized_72h_rows
    return scope.diagnostic.realized_fused_rows


def _prediction_values_for_horizon(
    scope: CausalAlphaV3SignalForensicsV2BoundScope, horizon: Horizon
) -> tuple[float, ...]:
    if horizon == "24h":
        field = "prediction_24h"
    elif horizon == "72h":
        field = "prediction_72h_24h_equivalent"
    else:
        field = "expected_return_24h_equivalent"
    return tuple(
        float(getattr(row, field)) for row in scope.diagnostic.prediction_rows
    )


def _prediction_distributions(
    scopes: tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...],
) -> tuple[CausalAlphaV3PredictionDistribution, ...]:
    if not scopes:
        raise ValueError("V3 signal forensics prediction scope is empty")
    result: list[CausalAlphaV3PredictionDistribution] = []
    for name, field in _PREDICTION_FIELDS:
        values = [
            float(getattr(row, field))
            for scope in scopes
            for row in scope.diagnostic.prediction_rows
        ]
        result.append(_prediction_distribution(name, values))
    return tuple(result)


def _paired_horizons(
    scope: CausalAlphaV3SignalForensicsV2BoundScope,
) -> CausalAlphaV3PairedHorizonDiagnostics:
    rows_24h = {row.decision_index: row for row in scope.diagnostic.realized_24h_rows}
    rows_72h = {row.decision_index: row for row in scope.diagnostic.realized_72h_rows}
    decisions = tuple(sorted(set(rows_24h) & set(rows_72h)))
    if len(decisions) < 2:
        return CausalAlphaV3PairedHorizonDiagnostics(
            decision_indices=decisions,
            sample_count=len(decisions),
            diagnostics_24h=None,
            diagnostics_72h=None,
            direction_accuracy_delta_24h_minus_72h=None,
            rank_correlation_delta_24h_minus_72h=None,
            unavailable_reason="fewer_than_two_matched_realized_decisions",
        )
    diagnostics_24h = _diagnostics_from_rows(
        tuple(rows_24h[index] for index in decisions)
    )
    diagnostics_72h = _diagnostics_from_rows(
        tuple(rows_72h[index] for index in decisions)
    )
    rank_delta = (
        None
        if diagnostics_24h.rank_correlation is None
        or diagnostics_72h.rank_correlation is None
        else diagnostics_24h.rank_correlation - diagnostics_72h.rank_correlation
    )
    return CausalAlphaV3PairedHorizonDiagnostics(
        decision_indices=decisions,
        sample_count=len(decisions),
        diagnostics_24h=diagnostics_24h,
        diagnostics_72h=diagnostics_72h,
        direction_accuracy_delta_24h_minus_72h=(
            diagnostics_24h.direction_accuracy - diagnostics_72h.direction_accuracy
        ),
        rank_correlation_delta_24h_minus_72h=rank_delta,
        unavailable_reason=None,
    )


def _availability_partition(
    rows: tuple[CausalAlphaV3SignalDiagnosticRealizedRow, ...],
    *,
    complete: bool,
) -> CausalAlphaV3AvailabilityPartition:
    selected = tuple(
        row for row in rows if (row.available_feature_fraction == 1.0) is complete
    )
    if len(selected) < 2:
        return CausalAlphaV3AvailabilityPartition(
            row_count=len(selected),
            diagnostics=None,
            unavailable_reason="fewer_than_two_realized_rows",
        )
    return CausalAlphaV3AvailabilityPartition(
        row_count=len(selected),
        diagnostics=_diagnostics_from_rows(selected),
        unavailable_reason=None,
    )


def _availability_summary(
    scope: CausalAlphaV3SignalForensicsV2BoundScope,
) -> CausalAlphaV3AvailabilitySummary:
    diagnostic = scope.diagnostic
    feature_names = diagnostic.model_24h.feature_names
    if feature_names != diagnostic.model_72h.feature_names:
        raise ValueError(
            "V3 signal forensics model feature order differs across horizons"
        )
    if len(feature_names) != len(diagnostic.per_feature_available_fraction):
        raise ValueError("V3 signal forensics feature availability width drifted")
    fractions = tuple(
        float(row.available_feature_fraction) for row in diagnostic.prediction_rows
    )
    horizons = tuple(
        CausalAlphaV3AvailabilityHorizon(
            horizon=horizon,
            complete=_availability_partition(
                _rows_for_horizon(scope, horizon), complete=True
            ),
            incomplete=_availability_partition(
                _rows_for_horizon(scope, horizon), complete=False
            ),
        )
        for horizon in _HORIZONS
    )
    return CausalAlphaV3AvailabilitySummary(
        feature_names=feature_names,
        per_feature_available_fraction=diagnostic.per_feature_available_fraction,
        complete_prediction_row_count=sum(value == 1.0 for value in fractions),
        incomplete_prediction_row_count=sum(value < 1.0 for value in fractions),
        row_available_fraction=_prediction_distribution(
            "available_feature_fraction", fractions
        ),
        horizons=horizons,
    )


def _scope_summary(
    scope: CausalAlphaV3SignalForensicsV2BoundScope,
) -> CausalAlphaV3ScopeSidecarSummary:
    return CausalAlphaV3ScopeSidecarSummary(
        fit_config_digest=scope.metric.fit_config_digest,
        symbol=scope.metric.symbol,
        episode_index=scope.metric.episode_index,
        contract_start=scope.metric.contract_start,
        contract_stop=scope.metric.contract_stop,
        metric_digest=scope.metric.digest,
        diagnostic_digest=scope.diagnostic.digest,
        horizon_24h=_diagnostics_from_rows(scope.diagnostic.realized_24h_rows),
        horizon_72h=_diagnostics_from_rows(scope.diagnostic.realized_72h_rows),
        horizon_fused=_diagnostics_from_rows(scope.diagnostic.realized_fused_rows),
        paired_24h_72h=_paired_horizons(scope),
        prediction_distributions=_prediction_distributions((scope,)),
        availability=_availability_summary(scope),
    )


def _chronological_metric(
    values: tuple[float | None, ...],
) -> CausalAlphaV3ChronologicalMetric:
    if not values:
        raise ValueError("V3 signal forensics chronological series is empty")
    defined = tuple(
        (index, float(value)) for index, value in enumerate(values) if value is not None
    )
    if any(not math.isfinite(value) for _, value in defined):
        raise ValueError("V3 signal forensics chronological values must be finite")
    defined_values = np.asarray([value for _, value in defined], dtype=np.float64)
    minimum = None if not defined else float(np.min(defined_values))
    mean = None if not defined else float(np.mean(defined_values, dtype=np.float64))
    maximum = None if not defined else float(np.max(defined_values))
    split = max(1, len(values) // 2)
    early_values = np.asarray(
        [float(value) for value in values[:split] if value is not None],
        dtype=np.float64,
    )
    late_slice = values[split:] if values[split:] else values[:split]
    late_values = np.asarray(
        [float(value) for value in late_slice if value is not None],
        dtype=np.float64,
    )
    early_mean = (
        None
        if early_values.size == 0
        else float(np.mean(early_values, dtype=np.float64))
    )
    late_mean = (
        None if late_values.size == 0 else float(np.mean(late_values, dtype=np.float64))
    )
    slope: float | None = None
    if len(defined) >= 2:
        x = np.asarray([index for index, _ in defined], dtype=np.float64)
        y = defined_values
        centered_x = x - float(np.mean(x, dtype=np.float64))
        denominator = float(np.sum(np.square(centered_x), dtype=np.float64))
        if denominator > _EPSILON:
            slope = float(
                np.sum(
                    centered_x * (y - float(np.mean(y, dtype=np.float64))),
                    dtype=np.float64,
                )
                / denominator
            )
    return CausalAlphaV3ChronologicalMetric(
        values=values,
        count=len(values),
        defined_count=len(defined),
        undefined_count=len(values) - len(defined),
        minimum=minimum,
        mean=mean,
        maximum=maximum,
        early_mean=early_mean,
        late_mean=late_mean,
        slope=slope,
    )


def _cluster_scopes(
    scopes: tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...],
) -> tuple[
    tuple[str, int, int, tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...]], ...
]:
    fit_order = tuple(dict.fromkeys(scope.metric.fit_config_digest for scope in scopes))
    clusters: list[
        tuple[str, int, int, tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...]]
    ] = []
    for fit_digest in fit_order:
        fit_scopes = tuple(
            scope for scope in scopes if scope.metric.fit_config_digest == fit_digest
        )
        intervals = tuple(
            sorted(
                {
                    (scope.metric.contract_start, scope.metric.contract_stop)
                    for scope in fit_scopes
                }
            )
        )
        for contract_start, contract_stop in intervals:
            cluster = tuple(
                scope
                for scope in fit_scopes
                if scope.metric.contract_start == contract_start
                and scope.metric.contract_stop == contract_stop
            )
            clusters.append((fit_digest, contract_start, contract_stop, cluster))
    return tuple(clusters)


def _models_match(
    left: CausalAlphaV3SignalDiagnosticModel,
    right: CausalAlphaV3SignalDiagnosticModel,
) -> bool:
    return left.to_payload() == right.to_payload()


def _deduplicated_snapshots(
    scopes: tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...],
) -> tuple[CausalAlphaV3ModelSnapshot, ...]:
    snapshots: list[CausalAlphaV3ModelSnapshot] = []
    for fit_digest, contract_start, contract_stop, cluster in _cluster_scopes(scopes):
        if not cluster:
            raise ValueError("V3 signal forensics pooled model cluster is empty")
        representative = cluster[0].diagnostic
        for other in cluster[1:]:
            if not _models_match(representative.model_24h, other.diagnostic.model_24h):
                raise ValueError(
                    "V3 signal forensics pooled model 24h evidence disagrees across symbols"
                )
            if not _models_match(representative.model_72h, other.diagnostic.model_72h):
                raise ValueError(
                    "V3 signal forensics pooled model 72h evidence disagrees across symbols"
                )
        snapshots.extend(
            (
                CausalAlphaV3ModelSnapshot(
                    fit_config_digest=fit_digest,
                    contract_start=contract_start,
                    contract_stop=contract_stop,
                    horizon="24h",
                    model=representative.model_24h,
                ),
                CausalAlphaV3ModelSnapshot(
                    fit_config_digest=fit_digest,
                    contract_start=contract_start,
                    contract_stop=contract_stop,
                    horizon="72h",
                    model=representative.model_72h,
                ),
            )
        )
    return tuple(snapshots)


def _model_transition(
    previous: CausalAlphaV3ModelSnapshot,
    current: CausalAlphaV3ModelSnapshot,
) -> CausalAlphaV3ModelTransition:
    if previous.model.feature_names != current.model.feature_names:
        raise ValueError(
            "V3 signal forensics model feature order changed across snapshots"
        )
    a = np.asarray(previous.model.coefficients, dtype=np.float64)
    b = np.asarray(current.model.coefficients, dtype=np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= _EPSILON or norm_b <= _EPSILON:
        cosine = None
        cosine_reason = "zero_coefficient_norm"
    else:
        cosine = float(np.dot(a, b) / (norm_a * norm_b))
        cosine_reason = None
    active = (a != 0.0) & (b != 0.0)
    active_count = int(np.count_nonzero(active))
    if active_count == 0:
        sign_flip_rate = None
        sign_flip_reason = "no_active_paired_coefficients"
    else:
        sign_flip_rate = float(
            np.count_nonzero(np.sign(a[active]) != np.sign(b[active])) / active_count
        )
        sign_flip_reason = None
    location_previous = np.asarray(previous.model.location, dtype=np.float64)
    location_current = np.asarray(current.model.location, dtype=np.float64)
    scale_previous = np.asarray(previous.model.scale, dtype=np.float64)
    scale_current = np.asarray(current.model.scale, dtype=np.float64)
    location_shift = float(
        np.sqrt(
            np.mean(
                np.square((location_current - location_previous) / scale_previous),
                dtype=np.float64,
            )
        )
    )
    log_scale_ratio = float(
        np.sqrt(
            np.mean(
                np.square(np.log(scale_current / scale_previous)),
                dtype=np.float64,
            )
        )
    )
    return CausalAlphaV3ModelTransition(
        previous_contract_start=previous.contract_start,
        previous_contract_stop=previous.contract_stop,
        current_contract_start=current.contract_start,
        current_contract_stop=current.contract_stop,
        coefficient_cosine_similarity=cosine,
        coefficient_cosine_unavailable_reason=cosine_reason,
        active_paired_coefficient_count=active_count,
        coefficient_sign_flip_rate=sign_flip_rate,
        coefficient_sign_flip_unavailable_reason=sign_flip_reason,
        location_shift_rms=location_shift,
        log_scale_ratio_rms=log_scale_ratio,
    )


def _model_series(
    snapshots: tuple[CausalAlphaV3ModelSnapshot, ...],
) -> tuple[CausalAlphaV3ModelSeries, ...]:
    fit_order = tuple(
        dict.fromkeys(snapshot.fit_config_digest for snapshot in snapshots)
    )
    result: list[CausalAlphaV3ModelSeries] = []
    for fit_digest in fit_order:
        for horizon in _MODEL_HORIZONS:
            selected = tuple(
                sorted(
                    (
                        snapshot
                        for snapshot in snapshots
                        if snapshot.fit_config_digest == fit_digest
                        and snapshot.horizon == horizon
                    ),
                    key=lambda item: (item.contract_start, item.contract_stop),
                )
            )
            if not selected:
                raise ValueError("V3 signal forensics model series is incomplete")
            transitions = tuple(
                _model_transition(previous, current)
                for previous, current in zip(selected, selected[1:])
            )
            digests = tuple(
                snapshot.model.overlap_weight_digest for snapshot in selected
            )
            symbols = tuple(
                symbol for symbol, _ in selected[0].model.per_symbol_weighted_ess
            )
            if any(
                tuple(symbol for symbol, _ in snapshot.model.per_symbol_weighted_ess)
                != symbols
                for snapshot in selected[1:]
            ):
                raise ValueError(
                    "V3 signal forensics per-symbol ESS identity changed across snapshots"
                )
            per_symbol_series = tuple(
                CausalAlphaV3PerSymbolEssSeries(
                    symbol=symbol,
                    weighted_ess=_chronological_metric(
                        tuple(
                            dict(snapshot.model.per_symbol_weighted_ess)[symbol]
                            for snapshot in selected
                        )
                    ),
                )
                for symbol in symbols
            )
            result.append(
                CausalAlphaV3ModelSeries(
                    fit_config_digest=fit_digest,
                    horizon=horizon,
                    snapshots=selected,
                    transitions=transitions,
                    weighted_residual_rmse=_chronological_metric(
                        tuple(
                            snapshot.model.weighted_residual_rmse
                            for snapshot in selected
                        )
                    ),
                    pooled_weighted_ess=_chronological_metric(
                        tuple(
                            snapshot.model.pooled_weighted_ess for snapshot in selected
                        )
                    ),
                    fitted_row_count=_chronological_metric(
                        tuple(
                            float(snapshot.model.fitted_row_count)
                            for snapshot in selected
                        )
                    ),
                    per_symbol_weighted_ess=per_symbol_series,
                    overlap_weight_digest_unique_count=len(set(digests)),
                    overlap_weight_digest_transition_count=sum(
                        current != previous
                        for previous, current in zip(digests, digests[1:])
                    ),
                )
            )
    return tuple(result)


def _fit_prediction_distributions(
    scopes: tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...],
) -> tuple[CausalAlphaV3FitPredictionDistributions, ...]:
    fit_order = tuple(dict.fromkeys(scope.metric.fit_config_digest for scope in scopes))
    return tuple(
        CausalAlphaV3FitPredictionDistributions(
            fit_config_digest=fit_digest,
            distributions=_prediction_distributions(
                tuple(
                    scope
                    for scope in scopes
                    if scope.metric.fit_config_digest == fit_digest
                )
            ),
        )
        for fit_digest in fit_order
    )


def _episode_prediction_distributions(
    scopes: tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...],
) -> tuple[CausalAlphaV3EpisodePredictionDistributions, ...]:
    return tuple(
        CausalAlphaV3EpisodePredictionDistributions(
            fit_config_digest=fit_digest,
            contract_start=contract_start,
            contract_stop=contract_stop,
            episode_indices=tuple(
                sorted({scope.metric.episode_index for scope in cluster})
            ),
            distributions=_prediction_distributions(cluster),
        )
        for fit_digest, contract_start, contract_stop, cluster in _cluster_scopes(
            scopes
        )
    )


def _representative_model(
    model_series: tuple[CausalAlphaV3ModelSeries, ...],
    *,
    fit_digest: str,
    horizon: Literal["24h", "72h"],
    contract_start: int,
    contract_stop: int,
) -> CausalAlphaV3SignalDiagnosticModel:
    series = next(
        item
        for item in model_series
        if item.fit_config_digest == fit_digest and item.horizon == horizon
    )
    return next(
        snapshot.model
        for snapshot in series.snapshots
        if snapshot.contract_start == contract_start
        and snapshot.contract_stop == contract_stop
    )


def _chronological_horizon_series(
    scopes: tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...],
    model_series: tuple[CausalAlphaV3ModelSeries, ...],
) -> tuple[CausalAlphaV3ChronologicalHorizonSeries, ...]:
    fit_order = tuple(dict.fromkeys(scope.metric.fit_config_digest for scope in scopes))
    clusters = _cluster_scopes(scopes)
    result: list[CausalAlphaV3ChronologicalHorizonSeries] = []
    for fit_digest in fit_order:
        fit_clusters = tuple(
            cluster for cluster in clusters if cluster[0] == fit_digest
        )
        intervals = tuple((item[1], item[2]) for item in fit_clusters)
        for horizon in _HORIZONS:
            direction: list[float | None] = []
            rank: list[float | None] = []
            pearson: list[float | None] = []
            prediction_std: list[float | None] = []
            residual_rmse: list[float | None] = []
            pooled_ess: list[float | None] = []
            availability: list[float | None] = []
            for _, contract_start, contract_stop, cluster in fit_clusters:
                rows = tuple(
                    row
                    for scope in cluster
                    for row in _rows_for_horizon(scope, horizon)
                )
                diagnostics = _diagnostics_from_rows(rows)
                direction.append(diagnostics.direction_accuracy)
                rank.append(diagnostics.rank_correlation)
                pearson.append(diagnostics.pearson_correlation)
                prediction_values = _finite_values(
                    [
                        value
                        for scope in cluster
                        for value in _prediction_values_for_horizon(scope, horizon)
                    ],
                    field=f"{horizon} prediction chronology",
                )
                prediction_std.append(
                    float(np.std(prediction_values, dtype=np.float64))
                )
                availability_values = _finite_values(
                    [
                        row.available_feature_fraction
                        for scope in cluster
                        for row in scope.diagnostic.prediction_rows
                    ],
                    field="feature availability",
                )
                availability.append(
                    float(np.mean(availability_values, dtype=np.float64))
                )
                if horizon == "24h":
                    model_horizon: Literal["24h", "72h"] | None = "24h"
                elif horizon == "72h":
                    model_horizon = "72h"
                else:
                    model_horizon = None
                if model_horizon is None:
                    residual_rmse.append(None)
                    pooled_ess.append(None)
                else:
                    model = _representative_model(
                        model_series,
                        fit_digest=fit_digest,
                        horizon=model_horizon,
                        contract_start=contract_start,
                        contract_stop=contract_stop,
                    )
                    residual_rmse.append(model.weighted_residual_rmse)
                    pooled_ess.append(model.pooled_weighted_ess)
            result.append(
                CausalAlphaV3ChronologicalHorizonSeries(
                    fit_config_digest=fit_digest,
                    horizon=horizon,
                    episode_count=len(fit_clusters),
                    contract_intervals=intervals,
                    direction_accuracy=_chronological_metric(tuple(direction)),
                    rank_correlation=_chronological_metric(tuple(rank)),
                    pearson_correlation=_chronological_metric(tuple(pearson)),
                    prediction_standard_deviation=_chronological_metric(
                        tuple(prediction_std)
                    ),
                    weighted_residual_rmse=_chronological_metric(tuple(residual_rmse)),
                    pooled_weighted_ess=_chronological_metric(tuple(pooled_ess)),
                    mean_feature_availability=_chronological_metric(
                        tuple(availability)
                    ),
                )
            )
    return tuple(result)


def build_causal_alpha_v3_signal_forensics_v2_analysis(
    scopes: tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...],
) -> CausalAlphaV3SignalForensicsV2Analysis:
    """Build deterministic summaries from already-bound diagnostic evidence."""

    bound = tuple(scopes)
    if not bound:
        raise ValueError(
            "V3 signal forensics V2 sidecar analysis requires bound scopes"
        )
    snapshots = _deduplicated_snapshots(bound)
    model_series = _model_series(snapshots)
    return CausalAlphaV3SignalForensicsV2Analysis(
        scope_summaries=tuple(_scope_summary(scope) for scope in bound),
        model_series=model_series,
        fit_prediction_distributions=_fit_prediction_distributions(bound),
        episode_prediction_distributions=_episode_prediction_distributions(bound),
        chronological_horizon_series=_chronological_horizon_series(bound, model_series),
    )


__all__ = [
    "CausalAlphaV3SignalForensicsV2Analysis",
    "build_causal_alpha_v3_signal_forensics_v2_analysis",
]
