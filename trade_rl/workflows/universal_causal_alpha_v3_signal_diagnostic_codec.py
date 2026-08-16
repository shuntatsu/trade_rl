"""Strict JSON decoding for causal alpha V3 Signal diagnostic sidecars."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticModel,
    CausalAlphaV3SignalDiagnosticPredictionRow,
    CausalAlphaV3SignalDiagnosticRealizedRow,
    CausalAlphaV3SignalDiagnosticScope,
)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise ValueError(f"{field} keys must be strings")
    return result


def _require_fields(
    raw: Mapping[str, object], expected: set[str], *, field: str
) -> None:
    observed = set(raw)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ValueError(
            f"{field} fields mismatch; missing={missing}, unknown={unknown}"
        )


def _sequence(value: object, *, field: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return tuple(value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _optional_number(value: object, *, field: str) -> float | None:
    return None if value is None else _number(value, field=field)


def _prediction_row_from_payload(
    payload: object,
) -> CausalAlphaV3SignalDiagnosticPredictionRow:
    raw = _mapping(payload, field="V3 diagnostic prediction row")
    expected = {
        "actionable",
        "available_feature_count",
        "available_feature_fraction",
        "decision_index",
        "expected_return_24h_equivalent",
        "prediction_24h",
        "prediction_72h",
        "prediction_72h_24h_equivalent",
        "signal_to_uncertainty",
        "uncertainty_24h_equivalent",
    }
    _require_fields(raw, expected, field="V3 diagnostic prediction row")
    actionable = raw["actionable"]
    if not isinstance(actionable, bool):
        raise ValueError("V3 diagnostic prediction actionable must be boolean")
    return CausalAlphaV3SignalDiagnosticPredictionRow(
        decision_index=_integer(
            raw["decision_index"], field="V3 diagnostic prediction decision_index"
        ),
        actionable=actionable,
        available_feature_count=_integer(
            raw["available_feature_count"],
            field="V3 diagnostic prediction available_feature_count",
        ),
        available_feature_fraction=_number(
            raw["available_feature_fraction"],
            field="V3 diagnostic prediction available_feature_fraction",
        ),
        prediction_24h=_number(
            raw["prediction_24h"], field="V3 diagnostic prediction_24h"
        ),
        prediction_72h=_number(
            raw["prediction_72h"], field="V3 diagnostic prediction_72h"
        ),
        prediction_72h_24h_equivalent=_number(
            raw["prediction_72h_24h_equivalent"],
            field="V3 diagnostic prediction_72h_24h_equivalent",
        ),
        expected_return_24h_equivalent=_number(
            raw["expected_return_24h_equivalent"],
            field="V3 diagnostic expected_return_24h_equivalent",
        ),
        uncertainty_24h_equivalent=_number(
            raw["uncertainty_24h_equivalent"],
            field="V3 diagnostic uncertainty_24h_equivalent",
        ),
        signal_to_uncertainty=_number(
            raw["signal_to_uncertainty"],
            field="V3 diagnostic signal_to_uncertainty",
        ),
    )


def _realized_row_from_payload(
    payload: object,
) -> CausalAlphaV3SignalDiagnosticRealizedRow:
    raw = _mapping(payload, field="V3 diagnostic realized row")
    expected = {
        "available_feature_count",
        "available_feature_fraction",
        "decision_index",
        "label_end_index",
        "prediction",
        "raw_prediction",
        "raw_realized_return",
        "realized_return",
    }
    _require_fields(raw, expected, field="V3 diagnostic realized row")
    return CausalAlphaV3SignalDiagnosticRealizedRow(
        decision_index=_integer(
            raw["decision_index"], field="V3 diagnostic realized decision_index"
        ),
        label_end_index=_integer(
            raw["label_end_index"], field="V3 diagnostic realized label_end_index"
        ),
        available_feature_count=_integer(
            raw["available_feature_count"],
            field="V3 diagnostic realized available_feature_count",
        ),
        available_feature_fraction=_number(
            raw["available_feature_fraction"],
            field="V3 diagnostic realized available_feature_fraction",
        ),
        prediction=_number(
            raw["prediction"], field="V3 diagnostic realized prediction"
        ),
        realized_return=_number(
            raw["realized_return"], field="V3 diagnostic realized return"
        ),
        raw_prediction=_optional_number(
            raw["raw_prediction"], field="V3 diagnostic realized raw_prediction"
        ),
        raw_realized_return=_optional_number(
            raw["raw_realized_return"],
            field="V3 diagnostic realized raw_realized_return",
        ),
    )


def _model_from_payload(payload: object) -> CausalAlphaV3SignalDiagnosticModel:
    raw = _mapping(payload, field="V3 diagnostic model")
    expected = {
        "coefficients",
        "constant_mask",
        "feature_names",
        "fitted_row_count",
        "intercept",
        "location",
        "model_digest",
        "overlap_weight_digest",
        "per_symbol_weighted_ess",
        "pooled_weighted_ess",
        "scale",
        "weighted_residual_rmse",
    }
    _require_fields(raw, expected, field="V3 diagnostic model")
    feature_names = tuple(
        _string(item, field="V3 diagnostic model feature name")
        for item in _sequence(raw["feature_names"], field="V3 diagnostic feature_names")
    )
    coefficients = tuple(
        _number(item, field="V3 diagnostic model coefficient")
        for item in _sequence(raw["coefficients"], field="V3 diagnostic coefficients")
    )
    location = tuple(
        _number(item, field="V3 diagnostic model location")
        for item in _sequence(raw["location"], field="V3 diagnostic location")
    )
    scale = tuple(
        _number(item, field="V3 diagnostic model scale")
        for item in _sequence(raw["scale"], field="V3 diagnostic scale")
    )
    constant_mask_values = _sequence(
        raw["constant_mask"], field="V3 diagnostic constant_mask"
    )
    if any(not isinstance(item, bool) for item in constant_mask_values):
        raise ValueError("V3 diagnostic constant_mask values must be boolean")
    per_symbol_items: list[tuple[str, float]] = []
    for item in _sequence(
        raw["per_symbol_weighted_ess"],
        field="V3 diagnostic per_symbol_weighted_ess",
    ):
        pair = _sequence(item, field="V3 diagnostic per-symbol ESS item")
        if len(pair) != 2:
            raise ValueError("V3 diagnostic per-symbol ESS item must contain two values")
        per_symbol_items.append(
            (
                _string(pair[0], field="V3 diagnostic ESS symbol"),
                _number(pair[1], field="V3 diagnostic ESS value"),
            )
        )
    return CausalAlphaV3SignalDiagnosticModel(
        model_digest=_string(
            raw["model_digest"], field="V3 diagnostic model digest"
        ),
        feature_names=feature_names,
        intercept=_number(raw["intercept"], field="V3 diagnostic model intercept"),
        coefficients=coefficients,
        location=location,
        scale=scale,
        constant_mask=tuple(constant_mask_values),
        fitted_row_count=_integer(
            raw["fitted_row_count"], field="V3 diagnostic fitted_row_count"
        ),
        weighted_residual_rmse=_number(
            raw["weighted_residual_rmse"],
            field="V3 diagnostic weighted_residual_rmse",
        ),
        pooled_weighted_ess=_number(
            raw["pooled_weighted_ess"], field="V3 diagnostic pooled_weighted_ess"
        ),
        per_symbol_weighted_ess=tuple(per_symbol_items),
        overlap_weight_digest=_string(
            raw["overlap_weight_digest"],
            field="V3 diagnostic overlap_weight_digest",
        ),
    )


def signal_diagnostic_scope_from_payload(
    payload: object,
) -> CausalAlphaV3SignalDiagnosticScope:
    """Decode one diagnostic sidecar without accepting schema/type drift."""

    raw = _mapping(payload, field="V3 diagnostic scope")
    expected = {
        "artifact_digest",
        "available_feature_fraction_maximum",
        "available_feature_fraction_mean",
        "available_feature_fraction_minimum",
        "canonical_cohort_indices",
        "complete_feature_row_count",
        "contract_digest",
        "contract_start",
        "contract_stop",
        "episode_index",
        "feature_schema_digest",
        "fit_config_digest",
        "fit_digest",
        "forecast_digest",
        "incomplete_feature_row_count",
        "model_24h",
        "model_72h",
        "per_feature_available_fraction",
        "prediction_rows",
        "promotion_eligible",
        "realized_24h_rows",
        "realized_72h_rows",
        "realized_fused_rows",
        "research_only",
        "run_manifest_digest",
        "schema_version",
        "signal_metric_digest",
        "symbol",
    }
    _require_fields(raw, expected, field="V3 diagnostic scope")
    research_only = raw["research_only"]
    promotion_eligible = raw["promotion_eligible"]
    if not isinstance(research_only, bool) or not isinstance(promotion_eligible, bool):
        raise ValueError("V3 diagnostic safety flags must be boolean")
    return CausalAlphaV3SignalDiagnosticScope(
        run_manifest_digest=_string(
            raw["run_manifest_digest"], field="V3 diagnostic run_manifest_digest"
        ),
        fit_config_digest=_string(
            raw["fit_config_digest"], field="V3 diagnostic fit_config_digest"
        ),
        symbol=_string(raw["symbol"], field="V3 diagnostic symbol"),
        episode_index=_integer(
            raw["episode_index"], field="V3 diagnostic episode_index"
        ),
        contract_start=_integer(
            raw["contract_start"], field="V3 diagnostic contract_start"
        ),
        contract_stop=_integer(
            raw["contract_stop"], field="V3 diagnostic contract_stop"
        ),
        contract_digest=_string(
            raw["contract_digest"], field="V3 diagnostic contract_digest"
        ),
        signal_metric_digest=_string(
            raw["signal_metric_digest"], field="V3 diagnostic signal_metric_digest"
        ),
        fit_digest=_string(raw["fit_digest"], field="V3 diagnostic fit_digest"),
        forecast_digest=_string(
            raw["forecast_digest"], field="V3 diagnostic forecast_digest"
        ),
        feature_schema_digest=_string(
            raw["feature_schema_digest"],
            field="V3 diagnostic feature_schema_digest",
        ),
        model_24h=_model_from_payload(raw["model_24h"]),
        model_72h=_model_from_payload(raw["model_72h"]),
        prediction_rows=tuple(
            _prediction_row_from_payload(item)
            for item in _sequence(
                raw["prediction_rows"], field="V3 diagnostic prediction_rows"
            )
        ),
        realized_24h_rows=tuple(
            _realized_row_from_payload(item)
            for item in _sequence(
                raw["realized_24h_rows"], field="V3 diagnostic realized_24h_rows"
            )
        ),
        realized_72h_rows=tuple(
            _realized_row_from_payload(item)
            for item in _sequence(
                raw["realized_72h_rows"], field="V3 diagnostic realized_72h_rows"
            )
        ),
        realized_fused_rows=tuple(
            _realized_row_from_payload(item)
            for item in _sequence(
                raw["realized_fused_rows"],
                field="V3 diagnostic realized_fused_rows",
            )
        ),
        canonical_cohort_indices=tuple(
            _integer(item, field="V3 diagnostic cohort index")
            for item in _sequence(
                raw["canonical_cohort_indices"],
                field="V3 diagnostic canonical_cohort_indices",
            )
        ),
        per_feature_available_fraction=tuple(
            _number(item, field="V3 diagnostic per-feature availability")
            for item in _sequence(
                raw["per_feature_available_fraction"],
                field="V3 diagnostic per_feature_available_fraction",
            )
        ),
        complete_feature_row_count=_integer(
            raw["complete_feature_row_count"],
            field="V3 diagnostic complete_feature_row_count",
        ),
        incomplete_feature_row_count=_integer(
            raw["incomplete_feature_row_count"],
            field="V3 diagnostic incomplete_feature_row_count",
        ),
        available_feature_fraction_minimum=_number(
            raw["available_feature_fraction_minimum"],
            field="V3 diagnostic available_feature_fraction_minimum",
        ),
        available_feature_fraction_mean=_number(
            raw["available_feature_fraction_mean"],
            field="V3 diagnostic available_feature_fraction_mean",
        ),
        available_feature_fraction_maximum=_number(
            raw["available_feature_fraction_maximum"],
            field="V3 diagnostic available_feature_fraction_maximum",
        ),
        schema_version=_string(
            raw["schema_version"], field="V3 diagnostic schema_version"
        ),
        research_only=research_only,
        promotion_eligible=promotion_eligible,
        digest=_string(raw["artifact_digest"], field="V3 diagnostic artifact_digest"),
    )


__all__ = ["signal_diagnostic_scope_from_payload"]
