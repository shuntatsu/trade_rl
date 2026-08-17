from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.workflows.test_universal_causal_alpha_v3_signal_forensics import (
    _write_json,
)
from tests.workflows.test_universal_causal_alpha_v3_signal_forensics_v2_sidecars import (
    _complete_sidecars,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics import (
    load_causal_alpha_v3_signal_forensics,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_loader import (
    load_causal_alpha_v3_signal_forensics_v2_sidecars,
)


def _analysis_api() -> Any:
    return importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_analysis"
    )


def _v2_api() -> Any:
    return importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2"
    )


def _distribution(scope: Any, name: str) -> Any:
    return next(item for item in scope.prediction_distributions if item.name == name)


def _availability_horizon(scope: Any, horizon: str) -> Any:
    return next(item for item in scope.availability.horizons if item.horizon == horizon)


def test_v2_complete_report_preserves_v1_and_exposes_sidecar_analysis(
    tmp_path: Path,
) -> None:
    _complete_sidecars(tmp_path)
    base = load_causal_alpha_v3_signal_forensics(tmp_path)

    report = _v2_api().load_causal_alpha_v3_signal_forensics_v2(tmp_path)

    assert report.sidecar_mode == "sidecar_complete"
    assert report.base_forensics_digest == base.digest
    assert report.base_forensics.to_payload() == base.to_payload()
    assert report.sidecar_analysis is not None
    assert report.sidecar_analysis.overlapping_realized_rows_are_descriptive is True
    assert report.research_only is True
    assert report.promotion_eligible is False
    assert {
        item.analysis for item in report.unavailable_analyses
    } == {
        "canonical_ridge_model_digest_reconstruction",
        "market_regime_classification",
        "overlapping_row_independent_confidence",
        "row_feature_error_attribution",
    }


def test_v2_scope_horizon_diagnostics_reuse_existing_oracle(tmp_path: Path) -> None:
    _complete_sidecars(tmp_path)
    bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
    analysis = _analysis_api().build_causal_alpha_v3_signal_forensics_v2_analysis(bound)
    first_bound = bound[0]
    first = analysis.scope_summaries[0]

    expected_24h = evaluate_causal_alpha_signal_diagnostics(
        np.asarray(
            [row.prediction for row in first_bound.diagnostic.realized_24h_rows],
            dtype=np.float64,
        ),
        np.asarray(
            [row.realized_return for row in first_bound.diagnostic.realized_24h_rows],
            dtype=np.float64,
        ),
    )
    expected_72h = evaluate_causal_alpha_signal_diagnostics(
        np.asarray(
            [row.prediction for row in first_bound.diagnostic.realized_72h_rows],
            dtype=np.float64,
        ),
        np.asarray(
            [row.realized_return for row in first_bound.diagnostic.realized_72h_rows],
            dtype=np.float64,
        ),
    )
    expected_fused = evaluate_causal_alpha_signal_diagnostics(
        np.asarray(
            [row.prediction for row in first_bound.diagnostic.realized_fused_rows],
            dtype=np.float64,
        ),
        np.asarray(
            [row.realized_return for row in first_bound.diagnostic.realized_fused_rows],
            dtype=np.float64,
        ),
    )

    assert first.horizon_24h.to_payload() == expected_24h.to_payload()
    assert first.horizon_72h.to_payload() == expected_72h.to_payload()
    assert first.horizon_fused.to_payload() == expected_fused.to_payload()
    assert first.paired_24h_72h.sample_count == len(
        first_bound.diagnostic.realized_24h_rows
    )
    assert first.paired_24h_72h.decision_indices == tuple(
        row.decision_index for row in first_bound.diagnostic.realized_24h_rows
    )
    assert first.paired_24h_72h.diagnostics_24h.to_payload() == expected_24h.to_payload()
    assert first.paired_24h_72h.diagnostics_72h.to_payload() == expected_72h.to_payload()


def test_v2_prediction_distributions_use_fixed_quantiles(tmp_path: Path) -> None:
    _complete_sidecars(tmp_path)
    bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
    analysis = _analysis_api().build_causal_alpha_v3_signal_forensics_v2_analysis(bound)
    first_bound = bound[0]
    first = analysis.scope_summaries[0]
    summary = _distribution(first, "prediction_24h")
    values = np.asarray(
        [row.prediction_24h for row in first_bound.diagnostic.prediction_rows],
        dtype=np.float64,
    )

    assert summary.count == values.size
    assert summary.mean == pytest.approx(float(np.mean(values)))
    assert summary.standard_deviation == pytest.approx(float(np.std(values)))
    assert summary.minimum == pytest.approx(float(np.min(values)))
    assert summary.maximum == pytest.approx(float(np.max(values)))
    assert tuple(item.quantile for item in summary.quantiles) == (
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        1.0,
    )
    assert tuple(item.value for item in summary.quantiles) == pytest.approx(
        tuple(
            float(value)
            for value in np.quantile(
                values,
                np.asarray((0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)),
            )
        )
    )


def test_v2_deduplicates_pooled_model_evidence_and_computes_stability(
    tmp_path: Path,
) -> None:
    _complete_sidecars(tmp_path)
    bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
    analysis = _analysis_api().build_causal_alpha_v3_signal_forensics_v2_analysis(bound)

    assert len(analysis.model_series) == 4
    assert sum(len(series.snapshots) for series in analysis.model_series) == 16
    first_series = analysis.model_series[0]
    assert len(first_series.snapshots) == 4
    assert len(first_series.transitions) == 3
    first_transition = first_series.transitions[0]
    first_model = first_series.snapshots[0].model
    second_model = first_series.snapshots[1].model
    a = np.asarray(first_model.coefficients, dtype=np.float64)
    b = np.asarray(second_model.coefficients, dtype=np.float64)
    mu0 = np.asarray(first_model.location, dtype=np.float64)
    mu1 = np.asarray(second_model.location, dtype=np.float64)
    scale0 = np.asarray(first_model.scale, dtype=np.float64)
    scale1 = np.asarray(second_model.scale, dtype=np.float64)

    expected_cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    expected_location = float(np.sqrt(np.mean(np.square((mu1 - mu0) / scale0))))
    expected_scale = float(np.sqrt(np.mean(np.square(np.log(scale1 / scale0)))))
    assert first_transition.coefficient_cosine_similarity == pytest.approx(
        expected_cosine
    )
    assert first_transition.active_paired_coefficient_count == 2
    assert first_transition.coefficient_sign_flip_rate == pytest.approx(0.0)
    assert first_transition.location_shift_rms == pytest.approx(expected_location)
    assert first_transition.log_scale_ratio_rms == pytest.approx(expected_scale)
    assert first_series.overlap_weight_digest_unique_count == 4
    assert first_series.overlap_weight_digest_transition_count == 3


def test_v2_rejects_cross_symbol_pooled_model_disagreement(tmp_path: Path) -> None:
    built = _complete_sidecars(tmp_path)
    target = next(
        path
        for (fit, symbol, episode), path in built["diagnostic_paths"].items()
        if symbol == "ETHUSDT" and episode == 0 and fit == built["fit_configs"][0]
    )
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["model_24h"]["coefficients"][0] += 0.25
    body = {key: value for key, value in raw.items() if key != "artifact_digest"}
    _write_json(target, {**body, "artifact_digest": content_digest(body)})
    bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)

    with pytest.raises(ValueError, match="pooled model"):
        _analysis_api().build_causal_alpha_v3_signal_forensics_v2_analysis(bound)


def test_v2_availability_partitions_use_existing_diagnostic_semantics(
    tmp_path: Path,
) -> None:
    _complete_sidecars(tmp_path)
    bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
    analysis = _analysis_api().build_causal_alpha_v3_signal_forensics_v2_analysis(bound)
    first_bound = bound[0]
    first = analysis.scope_summaries[0]
    availability = _availability_horizon(first, "24h")
    complete_rows = tuple(
        row
        for row in first_bound.diagnostic.realized_24h_rows
        if row.available_feature_fraction == 1.0
    )
    incomplete_rows = tuple(
        row
        for row in first_bound.diagnostic.realized_24h_rows
        if row.available_feature_fraction < 1.0
    )

    assert availability.complete.row_count == len(complete_rows)
    assert availability.incomplete.row_count == len(incomplete_rows)
    assert availability.complete.diagnostics is not None
    assert availability.incomplete.diagnostics is not None
    assert availability.complete.unavailable_reason is None
    assert availability.incomplete.unavailable_reason is None
    assert first.availability.per_feature_available_fraction == (
        first_bound.diagnostic.per_feature_available_fraction
    )


def test_v2_chronological_summaries_use_authored_episode_order(tmp_path: Path) -> None:
    _complete_sidecars(tmp_path)
    bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
    analysis = _analysis_api().build_causal_alpha_v3_signal_forensics_v2_analysis(bound)
    first_series = next(
        item for item in analysis.chronological_horizon_series if item.horizon == "24h"
    )

    assert first_series.episode_count == 4
    assert first_series.direction_accuracy.count == 4
    values = np.asarray(first_series.direction_accuracy.values, dtype=np.float64)
    x = np.arange(values.size, dtype=np.float64)
    expected_slope = float(
        np.sum((x - np.mean(x)) * (values - np.mean(values)))
        / np.sum(np.square(x - np.mean(x)))
    )
    assert first_series.direction_accuracy.slope == pytest.approx(expected_slope)
    assert math.isfinite(first_series.direction_accuracy.early_mean)
    assert math.isfinite(first_series.direction_accuracy.late_mean)
    assert len(analysis.fit_prediction_distributions) == 2
    assert len(analysis.episode_prediction_distributions) == 8
