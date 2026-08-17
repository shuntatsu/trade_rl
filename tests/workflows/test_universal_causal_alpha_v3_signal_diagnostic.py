from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3Forecast,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3 import (
    build_causal_alpha_v3_symbol_balanced_weights,
    causal_alpha_v3_weight_digest,
    fit_causal_alpha_v3,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3Candidate
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticModel,
    CausalAlphaV3SignalDiagnosticPredictionRow,
    CausalAlphaV3SignalDiagnosticRealizedRow,
    CausalAlphaV3SignalDiagnosticScope,
    weighted_effective_sample_size,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    build_causal_alpha_v3_signal_scope,
    build_causal_alpha_v3_signal_scope_metric,
)


def _sha(token: str) -> str:
    return token * 64


def _samples(symbol: str, *, rows: int, offset: float) -> CausalAlphaSymbolSamples:
    decisions = np.arange(2, 2 + rows, dtype=np.int64)
    signal = decisions.astype(np.float64) + offset
    features = np.column_stack((signal, np.full(signal.shape, offset + 5.0)))
    return CausalAlphaSymbolSamples(
        symbol=symbol,
        dataset_id=content_digest(f"dataset:{symbol}"),
        feature_names=("signal", "descriptor"),
        feature_schema_digest=content_digest("feature-schema"),
        context_digest=content_digest(f"context:{symbol}"),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        labels_24h=0.01 * signal,
        label_end_indices_24h=decisions + 2,
        labels_72h=0.03 * signal,
        label_end_indices_72h=decisions + 4,
    )


def _paired_samples() -> CausalAlphaSymbolSamples:
    decisions = np.asarray(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16],
        dtype=np.int64,
    )
    signal = decisions.astype(np.float64)
    features = np.column_stack((signal, 0.5 * signal))
    available = np.ones_like(features, dtype=np.bool_)
    available[decisions.tolist().index(11), 1] = False
    return CausalAlphaSymbolSamples(
        symbol="AAAUSDT",
        dataset_id=_sha("d"),
        feature_names=("signal", "descriptor"),
        feature_schema_digest=content_digest("paired-feature-schema"),
        context_digest=content_digest("paired-context"),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=available,
        labels_24h=0.001 * signal,
        label_end_indices_24h=decisions + 1,
        labels_72h=0.003 * signal,
        label_end_indices_72h=decisions + 2,
    )


def _candidate() -> CausalAlphaV3Candidate:
    return CausalAlphaV3Candidate(
        name="diagnostic",
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.05),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            alpha_rebalance_decisions=2,
            strong_reversal_threshold=0.02,
            max_target_delta=0.05,
        ),
    )


def _contract() -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=_sha("d"),
        episode_index=0,
        start=10,
        stop=16,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )


def _model(*, suffix: str) -> CausalAlphaV3SignalDiagnosticModel:
    return CausalAlphaV3SignalDiagnosticModel(
        model_digest=_sha(suffix),
        feature_names=("signal", "descriptor"),
        intercept=0.001,
        coefficients=(0.1, -0.2),
        location=(1.0, 2.0),
        scale=(0.5, 1.5),
        constant_mask=(False, False),
        fitted_row_count=12,
        weighted_residual_rmse=0.02,
        pooled_weighted_ess=8.5,
        per_symbol_weighted_ess=(("AAAUSDT", 4.2), ("BBBUSDT", 4.3)),
        overlap_weight_digest=_sha("d"),
    )


def _prediction_row() -> CausalAlphaV3SignalDiagnosticPredictionRow:
    return CausalAlphaV3SignalDiagnosticPredictionRow(
        decision_index=100,
        actionable=True,
        available_feature_count=2,
        available_feature_fraction=1.0,
        prediction_24h=0.01,
        prediction_72h=0.03,
        prediction_72h_24h_equivalent=0.01,
        expected_return_24h_equivalent=0.01,
        uncertainty_24h_equivalent=0.02,
        signal_to_uncertainty=0.5,
    )


def _realized_row(
    *, decision_index: int = 100
) -> CausalAlphaV3SignalDiagnosticRealizedRow:
    return CausalAlphaV3SignalDiagnosticRealizedRow(
        decision_index=decision_index,
        label_end_index=104,
        available_feature_count=2,
        available_feature_fraction=1.0,
        prediction=0.01,
        realized_return=0.012,
        raw_prediction=0.03,
        raw_realized_return=0.036,
    )


def _forecast_digest(
    prediction_rows: tuple[CausalAlphaV3SignalDiagnosticPredictionRow, ...],
    model_24h: CausalAlphaV3SignalDiagnosticModel,
    model_72h: CausalAlphaV3SignalDiagnosticModel,
) -> str:
    return CausalAlphaV3Forecast(
        prediction_24h=np.asarray(
            [row.prediction_24h for row in prediction_rows], dtype=np.float64
        ),
        prediction_72h=np.asarray(
            [row.prediction_72h for row in prediction_rows], dtype=np.float64
        ),
        expected_return_24h_equivalent=np.asarray(
            [row.expected_return_24h_equivalent for row in prediction_rows],
            dtype=np.float64,
        ),
        uncertainty_24h_equivalent=np.asarray(
            [row.uncertainty_24h_equivalent for row in prediction_rows],
            dtype=np.float64,
        ),
        signal_to_uncertainty=np.asarray(
            [row.signal_to_uncertainty for row in prediction_rows], dtype=np.float64
        ),
        residual_rmse_24h=model_24h.weighted_residual_rmse,
        residual_rmse_72h=model_72h.weighted_residual_rmse,
    ).digest


def _scope() -> CausalAlphaV3SignalDiagnosticScope:
    model_24h = _model(suffix="8")
    model_72h = _model(suffix="9")
    prediction_rows = (_prediction_row(),)
    return CausalAlphaV3SignalDiagnosticScope(
        run_manifest_digest=_sha("1"),
        fit_config_digest=_sha("2"),
        symbol="AAAUSDT",
        episode_index=3,
        contract_start=100,
        contract_stop=110,
        contract_digest=_sha("3"),
        signal_metric_digest=_sha("4"),
        fit_digest=_sha("5"),
        forecast_digest=_forecast_digest(prediction_rows, model_24h, model_72h),
        feature_schema_digest=_sha("7"),
        model_24h=model_24h,
        model_72h=model_72h,
        prediction_rows=prediction_rows,
        realized_24h_rows=(_realized_row(),),
        realized_72h_rows=(_realized_row(),),
        realized_fused_rows=(_realized_row(),),
        canonical_cohort_indices=(100,),
        per_feature_available_fraction=(1.0, 1.0),
        complete_feature_row_count=1,
        incomplete_feature_row_count=0,
        available_feature_fraction_minimum=1.0,
        available_feature_fraction_mean=1.0,
        available_feature_fraction_maximum=1.0,
    )


def test_weighted_effective_sample_size_uses_positive_weight_concentration() -> None:
    weights = np.asarray([0.0, 0.1, 0.2, 0.7], dtype=np.float64)
    expected = float(np.square(weights.sum()) / np.square(weights).sum())

    assert weighted_effective_sample_size(weights) == pytest.approx(expected)


@pytest.mark.parametrize(
    "weights",
    (
        np.asarray([], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([1.0, -0.1], dtype=np.float64),
        np.asarray([1.0, np.nan], dtype=np.float64),
    ),
)
def test_weighted_effective_sample_size_rejects_invalid_weights(
    weights: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="weights"):
        weighted_effective_sample_size(weights)


def test_public_weight_digest_reproduces_fit_weight_identity() -> None:
    samples = {
        "AAAUSDT": _samples("AAAUSDT", rows=12, offset=0.0),
        "BBBUSDT": _samples("BBBUSDT", rows=9, offset=10.0),
    }
    fit = fit_causal_alpha_v3(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        samples=samples,
        knowledge_cutoff=12,
        config=CausalAlphaV3FitConfig(ridge_strength=0.1),
    )

    weights_24h = build_causal_alpha_v3_symbol_balanced_weights(
        train_symbols=fit.train_symbols,
        samples=samples,
        knowledge_cutoff=fit.knowledge_cutoff,
        horizon="24h",
    )
    weights_72h = build_causal_alpha_v3_symbol_balanced_weights(
        train_symbols=fit.train_symbols,
        samples=samples,
        knowledge_cutoff=fit.knowledge_cutoff,
        horizon="72h",
    )

    assert (
        causal_alpha_v3_weight_digest(
            fit.train_symbols,
            weights_24h,
            horizon="24h",
            knowledge_cutoff=fit.knowledge_cutoff,
        )
        == fit.weight_digest_24h
    )
    assert (
        causal_alpha_v3_weight_digest(
            fit.train_symbols,
            weights_72h,
            horizon="72h",
            knowledge_cutoff=fit.knowledge_cutoff,
        )
        == fit.weight_digest_72h
    )

    mutated = dict(weights_24h)
    changed = np.asarray(mutated["AAAUSDT"]).copy()
    positive = np.flatnonzero(changed > 0.0)
    changed[int(positive[0])] *= 1.01
    mutated["AAAUSDT"] = changed
    assert (
        causal_alpha_v3_weight_digest(
            fit.train_symbols,
            mutated,
            horizon="24h",
            knowledge_cutoff=fit.knowledge_cutoff,
        )
        != fit.weight_digest_24h
    )


def test_signal_diagnostic_scope_is_research_only_and_content_addressed() -> None:
    diagnostic = _scope()
    payload = diagnostic.to_payload()

    assert diagnostic.schema_version == "causal_alpha_v3_signal_diagnostic_scope_v1"
    assert diagnostic.research_only is True
    assert diagnostic.promotion_eligible is False
    assert payload["artifact_digest"] == diagnostic.digest
    assert diagnostic.model_24h.model_digest == _sha("8")
    assert diagnostic.model_72h.model_digest == _sha("9")
    assert diagnostic.canonical_cohort_indices == (100,)
    assert "features" not in payload
    assert "targets" not in payload


def test_signal_diagnostic_scope_rejects_promotion_and_feature_order_drift() -> None:
    diagnostic = _scope()

    with pytest.raises(ValueError, match="research-only"):
        replace(diagnostic, promotion_eligible=True, digest="")

    with pytest.raises(ValueError, match="feature"):
        replace(
            diagnostic,
            model_72h=replace(
                diagnostic.model_72h,
                feature_names=("descriptor", "signal"),
            ),
            digest="",
        )


def test_signal_diagnostic_rows_reject_non_finite_or_invalid_availability() -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(_prediction_row(), prediction_24h=float("nan"))

    with pytest.raises(ValueError, match="available"):
        replace(_prediction_row(), available_feature_count=-1)

    with pytest.raises(ValueError, match="available"):
        replace(_realized_row(), available_feature_fraction=1.1)


def test_signal_diagnostic_scope_rejects_row_availability_inconsistent_with_width() -> (
    None
):
    diagnostic = _scope()
    inconsistent = replace(
        _prediction_row(),
        available_feature_count=3,
        available_feature_fraction=1.0,
    )

    with pytest.raises(ValueError, match="available"):
        replace(diagnostic, prediction_rows=(inconsistent,), digest="")


def test_paired_signal_scope_preserves_canonical_metric_exactly() -> None:
    samples = {"AAAUSDT": _paired_samples()}
    kwargs = {
        "run_manifest_digest": _sha("a"),
        "symbol": "AAAUSDT",
        "train_symbols": ("AAAUSDT",),
        "samples": samples,
        "contract": _contract(),
        "candidate": _candidate(),
    }

    canonical = build_causal_alpha_v3_signal_scope_metric(**kwargs)
    paired = build_causal_alpha_v3_signal_scope(**kwargs)

    assert paired.metric.to_payload() == canonical.to_payload()


def test_paired_signal_scope_preserves_horizons_availability_and_all_realized_rows() -> (
    None
):
    paired = build_causal_alpha_v3_signal_scope(
        run_manifest_digest=_sha("a"),
        symbol="AAAUSDT",
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": _paired_samples()},
        contract=_contract(),
        candidate=_candidate(),
    )
    diagnostic = paired.diagnostic

    assert tuple(row.decision_index for row in diagnostic.prediction_rows) == (
        10,
        11,
        12,
        13,
        14,
    )
    assert tuple(row.actionable for row in diagnostic.prediction_rows) == (
        True,
        True,
        False,
        True,
        True,
    )
    assert tuple(row.available_feature_count for row in diagnostic.prediction_rows) == (
        2,
        1,
        0,
        2,
        2,
    )
    assert tuple(row.decision_index for row in diagnostic.realized_24h_rows) == (
        10,
        11,
        13,
        14,
    )
    assert tuple(row.decision_index for row in diagnostic.realized_72h_rows) == (
        10,
        11,
        13,
    )
    assert tuple(row.decision_index for row in diagnostic.realized_fused_rows) == (
        10,
        11,
        13,
    )
    assert (
        diagnostic.canonical_cohort_indices == paired.metric.cohort_indices == (10, 13)
    )
    assert diagnostic.per_feature_available_fraction == pytest.approx((0.8, 0.6))
    assert diagnostic.complete_feature_row_count == 3
    assert diagnostic.incomplete_feature_row_count == 2
    assert diagnostic.available_feature_fraction_minimum == pytest.approx(0.0)
    assert diagnostic.available_feature_fraction_mean == pytest.approx(0.7)
    assert diagnostic.available_feature_fraction_maximum == pytest.approx(1.0)
    assert all(
        row.label_end_index < _contract().stop for row in diagnostic.realized_72h_rows
    )
    assert diagnostic.model_24h.model_digest
    assert diagnostic.model_72h.model_digest
    assert diagnostic.model_24h.pooled_weighted_ess > 0.0
    assert diagnostic.model_72h.pooled_weighted_ess > 0.0
