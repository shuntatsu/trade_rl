from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import CausalAlphaV3FitConfig
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3 import (
    build_causal_alpha_v3_symbol_balanced_weights,
    causal_alpha_v3_weight_digest,
    fit_causal_alpha_v3,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticModel,
    CausalAlphaV3SignalDiagnosticPredictionRow,
    CausalAlphaV3SignalDiagnosticRealizedRow,
    CausalAlphaV3SignalDiagnosticScope,
    weighted_effective_sample_size,
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


def _realized_row(*, decision_index: int = 100) -> CausalAlphaV3SignalDiagnosticRealizedRow:
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


def _scope() -> CausalAlphaV3SignalDiagnosticScope:
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
        forecast_digest=_sha("6"),
        feature_schema_digest=_sha("7"),
        model_24h=_model(suffix="8"),
        model_72h=_model(suffix="9"),
        prediction_rows=(_prediction_row(),),
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
def test_weighted_effective_sample_size_rejects_invalid_weights(weights: np.ndarray) -> None:
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

    assert causal_alpha_v3_weight_digest(
        fit.train_symbols,
        weights_24h,
        horizon="24h",
        knowledge_cutoff=fit.knowledge_cutoff,
    ) == fit.weight_digest_24h
    assert causal_alpha_v3_weight_digest(
        fit.train_symbols,
        weights_72h,
        horizon="72h",
        knowledge_cutoff=fit.knowledge_cutoff,
    ) == fit.weight_digest_72h

    mutated = dict(weights_24h)
    changed = np.asarray(mutated["AAAUSDT"]).copy()
    positive = np.flatnonzero(changed > 0.0)
    changed[int(positive[0])] *= 1.01
    mutated["AAAUSDT"] = changed
    assert causal_alpha_v3_weight_digest(
        fit.train_symbols,
        mutated,
        horizon="24h",
        knowledge_cutoff=fit.knowledge_cutoff,
    ) != fit.weight_digest_24h


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
        replace(_prediction_row(), available_feature_count=3)

    with pytest.raises(ValueError, match="available"):
        replace(_realized_row(), available_feature_fraction=1.1)
