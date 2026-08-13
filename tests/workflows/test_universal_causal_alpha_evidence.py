from __future__ import annotations

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import CausalAlphaRidgeConfig
from trade_rl.workflows.universal_causal_alpha_fitting import (
    causal_alpha_prediction_diagnostics,
    fit_expanding_causal_alpha_models,
)
from trade_rl.workflows.universal_causal_alpha_teacher import CausalAlphaSymbolSamples


def _samples(symbol: str, offset: float) -> CausalAlphaSymbolSamples:
    decisions = np.arange(2, 30, dtype=np.int64)
    signal = decisions.astype(np.float64) + offset
    features = np.column_stack((signal, np.full(signal.shape, offset + 5.0)))
    return CausalAlphaSymbolSamples(
        symbol=symbol,
        dataset_id=content_digest((symbol, "dataset")),
        feature_names=("signal", "descriptor"),
        feature_schema_digest=content_digest("feature-schema"),
        context_digest=content_digest((symbol, "context")),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        labels_24h=0.01 * signal + 0.1,
        label_end_indices_24h=decisions + 2,
        labels_72h=-0.02 * signal + 0.5,
        label_end_indices_72h=decisions + 4,
    )


def test_prediction_diagnostics_persist_correlation_direction_and_distribution() -> (
    None
):
    evidence = causal_alpha_prediction_diagnostics(
        np.asarray([-0.3, -0.1, 0.2, 0.5], dtype=np.float64),
        np.asarray([-0.2, 0.1, 0.4, 0.6], dtype=np.float64),
    )

    assert evidence.sample_count == 4
    assert evidence.pearson_correlation is not None
    assert evidence.pearson_correlation > 0.8
    assert evidence.directional_accuracy == pytest.approx(0.75)
    payload = evidence.to_payload()
    assert payload["prediction_quantiles"]["p00"] == pytest.approx(-0.3)
    assert payload["prediction_quantiles"]["p50"] == pytest.approx(0.05)
    assert payload["prediction_quantiles"]["p100"] == pytest.approx(0.5)
    assert len(payload["artifact_digest"]) == 64


def test_prediction_diagnostics_use_only_fully_realized_labels() -> None:
    evidence = causal_alpha_prediction_diagnostics(
        np.asarray([-0.3, -0.1, 0.2, 0.5], dtype=np.float64),
        np.asarray([-0.2, np.nan, 0.4, np.nan], dtype=np.float64),
    )

    assert evidence.sample_count == 2
    assert evidence.pearson_correlation == pytest.approx(1.0)
    assert evidence.directional_accuracy == pytest.approx(1.0)
    payload = evidence.to_payload()
    assert payload["prediction_quantiles"]["p00"] == pytest.approx(-0.3)
    assert payload["prediction_quantiles"]["p100"] == pytest.approx(0.2)


def test_constant_prediction_correlation_is_explicitly_unavailable() -> None:
    evidence = causal_alpha_prediction_diagnostics(
        np.ones(4, dtype=np.float64),
        np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float64),
    )
    assert evidence.pearson_correlation is None
    assert evidence.directional_accuracy == pytest.approx(0.5)


def test_expanding_fit_payload_contains_compact_models_and_cutoffs() -> None:
    fitted = fit_expanding_causal_alpha_models(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        samples={
            "AAAUSDT": _samples("AAAUSDT", 0.0),
            "BBBUSDT": _samples("BBBUSDT", 10.0),
        },
        knowledge_cutoff=16,
        ridge_config=CausalAlphaRidgeConfig(ridge_strength=0.1),
    )

    payload = fitted.to_payload()
    assert payload["knowledge_cutoff"] == 16
    assert payload["sample_count_24h"] == fitted.sample_count_24h
    assert payload["sample_count_72h"] == fitted.sample_count_72h
    assert payload["max_label_end_24h"] < 16
    assert payload["max_label_end_72h"] < 16
    for key in ("model_24h", "model_72h"):
        model = payload[key]
        assert "coefficients" in model
        assert "location" in model
        assert "scale" in model
        assert "constant_mask" in model
        assert "eligible_indices" not in model
        assert len(model["artifact_digest"]) == 64
