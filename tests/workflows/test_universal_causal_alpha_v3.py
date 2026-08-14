from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import CausalAlphaV3FitConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_v3 import (
    build_causal_alpha_v3_symbol_balanced_weights,
    fit_causal_alpha_v3,
)


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


def test_v3_overlap_weights_balance_total_eligible_weight_across_symbols() -> None:
    samples = {
        "AAAUSDT": _samples("AAAUSDT", rows=12, offset=0.0),
        "BBBUSDT": _samples("BBBUSDT", rows=7, offset=10.0),
    }

    weights = build_causal_alpha_v3_symbol_balanced_weights(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        samples=samples,
        knowledge_cutoff=12,
        horizon="24h",
    )

    assert set(weights) == set(samples)
    assert weights["AAAUSDT"].shape == (12,)
    assert weights["BBBUSDT"].shape == (7,)
    assert weights["AAAUSDT"].sum() == pytest.approx(weights["BBBUSDT"].sum())
    assert weights["AAAUSDT"][-1] == 0.0


def test_v3_fit_cannot_be_changed_by_labels_unrealized_at_cutoff() -> None:
    original = _samples("AAAUSDT", rows=12, offset=0.0)
    changed_labels = np.asarray(original.labels_24h).copy()
    changed_labels[-1] = 1_000.0
    changed = replace(original, labels_24h=changed_labels, digest="")
    config = CausalAlphaV3FitConfig(ridge_strength=0.1)

    first = fit_causal_alpha_v3(
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": original},
        knowledge_cutoff=12,
        config=config,
    )
    second = fit_causal_alpha_v3(
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": changed},
        knowledge_cutoff=12,
        config=config,
    )

    assert second.model_24h.coefficients == pytest.approx(first.model_24h.coefficients)
    assert second.model_24h.intercept == pytest.approx(first.model_24h.intercept)


def test_v3_fit_predicts_a_24h_equivalent_forecast_bundle() -> None:
    block = _samples("AAAUSDT", rows=16, offset=0.0)
    fitted = fit_causal_alpha_v3(
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": block},
        knowledge_cutoff=14,
        config=CausalAlphaV3FitConfig(ridge_strength=0.1),
    )

    forecast = fitted.predict(
        block.features[:2],
        feature_available=block.feature_available[:2],
    )

    assert forecast.expected_return_24h_equivalent.shape == (2,)
    assert forecast.uncertainty_24h_equivalent.shape == (2,)
    assert fitted.residual_rmse_24h >= 0.0
    assert fitted.residual_rmse_72h >= 0.0
    assert len(fitted.digest) == 64
