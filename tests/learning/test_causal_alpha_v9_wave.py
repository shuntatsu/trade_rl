from __future__ import annotations

import numpy as np

from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Config
from trade_rl.learning.causal_alpha_v9_wave import (
    CausalAlphaV9TrainingRows,
    causal_alpha_v9_wave_target_path,
    fit_causal_alpha_v9_wave,
)


def _rows(symbol: str, phase: float) -> CausalAlphaV9TrainingRows:
    decisions = np.arange(0, 320, 16, dtype=np.int64)
    x = np.arange(len(decisions), dtype=np.float64)
    features = np.column_stack(
        (
            np.sin(0.2 * x + phase),
            np.cos(0.1 * x + phase),
            np.full_like(x, phase),
        )
    )
    labels = 0.02 * np.sin(0.2 * x + phase) - 0.01 * features[:, 1] ** 2
    return CausalAlphaV9TrainingRows(
        symbol=symbol,
        decision_indices=decisions,
        label_end_indices=decisions + 16,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        labels=labels,
        feature_names=("return_4h", "return_24h", "liquidity"),
    )


def test_v9_fit_is_deterministic_causal_and_symbol_free() -> None:
    rows = {"A": _rows("A", 0.1), "B": _rows("B", 0.7)}
    config = CausalAlphaV9Config()

    first = fit_causal_alpha_v9_wave(rows, knowledge_cutoff=272, config=config)
    second = fit_causal_alpha_v9_wave(rows, knowledge_cutoff=272, config=config)

    assert first.digest == second.digest
    assert first.maximum_label_end_index < 272
    assert all("symbol" not in name.lower() for name in first.feature_names)
    predictions = first.predict_heads(rows["A"].features[-3:])
    assert predictions.shape == (3, 3)
    np.testing.assert_allclose(predictions, second.predict_heads(rows["A"].features[-3:]))


def test_v9_fit_rejects_symbol_identity_features() -> None:
    rows = _rows("A", 0.1)

    try:
        CausalAlphaV9TrainingRows(
            symbol=rows.symbol,
            decision_indices=rows.decision_indices,
            label_end_indices=rows.label_end_indices,
            features=rows.features,
            feature_available=rows.feature_available,
            labels=rows.labels,
            feature_names=("return_4h", "symbol_id", "liquidity"),
        )
    except ValueError as error:
        assert "symbol identity" in str(error)
    else:
        raise AssertionError("symbol identity feature was accepted")


def test_v9_wave_holds_neutral_and_exits_before_reversal() -> None:
    rows = 113
    heads = np.zeros((3, rows), dtype=np.float64)
    for index in (0, 16):
        heads[:, index] = 0.01
    for index in (48, 64, 80, 96):
        heads[:, index] = -0.01

    path = causal_alpha_v9_wave_target_path(
        decision_indices=np.arange(rows),
        head_predictions=heads,
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=np.full(rows, 0.25),
        risk_weight_caps=np.full(rows, 0.25),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        config=CausalAlphaV9Config(),
        initial_weight=0.0,
    )

    assert path.targets[16] == 0.025
    assert path.targets[32] == 0.025
    assert path.targets[64] == 0.0
    assert path.targets[80] == 0.0
    assert path.targets[96] == -0.025
    assert path.sign_flip_count == 0


def test_v9_inherited_position_must_earn_continuation() -> None:
    rows = 33
    path = causal_alpha_v9_wave_target_path(
        decision_indices=np.arange(rows),
        head_predictions=np.zeros((3, rows)),
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=np.full(rows, 0.25),
        risk_weight_caps=np.full(rows, 0.25),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="b" * 64,
        config=CausalAlphaV9Config(),
        initial_weight=0.25,
    )

    assert path.targets[0] == 0.025
    assert path.targets[15] == 0.025
    assert path.targets[16] == 0.0
