from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config
from trade_rl.learning.causal_alpha_v10_fit import (
    CausalAlphaV10TrainingRows,
    fit_causal_alpha_v10,
)


def _rows(symbol: str, phase: float) -> CausalAlphaV10TrainingRows:
    decisions = np.arange(0, 10_000, 16, dtype=np.int64)
    x = decisions.astype(np.float64) / 16.0
    features = np.column_stack(
        (
            np.sin(0.031 * x + phase),
            np.cos(0.017 * x + phase),
            np.tanh(0.009 * x - phase),
        )
    )
    fast = 0.01 * features[:, 0] - 0.003 * features[:, 1] ** 2
    slow = 0.04 * features[:, 2] + 0.01 * features[:, 0]
    return CausalAlphaV10TrainingRows(
        symbol=symbol,
        decision_indices=decisions,
        fast_label_end_indices=decisions + 16,
        slow_label_end_indices=decisions + 288,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        fast_labels=fast,
        slow_labels=slow,
        feature_names=("return_4h", "return_24h", "liquidity"),
    )


def test_v10_dual_fit_is_deterministic_causal_and_non_overlapping() -> None:
    rows = {
        "A": _rows("A", 0.1),
        "B": _rows("B", 0.7),
        "C": _rows("C", 1.1),
    }
    config = CausalAlphaV10Config()
    cutoff = 8_992

    first = fit_causal_alpha_v10(rows, knowledge_cutoff=cutoff, config=config)
    second = fit_causal_alpha_v10(rows, knowledge_cutoff=cutoff, config=config)

    assert first.digest == second.digest
    assert first.fast.digest != first.slow.digest
    assert first.fast.maximum_label_end_index < cutoff
    assert first.slow.maximum_label_end_index < cutoff
    expected_fast = sum(
        int(
            np.count_nonzero(
                (record.decision_indices >= cutoff - config.fast_lookback_decisions)
                & (record.fast_label_end_indices < cutoff)
                & ((cutoff - record.decision_indices) % config.fast_horizon_decisions == 0)
            )
        )
        for record in rows.values()
    )
    expected_slow = sum(
        int(
            np.count_nonzero(
                (record.decision_indices >= cutoff - config.slow_lookback_decisions)
                & (record.slow_label_end_indices < cutoff)
                & ((cutoff - record.decision_indices) % config.slow_horizon_decisions == 0)
            )
        )
        for record in rows.values()
    )
    assert first.fast.training_row_count == expected_fast
    assert first.slow.training_row_count == expected_slow
    assert first.slow.training_row_count >= 2 * config.slow_hidden_feature_count
    assert first.fast.coefficients.shape == (
        3,
        len(first.fast.feature_names) + config.hidden_feature_count,
    )
    assert first.slow.coefficients.shape == (3, config.slow_hidden_feature_count)
    assert first.fast.predict_heads(rows["A"].features[-5:]).shape == (3, 5)
    assert first.slow.predict_heads(rows["A"].features[-5:]).shape == (3, 5)


def test_v10_training_rows_reject_symbol_identity_features() -> None:
    source = _rows("A", 0.1)

    with pytest.raises(ValueError, match="symbol identity"):
        CausalAlphaV10TrainingRows(
            symbol=source.symbol,
            decision_indices=source.decision_indices,
            fast_label_end_indices=source.fast_label_end_indices,
            slow_label_end_indices=source.slow_label_end_indices,
            features=source.features,
            feature_available=source.feature_available,
            fast_labels=source.fast_labels,
            slow_labels=source.slow_labels,
            feature_names=("return_4h", "symbol_id", "liquidity"),
        )
