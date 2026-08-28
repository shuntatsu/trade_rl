from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config
from trade_rl.learning.causal_alpha_v10_fit import (
    CausalAlphaV10TrainingRows,
    fit_causal_alpha_v10,
)
from trade_rl.learning.causal_alpha_v10_hierarchy import (
    CausalAlphaV10ExecutionContract,
    prepare_causal_alpha_v10_hierarchy_policy,
)


def _boundaries() -> SimpleNamespace:
    return SimpleNamespace(
        liquidity=(10.0, 20.0, 30.0),
        realized_volatility=(1.0, 2.0, 3.0),
    )


def test_risk_partial_reduction_just_below_no_trade_band_flattens() -> None:
    rows = 1
    policy = prepare_causal_alpha_v10_hierarchy_policy(
        decision_indices=np.asarray([0], dtype=np.int64),
        fast_head_predictions=np.zeros((3, rows), dtype=np.float64),
        slow_head_predictions=np.zeros((3, rows), dtype=np.float64),
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=np.full(rows, 0.10),
        risk_weight_caps=np.full(rows, 0.0500005),
        realized_volatility=np.full(rows, 2.5),
        liquidity=np.full(rows, 25.0),
        attribution_boundaries=_boundaries(),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        dual_fit_digest="b" * 64,
        config=CausalAlphaV10Config(),
        initial_weight=0.10,
        execution_contract=CausalAlphaV10ExecutionContract(
            entry_threshold=0.05,
            exit_threshold=0.03,
            no_trade_band=0.05,
        ),
    )

    action, _ = policy.predict({"current_weights": np.asarray([0.10])})

    assert float(action[0]) == 0.0
    assert policy.result().hierarchy_reasons == ("risk_cap_flatten",)


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
    return CausalAlphaV10TrainingRows(
        symbol=symbol,
        decision_indices=decisions,
        fast_label_end_indices=decisions + 16,
        slow_label_end_indices=decisions + 288,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        fast_labels=0.01 * features[:, 0] - 0.003 * features[:, 1] ** 2,
        slow_labels=0.04 * features[:, 2] + 0.01 * features[:, 0],
        feature_names=("return_4h", "return_24h", "liquidity"),
    )


def test_v10_fit_sampling_uses_same_absolute_phase_as_inference() -> None:
    rows = {
        "A": _rows("A", 0.1),
        "B": _rows("B", 0.7),
        "C": _rows("C", 1.1),
    }
    config = CausalAlphaV10Config()
    cutoff = 8_993

    fit = fit_causal_alpha_v10(rows, knowledge_cutoff=cutoff, config=config)

    expected_fast = sum(
        int(
            np.count_nonzero(
                (record.decision_indices >= cutoff - config.fast_lookback_decisions)
                & (record.fast_label_end_indices < cutoff)
                & (record.decision_indices % config.fast_horizon_decisions == 0)
            )
        )
        for record in rows.values()
    )
    expected_slow = sum(
        int(
            np.count_nonzero(
                (record.decision_indices >= cutoff - config.slow_lookback_decisions)
                & (record.slow_label_end_indices < cutoff)
                & (record.decision_indices % config.slow_horizon_decisions == 0)
            )
        )
        for record in rows.values()
    )
    assert fit.fast.training_row_count == expected_fast
    assert fit.slow.training_row_count == expected_slow
