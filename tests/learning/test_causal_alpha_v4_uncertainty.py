from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.learning.causal_alpha_v4 import (
    V4ForecastState,
    fit_causal_alpha_v4_uncertainty,
)


HORIZONS = ("4h", "24h", "72h")


def _horizon_map(values: np.ndarray) -> dict[str, np.ndarray]:
    return {horizon: values.copy() for horizon in HORIZONS}


def _fit(*, rows: int = 100):
    index = np.arange(rows, dtype=np.float64)
    prediction = 0.001 * np.sin(index / 7.0)
    labels = prediction + 0.002 * np.cos(index / 5.0)
    weights = np.ones(rows, dtype=np.float64)
    return fit_causal_alpha_v4_uncertainty(
        final_predictions=_horizon_map(prediction),
        labels=_horizon_map(labels),
        weights=_horizon_map(weights),
        state_eligible=np.ones(rows, dtype=np.bool_),
        realized_volatility=index,
        liquidity=index,
        basis_positioning_stress=index - 50.0,
    )


def test_v4_uncertainty_state_thresholds_and_precedence_are_frozen() -> None:
    model = _fit(rows=100)

    assert model.high_realized_volatility_threshold == np.quantile(
        np.arange(100, dtype=np.float64), 0.80
    )
    assert model.low_liquidity_threshold == np.quantile(
        np.arange(100, dtype=np.float64), 0.20
    )
    assert model.basis_positioning_stress_threshold == np.quantile(
        np.abs(np.arange(100, dtype=np.float64) - 50.0), 0.80
    )

    states = model.resolve_states(
        realized_volatility=np.asarray([99.0, 99.0, 99.0, 10.0]),
        liquidity=np.asarray([0.0, 0.0, 99.0, 99.0]),
        basis_positioning_stress=np.asarray([99.0, 0.0, 0.0, 0.0]),
    )
    assert tuple(states) == (
        V4ForecastState.BASIS_POSITIONING_STRESS,
        V4ForecastState.LOW_LIQUIDITY,
        V4ForecastState.HIGH_REALIZED_VOLATILITY,
        V4ForecastState.NORMAL,
    )


def test_v4_uncertainty_low_ess_falls_back_to_global_rmse() -> None:
    model = _fit(rows=100)
    cell = model.cells["4h"][V4ForecastState.BASIS_POSITIONING_STRESS]

    assert cell.support > 0
    assert cell.effective_sample_size < 30.0
    assert cell.fallback_reason == "insufficient_state_ess"
    assert cell.selected_uncertainty == cell.global_rmse


def test_v4_uncertainty_uses_weighted_final_forecast_residual() -> None:
    prediction = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    labels = np.asarray([0.2, 0.0, 0.5, 0.4], dtype=np.float64)
    weights = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    model = fit_causal_alpha_v4_uncertainty(
        final_predictions=_horizon_map(prediction),
        labels=_horizon_map(labels),
        weights=_horizon_map(weights),
        state_eligible=np.ones(4, dtype=np.bool_),
        realized_volatility=np.asarray([1.0, 2.0, 3.0, 4.0]),
        liquidity=np.asarray([4.0, 3.0, 2.0, 1.0]),
        basis_positioning_stress=np.asarray([0.0, 0.1, 0.2, 0.3]),
    )
    expected = np.sqrt(np.sum(weights * np.square(labels - prediction)) / weights.sum())

    assert np.isclose(model.global_rmse["4h"], expected)


def test_v4_uncertainty_ignores_rows_outside_authored_train_prefix() -> None:
    rows = 100
    index = np.arange(rows, dtype=np.float64)
    prediction = np.sin(index / 7.0) * 0.001
    labels = prediction + np.cos(index / 9.0) * 0.002
    weights = np.ones(rows, dtype=np.float64)
    weights[-10:] = 0.0
    state_eligible = np.ones(rows, dtype=np.bool_)
    state_eligible[-10:] = False
    kwargs = dict(
        final_predictions=_horizon_map(prediction),
        labels=_horizon_map(labels),
        weights=_horizon_map(weights),
        state_eligible=state_eligible,
        realized_volatility=index.copy(),
        liquidity=index.copy(),
        basis_positioning_stress=index.copy(),
    )
    first = fit_causal_alpha_v4_uncertainty(**kwargs)

    mutated_prediction = prediction.copy()
    mutated_labels = labels.copy()
    mutated_volatility = index.copy()
    mutated_liquidity = index.copy()
    mutated_stress = index.copy()
    mutated_prediction[-10:] = 999.0
    mutated_labels[-10:] = -999.0
    mutated_volatility[-10:] = 1e9
    mutated_liquidity[-10:] = -1e9
    mutated_stress[-10:] = 1e9
    second = fit_causal_alpha_v4_uncertainty(
        final_predictions=_horizon_map(mutated_prediction),
        labels=_horizon_map(mutated_labels),
        weights=_horizon_map(weights),
        state_eligible=state_eligible,
        realized_volatility=mutated_volatility,
        liquidity=mutated_liquidity,
        basis_positioning_stress=mutated_stress,
    )

    assert second.digest == first.digest
    assert second.threshold_digest == first.threshold_digest
