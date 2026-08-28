from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config
from trade_rl.learning.causal_alpha_v10_hierarchy import (
    causal_alpha_v10_hierarchical_target_path,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionBoundaries,
)
from trade_rl.workflows.universal_causal_alpha_v10_stage_entry import (
    _execution_no_trade_band,
)


def _heads(rows: int) -> np.ndarray:
    result = np.zeros((3, rows), dtype=np.float64)
    result[:, 0] = 0.01
    result[:, 16] = 0.01
    return result


def _target(*, cap: float, band: float) -> float:
    rows = 33
    heads = _heads(rows)
    path = causal_alpha_v10_hierarchical_target_path(
        decision_indices=np.arange(rows),
        fast_head_predictions=heads,
        slow_head_predictions=heads,
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=np.full(rows, cap),
        risk_weight_caps=np.full(rows, 0.25),
        realized_volatility=np.full(rows, 2.5),
        liquidity=np.full(rows, 25.0),
        attribution_boundaries=CausalAlphaV7AttributionBoundaries(
            confidence=(0.1, 0.2, 0.3),
            realized_volatility=(1.0, 2.0, 3.0),
            liquidity=(10.0, 20.0, 30.0),
            calibration_range_digest="d" * 64,
        ),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        dual_fit_digest="b" * 64,
        config=CausalAlphaV10Config(),
        initial_weight=0.0,
        execution_no_trade_band=band,
    )
    return float(path.targets[16])


def _pretrade(*, band: float) -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_abs_weight=1.0,
            max_turnover=2.0,
            no_trade_band=band,
        )
    )


def test_v10_compiler_and_pretrade_agree_on_execution_band_boundary() -> None:
    band = 0.05
    below = _target(cap=0.049, band=band)
    equal = _target(cap=0.05, band=band)

    assert below == 0.0
    assert equal == 0.05

    result = _pretrade(band=band).constrain(
        np.array([equal]),
        current=np.array([0.0]),
        drawdown=0.0,
    )
    np.testing.assert_allclose(result.weights, np.array([0.05]))
    assert "no_trade_band" not in result.reasons


def test_v10_execution_band_resolver_uses_environment_contract_and_closes() -> None:
    closed: list[bool] = []

    class Environment:
        pre_trade_risk = _pretrade(band=0.05)

        def close(self) -> None:
            closed.append(True)

    prepared = SimpleNamespace(
        prepared_v3=SimpleNamespace(
            environment_factories={"BTCUSDT": Environment},
        )
    )

    assert _execution_no_trade_band(prepared, "BTCUSDT") == 0.05
    assert closed == [True]
