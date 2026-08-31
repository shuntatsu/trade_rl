from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config
from trade_rl.learning.causal_alpha_v10_hierarchy import (
    CausalAlphaV10BoundaryMode,
    CausalAlphaV10ExecutionContract,
    prepare_causal_alpha_v10_hierarchy_policy,
)


def _policy(
    *,
    initial_weight: float,
    risk_cap: float,
    boundary_mode: CausalAlphaV10BoundaryMode = CausalAlphaV10BoundaryMode.INHERIT_CONFIRM,
):
    rows = 2
    return prepare_causal_alpha_v10_hierarchy_policy(
        decision_indices=np.asarray([0, 1], dtype=np.int64),
        fast_head_predictions=np.zeros((3, rows), dtype=np.float64),
        slow_head_predictions=np.zeros((3, rows), dtype=np.float64),
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=np.full(rows, 0.10),
        risk_weight_caps=np.full(rows, risk_cap),
        realized_volatility=np.full(rows, 2.5),
        liquidity=np.full(rows, 25.0),
        attribution_boundaries=SimpleNamespace(
            liquidity=(10.0, 20.0, 30.0),
            realized_volatility=(1.0, 2.0, 3.0),
        ),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        dual_fit_digest="b" * 64,
        config=CausalAlphaV10Config(),
        initial_weight=initial_weight,
        execution_contract=CausalAlphaV10ExecutionContract(
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.05,
        ),
        boundary_mode=boundary_mode,
    )


def test_v10_micro_risk_reduction_below_no_trade_band_flattens_reduce_only() -> None:
    policy = _policy(initial_weight=0.1004, risk_cap=0.10)

    action, _ = policy.predict(
        {"current_weights": np.asarray([0.1004], dtype=np.float64)}
    )

    assert float(action[0]) == 0.0
    metadata = policy.last_step_trace_metadata
    assert metadata["hierarchy_reason"] == "risk_cap_flatten"
    assert metadata["reduce_only"] is True


def test_v10_flatten_on_risk_breach_remains_explicit_flat() -> None:
    policy = _policy(
        initial_weight=0.1004,
        risk_cap=0.10,
        boundary_mode=CausalAlphaV10BoundaryMode.FLATTEN_ON_RISK_BREACH,
    )

    action, _ = policy.predict(
        {"current_weights": np.asarray([0.1004], dtype=np.float64)}
    )

    assert float(action[0]) == 0.0
    metadata = policy.last_step_trace_metadata
    assert metadata["hierarchy_reason"] == "risk_cap_flatten"
    assert metadata["reduce_only"] is True


def test_v10_ordinary_hold_does_not_claim_reduce_only() -> None:
    policy = _policy(initial_weight=0.10, risk_cap=0.10)

    action, _ = policy.predict(
        {"current_weights": np.asarray([0.10], dtype=np.float64)}
    )

    assert float(action[0]) == np.float32(0.10)
    assert policy.last_step_trace_metadata["reduce_only"] is False
