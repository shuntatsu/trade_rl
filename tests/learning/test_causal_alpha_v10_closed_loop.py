from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import trade_rl.learning.causal_alpha_v10_hierarchy as hierarchy
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config


def _boundaries() -> SimpleNamespace:
    return SimpleNamespace(
        liquidity=(10.0, 20.0, 30.0),
        realized_volatility=(1.0, 2.0, 3.0),
    )


def _heads(rows: int, offsets: tuple[int, ...], direction: int = 1) -> np.ndarray:
    values = np.zeros((3, rows), dtype=np.float64)
    for offset in offsets:
        values[:, offset] = 0.01 * direction
    return values


def _path(
    *,
    decision_indices: np.ndarray,
    fast_offsets: tuple[int, ...],
    slow_offsets: tuple[int, ...],
    initial_weight: float = 0.0,
    liquidity_caps: np.ndarray | None = None,
    risk_caps: np.ndarray | None = None,
    entry_threshold: float = 0.10,
    no_trade_band: float = 0.05,
):
    rows = len(decision_indices)
    return hierarchy.causal_alpha_v10_hierarchical_target_path(
        decision_indices=decision_indices,
        fast_head_predictions=_heads(rows, fast_offsets),
        slow_head_predictions=_heads(rows, slow_offsets),
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=(
            np.full(rows, 0.10) if liquidity_caps is None else liquidity_caps
        ),
        risk_weight_caps=np.full(rows, 0.25) if risk_caps is None else risk_caps,
        realized_volatility=np.full(rows, 2.5),
        liquidity=np.full(rows, 25.0),
        attribution_boundaries=_boundaries(),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        dual_fit_digest="b" * 64,
        config=CausalAlphaV10Config(),
        initial_weight=initial_weight,
        execution_entry_threshold=entry_threshold,
        execution_no_trade_band=no_trade_band,
    )


def test_v10_closed_loop_policy_uses_realized_weight_on_next_hold() -> None:
    rows = 18
    policy = hierarchy.prepare_causal_alpha_v10_hierarchy_policy(
        decision_indices=np.arange(rows, dtype=np.int64),
        fast_head_predictions=_heads(rows, (0, 16)),
        slow_head_predictions=_heads(rows, (0, 16)),
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=np.full(rows, 0.10),
        risk_weight_caps=np.full(rows, 0.25),
        realized_volatility=np.full(rows, 2.5),
        liquidity=np.full(rows, 25.0),
        attribution_boundaries=_boundaries(),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        dual_fit_digest="b" * 64,
        config=CausalAlphaV10Config(),
        initial_weight=0.0,
        execution_contract=hierarchy.CausalAlphaV10ExecutionContract(
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.05,
        ),
    )

    realized = 0.0
    for offset in range(17):
        action, _ = policy.predict(
            {"current_weights": np.asarray([realized], dtype=np.float32)}
        )
        if offset == 16:
            assert float(action[0]) == np.float32(0.10)
        realized = float(action[0])

    action, _ = policy.predict(
        {"current_weights": np.asarray([0.05], dtype=np.float32)}
    )
    assert float(action[0]) == np.float32(0.05)


def test_v10_fast_cadence_is_bound_to_absolute_decision_index() -> None:
    decisions = np.arange(1, 34, dtype=np.int64)
    # Absolute decisions 16 and 32 are offsets 15 and 31 in this slice.
    path = _path(
        decision_indices=decisions,
        fast_offsets=(15, 31),
        slow_offsets=(15, 31),
    )

    assert path.targets[31] == 0.10


def test_v10_micro_hard_risk_reduction_projects_to_cap() -> None:
    decisions = np.arange(49, dtype=np.int64)
    risk_caps = np.full(49, 0.25)
    risk_caps[32] = 0.04
    path = _path(
        decision_indices=decisions,
        fast_offsets=(0, 16, 32),
        slow_offsets=(0, 16, 32),
        risk_caps=risk_caps,
    )

    assert path.targets[16] == 0.10
    assert path.targets[32] == 0.04
    assert path.reasons[32] == "risk_projection"


def test_v10_risk_projection_releases_once_realized_exposure_is_within_cap() -> None:
    rows = 2
    policy = hierarchy.prepare_causal_alpha_v10_hierarchy_policy(
        decision_indices=np.asarray([0, 1], dtype=np.int64),
        fast_head_predictions=np.zeros((3, rows), dtype=np.float64),
        slow_head_predictions=np.zeros((3, rows), dtype=np.float64),
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=np.full(rows, 0.10),
        risk_weight_caps=np.full(rows, 0.04),
        realized_volatility=np.full(rows, 2.5),
        liquidity=np.full(rows, 25.0),
        attribution_boundaries=_boundaries(),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        dual_fit_digest="b" * 64,
        config=CausalAlphaV10Config(),
        initial_weight=0.10,
        execution_contract=hierarchy.CausalAlphaV10ExecutionContract(
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.05,
        ),
    )

    first, _ = policy.predict({"current_weights": np.asarray([0.10], dtype=np.float32)})
    assert float(first[0]) == np.float32(0.04)
    assert policy.last_step_trace_metadata["hierarchy_reason"] == "risk_cap_projection"
    assert policy.last_step_trace_metadata["reduce_only"] is True

    second, _ = policy.predict(
        {"current_weights": np.asarray([0.04], dtype=np.float32)}
    )
    assert float(second[0]) == np.float32(0.04)
    assert policy.last_step_trace_metadata["hierarchy_reason"] == "cadence_hold"
    assert policy.last_step_trace_metadata["reduce_only"] is False


def test_v10_executable_hard_risk_reduction_projects_partially() -> None:
    decisions = np.arange(49, dtype=np.int64)
    risk_caps = np.full(49, 0.25)
    risk_caps[32] = 0.05
    path = _path(
        decision_indices=decisions,
        fast_offsets=(0, 16, 32),
        slow_offsets=(0, 16, 32),
        risk_caps=risk_caps,
        entry_threshold=0.05,
        no_trade_band=0.05,
    )

    assert path.targets[16] == 0.10
    assert path.targets[32] == 0.05
    assert path.reasons[32] == "risk_projection"


def test_v10_entry_floor_hold_uses_generic_v6_hold_reason() -> None:
    decisions = np.arange(33, dtype=np.int64)
    caps = np.full(33, 0.099)
    path = _path(
        decision_indices=decisions,
        fast_offsets=(0, 16),
        slow_offsets=(0, 16),
        liquidity_caps=caps,
    )

    assert path.targets[16] == 0.0
    assert path.reasons[16] == "hold_flat"


def test_v10_soft_liquidity_capacity_hold_uses_generic_v6_hold_reason() -> None:
    decisions = np.arange(49, dtype=np.int64)
    caps = np.full(49, 0.10)
    caps[32] = 0.04
    path = _path(
        decision_indices=decisions,
        fast_offsets=(0, 16, 32),
        slow_offsets=(0, 16, 32),
        liquidity_caps=caps,
    )

    assert path.targets[16] == 0.10
    assert path.targets[32] == 0.10
    assert path.reasons[32] == "hold_position"
