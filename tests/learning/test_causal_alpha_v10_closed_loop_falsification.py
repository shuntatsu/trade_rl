from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v10 import (
    CAUSAL_ALPHA_V10_HIERARCHY_REASONS,
    CausalAlphaV10Config,
)
from trade_rl.learning.causal_alpha_v10_hierarchy import (
    CausalAlphaV10BoundaryMode,
    CausalAlphaV10ExecutionContract,
    CausalAlphaV10HierarchyPolicy,
    prepare_causal_alpha_v10_hierarchy_policy,
)


def _boundaries() -> SimpleNamespace:
    return SimpleNamespace(
        liquidity=(10.0, 20.0, 30.0),
        realized_volatility=(1.0, 2.0, 3.0),
    )


def _heads(rows: int, offsets: tuple[int, ...]) -> np.ndarray:
    result = np.zeros((3, rows), dtype=np.float64)
    for offset in offsets:
        result[:, offset] = 0.01
    return result


def _head_values(rows: int, values: tuple[tuple[int, float], ...]) -> np.ndarray:
    result = np.zeros((3, rows), dtype=np.float64)
    for offset, value in values:
        result[:, offset] = value
    return result


def _policy(
    *,
    rows: int,
    initial_weight: float = 0.0,
    signal_offsets: tuple[int, ...] = (),
    fast_head_values: tuple[tuple[int, float], ...] | None = None,
    slow_head_values: tuple[tuple[int, float], ...] | None = None,
    liquidity_caps: np.ndarray | None = None,
    risk_caps: np.ndarray | None = None,
    boundary_mode: CausalAlphaV10BoundaryMode = CausalAlphaV10BoundaryMode.INHERIT_CONFIRM,
) -> CausalAlphaV10HierarchyPolicy:
    return prepare_causal_alpha_v10_hierarchy_policy(
        decision_indices=np.arange(rows, dtype=np.int64),
        fast_head_predictions=(
            _heads(rows, signal_offsets)
            if fast_head_values is None
            else _head_values(rows, fast_head_values)
        ),
        slow_head_predictions=(
            _heads(rows, signal_offsets)
            if slow_head_values is None
            else _head_values(rows, slow_head_values)
        ),
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
        execution_contract=CausalAlphaV10ExecutionContract(
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.05,
        ),
        boundary_mode=boundary_mode,
    )


def _drive_requested_as_realized(policy: CausalAlphaV10HierarchyPolicy) -> None:
    realized = float(policy.input.initial_weight)
    for _ in policy.input.decision_indices:
        action, _state = policy.predict(
            {"current_weights": np.asarray([realized], dtype=np.float64)}
        )
        realized = float(action[0])


def test_v10_trace_distinguishes_entry_floor_from_generic_v6_hold() -> None:
    caps = np.full(17, 0.099)
    policy = _policy(rows=17, signal_offsets=(0, 16), liquidity_caps=caps)

    _drive_requested_as_realized(policy)
    result = policy.result()

    assert result.v6_target_path.reasons[16] == "hold_flat"
    assert result.hierarchy_reasons[16] == "entry_floor_hold"
    assert ("entry_floor_hold", 2) in result.hierarchy_reason_counts


def test_v10_trace_distinguishes_held_soft_liquidity_capacity() -> None:
    caps = np.full(33, 0.10)
    caps[32] = 0.04
    policy = _policy(rows=33, signal_offsets=(0, 16, 32), liquidity_caps=caps)

    _drive_requested_as_realized(policy)
    result = policy.result()

    np.testing.assert_allclose(
        result.v6_target_path.targets[32], 0.10, rtol=0.0, atol=1e-8
    )
    assert result.v6_target_path.reasons[32] == "hold_position"
    assert result.hierarchy_reasons[32] == "liquidity_capacity_hold"


def test_v11_neutral_fast_expiry_reason_is_supported_by_target_contract() -> None:
    policy = _policy(
        rows=113,
        signal_offsets=(0, 16),
        boundary_mode=CausalAlphaV10BoundaryMode.NEUTRAL_FAST_EXPIRY,
    )

    _drive_requested_as_realized(policy)
    result = policy.result()

    assert result.hierarchy_reasons[112] == "neutral_fast_expiry"
    assert "neutral_fast_expiry" in CAUSAL_ALPHA_V10_HIERARCHY_REASONS


def test_v12_flat_on_risk_breach_holds_flat_until_realized_flat() -> None:
    policy = _policy(
        rows=3,
        initial_weight=0.10,
        risk_caps=np.full(3, 0.05),
        boundary_mode=CausalAlphaV10BoundaryMode.FLATTEN_ON_RISK_BREACH,
    )

    first, _state = policy.predict(
        {"current_weights": np.asarray([0.10], dtype=np.float64)}
    )
    second, _state = policy.predict(
        {"current_weights": np.asarray([0.04], dtype=np.float64)}
    )
    third, _state = policy.predict(
        {"current_weights": np.asarray([0.0], dtype=np.float64)}
    )

    assert float(first[0]) == 0.0
    assert float(second[0]) == 0.0
    assert float(third[0]) == 0.0
    result = policy.result()
    assert result.hierarchy_reasons[:2] == (
        "risk_cap_flatten",
        "risk_cap_flatten",
    )
    assert result.hierarchy_reasons[2] == "realized_state_reset"
    assert "risk_cap_flatten" in CAUSAL_ALPHA_V10_HIERARCHY_REASONS


def test_v13_fast_only_ownership_enters_without_slow_support() -> None:
    policy = _policy(
        rows=17,
        fast_head_values=((0, 0.01), (16, 0.01)),
        slow_head_values=(),
        boundary_mode=CausalAlphaV10BoundaryMode.FAST_ONLY_OWNERSHIP,
    )

    actions: list[float] = []
    realized = 0.0
    for _ in policy.input.decision_indices:
        action, _state = policy.predict(
            {"current_weights": np.asarray([realized], dtype=np.float64)}
        )
        actions.append(float(action[0]))
        realized = float(action[0])

    assert actions[0] == 0.0
    assert actions[16] == pytest.approx(0.10)
    result = policy.result()
    assert result.hierarchy_reasons[0] == "confirmation_hold"
    assert result.hierarchy_reasons[16] == "entry"


def test_v13_fast_only_ownership_ignores_slow_opposite_for_inherited_position() -> None:
    policy = _policy(
        rows=17,
        initial_weight=0.10,
        fast_head_values=((0, 0.01), (16, 0.01)),
        slow_head_values=((0, -0.01), (16, -0.01)),
        boundary_mode=CausalAlphaV10BoundaryMode.FAST_ONLY_OWNERSHIP,
    )

    actions: list[float] = []
    realized = 0.10
    for _ in policy.input.decision_indices:
        action, _state = policy.predict(
            {"current_weights": np.asarray([realized], dtype=np.float64)}
        )
        actions.append(float(action[0]))
        realized = float(action[0])

    assert actions[0] == pytest.approx(0.10)
    assert actions[16] == pytest.approx(0.10)
    result = policy.result()
    assert result.hierarchy_reasons[0] == "confirmation_hold"
    assert result.hierarchy_reasons[16] == "fast_support_hold"


def test_v10_external_realized_flatten_resets_hierarchy_state() -> None:
    policy = _policy(rows=2, initial_weight=0.10)

    first, _state = policy.predict(
        {"current_weights": np.asarray([0.10], dtype=np.float64)}
    )
    np.testing.assert_allclose(float(first[0]), 0.10, rtol=0.0, atol=1e-8)

    second, _state = policy.predict(
        {"current_weights": np.asarray([0.0], dtype=np.float64)}
    )
    assert float(second[0]) == 0.0

    result = policy.result()
    assert result.v6_target_path.reasons[1] == "hold_flat"
    assert result.hierarchy_reasons[1] == "realized_state_reset"


def test_v10_unrequested_realized_sign_flip_fails_closed() -> None:
    policy = _policy(rows=2, initial_weight=0.10)
    policy.predict({"current_weights": np.asarray([0.10], dtype=np.float64)})

    with pytest.raises(RuntimeError, match="flipped without an intervening flat"):
        policy.predict({"current_weights": np.asarray([-0.10], dtype=np.float64)})


@pytest.mark.parametrize(
    "observation",
    (
        {},
        {"current_weights": np.asarray([np.nan])},
        {"current_weights": np.asarray([0.0, 0.0])},
    ),
)
def test_v10_invalid_realized_weight_observation_fails_closed(
    observation: object,
) -> None:
    policy = _policy(rows=1)

    with pytest.raises(ValueError, match="current_weights"):
        policy.predict(observation)
