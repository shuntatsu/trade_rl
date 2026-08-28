from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config
from trade_rl.learning.causal_alpha_v10_hierarchy import (
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


def _policy(
    *,
    rows: int,
    initial_weight: float = 0.0,
    signal_offsets: tuple[int, ...] = (),
    liquidity_caps: np.ndarray | None = None,
    risk_caps: np.ndarray | None = None,
) -> CausalAlphaV10HierarchyPolicy:
    return prepare_causal_alpha_v10_hierarchy_policy(
        decision_indices=np.arange(rows, dtype=np.int64),
        fast_head_predictions=_heads(rows, signal_offsets),
        slow_head_predictions=_heads(rows, signal_offsets),
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

    assert result.v6_target_path.targets[32] == 0.10
    assert result.v6_target_path.reasons[32] == "hold_position"
    assert result.hierarchy_reasons[32] == "liquidity_capacity_hold"


def test_v10_external_realized_flatten_resets_hierarchy_state() -> None:
    policy = _policy(rows=2, initial_weight=0.10)

    first, _state = policy.predict(
        {"current_weights": np.asarray([0.10], dtype=np.float64)}
    )
    assert float(first[0]) == 0.10

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
