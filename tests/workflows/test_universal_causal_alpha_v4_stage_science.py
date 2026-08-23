from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.workflows.universal_causal_alpha_v4_stage_science import (
    resolve_causal_alpha_v4_contract_rows,
    resolve_causal_alpha_v4_stage_state_inputs,
)


def _sample() -> SimpleNamespace:
    decisions = np.arange(100, 108, dtype=np.int64)
    names = (
        "15m__garman_klass_volatility_32bar",
        "15m__relative_volume_32bar",
        "15m__other",
    )
    values = np.column_stack(
        (
            np.linspace(0.1, 0.8, len(decisions)),
            np.linspace(8.0, 1.0, len(decisions)),
            np.ones(len(decisions)),
        )
    )
    available = np.ones_like(values, dtype=np.bool_)
    local_names = ("spot_perp_basis_robust_z_7d", "other_local")
    local_values = np.column_stack(
        (np.linspace(-2.0, 2.0, len(decisions)), np.ones(len(decisions)))
    )
    local_available = np.ones_like(local_values, dtype=np.bool_)
    global_values = np.ones((len(decisions), 2), dtype=np.float64)
    global_available = np.ones_like(global_values, dtype=np.bool_)
    return SimpleNamespace(
        decision_indices=decisions,
        target_local_feature_names=names,
        target_local_features=values,
        target_local_available=available,
        local_context=SimpleNamespace(
            feature_names=local_names,
            values=local_values,
            available=local_available,
        ),
        global_context=SimpleNamespace(
            feature_names=("g1", "g2"),
            values=global_values,
            available=global_available,
        ),
        beta=np.ones(len(decisions), dtype=np.float64),
        beta_available=np.ones(len(decisions), dtype=np.bool_),
    )


def test_v4_contract_rows_require_complete_decision_coverage() -> None:
    sample = _sample()
    rows = resolve_causal_alpha_v4_contract_rows(sample, start=101, stop=106)
    np.testing.assert_array_equal(rows, np.asarray([1, 2, 3, 4], dtype=np.int64))

    broken = _sample()
    broken.decision_indices = np.asarray([100, 101, 103, 104, 105, 106, 107, 108])
    with pytest.raises(ValueError, match="complete decision coverage"):
        resolve_causal_alpha_v4_contract_rows(broken, start=101, stop=106)


def test_v4_stage_state_inputs_use_frozen_channels_and_required_context_availability() -> (
    None
):
    sample = _sample()
    resolved = resolve_causal_alpha_v4_stage_state_inputs(sample)

    np.testing.assert_allclose(
        resolved.realized_volatility,
        sample.target_local_features[:, 0],
    )
    np.testing.assert_allclose(resolved.liquidity, sample.target_local_features[:, 1])
    np.testing.assert_allclose(
        resolved.basis_positioning_stress,
        sample.local_context.values[:, 0],
    )
    assert resolved.state_eligible.all()
    assert resolved.actionable.all()

    sample.local_context.available[2, 1] = False
    sample.global_context.available[3, 0] = False
    sample.beta_available[4] = False
    sample.target_local_available[5, 0] = False
    resolved = resolve_causal_alpha_v4_stage_state_inputs(sample)

    assert resolved.actionable[2] == 0
    assert resolved.actionable[3] == 0
    assert resolved.actionable[4] == 0
    assert resolved.state_eligible[5] == 0
    assert resolved.actionable[5] == 1
