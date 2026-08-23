from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.workflows.universal_causal_alpha_v4_stage_execution import (
    _horizon_weights,
    run_causal_alpha_v4_admission_stage,
    run_causal_alpha_v4_selection_stage,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_science import (
    CausalAlphaV4StageStateInputs,
)


def test_v4_selection_stage_cannot_bypass_failed_signal_gate() -> None:
    with pytest.raises(ValueError, match="cannot bypass failed Signal"):
        run_causal_alpha_v4_selection_stage(
            object(),
            SimpleNamespace(passed=False),
            config=object(),
            store=object(),
            slice_forecast=object(),
        )


def test_v4_admission_stage_cannot_bypass_failed_upstream_gate() -> None:
    with pytest.raises(ValueError, match="cannot bypass upstream gates"):
        run_causal_alpha_v4_admission_stage(
            object(),
            SimpleNamespace(passed=True),
            SimpleNamespace(passed=False),
            config=object(),
            store=object(),
            slice_forecast=object(),
        )


def test_v4_uncertainty_weights_exclude_unrealized_nan_and_state_ineligible_rows() -> (
    None
):
    decisions = np.arange(6, dtype=np.int64)
    labels = np.asarray([0.1, 0.2, np.nan, 0.4, 0.5, 0.6], dtype=np.float64)
    ends = np.asarray([1, 2, -1, 4, 5, 6], dtype=np.int64)
    sample = SimpleNamespace(
        decision_indices=decisions,
        labels_4h=labels,
        labels_24h=labels,
        labels_72h=labels,
        label_end_indices_4h=ends,
        label_end_indices_24h=ends,
        label_end_indices_72h=ends,
    )
    state = CausalAlphaV4StageStateInputs(
        realized_volatility=np.ones(6),
        liquidity=np.ones(6),
        basis_positioning_stress=np.zeros(6),
        state_eligible=np.asarray([True, True, True, False, True, True]),
        actionable=np.ones(6, dtype=np.bool_),
    )

    weights = _horizon_weights(sample=sample, cutoff=5, state=state)

    for horizon in ("4h", "24h", "72h"):
        value = weights[horizon]
        assert value[0] > 0.0
        assert value[1] > 0.0
        np.testing.assert_array_equal(value[2:], np.zeros(4, dtype=np.float64))
