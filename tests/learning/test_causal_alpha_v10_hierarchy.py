from __future__ import annotations

import numpy as np

from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config
from trade_rl.learning.causal_alpha_v10_hierarchy import (
    causal_alpha_v10_hierarchical_target_path,
)
from trade_rl.simulation.target_exposure_controller import (
    TargetExposureController,
    TargetExposureInput,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionBoundaries,
)


def _boundaries() -> CausalAlphaV7AttributionBoundaries:
    return CausalAlphaV7AttributionBoundaries(
        confidence=(0.1, 0.2, 0.3),
        realized_volatility=(1.0, 2.0, 3.0),
        liquidity=(10.0, 20.0, 30.0),
        calibration_range_digest="d" * 64,
    )


def _heads(rows: int, values: dict[int, int]) -> np.ndarray:
    result = np.zeros((3, rows), dtype=np.float64)
    for index, direction in values.items():
        result[:, index] = 0.01 * direction
    return result


def _path(
    *,
    fast: dict[int, int],
    slow: dict[int, int],
    rows: int = 145,
    initial_weight: float = 0.0,
    liquidity: np.ndarray | None = None,
    liquidity_caps: np.ndarray | None = None,
    risk_caps: np.ndarray | None = None,
    costs: np.ndarray | None = None,
    execution_entry_threshold: float = 0.10,
    execution_no_trade_band: float = 0.05,
) -> CausalAlphaV6TargetPath:
    return causal_alpha_v10_hierarchical_target_path(
        decision_indices=np.arange(rows),
        fast_head_predictions=_heads(rows, fast),
        slow_head_predictions=_heads(rows, slow),
        one_way_cost_rates=(np.full(rows, 0.0001) if costs is None else costs),
        liquidity_weight_caps=(
            np.full(rows, 0.25) if liquidity_caps is None else liquidity_caps
        ),
        risk_weight_caps=np.full(rows, 0.25) if risk_caps is None else risk_caps,
        realized_volatility=np.full(rows, 2.5),
        liquidity=np.full(rows, 25.0) if liquidity is None else liquidity,
        attribution_boundaries=_boundaries(),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        dual_fit_digest="b" * 64,
        config=CausalAlphaV10Config(),
        initial_weight=initial_weight,
        execution_entry_threshold=execution_entry_threshold,
        execution_no_trade_band=execution_no_trade_band,
    )


def test_v10_requires_coherent_fast_slow_entry_and_execution_regime() -> None:
    entered = _path(fast={0: 1, 16: 1}, slow={0: 1, 16: 1})
    bad_liquidity = np.full(145, 25.0)
    bad_liquidity[[0, 16]] = 19.0
    vetoed = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
        liquidity=bad_liquidity,
    )

    assert entered.targets[16] == 0.1
    assert vetoed.targets[16] == 0.0
    plan = TargetExposureController(no_trade_band=0.05).plan(
        TargetExposureInput(
            target_exposure=float(entered.targets[16]),
            allocated_equity=1_000.0,
            reference_price=100.0,
            contract_multiplier=1.0,
            realized_quantity=0.0,
            working_remaining_quantities=(),
        )
    )
    assert plan.child_order is not None


def test_v10_flat_entry_below_pretrade_entry_floor_stays_flat() -> None:
    caps = np.full(145, 0.099)
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
        liquidity_caps=caps,
    )

    assert path.targets[16] == 0.0
    assert path.reasons[16] == "hold_flat"


def test_v10_entry_threshold_equality_is_executable() -> None:
    caps = np.full(145, 0.10)
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
        liquidity_caps=caps,
    )

    assert path.targets[16] == 0.10


def test_v10_no_trade_band_equality_is_executable_when_entry_threshold_is_zero() -> (
    None
):
    caps = np.full(145, 0.05)
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
        liquidity_caps=caps,
        execution_entry_threshold=0.0,
        execution_no_trade_band=0.05,
    )

    assert path.targets[16] == 0.05


def test_v10_coherent_entry_must_clear_after_cost_hurdle() -> None:
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
        costs=np.full(145, 0.01),
    )

    assert path.targets[16] == 0.0
    assert path.reasons[16] == "cost_or_uncertainty_hold"


def test_v10_liquidity_cap_jitter_does_not_resize_between_fast_decisions() -> None:
    caps = np.full(145, 0.10)
    caps[17:32] = 0.07
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1, 32: 1},
        liquidity_caps=caps,
    )

    assert np.all(path.targets[17:32] == 0.10)


def test_v10_soft_liquidity_cap_does_not_resize_held_position_at_fast_decision() -> (
    None
):
    caps = np.full(145, 0.10)
    caps[32] = 0.04
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1, 32: 1},
        liquidity_caps=caps,
    )

    assert path.targets[32] == 0.10
    assert path.reasons[32] == "hold_position"


def test_v10_execution_contract_is_bound_into_target_identity() -> None:
    first = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
        execution_entry_threshold=0.10,
        execution_no_trade_band=0.05,
    )
    second = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
        execution_entry_threshold=0.09,
        execution_no_trade_band=0.05,
    )

    np.testing.assert_array_equal(first.targets, second.targets)
    assert first.config_digest != second.config_digest
    assert first.digest != second.digest


def test_v10_slow_regime_holds_through_neutral_fast_signal() -> None:
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1, 32: 1, 48: 1, 64: 1},
    )

    assert path.targets[16] == 0.1
    assert path.targets[64] == 0.1


def test_v10_latches_sparse_slow_regime_between_qualified_observations() -> None:
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1},
    )

    assert path.targets[16] == 0.1


def test_v10_two_fast_opposites_exit_without_direct_flip() -> None:
    path = _path(
        fast={0: 1, 16: 1, 32: -1, 48: -1, 64: -1, 80: -1},
        slow={0: 1, 16: 1, 32: 1, 48: 1, 64: -1, 80: -1},
    )

    assert path.targets[32] == 0.1
    assert path.targets[48] == 0.0
    assert path.targets[80] == -0.1
    assert path.sign_flip_count == 0
    assert path.candidate is CausalAlphaV6Candidate.FAST_ONLY


def test_v10_two_slow_opposites_exit() -> None:
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1, 32: -1, 48: -1},
    )

    assert path.targets[32] == 0.1
    assert path.targets[48] == 0.0


def test_v10_slow_opposite_confirmation_requires_consecutive_observations() -> None:
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1, 32: -1, 48: 1, 64: -1},
    )

    assert path.targets[48] == 0.1
    assert path.targets[64] == 0.1


def test_v10_slow_neutral_breaks_opposite_confirmation() -> None:
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1, 32: -1, 48: 0, 64: -1},
    )

    assert path.targets[48] == 0.1
    assert path.targets[64] == 0.1


def test_v10_six_slow_neutral_observations_expire_wave() -> None:
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
    )

    assert path.targets[96] == 0.1
    assert path.targets[112] == 0.0


def test_v10_inherited_position_must_earn_coherent_confirmation() -> None:
    retained = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1},
        initial_weight=0.25,
    )
    rejected = _path(fast={}, slow={}, initial_weight=0.25)

    assert retained.targets[16] == 0.1
    assert rejected.targets[16] == 0.0
