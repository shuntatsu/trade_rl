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
) -> CausalAlphaV6TargetPath:
    return causal_alpha_v10_hierarchical_target_path(
        decision_indices=np.arange(rows),
        fast_head_predictions=_heads(rows, fast),
        slow_head_predictions=_heads(rows, slow),
        one_way_cost_rates=np.full(rows, 0.0001),
        liquidity_weight_caps=np.full(rows, 0.25),
        risk_weight_caps=np.full(rows, 0.25),
        realized_volatility=np.full(rows, 2.5),
        liquidity=np.full(rows, 25.0) if liquidity is None else liquidity,
        attribution_boundaries=_boundaries(),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        source_forecast_digest="a" * 64,
        dual_fit_digest="b" * 64,
        config=CausalAlphaV10Config(),
        initial_weight=initial_weight,
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


def test_v10_slow_regime_holds_through_neutral_fast_signal() -> None:
    path = _path(
        fast={0: 1, 16: 1},
        slow={0: 1, 16: 1, 32: 1, 48: 1, 64: 1},
    )

    assert path.targets[16] == 0.1
    assert path.targets[64] == 0.1


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
