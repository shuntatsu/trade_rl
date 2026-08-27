from __future__ import annotations

import numpy as np

from trade_rl.learning.causal_alpha_v4 import build_causal_alpha_v4_forecast
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8TargetConfig
from trade_rl.learning.causal_alpha_v8_target import causal_alpha_v8_exposure_path


def _digest(char: str) -> str:
    return char * 64


def _forecast(
    fast_return: np.ndarray,
    *,
    fast_direction: np.ndarray | None = None,
) -> object:
    rows = len(fast_return)
    direction = (
        np.sign(fast_return)
        if fast_direction is None
        else np.asarray(fast_direction, dtype=np.float64)
    )
    zeros = np.zeros(rows, dtype=np.float64)
    return build_causal_alpha_v4_forecast(
        symbol="BTCUSDT",
        decision_indices=np.arange(100, 100 + rows, dtype=np.int64),
        beta=np.ones(rows),
        beta_available=np.ones(rows, dtype=np.bool_),
        market_predictions={"4h": zeros, "24h": zeros, "72h": zeros},
        residual_predictions={
            "4h": np.asarray(fast_return, dtype=np.float64),
            "24h": zeros,
            "72h": zeros,
        },
        direction_scores={"4h": direction, "24h": zeros, "72h": zeros},
        market_model_digests={h: _digest("a") for h in ("4h", "24h", "72h")},
        residual_model_digests={h: _digest("b") for h in ("4h", "24h", "72h")},
        direction_model_digests={h: _digest("c") for h in ("4h", "24h", "72h")},
        fit_digest=_digest("d"),
    )


def _path(
    fast_return: np.ndarray,
    *,
    initial_weight: float,
    uncertainty: float = 0.01,
    cost: float = 0.001,
    cap: float = 0.25,
) -> object:
    rows = len(fast_return)
    return causal_alpha_v8_exposure_path(
        _forecast(fast_return),
        uncertainty={h: np.full(rows, uncertainty) for h in ("4h", "24h", "72h")},
        one_way_cost_rates=np.full(rows, cost),
        liquidity_weight_caps=np.full(rows, cap),
        risk_weight_caps=np.full(rows, cap),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        config=CausalAlphaV8TargetConfig(),
        initial_weight=initial_weight,
    )


def test_v8_exits_unsupported_inherited_exposure() -> None:
    path = _path(np.zeros(5), initial_weight=0.10)

    assert path.targets[0] == 0.0
    assert path.reasons[0] == "exit"
    assert np.all(path.targets == 0.0)
    assert path.sign_flip_count == 0


def test_v8_holds_a_positive_robust_continuation_edge() -> None:
    path = _path(np.full(5, 0.03), initial_weight=0.10, cap=0.10)

    assert np.all(path.targets >= 0.10)
    assert path.targets[0] == 0.10
    assert path.reasons[0] == "hold_position"


def test_v8_requires_confirmation_for_entry() -> None:
    path = _path(np.full(9, 0.03), initial_weight=0.0)

    assert np.all(path.targets[:4] == 0.0)
    assert path.reasons[0] == "confirmation_hold"
    assert path.targets[4] > 0.0
    assert path.reasons[4] == "entry"


def test_v8_reversal_exits_through_flat_without_direct_flip() -> None:
    path = _path(np.full(13, -0.03), initial_weight=0.25)

    assert path.targets[0] == 0.0
    assert path.reasons[0] == "exit"
    assert np.all(path.targets[:4] == 0.0)
    assert path.targets[4] < 0.0
    assert path.reasons[4] == "entry"
    assert path.sign_flip_count == 0


def test_v8_liquidity_reduction_overrides_cadence() -> None:
    rows = 2
    path = causal_alpha_v8_exposure_path(
        _forecast(np.full(rows, 0.03)),
        uncertainty={h: np.full(rows, 0.001) for h in ("4h", "24h", "72h")},
        one_way_cost_rates=np.zeros(rows),
        liquidity_weight_caps=np.asarray([0.25, 0.05]),
        risk_weight_caps=np.full(rows, 0.25),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        config=CausalAlphaV8TargetConfig(),
        initial_weight=0.10,
    )

    assert path.targets[1] == 0.05
    assert path.reasons[1] == "liquidity_deleverage"
