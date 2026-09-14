from __future__ import annotations

import numpy as np
import pytest

import trade_rl.evaluation.loss_attribution as loss_attribution
from trade_rl.evaluation.loss_attribution import _market_sensitivity


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    strategy = np.asarray(
        [
            -0.009891213503478508,
            -0.0036778665146788322,
            0.012879252612892488,
            0.0019397441913261322,
            0.009202308996398569,
            0.005771037912572513,
            -0.006364636463709806,
            0.005419522204102934,
            -0.003165954511658161,
            -0.0032238911615896015,
        ],
        dtype=np.float64,
    )
    market = np.asarray(
        [
            0.001943346373409144,
            -0.03051860813037903,
            0.02384332208203317,
            -0.013421793503482191,
            0.02000538839318921,
            0.00272642247706235,
            0.03064066159257593,
            -0.013199388275836416,
            -0.006235897129398351,
            0.00675538253117652,
        ],
        dtype=np.float64,
    )
    return strategy, market


def test_market_sensitivity_is_fixed_order_and_blas_independent() -> None:
    strategy, market = _inputs()

    correlation, beta = _market_sensitivity(strategy, market)

    assert correlation == 0.22849169250485715
    assert beta == 0.08809694283757973


def test_market_sensitivity_does_not_use_numpy_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy, market = _inputs()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("portable sensitivity must not use NumPy reductions")

    monkeypatch.setattr(loss_attribution.np, "mean", forbidden)
    monkeypatch.setattr(loss_attribution.np, "dot", forbidden)

    correlation, beta = _market_sensitivity(strategy, market)

    assert correlation == 0.22849169250485715
    assert beta == 0.08809694283757973
