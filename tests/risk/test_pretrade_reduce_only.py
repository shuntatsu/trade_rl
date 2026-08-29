from __future__ import annotations

import numpy as np
import pytest

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


def _risk() -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=2.0,
            entry_threshold=0.10,
            exit_threshold=0.03,
            no_trade_band=0.05,
        )
    )


def test_reduce_only_micro_reduction_bypasses_hysteresis_and_no_trade_band() -> None:
    risk = _risk()

    ordinary = risk.constrain(
        np.array([0.1000]),
        current=np.array([0.1004]),
        drawdown=0.0,
    )
    reduced = risk.constrain(
        np.array([0.1000]),
        current=np.array([0.1004]),
        drawdown=0.0,
        reduce_only_mask=np.array([True]),
    )

    np.testing.assert_allclose(ordinary.weights, np.array([0.1004]))
    np.testing.assert_allclose(reduced.weights, np.array([0.1000]))
    assert "reduce_only" in reduced.reasons
    assert "hold_hysteresis" not in reduced.reasons
    assert "no_trade_band" not in reduced.reasons


def test_reduce_only_add_fails_closed() -> None:
    with pytest.raises(ValueError, match="reduce-only.*increase"):
        _risk().constrain(
            np.array([0.1004]),
            current=np.array([0.1000]),
            drawdown=0.0,
            reduce_only_mask=np.array([True]),
        )


def test_reduce_only_sign_flip_fails_closed() -> None:
    with pytest.raises(ValueError, match="reduce-only.*sign"):
        _risk().constrain(
            np.array([-0.05]),
            current=np.array([0.10]),
            drawdown=0.0,
            reduce_only_mask=np.array([True]),
        )


@pytest.mark.parametrize(
    "mask",
    (
        np.array([1], dtype=np.int64),
        np.array(["false"], dtype=object),
        np.array([True, False], dtype=np.bool_),
    ),
)
def test_reduce_only_mask_is_strict_boolean_and_shape_aligned(mask: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="reduce_only_mask"):
        _risk().constrain(
            np.array([0.05]),
            current=np.array([0.10]),
            drawdown=0.0,
            reduce_only_mask=mask,
        )


def test_reduce_only_does_not_change_emergency_flatten() -> None:
    result = _risk().constrain(
        np.array([0.10]),
        current=np.array([0.10]),
        drawdown=0.0,
        emergency_flatten_mask=np.array([True]),
        reduce_only_mask=np.array([False]),
    )

    np.testing.assert_array_equal(result.weights, np.array([0.0]))
    assert "emergency_flatten" in result.reasons
