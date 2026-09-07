from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.rl.universal_trade_action import parse_normalized_target_exposure


@pytest.mark.parametrize(
    ("raw", "scale", "expected"),
    (
        (-1.0, 1.0, -1.0),
        (-0.5, 1.0, -0.5),
        (0.0, 1.0, 0.0),
        (0.5, 0.4, 0.2),
        (1.0, 0.4, 0.4),
    ),
)
def test_action_mapping_is_linear_and_static(
    raw: float,
    scale: float,
    expected: float,
) -> None:
    parsed = parse_normalized_target_exposure(
        np.asarray([raw], dtype=np.float32),
        policy_weight_scale=scale,
    )
    assert parsed.normalized == pytest.approx(raw)
    assert parsed.policy_requested_weight == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    (
        np.asarray([1.01], dtype=np.float64),
        np.asarray([-1.01], dtype=np.float64),
        np.asarray([math.nan], dtype=np.float64),
        np.asarray([math.inf], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([], dtype=np.float64),
    ),
)
def test_action_rejects_invalid_policy_output(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        parse_normalized_target_exposure(value, policy_weight_scale=1.0)


@pytest.mark.parametrize(
    "scale",
    (0.0, -0.1, 1.01, math.inf, math.nan, True),
)
def test_action_rejects_invalid_static_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="policy_weight_scale"):
        parse_normalized_target_exposure(
            np.asarray([0.5], dtype=np.float32),
            policy_weight_scale=scale,
        )


def test_action_parser_does_not_clip_out_of_range_input() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        parse_normalized_target_exposure(
            np.asarray([2.0], dtype=np.float32),
            policy_weight_scale=0.25,
        )
