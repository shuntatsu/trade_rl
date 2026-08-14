from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.actions import (
    ActionMode,
    ActionSpec,
    AnchoredTargetResidualAction,
    BaselineResidualComposer,
)
from trade_rl.strategies.trend import TrendTargets


def _spec(**overrides: object) -> ActionSpec:
    values: dict[str, object] = {
        "mode": ActionMode.ANCHORED_TARGET_RESIDUAL,
        "alpha_enabled": True,
        "risk_tilt_enabled": False,
        "n_factors": 0,
        "target_weight_count": 2,
        "residual_scale": 0.1,
    }
    values.update(overrides)
    return ActionSpec(**values)  # type: ignore[arg-type]


def _trends() -> TrendTargets:
    return TrendTargets(
        fast=np.asarray([0.2, -0.2]),
        base=np.asarray([0.1, -0.1]),
        slow=np.asarray([0.05, -0.05]),
    )


def test_anchored_mode_requires_target_weight_alpha_and_no_other_residual_controls() -> (
    None
):
    with pytest.raises(ValueError, match="alpha_enabled"):
        _spec(alpha_enabled=False)
    with pytest.raises(ValueError, match="target_weight_count"):
        _spec(target_weight_count=0)
    with pytest.raises(ValueError, match="risk_tilt"):
        _spec(risk_tilt_enabled=True)
    with pytest.raises(ValueError, match="n_factors"):
        _spec(n_factors=1)


def test_anchored_action_parsing_scales_policy_residuals() -> None:
    action = _spec().parse(np.asarray([1.0, -0.5], dtype=np.float64))

    assert isinstance(action, AnchoredTargetResidualAction)
    assert action.residuals.tolist() == pytest.approx([0.1, -0.05])
    assert action.as_array().tolist() == pytest.approx([0.1, -0.05])


def test_zero_anchored_residual_reproduces_target_weight_alpha_anchor() -> None:
    action = _spec().parse(np.zeros(2, dtype=np.float64))
    anchor = np.asarray([0.4, -0.2], dtype=np.float64)

    composition = BaselineResidualComposer().compose(
        action,
        _trends(),
        anchor,
        alpha_enabled=True,
        max_gross=1.0,
    )

    assert composition.baseline.tolist() == pytest.approx(anchor)
    assert composition.proposal.tolist() == pytest.approx(anchor)
    assert composition.residual_component.tolist() == pytest.approx([0.0, 0.0])


def test_anchored_residual_adds_bounded_policy_delta_then_normalizes_gross() -> None:
    action = _spec(residual_scale=0.2).parse(
        np.asarray([1.0, -1.0], dtype=np.float64)
    )
    anchor = np.asarray([0.45, -0.35], dtype=np.float64)

    composition = BaselineResidualComposer().compose(
        action,
        _trends(),
        anchor,
        alpha_enabled=True,
        max_gross=1.0,
    )

    expected_raw = np.asarray([0.65, -0.55])
    expected = expected_raw / np.abs(expected_raw).sum()
    assert composition.proposal.tolist() == pytest.approx(expected)
    assert np.abs(composition.proposal).sum() == pytest.approx(1.0)


def test_existing_target_weight_mode_keeps_its_direct_action_semantics() -> None:
    spec = ActionSpec(
        mode=ActionMode.TARGET_WEIGHT,
        alpha_enabled=False,
        risk_tilt_enabled=False,
        target_weight_count=2,
    )
    action = spec.parse(np.asarray([0.4, -0.2]))
    composition = BaselineResidualComposer().compose(
        action,
        _trends(),
        np.asarray([0.9, -0.9]),
        alpha_enabled=False,
        max_gross=1.0,
    )

    assert composition.proposal.tolist() == pytest.approx([0.4, -0.2])
    assert composition.baseline.tolist() == pytest.approx([0.1, -0.1])
