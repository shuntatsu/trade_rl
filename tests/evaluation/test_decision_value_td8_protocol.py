"""Pre-registered TD8 protocol parameters and their authorization binding (#1025)."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from trade_rl.evaluation.decision_value_dev_study import (
    default_2024q1_decision_value_dev_protocol,
)
from trade_rl.evaluation.decision_value_fit import (
    fit_conservative_linear_decision_value,
)


def test_default_protocol_freezes_the_td8_parameters_before_any_result() -> None:
    proto = default_2024q1_decision_value_dev_protocol()
    assert proto.label_prefix_bars == 48
    assert proto.margin_multiplier_grid == (0.0, 0.5, 1.0, 2.0)
    assert proto.margin_calibration_fraction == 0.25
    assert proto.feature_clip == 3.0
    assert proto.unexcited_prior_std == 1.0
    # Non-overlapping branch labels: step equals the label horizon.
    assert all(
        block.train_step == proto.study_plan.horizon_bars for block in proto.blocks
    )
    mapping = proto.to_mapping()
    assert mapping["schema_version"] == "decision_value_dev_study_protocol_v2"
    for key in (
        "label_prefix_bars",
        "margin_multiplier_grid",
        "margin_calibration_fraction",
        "feature_clip",
        "unexcited_prior_std",
    ):
        assert key in mapping


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("label_prefix_bars", 24),
        ("margin_multiplier_grid", (0.0, 1.0)),
        ("margin_calibration_fraction", 0.5),
        ("feature_clip", 5.0),
        ("unexcited_prior_std", 1e-3),
    ),
)
def test_every_td8_parameter_is_bound_to_the_authorization_digest(
    field: str, value: object
) -> None:
    proto = default_2024q1_decision_value_dev_protocol()
    with pytest.raises(ValueError, match="authorization"):
        replace(proto, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("label_prefix_bars", -1, "label_prefix_bars"),
        ("margin_multiplier_grid", (), "margin_multiplier_grid"),
        ("margin_multiplier_grid", (1.0, 0.5), "margin_multiplier_grid"),
        ("margin_calibration_fraction", 0.0, "margin_calibration_fraction"),
        ("margin_calibration_fraction", 1.0, "margin_calibration_fraction"),
        ("feature_clip", 0.0, "feature_clip"),
        ("unexcited_prior_std", 0.0, "unexcited_prior_std"),
    ),
)
def test_invalid_td8_parameters_are_rejected(
    field: str, value: object, match: str
) -> None:
    proto = default_2024q1_decision_value_dev_protocol()
    with pytest.raises((ValueError, TypeError), match=match):
        replace(proto, **{field: value})


def test_overlapping_branch_step_is_rejected() -> None:
    proto = default_2024q1_decision_value_dev_protocol()
    short = replace(proto.blocks[0], train_step=6)
    with pytest.raises(ValueError, match="train_step"):
        replace(proto, blocks=(short, *proto.blocks[1:]))


def test_post_hoc_prior_is_withdrawn_back_to_the_uninformative_default() -> None:
    signature = inspect.signature(fit_conservative_linear_decision_value)
    assert signature.parameters["unexcited_prior_std"].default == 1.0
