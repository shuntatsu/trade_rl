from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.lagrangian import (
    ConstraintEstimate,
    LagrangianConstraintSpec,
    LagrangianDualController,
    LagrangianSchema,
    canonical_constraint_aggregation,
)


def _spec(
    name: str,
    *,
    budget: float = 0.1,
    dual_learning_rate: float = 0.5,
    ema_beta: float = 0.5,
    initial_multiplier: float = 0.0,
    max_multiplier: float = 10.0,
    warmup_rollouts: int = 0,
    update_interval_rollouts: int = 1,
) -> LagrangianConstraintSpec:
    return LagrangianConstraintSpec(
        name=name,
        aggregation=canonical_constraint_aggregation(name),
        budget=budget,
        dual_learning_rate=dual_learning_rate,
        ema_beta=ema_beta,
        initial_multiplier=initial_multiplier,
        max_multiplier=max_multiplier,
        warmup_rollouts=warmup_rollouts,
        update_interval_rollouts=update_interval_rollouts,
    )


def _estimate(name: str, value: float, denominator: int = 2) -> ConstraintEstimate:
    return ConstraintEstimate(
        name=name,
        numerator=value * denominator,
        denominator=denominator,
    )


def test_dual_controller_freezes_rollout_snapshot_and_updates_afterward() -> None:
    schema = LagrangianSchema((_spec("drawdown_excess"),))
    controller = LagrangianDualController(schema)

    frozen = controller.begin_rollout()
    assert frozen.tolist() == [0.0]
    assert frozen.flags.writeable is False
    with pytest.raises(ValueError):
        frozen[0] = 1.0

    reports = controller.update_after_rollout(
        {"drawdown_excess": _estimate("drawdown_excess", 0.3)}
    )
    report = reports["drawdown_excess"]
    assert report.updated is True
    assert report.raw_estimate == pytest.approx(0.3)
    assert report.ema_estimate == pytest.approx(0.3)
    assert report.multiplier_before == pytest.approx(0.0)
    assert report.multiplier_after == pytest.approx(0.1)
    assert report.skip_reason is None
    assert controller.begin_rollout().tolist() == pytest.approx([0.1])
    assert frozen.tolist() == [0.0]


def test_dual_controller_decreases_and_clips_each_multiplier_independently() -> None:
    schema = LagrangianSchema(
        (
            _spec(
                "drawdown_excess",
                initial_multiplier=0.2,
                dual_learning_rate=0.5,
            ),
            _spec(
                "margin_deficit_fraction",
                budget=0.0,
                dual_learning_rate=1.0,
                initial_multiplier=0.9,
                max_multiplier=1.0,
            ),
        )
    )
    controller = LagrangianDualController(schema)

    reports = controller.update_after_rollout(
        {
            "drawdown_excess": _estimate("drawdown_excess", 0.0),
            "margin_deficit_fraction": _estimate(
                "margin_deficit_fraction",
                10.0,
            ),
        }
    )

    assert reports["drawdown_excess"].multiplier_after == pytest.approx(0.15)
    margin = reports["margin_deficit_fraction"]
    assert margin.multiplier_after == pytest.approx(1.0)
    assert margin.saturated is True
    assert controller.begin_rollout().tolist() == pytest.approx([0.15, 1.0])


def test_dual_controller_ema_uses_previous_eligible_estimate() -> None:
    schema = LagrangianSchema((_spec("drawdown_excess", budget=0.0),))
    controller = LagrangianDualController(schema)

    first = controller.update_after_rollout(
        {"drawdown_excess": _estimate("drawdown_excess", 0.2)}
    )["drawdown_excess"]
    second = controller.update_after_rollout(
        {"drawdown_excess": _estimate("drawdown_excess", 0.6)}
    )["drawdown_excess"]

    assert first.ema_estimate == pytest.approx(0.2)
    assert second.ema_estimate == pytest.approx(0.4)
    assert second.multiplier_after == pytest.approx(0.3)
    assert second.update_count == 2


def test_dual_controller_warmup_and_interval_do_not_mutate_state() -> None:
    schema = LagrangianSchema(
        (
            _spec(
                "drawdown_excess",
                budget=0.0,
                warmup_rollouts=1,
                update_interval_rollouts=2,
            ),
        )
    )
    controller = LagrangianDualController(schema)
    estimate = {"drawdown_excess": _estimate("drawdown_excess", 0.5)}

    warmup = controller.update_after_rollout(estimate)["drawdown_excess"]
    interval = controller.update_after_rollout(estimate)["drawdown_excess"]
    eligible = controller.update_after_rollout(estimate)["drawdown_excess"]

    assert warmup.updated is False
    assert warmup.skip_reason == "warmup"
    assert warmup.ema_estimate is None
    assert interval.updated is False
    assert interval.skip_reason == "update_interval"
    assert interval.ema_estimate is None
    assert eligible.updated is True
    assert eligible.ema_estimate == pytest.approx(0.5)
    assert eligible.multiplier_after == pytest.approx(0.25)
    assert eligible.rollout_count == 3


def test_dual_controller_missing_estimate_retains_ema_and_multiplier() -> None:
    schema = LagrangianSchema((_spec("drawdown_stop_event", budget=0.0),))
    controller = LagrangianDualController(schema)
    updated = controller.update_after_rollout(
        {"drawdown_stop_event": _estimate("drawdown_stop_event", 0.5)}
    )["drawdown_stop_event"]
    before = controller.state_dict()

    skipped = controller.update_after_rollout({"drawdown_stop_event": None})[
        "drawdown_stop_event"
    ]

    assert skipped.updated is False
    assert skipped.skip_reason == "missing_estimate"
    assert skipped.ema_estimate == pytest.approx(updated.ema_estimate)
    assert skipped.multiplier_after == pytest.approx(updated.multiplier_after)
    after = controller.state_dict()
    assert after["ema_estimates"] == before["ema_estimates"]
    assert after["multipliers"] == before["multipliers"]
    assert after["update_counts"] == before["update_counts"]


def test_dual_controller_state_round_trip_reproduces_next_update() -> None:
    schema = LagrangianSchema(
        (
            _spec("drawdown_excess", budget=0.05),
            _spec("drawdown_stop_event", budget=0.1),
        )
    )
    original = LagrangianDualController(schema)
    original.update_after_rollout(
        {
            "drawdown_excess": _estimate("drawdown_excess", 0.2),
            "drawdown_stop_event": _estimate("drawdown_stop_event", 0.5),
        }
    )
    state = original.state_dict()
    restored = LagrangianDualController(schema)
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    next_estimates = {
        "drawdown_excess": _estimate("drawdown_excess", 0.1),
        "drawdown_stop_event": _estimate("drawdown_stop_event", 0.0),
    }
    assert restored.update_after_rollout(next_estimates) == (
        original.update_after_rollout(next_estimates)
    )
    np.testing.assert_array_equal(
        restored.begin_rollout(),
        original.begin_rollout(),
    )


def test_dual_controller_rejects_bad_estimate_mapping_and_state_identity() -> None:
    schema = LagrangianSchema((_spec("drawdown_excess"),))
    controller = LagrangianDualController(schema)

    with pytest.raises(ValueError, match="constraint names"):
        controller.update_after_rollout({})
    with pytest.raises(ValueError, match="constraint names"):
        controller.update_after_rollout(
            {
                "drawdown_excess": _estimate("drawdown_excess", 0.1),
                "margin_deficit_fraction": _estimate(
                    "margin_deficit_fraction",
                    0.1,
                ),
            }
        )
    with pytest.raises(ValueError, match="estimate name"):
        controller.update_after_rollout(
            {
                "drawdown_excess": _estimate(
                    "margin_deficit_fraction",
                    0.1,
                )
            }
        )

    wrong_schema = LagrangianDualController(
        LagrangianSchema((_spec("margin_deficit_fraction"),))
    )
    with pytest.raises(ValueError, match="schema"):
        wrong_schema.load_state_dict(controller.state_dict())
