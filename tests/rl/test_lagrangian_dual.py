from __future__ import annotations

import copy

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
    minimum_completed_episodes: int = 1,
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
        minimum_completed_episodes=minimum_completed_episodes,
    )


def _schema(name: str = "drawdown_excess", **kwargs: object) -> LagrangianSchema:
    return LagrangianSchema((_spec(name, **kwargs),))


def _estimate(name: str, value: float, denominator: int = 1) -> ConstraintEstimate:
    return ConstraintEstimate(
        name=name,
        numerator=value * denominator,
        denominator=denominator,
    )


def _update(
    controller: LagrangianDualController,
    estimate: ConstraintEstimate | None,
    *,
    censored_episode_count: int = 0,
):
    name = controller.schema.names[0]
    return controller.update_after_rollout(
        {name: estimate},
        censored_episode_count=censored_episode_count,
    )[name]


def test_dual_controller_freezes_rollout_snapshot_and_updates_afterward() -> None:
    controller = LagrangianDualController(_schema())

    frozen = controller.begin_rollout()
    assert frozen.tolist() == [0.0]
    assert frozen.flags.writeable is False
    with pytest.raises(ValueError):
        frozen[0] = 1.0

    report = _update(controller, _estimate("drawdown_excess", 0.3, denominator=2))

    assert report.updated is True
    assert report.raw_estimate == pytest.approx(0.3)
    assert report.ema_estimate == pytest.approx(0.3)
    assert report.constraint_residual == pytest.approx(0.2)
    assert report.pending_numerator_before == pytest.approx(0.0)
    assert report.pending_denominator_before == 0
    assert report.consumed_denominator == 2
    assert report.multiplier_before == pytest.approx(0.0)
    assert report.multiplier_after == pytest.approx(0.1)
    assert report.at_lower_bound is False
    assert report.at_upper_cap is False
    assert report.saturated is False
    assert report.skip_reason is None
    assert controller.begin_rollout().tolist() == pytest.approx([0.1])
    assert frozen.tolist() == [0.0]


def test_warmup_observations_feed_first_eligible_dual_update() -> None:
    controller = LagrangianDualController(
        _schema(
            minimum_completed_episodes=3,
            warmup_rollouts=1,
            update_interval_rollouts=2,
            ema_beta=0.5,
            budget=0.0,
            dual_learning_rate=1.0,
        )
    )

    warmup = _update(controller, _estimate("drawdown_excess", 1.0))
    interval = _update(controller, _estimate("drawdown_excess", 2.0))
    updated = _update(controller, _estimate("drawdown_excess", 3.0))

    assert warmup.skip_reason == "warmup"
    assert warmup.pending_denominator_before == 0
    assert warmup.consumed_denominator == 0
    assert interval.skip_reason == "update_interval"
    assert interval.pending_denominator_before == 1
    assert interval.consumed_denominator == 0
    assert updated.updated is True
    assert updated.pending_denominator_before == 2
    assert updated.consumed_denominator == 3
    assert updated.raw_estimate == pytest.approx(2.0)
    assert updated.ema_estimate == pytest.approx(2.0)
    state = controller.state_dict()
    assert state["pending_numerators"] == [0.0]
    assert state["pending_denominators"] == [0]


def test_denominator_aware_ema_uses_beta_power_support() -> None:
    controller = LagrangianDualController(
        _schema(ema_beta=0.9, budget=0.0, dual_learning_rate=1.0)
    )

    first = _update(
        controller,
        _estimate("drawdown_excess", 0.2, denominator=1),
    )
    second = _update(
        controller,
        _estimate("drawdown_excess", 0.6, denominator=4),
    )

    expected = (0.9**4) * 0.2 + (1.0 - 0.9**4) * 0.6
    assert first.ema_estimate == pytest.approx(0.2)
    assert second.ema_estimate == pytest.approx(expected)
    assert second.consumed_denominator == 4


def test_pooled_raw_estimate_is_invariant_to_rollout_partition() -> None:
    episodes = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    def pooled(partitions: tuple[int, ...]):
        controller = LagrangianDualController(
            _schema(minimum_completed_episodes=6, budget=0.0)
        )
        cursor = 0
        report = None
        for size in partitions:
            values = episodes[cursor : cursor + size]
            cursor += size
            report = _update(
                controller,
                ConstraintEstimate(
                    name="drawdown_excess",
                    numerator=float(values.sum()),
                    denominator=size,
                ),
            )
        assert cursor == len(episodes)
        assert report is not None
        return report, controller.state_dict()

    one_four, state_one_four = pooled((1, 4))
    two_three, state_two_three = pooled((2, 3))
    five, state_five = pooled((5,))

    for report in (one_four, two_three, five):
        assert report.updated is False
        assert report.skip_reason == "insufficient_completed_episodes"
        assert report.raw_estimate == pytest.approx(0.3)
    assert state_one_four["pending_numerators"] == pytest.approx([1.5])
    assert state_two_three["pending_numerators"] == pytest.approx([1.5])
    assert state_five["pending_numerators"] == pytest.approx([1.5])
    assert state_one_four["pending_denominators"] == [5]
    assert state_two_three["pending_denominators"] == [5]
    assert state_five["pending_denominators"] == [5]


def test_event_constraint_waits_for_minimum_episode_support() -> None:
    controller = LagrangianDualController(
        _schema(
            "drawdown_stop_event",
            budget=0.0,
            minimum_completed_episodes=20,
        )
    )

    insufficient = _update(
        controller,
        _estimate("drawdown_stop_event", 1.0 / 19.0, denominator=19),
    )
    updated = _update(
        controller,
        _estimate("drawdown_stop_event", 0.0, denominator=1),
    )

    assert insufficient.updated is False
    assert insufficient.skip_reason == "insufficient_completed_episodes"
    assert insufficient.raw_estimate == pytest.approx(1.0 / 19.0)
    assert updated.updated is True
    assert updated.consumed_denominator == 20
    assert updated.raw_estimate == pytest.approx(1.0 / 20.0)


def test_missing_current_estimate_can_consume_retained_pending_support() -> None:
    controller = LagrangianDualController(
        _schema(
            warmup_rollouts=1,
            minimum_completed_episodes=2,
            budget=0.0,
        )
    )

    warmup = _update(
        controller,
        _estimate("drawdown_excess", 0.4, denominator=2),
    )
    updated = _update(controller, None)

    assert warmup.skip_reason == "warmup"
    assert updated.updated is True
    assert updated.raw_estimate == pytest.approx(0.4)
    assert updated.consumed_denominator == 2
    assert updated.denominator is None


def test_missing_estimate_without_pending_support_is_explicit() -> None:
    controller = LagrangianDualController(_schema())

    report = _update(controller, None)

    assert report.updated is False
    assert report.skip_reason == "missing_estimate_or_pending_support"
    assert report.raw_estimate is None
    assert report.pending_denominator_before == 0
    assert report.consumed_denominator == 0


def test_successful_update_resets_pending_but_skipped_updates_retain_it() -> None:
    controller = LagrangianDualController(
        _schema(minimum_completed_episodes=3, budget=0.0)
    )

    first = _update(controller, _estimate("drawdown_excess", 0.2))
    second = _update(controller, _estimate("drawdown_excess", 0.4))
    before_update = controller.state_dict()
    third = _update(controller, _estimate("drawdown_excess", 0.6))
    after_update = controller.state_dict()

    assert first.skip_reason == "insufficient_completed_episodes"
    assert second.skip_reason == "insufficient_completed_episodes"
    assert before_update["pending_denominators"] == [2]
    assert third.updated is True
    assert third.raw_estimate == pytest.approx(0.4)
    assert after_update["pending_numerators"] == [0.0]
    assert after_update["pending_denominators"] == [0]


def test_validation_failure_leaves_controller_state_unchanged() -> None:
    controller = LagrangianDualController(_schema(minimum_completed_episodes=3))
    _update(controller, _estimate("drawdown_excess", 0.2))
    before = copy.deepcopy(controller.state_dict())

    with pytest.raises(ValueError, match="constraint names"):
        controller.update_after_rollout(
            {},
            censored_episode_count=7,
        )

    assert controller.state_dict() == before


def test_censored_episode_count_is_cumulative_and_reported() -> None:
    controller = LagrangianDualController(_schema(minimum_completed_episodes=2))

    first = _update(
        controller,
        _estimate("drawdown_excess", 0.2),
        censored_episode_count=3,
    )
    second = _update(
        controller,
        _estimate("drawdown_excess", 0.4),
        censored_episode_count=2,
    )

    assert first.censored_episode_count == 3
    assert second.censored_episode_count == 5
    assert controller.state_dict()["censored_episode_count"] == 5


def test_lower_bound_and_upper_cap_are_distinct() -> None:
    lower = LagrangianDualController(
        _schema(
            initial_multiplier=0.1,
            budget=1.0,
            dual_learning_rate=1.0,
        )
    )
    upper = LagrangianDualController(
        _schema(
            initial_multiplier=0.9,
            max_multiplier=1.0,
            budget=0.0,
            dual_learning_rate=1.0,
        )
    )

    lower_report = _update(lower, _estimate("drawdown_excess", 0.0))
    upper_report = _update(upper, _estimate("drawdown_excess", 10.0))

    assert lower_report.multiplier_after == pytest.approx(0.0)
    assert lower_report.at_lower_bound is True
    assert lower_report.at_upper_cap is False
    assert lower_report.saturated is False
    assert upper_report.multiplier_after == pytest.approx(1.0)
    assert upper_report.at_lower_bound is False
    assert upper_report.at_upper_cap is True
    assert upper_report.saturated is True


def test_dual_controller_updates_each_multiplier_independently() -> None:
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
        },
        censored_episode_count=0,
    )

    assert reports["drawdown_excess"].multiplier_after == pytest.approx(0.15)
    margin = reports["margin_deficit_fraction"]
    assert margin.multiplier_after == pytest.approx(1.0)
    assert margin.at_upper_cap is True
    assert controller.begin_rollout().tolist() == pytest.approx([0.15, 1.0])


def test_dual_controller_state_round_trip_reproduces_next_update() -> None:
    schema = LagrangianSchema(
        (
            _spec("drawdown_excess", budget=0.05, minimum_completed_episodes=3),
            _spec("drawdown_stop_event", budget=0.1, minimum_completed_episodes=3),
        )
    )
    original = LagrangianDualController(schema)
    original.update_after_rollout(
        {
            "drawdown_excess": _estimate("drawdown_excess", 0.2),
            "drawdown_stop_event": _estimate("drawdown_stop_event", 0.5),
        },
        censored_episode_count=2,
    )
    state = original.state_dict()
    restored = LagrangianDualController(schema)
    restored.load_state_dict(state)

    assert state["schema_version"] == "lagrangian_dual_controller_v2"
    assert restored.state_dict() == state
    next_estimates = {
        "drawdown_excess": _estimate("drawdown_excess", 0.1, denominator=2),
        "drawdown_stop_event": _estimate("drawdown_stop_event", 0.0, denominator=2),
    }
    assert restored.update_after_rollout(
        next_estimates,
        censored_episode_count=1,
    ) == original.update_after_rollout(
        next_estimates,
        censored_episode_count=1,
    )
    np.testing.assert_array_equal(
        restored.begin_rollout(),
        original.begin_rollout(),
    )


def test_old_controller_state_version_fails_closed() -> None:
    controller = LagrangianDualController(_schema())
    state = controller.state_dict()
    state["schema_version"] = "lagrangian_dual_controller_v1"

    with pytest.raises(ValueError, match="schema version"):
        controller.load_state_dict(state)


@pytest.mark.parametrize("minimum", [0, -1, True])
def test_minimum_completed_episodes_must_be_positive_integer(
    minimum: object,
) -> None:
    with pytest.raises(ValueError, match="minimum_completed_episodes"):
        _spec("drawdown_excess", minimum_completed_episodes=minimum)  # type: ignore[arg-type]


@pytest.mark.parametrize("censored", [-1, True, 1.5])
def test_censored_episode_count_must_be_non_negative_integer(
    censored: object,
) -> None:
    controller = LagrangianDualController(_schema())
    before = controller.state_dict()

    with pytest.raises(ValueError, match="censored_episode_count"):
        controller.update_after_rollout(
            {"drawdown_excess": _estimate("drawdown_excess", 0.1)},
            censored_episode_count=censored,  # type: ignore[arg-type]
        )

    assert controller.state_dict() == before
