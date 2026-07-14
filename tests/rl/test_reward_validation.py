from __future__ import annotations

import math

import pytest

from trade_rl.rl.rewards import (
    AbsoluteGrowthRewardConfig,
    RewardBreakdown,
    RewardConfig,
    RewardContext,
    RewardTracker,
    absolute_growth_reward,
    build_reward_context,
    drawdown_severity,
    relative_interval_reward,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": math.inf},
        {"scale": 0.0},
        {"absolute_growth_weight": -1.0},
        {"baseline_progressive_power": 0.5},
        {"baseline_window_hours": 0.0},
        {"baseline_minimum_history_hours": 0.0},
        {"baseline_window_hours": 720.0, "baseline_minimum_history_hours": 721.0},
        {"baseline_window_steps": True},
        {"baseline_window_steps": 0},
        {"baseline_minimum_history_steps": 0},
        {"baseline_window_steps": 2, "baseline_minimum_history_steps": 3},
        {"equity_floor_fraction": 0.0},
        {"equity_floor_fraction": 1.1},
    ],
)
def test_reward_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RewardConfig(**kwargs)  # type: ignore[arg-type]


def test_reward_config_conversion_preserves_public_contract() -> None:
    contract = AbsoluteGrowthRewardConfig(
        scale=7.0,
        baseline_tolerance=0.02,
        baseline_penalty_weight=0.3,
        drawdown_penalty_weight=0.4,
    )
    config = RewardConfig.from_absolute_growth(contract)
    assert config.absolute_growth_contract() == contract


def test_reward_breakdown_rejects_nonfinite_component() -> None:
    values = dict.fromkeys(RewardBreakdown.__dataclass_fields__, 0.0)
    values["scaled_total"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        RewardBreakdown(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, -0.1, 1.1])
def test_drawdown_severity_rejects_invalid_drawdown(value: float) -> None:
    with pytest.raises(ValueError, match="drawdown"):
        drawdown_severity(value, RewardConfig())


def test_reward_context_rejects_invalid_history_sizes_and_returns() -> None:
    config = AbsoluteGrowthRewardConfig()
    for window, minimum in ((0, 1), (1, 0), (1, 2)):
        with pytest.raises(ValueError):
            build_reward_context(
                hybrid_returns=(),
                shadow_returns=(),
                hybrid_drawdown=0.0,
                window_bars=window,
                minimum_history_bars=minimum,
                config=config,
            )
    with pytest.raises(ValueError, match="greater than -1"):
        build_reward_context(
            hybrid_returns=(-1.0,),
            shadow_returns=(0.0,),
            hybrid_drawdown=0.0,
            window_bars=1,
            minimum_history_bars=1,
            config=config,
        )


def test_absolute_reward_rejects_nonfinite_growth() -> None:
    context = RewardContext(0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 0)
    with pytest.raises(ValueError, match="finite"):
        absolute_growth_reward(
            hybrid_log_return=math.nan,
            before=context,
            after=context,
            config=AbsoluteGrowthRewardConfig(),
        )


def test_tracker_validates_constructor_reset_and_step_inputs() -> None:
    with pytest.raises(ValueError, match="decision_hours"):
        RewardTracker(decision_hours=0.0)
    tracker = RewardTracker(
        RewardConfig(
            baseline_window_steps=3,
            baseline_minimum_history_steps=2,
            baseline_progressive_power=2.0,
        )
    )
    with pytest.raises(ValueError, match="finite log returns"):
        tracker.reset(hybrid_history=(math.nan,), shadow_history=(0.0,))
    for kwargs in (
        {"hybrid_log_return": math.nan},
        {"projection_distance": -1.0},
        {"hybrid_margin_deficit_fraction": -1.0},
        {"hybrid_equity_fraction": -1.0},
        {"hybrid_drawdown": 1.1},
    ):
        values = {
            "hybrid_log_return": 0.0,
            "shadow_log_return": 0.0,
            "hybrid_drawdown": 0.0,
            "shadow_drawdown": 0.0,
        }
        values.update(kwargs)
        with pytest.raises(ValueError):
            tracker.step(**values)


def test_tracker_progressive_hinge_terminal_margin_and_projection_components() -> None:
    tracker = RewardTracker(
        RewardConfig(
            baseline_window_steps=2,
            baseline_minimum_history_steps=1,
            baseline_tolerance=0.01,
            baseline_progressive_power=2.0,
            projection_penalty_weight=0.5,
        )
    )
    result = tracker.step(
        hybrid_log_return=-0.03,
        shadow_log_return=0.0,
        hybrid_drawdown=0.12,
        shadow_drawdown=0.0,
        projection_distance=0.2,
        hybrid_margin_deficit_fraction=0.1,
        hybrid_equity_fraction=0.5,
        hybrid_terminated=True,
    )
    assert result.baseline_penalty > 0.0
    assert result.terminal_penalty == pytest.approx(-math.log(0.5))
    assert result.margin_penalty == pytest.approx(0.1)
    assert result.projection_penalty == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("hybrid_terminated", "shadow_terminated", "expected"),
    [(True, False, -10.0), (False, True, 10.0)],
)
def test_relative_reward_terminal_compatibility(
    hybrid_terminated: bool, shadow_terminated: bool, expected: float
) -> None:
    assert relative_interval_reward(
        hybrid_log_return=0.0,
        shadow_log_return=0.0,
        scale=10.0,
        hybrid_terminated=hybrid_terminated,
        shadow_terminated=shadow_terminated,
        hybrid_drawdown=0.0,
        shadow_drawdown=0.0,
    ) == pytest.approx(expected)


def test_relative_reward_validation_and_penalties() -> None:
    with pytest.raises(ValueError):
        relative_interval_reward(
            hybrid_log_return=math.nan,
            shadow_log_return=0.0,
            scale=1.0,
            hybrid_terminated=False,
            shadow_terminated=False,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
        )
    with pytest.raises(ValueError, match="positive"):
        relative_interval_reward(
            hybrid_log_return=0.0,
            shadow_log_return=0.0,
            scale=0.0,
            hybrid_terminated=False,
            shadow_terminated=False,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        relative_interval_reward(
            hybrid_log_return=0.0,
            shadow_log_return=0.0,
            scale=1.0,
            hybrid_terminated=False,
            shadow_terminated=False,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
            downside_penalty=-1.0,
        )
    value = relative_interval_reward(
        hybrid_log_return=-0.02,
        shadow_log_return=-0.01,
        scale=100.0,
        hybrid_terminated=False,
        shadow_terminated=False,
        hybrid_drawdown=0.2,
        shadow_drawdown=0.1,
        downside_penalty=0.5,
        excess_drawdown_penalty=0.25,
    )
    assert value == pytest.approx(100.0 * (-0.01 - 0.01 - 0.025))
