from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.rl.actions import ActionMode, ActionSpec
from trade_rl.rl.environment_constraints import (
    CONSTRAINT_COST_NAMES,
    ConstraintCostVector,
)
from trade_rl.rl.lagrangian import canonical_lagrangian_schema
from trade_rl.rl.lagrangian_probe import (
    CanonicalActionSemantic,
    run_canonical_action_feasibility_probe,
)


def _schema(*, budget: float = 1.0):
    count = len(CONSTRAINT_COST_NAMES)
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(budget,) * count,
        dual_learning_rates=(0.1,) * count,
        ema_betas=(0.0,) * count,
        initial_multipliers=(0.0,) * count,
        max_multipliers=(10.0,) * count,
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
        minimum_completed_episodes=(1,) * count,
    )


def _costs(*, value: float = 0.0, elapsed: float = 1.0) -> ConstraintCostVector:
    return ConstraintCostVector(
        drawdown_excess=value,
        drawdown_stop_event=0.0,
        margin_deficit_fraction=value,
        forced_liquidation_event=0.0,
        gross_exposure_request_excess=value,
        daily_turnover=value,
        execution_cost_fraction=value,
        funding_credit_fraction=0.0,
        transition_elapsed_hours=elapsed,
    )


class _ProbeEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        mode: ActionMode,
        recorded_actions: list[np.ndarray],
        close_calls: list[int],
        cost_value: float = 0.0,
        malformed_info: Callable[[dict[str, object]], dict[str, object]] | None = None,
    ) -> None:
        super().__init__()
        self.action_spec = (
            ActionSpec(
                mode=ActionMode.TARGET_WEIGHT,
                risk_tilt_enabled=False,
                target_weight_count=3,
            )
            if mode is ActionMode.TARGET_WEIGHT
            else ActionSpec(mode=ActionMode.RESIDUAL)
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.action_spec.size,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.recorded_actions = recorded_actions
        self.close_calls = close_calls
        self.cost_value = cost_value
        self.malformed_info = malformed_info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.recorded_actions.append(np.asarray(action).copy())
        info: dict[str, object] = {
            "constraint_costs": _costs(value=self.cost_value),
            "transition_elapsed_hours": 1.0,
            "termination_reason": "probe_complete",
        }
        if self.malformed_info is not None:
            info = self.malformed_info(info)
        return np.zeros(2, dtype=np.float32), 0.0, True, False, info

    def close(self) -> None:
        self.close_calls.append(1)


def _factory(
    *,
    mode: ActionMode,
    recorded_actions: list[np.ndarray],
    close_calls: list[int],
    cost_value: float = 0.0,
    malformed_info: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> Callable[[], _ProbeEnvironment]:
    return lambda: _ProbeEnvironment(
        mode=mode,
        recorded_actions=recorded_actions,
        close_calls=close_calls,
        cost_value=cost_value,
        malformed_info=malformed_info,
    )


def test_target_weight_zero_action_is_recorded_as_cash() -> None:
    actions: list[np.ndarray] = []
    closes: list[int] = []

    evidence = run_canonical_action_feasibility_probe(
        environment_factory=_factory(
            mode=ActionMode.TARGET_WEIGHT,
            recorded_actions=actions,
            close_calls=closes,
        ),
        schema=_schema(),
        episode_count=2,
        max_steps_per_episode=4,
    )

    assert evidence.action_semantic is CanonicalActionSemantic.TARGET_WEIGHT_CASH
    assert evidence.action.tolist() == [0.0, 0.0, 0.0]
    assert len(actions) == 2
    assert all(action.tolist() == [0.0, 0.0, 0.0] for action in actions)
    assert evidence.completed_episode_count == 2
    assert evidence.censored_episode_count == 0
    assert evidence.warning is False
    assert evidence.violated_costs == ()
    assert len(closes) == 2


def test_residual_zero_action_is_recorded_as_baseline() -> None:
    actions: list[np.ndarray] = []
    closes: list[int] = []

    evidence = run_canonical_action_feasibility_probe(
        environment_factory=_factory(
            mode=ActionMode.RESIDUAL,
            recorded_actions=actions,
            close_calls=closes,
        ),
        schema=_schema(),
        episode_count=1,
        max_steps_per_episode=4,
    )

    assert evidence.action_semantic is CanonicalActionSemantic.RESIDUAL_BASELINE
    assert evidence.action.tolist() == [0.0, 0.0, 0.0]
    assert actions[0].tolist() == [0.0, 0.0, 0.0]
    assert len(closes) == 1


def test_probe_budget_violation_is_warning_evidence() -> None:
    actions: list[np.ndarray] = []
    closes: list[int] = []

    evidence = run_canonical_action_feasibility_probe(
        environment_factory=_factory(
            mode=ActionMode.TARGET_WEIGHT,
            recorded_actions=actions,
            close_calls=closes,
            cost_value=12.0,
        ),
        schema=_schema(budget=0.1),
        episode_count=2,
        max_steps_per_episode=4,
    )

    assert evidence.warning is True
    assert set(evidence.violated_costs) == {
        "drawdown_excess",
        "margin_deficit_fraction",
        "gross_exposure_request_excess",
        "daily_turnover",
        "execution_cost_fraction",
    }
    assert len(evidence.digest) == 64
    assert evidence.digest_payload()["warning"] is True


def test_parallel_probe_preserves_serial_evidence_digest() -> None:
    serial = run_canonical_action_feasibility_probe(
        environment_factory=_factory(
            mode=ActionMode.TARGET_WEIGHT,
            recorded_actions=[],
            close_calls=[],
            cost_value=0.25,
        ),
        schema=_schema(),
        episode_count=8,
        max_steps_per_episode=4,
        max_workers=1,
    )
    parallel = run_canonical_action_feasibility_probe(
        environment_factory=_factory(
            mode=ActionMode.TARGET_WEIGHT,
            recorded_actions=[],
            close_calls=[],
            cost_value=0.25,
        ),
        schema=_schema(),
        episode_count=8,
        max_steps_per_episode=4,
        max_workers=8,
    )

    assert parallel.digest == serial.digest


def test_probe_fails_closed_on_missing_constraint_costs_and_closes_environment() -> (
    None
):
    actions: list[np.ndarray] = []
    closes: list[int] = []

    def remove_costs(info: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in info.items() if key != "constraint_costs"}

    with pytest.raises(ValueError, match="constraint_costs"):
        run_canonical_action_feasibility_probe(
            environment_factory=_factory(
                mode=ActionMode.TARGET_WEIGHT,
                recorded_actions=actions,
                close_calls=closes,
                malformed_info=remove_costs,
            ),
            schema=_schema(),
            episode_count=1,
            max_steps_per_episode=4,
        )

    assert len(closes) == 1
