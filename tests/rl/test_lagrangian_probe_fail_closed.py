from __future__ import annotations

from typing import Any

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
    CanonicalActionProbeEvidence,
    CanonicalActionSemantic,
    run_canonical_action_feasibility_probe,
)


def _schema():
    count = len(CONSTRAINT_COST_NAMES)
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(1.0,) * count,
        dual_learning_rates=(0.1,) * count,
        ema_betas=(0.0,) * count,
        initial_multipliers=(0.0,) * count,
        max_multipliers=(10.0,) * count,
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
        minimum_completed_episodes=(1,) * count,
    )


def _costs() -> ConstraintCostVector:
    return ConstraintCostVector(
        drawdown_excess=0.0,
        drawdown_stop_event=0.0,
        margin_deficit_fraction=0.0,
        forced_liquidation_event=0.0,
        gross_exposure_request_excess=0.0,
        daily_turnover=0.0,
        execution_cost_fraction=0.0,
        funding_credit_fraction=0.0,
        transition_elapsed_hours=1.0,
    )


class _BoundaryEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        action_space: spaces.Space[Any] | None = None,
        completion: str = "terminated",
        elapsed: float = 1.0,
        reset_seeds: list[int | None] | None = None,
        close_calls: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.action_spec = ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=3,
        )
        self.action_space = (
            spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            if action_space is None
            else action_space
        )
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.completion = completion
        self.elapsed = elapsed
        self.reset_seeds = [] if reset_seeds is None else reset_seeds
        self.close_calls = [] if close_calls is None else close_calls

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        terminated = self.completion == "terminated"
        truncated = self.completion in {"censored", "native_time_limit", "unknown"}
        info: dict[str, object] = {
            "constraint_costs": _costs(),
            "transition_elapsed_hours": self.elapsed,
        }
        if self.completion == "censored":
            info["TimeLimit.truncated"] = True
            info["termination_reason"] = "shadow_probe_interrupt"
        elif self.completion == "unknown":
            info["TimeLimit.truncated"] = True
            info["termination_reason"] = "unexpected_probe_reason"
        elif self.completion == "terminated":
            info["termination_reason"] = "probe_complete"
        return np.zeros(2, dtype=np.float32), 0.0, terminated, truncated, info

    def close(self) -> None:
        self.close_calls.append(1)


def test_probe_rejects_action_space_shape_mismatch_and_closes() -> None:
    closes: list[int] = []

    with pytest.raises(ValueError, match="does not match ActionSpec"):
        run_canonical_action_feasibility_probe(
            environment_factory=lambda: _BoundaryEnvironment(
                action_space=spaces.Box(
                    -1.0,
                    1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
                close_calls=closes,
            ),
            schema=_schema(),
            episode_count=1,
            max_steps_per_episode=2,
        )

    assert closes == [1]


def test_probe_accepts_native_gymnasium_truncation_without_legacy_sb3_key() -> None:
    closes: list[int] = []

    evidence = run_canonical_action_feasibility_probe(
        environment_factory=lambda: _BoundaryEnvironment(
            completion="native_time_limit",
            close_calls=closes,
        ),
        schema=_schema(),
        episode_count=1,
        max_steps_per_episode=2,
    )

    assert evidence.completed_episode_count == 1
    assert evidence.censored_episode_count == 0
    assert closes == [1]


def test_probe_rejects_discrete_action_space_and_closes() -> None:
    closes: list[int] = []

    with pytest.raises(ValueError, match="Box action space"):
        run_canonical_action_feasibility_probe(
            environment_factory=lambda: _BoundaryEnvironment(
                action_space=spaces.Discrete(3),
                close_calls=closes,
            ),
            schema=_schema(),
            episode_count=1,
            max_steps_per_episode=2,
        )

    assert closes == [1]


def test_probe_rejects_non_finite_elapsed_time() -> None:
    closes: list[int] = []

    with pytest.raises(ValueError, match="transition_elapsed_hours"):
        run_canonical_action_feasibility_probe(
            environment_factory=lambda: _BoundaryEnvironment(
                elapsed=float("nan"),
                close_calls=closes,
            ),
            schema=_schema(),
            episode_count=1,
            max_steps_per_episode=2,
        )

    assert closes == [1]


def test_probe_rejects_unknown_completion_reason() -> None:
    closes: list[int] = []

    with pytest.raises(ValueError, match="unknown truncation reason"):
        run_canonical_action_feasibility_probe(
            environment_factory=lambda: _BoundaryEnvironment(
                completion="unknown",
                close_calls=closes,
            ),
            schema=_schema(),
            episode_count=1,
            max_steps_per_episode=2,
        )

    assert closes == [1]


def test_probe_rejects_episode_that_never_completes() -> None:
    closes: list[int] = []

    with pytest.raises(ValueError, match="did not complete within the step limit"):
        run_canonical_action_feasibility_probe(
            environment_factory=lambda: _BoundaryEnvironment(
                completion="none",
                close_calls=closes,
            ),
            schema=_schema(),
            episode_count=1,
            max_steps_per_episode=2,
        )

    assert closes == [1]


def test_censored_episode_does_not_satisfy_probe_completion_count() -> None:
    attempts: list[str] = []
    closes: list[int] = []

    def factory() -> _BoundaryEnvironment:
        completion = "censored" if not attempts else "terminated"
        attempts.append(completion)
        return _BoundaryEnvironment(completion=completion, close_calls=closes)

    evidence = run_canonical_action_feasibility_probe(
        environment_factory=factory,
        schema=_schema(),
        episode_count=1,
        max_steps_per_episode=2,
    )

    assert attempts == ["censored", "terminated"]
    assert closes == [1, 1]
    assert evidence.completed_episode_count == 1
    assert evidence.censored_episode_count == 1


def test_probe_uses_completed_episode_index_as_deterministic_seed() -> None:
    seeds: list[int | None] = []

    evidence = run_canonical_action_feasibility_probe(
        environment_factory=lambda: _BoundaryEnvironment(reset_seeds=seeds),
        schema=_schema(),
        episode_count=2,
        max_steps_per_episode=2,
    )

    assert evidence.completed_episode_count == 2
    assert seeds == [0, 1]


def _evidence(
    *,
    semantic: CanonicalActionSemantic = CanonicalActionSemantic.TARGET_WEIGHT_CASH,
    action: np.ndarray | None = None,
    estimate: float = 0.1,
    denominator: int = 1,
    budget: float = 1.0,
    completed: int = 1,
    censored: int = 0,
    warning: bool = False,
) -> CanonicalActionProbeEvidence:
    violated = ("cost",) if warning else ()
    return CanonicalActionProbeEvidence(
        action_semantic=semantic,
        action=(
            np.zeros(2, dtype=np.float32)
            if action is None
            else np.asarray(action, dtype=np.float32)
        ),
        estimates={"cost": estimate},
        denominators={"cost": denominator},
        budgets={"cost": budget},
        violated_costs=violated,
        completed_episode_count=completed,
        censored_episode_count=censored,
        episode_count=completed,
        max_steps_per_episode=8,
        warning=warning,
    )


def test_probe_digest_tracks_every_evidence_semantic() -> None:
    baseline = _evidence()
    variants = (
        _evidence(semantic=CanonicalActionSemantic.RESIDUAL_BASELINE),
        _evidence(action=np.asarray([0.0, 0.5], dtype=np.float32)),
        _evidence(estimate=0.2),
        _evidence(denominator=2),
        _evidence(budget=2.0),
        _evidence(completed=2),
        _evidence(censored=1),
        _evidence(estimate=2.0, warning=True),
    )

    assert baseline.action.flags.writeable is False
    assert len({baseline.digest, *(variant.digest for variant in variants)}) == 9


def test_probe_evidence_rejects_inconsistent_warning_flag() -> None:
    with pytest.raises(ValueError, match="warning"):
        CanonicalActionProbeEvidence(
            action_semantic=CanonicalActionSemantic.TARGET_WEIGHT_CASH,
            action=np.zeros(1, dtype=np.float32),
            estimates={"cost": 0.1},
            denominators={"cost": 1},
            budgets={"cost": 1.0},
            violated_costs=(),
            completed_episode_count=1,
            censored_episode_count=0,
            episode_count=1,
            max_steps_per_episode=1,
            warning=True,
        )
