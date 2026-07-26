from __future__ import annotations

import numpy as np

from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import ConstraintCostVector
from trade_rl.rl.lagrangian_episode import EpisodeCompletionKind


def _vector(value: float) -> ConstraintCostVector:
    return ConstraintCostVector(
        drawdown_excess=value,
        drawdown_stop_event=0.0,
        margin_deficit_fraction=0.0,
        forced_liquidation_event=0.0,
        gross_exposure_request_excess=0.0,
        daily_turnover=0.0,
        execution_cost_fraction=0.0,
        funding_credit_fraction=0.0,
    )


def test_cost_rollout_sampling_matches_sb3_env_major_flatten_order() -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

    storage = CostRolloutStorage(
        buffer_size=2,
        n_envs=2,
        schema=canonical_cost_learning_schema(),
    )
    zeros = np.zeros((2, 7), dtype=np.float32)
    storage.add_from_infos(
        infos=(
            {"constraint_costs": _vector(10.0)},
            {"constraint_costs": _vector(20.0)},
        ),
        cost_values=zeros,
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        terminal_cost_values=zeros,
    )
    storage.add_from_infos(
        infos=(
            {"constraint_costs": _vector(11.0)},
            {"constraint_costs": _vector(21.0)},
        ),
        cost_values=zeros,
        terminated=np.array([True, True]),
        truncated=np.array([False, False]),
        terminal_cost_values=zeros,
    )
    storage.finalize(last_cost_values=zeros)

    batch = storage.sample(np.arange(4, dtype=np.int64))

    # SB3 swaps [step, env, ...] to [env, step, ...] before flattening.
    np.testing.assert_array_equal(batch.costs[:, 0], [10.0, 11.0, 20.0, 21.0])


def test_completion_metadata_preserves_step_and_environment_alignment() -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

    storage = CostRolloutStorage(
        buffer_size=2,
        n_envs=2,
        schema=canonical_cost_learning_schema(),
        require_episode_metadata=True,
    )
    zeros = np.zeros((2, 7), dtype=np.float32)
    storage.add_from_infos(
        infos=(
            {
                "constraint_costs": _vector(10.0),
                "transition_elapsed_hours": 0.25,
                "termination_reason": None,
            },
            {
                "constraint_costs": _vector(20.0),
                "transition_elapsed_hours": 1.0,
                "termination_reason": None,
            },
        ),
        cost_values=zeros,
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        terminal_cost_values=zeros,
    )
    storage.add_from_infos(
        infos=(
            {
                "constraint_costs": _vector(11.0),
                "transition_elapsed_hours": 0.5,
                "termination_reason": "margin_call",
            },
            {
                "constraint_costs": _vector(21.0),
                "transition_elapsed_hours": 2.0,
                "termination_reason": "shadow_minimum_equity",
                "TimeLimit.truncated": True,
            },
        ),
        cost_values=zeros,
        terminated=np.array([True, False]),
        truncated=np.array([False, True]),
        terminal_cost_values=zeros,
    )

    np.testing.assert_array_equal(
        storage.elapsed_hours,
        np.asarray([[0.25, 1.0], [0.5, 2.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        storage.completion_kinds,
        np.asarray(
            [
                [EpisodeCompletionKind.NONE, EpisodeCompletionKind.NONE],
                [
                    EpisodeCompletionKind.ECONOMIC_TERMINATION,
                    EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION,
                ],
            ],
            dtype=np.int8,
        ),
    )
