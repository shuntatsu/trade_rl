from __future__ import annotations

import numpy as np
import pytest

from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import ConstraintCostVector
from trade_rl.rl.lagrangian_episode import EpisodeCompletionKind


def _costs(*, elapsed_hours: float) -> ConstraintCostVector:
    return ConstraintCostVector(
        drawdown_excess=0.0,
        drawdown_stop_event=0.0,
        margin_deficit_fraction=0.0,
        forced_liquidation_event=0.0,
        gross_exposure_request_excess=0.0,
        daily_turnover=0.0,
        execution_cost_fraction=0.0,
        funding_credit_fraction=0.0,
        transition_elapsed_hours=elapsed_hours,
    )


def test_required_storage_reads_elapsed_time_from_constraint_vector() -> None:
    storage = CostRolloutStorage(
        buffer_size=1,
        n_envs=2,
        schema=canonical_cost_learning_schema(),
        require_episode_metadata=True,
    )
    zeros = np.zeros((2, 7), dtype=np.float32)

    storage.add_from_infos(
        infos=(
            {
                "constraint_costs": _costs(elapsed_hours=0.25),
                "termination_reason": "margin_call",
            },
            {
                "constraint_costs": _costs(elapsed_hours=1.5),
                "termination_reason": "shadow_minimum_equity",
                "TimeLimit.truncated": True,
            },
        ),
        cost_values=zeros,
        terminated=np.asarray([True, False]),
        truncated=np.asarray([False, True]),
        terminal_cost_values=zeros,
    )

    np.testing.assert_array_equal(storage.elapsed_hours[0], [0.25, 1.5])
    np.testing.assert_array_equal(
        storage.completion_kinds[0],
        [
            EpisodeCompletionKind.ECONOMIC_TERMINATION,
            EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION,
        ],
    )


def test_explicit_info_elapsed_time_must_match_vector_source() -> None:
    storage = CostRolloutStorage(
        buffer_size=1,
        n_envs=1,
        schema=canonical_cost_learning_schema(),
        require_episode_metadata=True,
    )
    zeros = np.zeros((1, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="elapsed metadata mismatch"):
        storage.add_from_infos(
            infos=(
                {
                    "constraint_costs": _costs(elapsed_hours=0.5),
                    "transition_elapsed_hours": 1.0,
                    "termination_reason": None,
                },
            ),
            cost_values=zeros,
            terminated=np.asarray([False]),
            truncated=np.asarray([False]),
            terminal_cost_values=zeros,
        )
