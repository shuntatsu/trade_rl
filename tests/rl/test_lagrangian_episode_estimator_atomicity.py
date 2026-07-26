from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import canonical_lagrangian_schema
from trade_rl.rl.lagrangian_episode_estimator import (
    CompletionKind,
    TimeAwareCompletedEpisodeCostAccumulator,
)


def _accumulator() -> TimeAwareCompletedEpisodeCostAccumulator:
    count = len(CONSTRAINT_COST_NAMES)
    schema = canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(0.0,) * count,
        dual_learning_rates=(0.1,) * count,
        ema_betas=(0.9,) * count,
        initial_multipliers=(0.0,) * count,
        max_multipliers=(10.0,) * count,
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
    )
    return TimeAwareCompletedEpisodeCostAccumulator(n_envs=1, schema=schema)


def test_failed_rollout_ingestion_preserves_committed_episode_state() -> None:
    accumulator = _accumulator()
    accumulator.ingest_rollout(
        costs=np.asarray([[[0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        transition_elapsed_hours=np.asarray([[6.0]]),
        completion_kinds=np.asarray([[CompletionKind.NONE]], dtype=object),
    )
    before = accumulator.state_dict()

    with pytest.raises(ValueError, match="occurred more than once"):
        accumulator.ingest_rollout(
            costs=np.asarray([[[0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
            transition_elapsed_hours=np.asarray([[18.0]]),
            completion_kinds=np.asarray(
                [[CompletionKind.ECONOMIC_TERMINATION]],
                dtype=object,
            ),
        )

    assert accumulator.state_dict() == before
