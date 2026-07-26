from __future__ import annotations

import numpy as np
import pytest

from trade_rl.integrations.cost_rollout_buffer import (
    CostRolloutStorage,
    estimate_cost_rollout_storage_bytes,
)
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import ConstraintCostVector
from trade_rl.rl.lagrangian_episode_estimator import CompletionKind


def _costs() -> ConstraintCostVector:
    return ConstraintCostVector(
        drawdown_excess=0.1,
        drawdown_stop_event=0.0,
        margin_deficit_fraction=0.2,
        forced_liquidation_event=0.0,
        gross_exposure_request_excess=0.3,
        daily_turnover=0.4,
        execution_cost_fraction=0.5,
        funding_credit_fraction=0.0,
    )


def _storage() -> CostRolloutStorage:
    return CostRolloutStorage(
        buffer_size=2,
        n_envs=2,
        schema=canonical_cost_learning_schema(),
        store_episode_metadata=True,
    )


def test_opt_in_rollout_storage_allocates_compact_episode_metadata() -> None:
    storage = _storage()

    assert storage.transition_elapsed_hours.shape == (2, 2)
    assert storage.transition_elapsed_hours.dtype == np.float64
    assert storage.completion_kind_codes.shape == (2, 2)
    assert storage.completion_kind_codes.dtype == np.uint8
    assert np.isnan(storage.transition_elapsed_hours).all()
    assert storage.completion_kinds.tolist() == [
        [CompletionKind.NONE, CompletionKind.NONE],
        [CompletionKind.NONE, CompletionKind.NONE],
    ]

    storage.reset()
    assert np.isnan(storage.transition_elapsed_hours).all()
    assert not storage.completion_kind_codes.any()


def test_opt_in_rollout_storage_classifies_explicit_completion_metadata() -> None:
    storage = _storage()
    zeros = np.zeros((2, 7), dtype=np.float32)
    infos = (
        {
            "constraint_costs": _costs(),
            "transition_elapsed_hours": 6.0,
            "termination_reason": None,
        },
        {
            "constraint_costs": _costs(),
            "transition_elapsed_hours": 18.0,
            "termination_reason": "shadow_minimum_equity",
        },
    )

    storage.add_from_infos(
        infos=infos,
        cost_values=zeros,
        terminated=np.asarray([False, False]),
        truncated=np.asarray([False, True]),
        terminal_cost_values=zeros,
    )

    np.testing.assert_array_equal(storage.transition_elapsed_hours[0], [6.0, 18.0])
    assert storage.completion_kinds[0].tolist() == [
        CompletionKind.NONE,
        CompletionKind.CENSORED_EXTERNAL_TRUNCATION,
    ]

    storage.add_from_infos(
        infos=(
            {
                "constraint_costs": _costs(),
                "transition_elapsed_hours": 12.0,
                "termination_reason": "drawdown_stop",
            },
            {
                "constraint_costs": _costs(),
                "transition_elapsed_hours": 24.0,
                "termination_reason": None,
                "TimeLimit.truncated": True,
            },
        ),
        cost_values=zeros,
        terminated=np.asarray([True, False]),
        truncated=np.asarray([False, True]),
        terminal_cost_values=zeros,
    )

    assert storage.completion_kinds[1].tolist() == [
        CompletionKind.ECONOMIC_TERMINATION,
        CompletionKind.TIME_LIMIT_COMPLETION,
    ]


def test_opt_in_episode_metadata_validation_is_transactional() -> None:
    storage = _storage()
    zeros = np.zeros((2, 7), dtype=np.float32)
    before_costs = storage.costs.copy()

    with pytest.raises(ValueError, match="transition_elapsed_hours"):
        storage.add_from_infos(
            infos=(
                {"constraint_costs": _costs(), "termination_reason": None},
                {
                    "constraint_costs": _costs(),
                    "transition_elapsed_hours": 1.0,
                    "termination_reason": None,
                },
            ),
            cost_values=zeros,
            terminated=np.asarray([False, False]),
            truncated=np.asarray([False, False]),
            terminal_cost_values=zeros,
        )

    assert storage.pos == 0
    np.testing.assert_array_equal(storage.costs, before_costs)
    assert np.isnan(storage.transition_elapsed_hours).all()

    with pytest.raises(ValueError, match="completion"):
        storage.add_from_infos(
            infos=(
                {
                    "constraint_costs": _costs(),
                    "transition_elapsed_hours": 1.0,
                    "termination_reason": "mystery",
                },
                {
                    "constraint_costs": _costs(),
                    "transition_elapsed_hours": 1.0,
                    "termination_reason": None,
                },
            ),
            cost_values=zeros,
            terminated=np.asarray([False, False]),
            truncated=np.asarray([True, False]),
            terminal_cost_values=zeros,
        )
    assert storage.pos == 0


def test_episode_metadata_memory_is_opt_in_and_exact() -> None:
    legacy = estimate_cost_rollout_storage_bytes(4, 3, 7)
    enriched = estimate_cost_rollout_storage_bytes(
        4,
        3,
        7,
        store_episode_metadata=True,
    )

    assert enriched - legacy == 4 * 3 * (8 + 1)
