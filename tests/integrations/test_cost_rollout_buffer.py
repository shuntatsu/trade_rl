from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import ConstraintCostVector
from trade_rl.rl.lagrangian_episode import EpisodeCompletionKind


def _cost_vector(scale: float) -> ConstraintCostVector:
    return ConstraintCostVector(
        drawdown_excess=0.1 * scale,
        drawdown_stop_event=1.0 if scale >= 2.0 else 0.0,
        margin_deficit_fraction=0.2 * scale,
        forced_liquidation_event=1.0 if scale >= 3.0 else 0.0,
        gross_exposure_request_excess=0.3 * scale,
        daily_turnover=0.4 * scale,
        execution_cost_fraction=0.5 * scale,
        funding_credit_fraction=0.6 * scale,
    )


def test_cost_rollout_storage_allocates_canonical_arrays_and_resets() -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

    schema = canonical_cost_learning_schema()
    storage = CostRolloutStorage(buffer_size=3, n_envs=2, schema=schema)

    expected_shape = (3, 2, len(schema.names))
    for array in (
        storage.costs,
        storage.values,
        storage.returns,
        storage.advantages,
        storage.terminal_values,
    ):
        assert array.shape == expected_shape
        assert array.dtype == np.float32
        assert not array.any()
    assert storage.terminated.shape == (3, 2)
    assert storage.truncated.shape == (3, 2)
    assert storage.terminated.dtype == np.bool_
    assert storage.truncated.dtype == np.bool_
    assert storage.elapsed_hours.shape == (3, 2)
    assert storage.elapsed_hours.dtype == np.float64
    assert np.isnan(storage.elapsed_hours).all()
    assert storage.completion_kinds.shape == (3, 2)
    assert storage.completion_kinds.dtype == np.int8
    assert not storage.completion_kinds.any()
    assert storage.cost_names == schema.names
    assert storage.pos == 0
    assert not storage.full
    assert not storage.finalized

    storage.costs.fill(3.0)
    storage.elapsed_hours.fill(2.0)
    storage.completion_kinds.fill(EpisodeCompletionKind.ECONOMIC_TERMINATION)
    storage.pos = 2
    storage.full = True
    storage.finalized = True
    storage.reset()

    assert storage.pos == 0
    assert not storage.full
    assert not storage.finalized
    assert not storage.costs.any()
    assert np.isnan(storage.elapsed_hours).all()
    assert not storage.completion_kinds.any()


def test_cost_rollout_storage_adds_compact_infos_in_schema_order() -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

    schema = canonical_cost_learning_schema()
    storage = CostRolloutStorage(buffer_size=2, n_envs=2, schema=schema)
    infos = (
        {"constraint_costs": _cost_vector(1.0)},
        {"constraint_costs": _cost_vector(2.0)},
    )
    values = np.arange(14, dtype=np.float32).reshape(2, 7)

    storage.add_from_infos(
        infos=infos,
        cost_values=values,
        terminated=np.array([False, True]),
        truncated=np.array([True, False]),
        terminal_cost_values=np.full((2, 7), 9.0, dtype=np.float32),
    )

    assert storage.pos == 1
    np.testing.assert_allclose(
        storage.costs[0, 0],
        np.array([0.1, 0.0, 0.2, 0.0, 0.3, 0.4, 0.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        storage.costs[0, 1],
        np.array([0.2, 1.0, 0.4, 0.0, 0.6, 0.8, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(storage.values[0], values)
    np.testing.assert_array_equal(storage.terminated[0], [False, True])
    np.testing.assert_array_equal(storage.truncated[0], [True, False])
    np.testing.assert_array_equal(storage.terminal_values[0, 0], np.full(7, 9.0))
    np.testing.assert_array_equal(storage.terminal_values[0, 1], np.zeros(7))
    assert np.isnan(storage.elapsed_hours[0]).all()
    np.testing.assert_array_equal(
        storage.completion_kinds[0],
        [EpisodeCompletionKind.NONE, EpisodeCompletionKind.NONE],
    )


def test_required_episode_metadata_preserves_vector_environment_order() -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

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
                "constraint_costs": _cost_vector(1.0),
                "transition_elapsed_hours": 0.25,
                "termination_reason": "margin_call",
            },
            {
                "constraint_costs": _cost_vector(2.0),
                "transition_elapsed_hours": 1.0,
                "termination_reason": "shadow_minimum_equity",
                "TimeLimit.truncated": True,
            },
        ),
        cost_values=zeros,
        terminated=np.asarray([True, False]),
        truncated=np.asarray([False, True]),
        terminal_cost_values=zeros,
    )

    np.testing.assert_array_equal(storage.elapsed_hours[0], [0.25, 1.0])
    np.testing.assert_array_equal(
        storage.completion_kinds[0],
        [
            EpisodeCompletionKind.ECONOMIC_TERMINATION,
            EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION,
        ],
    )


@pytest.mark.parametrize("elapsed", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_required_episode_metadata_rejects_missing_or_invalid_elapsed_time(
    elapsed: float | None,
) -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

    storage = CostRolloutStorage(
        buffer_size=1,
        n_envs=1,
        schema=canonical_cost_learning_schema(),
        require_episode_metadata=True,
    )
    info: dict[str, object] = {
        "constraint_costs": _cost_vector(1.0),
        "termination_reason": None,
    }
    if elapsed is not None:
        info["transition_elapsed_hours"] = elapsed
    zeros = np.zeros((1, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="transition_elapsed_hours"):
        storage.add_from_infos(
            infos=(info,),
            cost_values=zeros,
            terminated=np.asarray([False]),
            truncated=np.asarray([False]),
            terminal_cost_values=zeros,
        )


def test_cost_rollout_storage_fails_closed_on_invalid_info_or_transition() -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

    storage = CostRolloutStorage(
        buffer_size=1,
        n_envs=1,
        schema=canonical_cost_learning_schema(),
    )
    valid = dict(
        infos=({"constraint_costs": _cost_vector(1.0)},),
        cost_values=np.zeros((1, 7), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        terminal_cost_values=np.zeros((1, 7), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="constraint_costs"):
        storage.add_from_infos(**{**valid, "infos": ({},)})
    with pytest.raises(ValueError, match="ConstraintCostVector"):
        storage.add_from_infos(**{**valid, "infos": ({"constraint_costs": object()},)})
    with pytest.raises(ValueError, match="both terminate and truncate"):
        storage.add_from_infos(
            **{
                **valid,
                "terminated": np.array([True]),
                "truncated": np.array([True]),
            }
        )
    bad_values = np.zeros((1, 7), dtype=np.float32)
    bad_values[0, 0] = np.nan
    with pytest.raises(ValueError, match="cost_values"):
        storage.add_from_infos(**{**valid, "cost_values": bad_values})


def test_cost_rollout_storage_finalizes_and_samples_independent_costs() -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

    schema = canonical_cost_learning_schema(
        continuous_gae_lambda=0.0,
        event_gae_lambda=1.0,
    )
    storage = CostRolloutStorage(buffer_size=2, n_envs=1, schema=schema)
    zeros = np.zeros((1, 7), dtype=np.float32)
    storage.add_from_infos(
        infos=({"constraint_costs": _cost_vector(1.0)},),
        cost_values=zeros,
        terminated=np.array([False]),
        truncated=np.array([False]),
        terminal_cost_values=zeros,
    )
    storage.add_from_infos(
        infos=({"constraint_costs": _cost_vector(2.0)},),
        cost_values=zeros,
        terminated=np.array([True]),
        truncated=np.array([False]),
        terminal_cost_values=zeros,
    )

    storage.finalize(last_cost_values=zeros)

    assert storage.full
    assert storage.finalized
    np.testing.assert_allclose(storage.advantages[:, 0, 0], [0.1, 0.2])
    np.testing.assert_allclose(storage.advantages[:, 0, 1], [1.0, 1.0])
    batch = storage.sample(np.array([1, 0], dtype=np.int64))
    assert batch.cost_names == schema.names
    assert batch.costs.shape == (2, 7)
    assert batch.old_cost_values.shape == (2, 7)
    assert batch.cost_advantages.shape == (2, 7)
    assert batch.cost_returns.shape == (2, 7)
    np.testing.assert_allclose(batch.costs[0], storage.costs[1, 0])
    np.testing.assert_allclose(batch.costs[1], storage.costs[0, 0])


def test_cost_rollout_storage_requires_full_rollout_and_valid_sample_indices() -> None:
    from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage

    storage = CostRolloutStorage(
        buffer_size=2,
        n_envs=1,
        schema=canonical_cost_learning_schema(),
    )
    zeros = np.zeros((1, 7), dtype=np.float32)
    storage.add_from_infos(
        infos=({"constraint_costs": _cost_vector(1.0)},),
        cost_values=zeros,
        terminated=np.array([False]),
        truncated=np.array([False]),
        terminal_cost_values=zeros,
    )

    with pytest.raises(RuntimeError, match="full rollout"):
        storage.finalize(last_cost_values=zeros)
    with pytest.raises(RuntimeError, match="finalized"):
        storage.sample(np.array([0], dtype=np.int64))

    storage.add_from_infos(
        infos=({"constraint_costs": _cost_vector(1.0)},),
        cost_values=zeros,
        terminated=np.array([True]),
        truncated=np.array([False]),
        terminal_cost_values=zeros,
    )
    storage.finalize(last_cost_values=zeros)
    with pytest.raises(IndexError, match="sample index"):
        storage.sample(np.array([2], dtype=np.int64))
    with pytest.raises(RuntimeError, match="full"):
        storage.add_from_infos(
            infos=({"constraint_costs": _cost_vector(1.0)},),
            cost_values=zeros,
            terminated=np.array([False]),
            truncated=np.array([False]),
            terminal_cost_values=zeros,
        )


def test_cost_rollout_memory_estimator_counts_all_cost_state() -> None:
    from trade_rl.integrations.cost_rollout_buffer import (
        estimate_cost_rollout_storage_bytes,
    )

    transitions = 4 * 3
    # Five float32 cost tensors, two bool masks, one float64 elapsed-time matrix,
    # and one int8 completion-kind matrix.
    expected = 4 * 3 * 7 * 5 * 4 + transitions * 2 + transitions * 8 + transitions
    assert estimate_cost_rollout_storage_bytes(4, 3, 7) == expected

    with pytest.raises(ValueError, match="positive"):
        estimate_cost_rollout_storage_bytes(0, 3, 7)
