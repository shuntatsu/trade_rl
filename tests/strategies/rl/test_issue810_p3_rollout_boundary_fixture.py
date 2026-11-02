from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np
import pytest

from tools.issue810_p3_rollout_boundary_fixture import (
    ALL_WINDOW_CUES,
    BLOCK_TRANSITIONS,
    DATASET_ROW_COUNT,
    EVENT_RETURN,
    FEE_RATE,
    IMPACT_RATE,
    INITIAL_CAPITAL,
    MAX_PARTICIPATION_RATE,
    SPREAD_RATE,
    WINDOW_ACTOR_TRANSITIONS,
    WINDOW_COUNT,
    WINDOW_STARTS,
    WINDOW_STOPS,
    build_synthetic_dataset,
    make_oos_env,
    make_training_env,
)
from trade_rl.simulation.stateful.execution import StatefulExecutionObservation
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioGymObservationWrapper,
    fit_portfolio_observation_transform,
)


def _expected_action(cue: float) -> int:
    if cue > 0.0:
        return ACTION_LONG
    if cue < 0.0:
        return ACTION_SHORT
    return ACTION_FLAT


def test_fixture_is_closed_balanced_and_has_four_disjoint_native_windows() -> None:
    dataset = build_synthetic_dataset(48)
    assert dataset.n_bars == DATASET_ROW_COUNT == 4_101
    assert WINDOW_COUNT == len(ALL_WINDOW_CUES) == 4
    assert WINDOW_ACTOR_TRANSITIONS == 1_024
    assert WINDOW_STARTS == (0, 1_025, 2_050, 3_075)
    assert WINDOW_STOPS == (1_025, 2_050, 3_075, 4_100)
    assert build_synthetic_dataset(48).dataset_id == dataset.dataset_id
    assert build_synthetic_dataset(72).dataset_id != dataset.dataset_id

    expected_volume_rows: set[int] = set()
    expected_event_rows: set[int] = set()
    for window_start, window_stop, cues in zip(
        WINDOW_STARTS,
        WINDOW_STOPS,
        ALL_WINDOW_CUES,
        strict=True,
    ):
        assert window_stop - window_start == 1_025
        assert sum(cue > 0 for cue in cues) == 3
        assert sum(cue < 0 for cue in cues) == 3
        assert sum(cue == 0 for cue in cues) == 2
        expected_volume_rows.add(window_stop)
        for block_index, cue in enumerate(cues):
            block_start = window_start + block_index * BLOCK_TRANSITIONS
            expected_volume_rows.add(block_start + 1)
            assert np.all(
                dataset.features[block_start : block_start + BLOCK_TRANSITIONS, 0, 0]
                == cue
            )
            payoff_row = block_start + 48
            if cue:
                expected_event_rows.add(payoff_row)
                assert dataset.open[payoff_row, 0] == pytest.approx(
                    dataset.close[payoff_row - 1, 0]
                )
                assert dataset.close[payoff_row, 0] / dataset.close[
                    payoff_row - 1, 0
                ] - 1.0 == pytest.approx(cue * EVENT_RETURN)

    assert set(np.flatnonzero(dataset.volume[:, 0])) == expected_volume_rows
    close_returns = dataset.close[1:, 0] / dataset.close[:-1, 0] - 1.0
    actual_event_rows = set(np.flatnonzero(np.abs(close_returns) > 1e-12) + 1)
    assert actual_event_rows == expected_event_rows
    assert np.count_nonzero(dataset.fee_rate) == 0
    assert np.count_nonzero(dataset.spread_rate) == 0


def test_cue_rule_uses_native_cash_cost_and_one_terminal_settlement() -> None:
    dataset = build_synthetic_dataset(72)
    env = make_training_env(dataset, execution_seed=2_003)
    observations: list[StatefulExecutionObservation] = []
    env.execution_observer = observations.append
    observation, _ = env.reset(seed=2_002)
    rewards: list[float] = []
    quantity_after_opportunity: list[float] = []
    terminal_info: dict[str, object] | None = None

    for actor_step in range(WINDOW_ACTOR_TRANSITIONS):
        cue = float(observation["features"][0, 0])
        action = (
            _expected_action(cue)
            if actor_step % BLOCK_TRANSITIONS == 0
            else ACTION_HOLD
        )
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.int64)
        )
        rewards.append(reward)
        assert truncated is False
        if actor_step % BLOCK_TRANSITIONS == 0:
            assert info["current_index"] == actor_step + 1
            quantity_after_opportunity.append(float(observation["account"][0, 0]))
        if actor_step < WINDOW_ACTOR_TRANSITIONS - 1:
            assert terminated is False
        else:
            assert terminated is True
            terminal_info = info

    assert terminal_info is not None
    assert len(quantity_after_opportunity) == 8
    for cue, quantity in zip(
        ALL_WINDOW_CUES[0], quantity_after_opportunity, strict=True
    ):
        if cue > 0:
            assert quantity > 0.0
        elif cue < 0:
            assert quantity < 0.0
        else:
            assert quantity == pytest.approx(0.0, abs=1e-12)

    final_state = terminal_info["terminal_state"]
    assert isinstance(final_state, dict)
    assert terminal_info["elapsed_intervals"] == 2
    assert terminal_info["settlement_complete"] is True
    assert terminal_info["objective_endpoint_valid"] is True
    assert final_state["current_index"] == WINDOW_STOPS[0]
    assert final_state["exact_quantities"] == ("0",)
    assert final_state["active_orders"] == ()

    fills = [
        event
        for item in observations
        for event in item.order_events
        if event.event_type in {"filled", "partial_fill"}
    ]
    terminal_fills = [
        event for event in fills if event.processing_index == WINDOW_STOPS[0]
    ]
    assert len(terminal_fills) == 1
    assert terminal_fills[0].reduce_only is True
    expected_costs: list[float] = []
    signed_fill_cashflows: list[float] = []
    for event in fills:
        assert event.execution_price is not None
        independent_notional = abs(event.filled_quantity * event.execution_price)
        signed_fill_cashflows.append(event.filled_quantity * event.execution_price)
        expected_costs.append(
            independent_notional
            * (
                FEE_RATE
                + SPREAD_RATE
                + IMPACT_RATE * math.sqrt(event.participation_rate)
            )
        )
    expected_cost = math.fsum(expected_costs)
    expected_cash = INITIAL_CAPITAL - math.fsum(signed_fill_cashflows) - expected_cost
    assert env.execution_cost.fee_rate == pytest.approx(FEE_RATE)
    assert env.execution_cost.spread_rate == pytest.approx(SPREAD_RATE)
    assert env.execution_cost.impact_rate == pytest.approx(IMPACT_RATE)
    assert env.execution_cost.max_participation_rate == pytest.approx(
        MAX_PARTICIPATION_RATE
    )
    assert final_state["total_cost"] == pytest.approx(expected_cost, rel=1e-10)
    assert final_state["cash"] == pytest.approx(expected_cash, rel=1e-10)
    assert final_state["portfolio_value"] == pytest.approx(expected_cash, rel=1e-10)
    assert sum(rewards) == pytest.approx(
        (expected_cash - INITIAL_CAPITAL) / INITIAL_CAPITAL
    )
    assert math.log(expected_cash / INITIAL_CAPITAL) > 0.0


def test_train_only_transform_binds_the_three_later_same_dataset_windows() -> None:
    dataset = build_synthetic_dataset(72)
    training_env = make_training_env(dataset, execution_seed=3_003)
    transform = fit_portfolio_observation_transform(training_env)

    assert transform.source_dataset_id == dataset.dataset_id
    assert transform.start_index == WINDOW_STARTS[0]
    assert transform.stop_index == WINDOW_STOPS[0]
    assert transform.agent_stop_index == WINDOW_STARTS[0] + WINDOW_ACTOR_TRANSITIONS
    assert transform.local_normalizer.start_index == WINDOW_STARTS[0]
    assert transform.local_normalizer.stop_index == transform.agent_stop_index

    for window_index in (1, 2, 3):
        runtime = make_oos_env(dataset, window_index, transform, training_env)
        assert isinstance(runtime, PortfolioGymObservationWrapper)
        assert runtime.runtime_binding is not None
        assert (
            runtime.runtime_binding.runtime_start_index == WINDOW_STARTS[window_index]
        )
        assert runtime.runtime_binding.runtime_stop_index == WINDOW_STOPS[window_index]
        assert runtime.runtime_binding.runtime_start_index >= transform.stop_index
        assert runtime.env.dataset.dataset_id == transform.source_dataset_id

    with pytest.raises(ValueError, match="window_index"):
        make_oos_env(dataset, 0, transform, training_env)


def test_oos_only_tail_poison_does_not_change_training_transform_statistics() -> None:
    dataset = build_synthetic_dataset(72)
    training_env = make_training_env(dataset, execution_seed=3_003)
    baseline = fit_portfolio_observation_transform(training_env)

    poisoned_features = dataset.features.copy()
    poisoned_features[WINDOW_STOPS[0] :, :, :] = 50_000.0
    poisoned_globals = dataset.global_features.copy()
    poisoned_globals[WINDOW_STOPS[0] :, :] = -75_000.0
    poisoned_id = hashlib.sha256(
        dataset.dataset_id.encode("ascii")
        + poisoned_features.tobytes()
        + poisoned_globals.tobytes()
    ).hexdigest()
    poisoned_dataset = replace(
        dataset,
        dataset_id=poisoned_id,
        features=poisoned_features,
        global_features=poisoned_globals,
    )
    poisoned_training_env = make_training_env(
        poisoned_dataset,
        execution_seed=3_003,
    )
    poisoned_transform = fit_portfolio_observation_transform(poisoned_training_env)

    assert poisoned_transform.source_dataset_id != baseline.source_dataset_id
    assert poisoned_transform.local_normalizer.mean == baseline.local_normalizer.mean
    assert poisoned_transform.local_normalizer.scale == baseline.local_normalizer.scale
    assert (
        poisoned_transform.local_normalizer.usable_counts
        == baseline.local_normalizer.usable_counts
    )
    assert poisoned_transform.global_mean == baseline.global_mean
    assert poisoned_transform.global_scale == baseline.global_scale
    assert poisoned_transform.reference_log_price_mean == (
        baseline.reference_log_price_mean
    )


@pytest.mark.parametrize("payoff_transition", (0, 47, 49, 64, 71, 73, 128))
def test_fixture_rejects_unregistered_payoff_transitions(
    payoff_transition: int,
) -> None:
    with pytest.raises(ValueError, match="48 or 72"):
        build_synthetic_dataset(payoff_transition)
