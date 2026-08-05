from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trade_rl.data.market import MarketDataset
from trade_rl.learning.episode_oracle_teacher import episode_oracle_target_path
from trade_rl.learning.oracle_bellman_contracts import (
    OracleEpisodeInputs,
    OracleSolverConfig,
)
from trade_rl.learning.oracle_bellman_numpy import (
    reconstruct_state_paths_numpy,
    reduce_candidates_numpy,
    solve_numpy_oracle_batch,
)
from trade_rl.learning.oracle_market_tape import build_oracle_market_tape
from trade_rl.learning.oracle_teacher import OracleTeacherConfig, portfolio_states
from trade_rl.simulation.execution import ExecutionCostConfig


def _market(close_values: np.ndarray) -> MarketDataset:
    close = np.asarray(close_values, dtype=np.float64)
    if close.ndim == 1:
        close = close[:, None]
    n_bars, n_symbols = close.shape
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=tuple(f"S{index}" for index in range(n_symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _episodes(
    *,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    initial_weights: np.ndarray,
) -> OracleEpisodeInputs:
    return OracleEpisodeInputs(
        episode_indices=np.arange(len(starts), dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        stops=np.asarray(stops, dtype=np.int64),
        initial_weights=np.asarray(initial_weights, dtype=np.float64),
    )


def test_lowest_prior_index_wins_within_tolerance() -> None:
    scores = np.array([[[1.0], [1.0 + 5e-13], [0.0]]], dtype=np.float64)

    best, pointers = reduce_candidates_numpy(scores, tie_tolerance=1e-12)

    assert pointers.tolist() == [[0]]
    np.testing.assert_allclose(best, [[1.0 + 5e-13]])


@pytest.mark.parametrize(
    ("state_count", "expected_dtype"),
    [(32_767, np.dtype(np.int16)), (32_768, np.dtype(np.int32))],
)
def test_reduction_uses_smallest_safe_pointer_dtype(
    state_count: int,
    expected_dtype: np.dtype[np.signedinteger],
) -> None:
    scores = np.zeros((1, state_count, 1), dtype=np.float64)

    _, pointers = reduce_candidates_numpy(scores, tie_tolerance=1e-12)

    assert pointers.dtype == expected_dtype
    assert pointers[0, 0] == 0


def test_reduction_marks_all_invalid_target_with_missing_pointer() -> None:
    scores = np.full((2, 3, 4), -np.inf, dtype=np.float64)

    best, pointers = reduce_candidates_numpy(scores, tie_tolerance=1e-12)

    assert np.isneginf(best).all()
    np.testing.assert_array_equal(pointers, np.full((2, 4), -1))


def test_reverse_reconstruction_follows_backpointers() -> None:
    pointers = np.full((1, 3, 3), -1, dtype=np.int16)
    pointers[0, 1, 1] = 0
    pointers[0, 2, 2] = 1

    paths = reconstruct_state_paths_numpy(
        pointers,
        np.array([2], dtype=np.int64),
    )

    assert paths.tolist() == [[0, 1, 2]]


def test_reverse_reconstruction_rejects_missing_pointer() -> None:
    pointers = np.full((1, 2, 2), -1, dtype=np.int16)

    with pytest.raises(RuntimeError, match="backpointer"):
        reconstruct_state_paths_numpy(
            pointers,
            np.array([1], dtype=np.int64),
        )


@pytest.mark.parametrize("signal_delay", [0, 1])
def test_batched_solver_matches_legacy_episode_paths(signal_delay: int) -> None:
    close = np.column_stack(
        [
            100.0 * np.exp(np.arange(8) * 0.025),
            100.0 * np.exp(np.arange(8) * -0.015),
        ]
    )
    market = _market(close)
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        signal_delay_decisions=signal_delay,
    )
    states = portfolio_states(market, config)
    episodes = _episodes(
        starts=(0, 1),
        stops=(7, 8),
        initial_weights=np.array([[0.0, 0.0], [0.2, -0.1]]),
    )
    tape = build_oracle_market_tape(market, (0, 8), config.bellman_parameters)

    result = solve_numpy_oracle_batch(
        tape=tape,
        states=states,
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=OracleSolverConfig(
            selection="numpy",
            episode_batch_size=2,
            target_state_block_size=2,
        ),
    )

    expected = tuple(
        episode_oracle_target_path(
            market,
            (int(start), int(stop)),
            config,
            initial_weights=initial,
        )
        for start, stop, initial in zip(
            episodes.starts,
            episodes.stops,
            episodes.initial_weights,
            strict=True,
        )
    )
    assert result.provenance.backend == "numpy"
    assert result.final_scores.shape == (2,)
    assert np.isfinite(result.final_scores).all()
    for actual, legacy in zip(result.targets, expected, strict=True):
        np.testing.assert_array_equal(actual, legacy)
        assert actual.dtype == np.float32
        assert not actual.flags.writeable


def test_target_state_blocks_do_not_change_solver_result() -> None:
    market = _market(100.0 * np.exp(np.arange(7) * 0.02))
    config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    states = portfolio_states(market, config)
    episodes = _episodes(
        starts=(0,),
        stops=(7,),
        initial_weights=np.zeros((1, 1)),
    )
    tape = build_oracle_market_tape(market, (0, 7), config.bellman_parameters)

    unblocked = solve_numpy_oracle_batch(
        tape=tape,
        states=states,
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=OracleSolverConfig(target_state_block_size=None),
    )
    blocked = solve_numpy_oracle_batch(
        tape=tape,
        states=states,
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=OracleSolverConfig(target_state_block_size=1),
    )

    np.testing.assert_array_equal(blocked.targets[0], unblocked.targets[0])
    np.testing.assert_allclose(blocked.final_scores, unblocked.final_scores)


def test_solver_rejects_mixed_episode_horizons() -> None:
    market = _market(np.linspace(100.0, 106.0, 7))
    config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    states = portfolio_states(market, config)
    episodes = _episodes(
        starts=(0, 1),
        stops=(7, 6),
        initial_weights=np.zeros((2, 1)),
    )
    tape = build_oracle_market_tape(market, (0, 7), config.bellman_parameters)

    with pytest.raises(ValueError, match="equal horizon"):
        solve_numpy_oracle_batch(
            tape=tape,
            states=states,
            episode_inputs=episodes,
            parameters=config.bellman_parameters,
            solver_config=OracleSolverConfig(),
        )


@given(
    growth=st.floats(
        min_value=-0.03,
        max_value=0.03,
        allow_nan=False,
        allow_infinity=False,
    ),
    nonzero_cost=st.booleans(),
    signal_delay=st.sampled_from((0, 1)),
    initial_weight=st.sampled_from((-0.2, 0.0, 0.2)),
)
@settings(max_examples=20, deadline=None)
def test_numpy_solver_matches_legacy_across_randomized_small_markets(
    growth: float,
    nonzero_cost: bool,
    signal_delay: int,
    initial_weight: float,
) -> None:
    market = _market(100.0 * np.exp(np.arange(7) * growth))
    cost = ExecutionCostConfig(
        fee_rate=0.001 if nonzero_cost else 0.0,
        spread_rate=0.0005 if nonzero_cost else 0.0,
        impact_rate=0.0002 if nonzero_cost else 0.0,
        max_participation_rate=1.0,
    )
    config = OracleTeacherConfig(
        execution_cost=cost,
        signal_delay_decisions=signal_delay,
    )
    states = portfolio_states(market, config)
    initial = np.array([[initial_weight]], dtype=np.float64)
    episodes = _episodes(
        starts=(0,),
        stops=(7,),
        initial_weights=initial,
    )
    tape = build_oracle_market_tape(market, (0, 7), config.bellman_parameters)

    result = solve_numpy_oracle_batch(
        tape=tape,
        states=states,
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=OracleSolverConfig(target_state_block_size=1),
    )
    legacy = episode_oracle_target_path(
        market,
        (0, 7),
        config,
        initial_weights=initial[0],
    )

    np.testing.assert_array_equal(result.targets[0], legacy)
