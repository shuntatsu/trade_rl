import numpy as np
import pytest

from mars_lite.env.observation import (
    ObservationSchema,
    ObservationState,
    build_observation,
)


def test_build_observation_matches_current_layout() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    globals_ = np.array([5.0, 6.0], dtype=np.float32)
    state = ObservationState(
        weights=np.array([0.2, -0.1]),
        portfolio_value=0.9,
        peak_value=1.0,
        progress=0.25,
    )
    actual = build_observation(
        features, globals_, state, ObservationSchema(progress_mode="episode")
    )
    expected = np.array(
        [1.0, 2.0, 0.2, 3.0, 4.0, -0.1, 5.0, 6.0, 0.1, 0.3, 0.25],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected)


def test_real_current_weights_change_policy_input() -> None:
    features = np.zeros((2, 1), dtype=np.float32)
    globals_ = np.zeros(1, dtype=np.float32)
    first = build_observation(
        features,
        globals_,
        ObservationState(np.array([0.0, 0.0]), 1.0, 1.0, 0.0),
        ObservationSchema(),
    )
    second = build_observation(
        features,
        globals_,
        ObservationState(np.array([0.2, -0.1]), 1.0, 1.0, 0.0),
        ObservationSchema(),
    )
    assert not np.array_equal(first, second)


def test_invalid_weight_dimension_fails_closed() -> None:
    with pytest.raises(ValueError, match="weights"):
        build_observation(
            np.zeros((2, 1), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            ObservationState(np.zeros(1), 1.0, 1.0, 0.0),
            ObservationSchema(),
        )


def test_zero_progress_mode_removes_training_episode_position() -> None:
    features = np.zeros((1, 1), dtype=np.float32)
    globals_ = np.zeros(1, dtype=np.float32)
    observation = build_observation(
        features,
        globals_,
        ObservationState(np.array([0.0]), 1.0, 1.0, 0.75),
        ObservationSchema(progress_mode="zero"),
    )
    assert observation[-1] == 0.0
