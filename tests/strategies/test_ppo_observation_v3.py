from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests.strategies.test_ppo_intent import market
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl import intent
from trade_rl.strategies.rl.ppo import (
    PPOIntentStrategy,
    PPOTradingEnv,
    fit_ppo_strategy,
)


class _FixedPolicy:
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        assert observation.ndim == 1
        assert deterministic is True
        return np.asarray(2), None


def _observation(position_age_bars: int) -> StrategyObservation:
    return StrategyObservation(
        index=0,
        timestamp=np.datetime64("2026-01-01", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([0.5]),
        feature_available=np.asarray([True]),
        feature_staleness=np.asarray([0.0], dtype=np.float32),
        global_features=np.zeros(1, dtype=np.float64),
        global_feature_available=np.ones(1, dtype=np.bool_),
        current_intent=PositionIntent.LONG,
        current_weight=0.25,
        position_age_bars=position_age_bars,
    )


def test_v3_appends_clipped_fill_age_without_changing_v2_layout() -> None:
    schema_v3 = getattr(intent, "PPO_OBSERVATION_SCHEMA_V3", None)
    assert schema_v3 == "ppo_observation_v3", (
        "the fill-age observation must have a separately versioned schema"
    )
    encoder_parameters = inspect.signature(intent._encode_observation).parameters
    assert "observation_schema" in encoder_parameters, (
        "the encoder must preserve v2 artifacts while emitting v3 observations"
    )

    observation = _observation(position_age_bars=252)
    v2 = intent._encode_observation(
        observation,
        (0,),
        observation_schema=intent.PPO_OBSERVATION_SCHEMA,
    )
    v3 = intent._encode_observation(
        observation,
        (0,),
        observation_schema=schema_v3,
    )

    assert v2.shape == (5,)
    assert v3.shape == (6,)
    assert v3[-1] == 0.5
    assert v3[-2] == v2[-1]


def test_v3_age_feature_saturates_at_registered_maximum_horizon() -> None:
    schema_v3 = getattr(intent, "PPO_OBSERVATION_SCHEMA_V3", None)
    assert schema_v3 == "ppo_observation_v3", (
        "the fill-age observation must have a separately versioned schema"
    )
    assert hasattr(intent, "PPO_POSITION_AGE_SCALE_BARS"), (
        "the age normalization limit must be part of the observation contract"
    )

    maximum_age = int(intent.PPO_POSITION_AGE_SCALE_BARS)
    encoded = intent._encode_observation(
        _observation(position_age_bars=maximum_age + 10),
        (0,),
        observation_schema=schema_v3,
    )

    assert encoded[-1] == 1.0


def test_v3_training_environment_and_inference_adapter_use_matching_shape() -> None:
    class CapturePolicy:
        observation: np.ndarray | None = None

        def predict(
            self,
            value: np.ndarray,
            *,
            deterministic: bool = True,
        ) -> tuple[np.ndarray, None]:
            assert deterministic is True
            self.observation = value.copy()
            return np.asarray(2), None

    env = PPOTradingEnv(
        market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.25,
        observation_schema=intent.PPO_OBSERVATION_SCHEMA_V3,
    )
    initial_observation, _ = env.reset(seed=29)
    assert env.observation_space.contains(initial_observation)
    assert initial_observation.shape == (6,)

    next_observation, _, _, _, info = env.step(2)
    assert env.observation_space.contains(next_observation)
    assert next_observation[-1] == pytest.approx(1 / intent.PPO_POSITION_AGE_SCALE_BARS)
    assert info["position_age_bars"] == 1

    policy = CapturePolicy()
    strategy = PPOIntentStrategy(
        policy,
        feature_indices=(0,),
        observation_schema=intent.PPO_OBSERVATION_SCHEMA_V3,
    )
    strategy.decide(_observation(position_age_bars=1))
    assert policy.observation is not None
    assert policy.observation.shape == env.observation_space.shape
    assert policy.observation[-1] == pytest.approx(next_observation[-1])


def test_ppo_minimum_hold_rejects_age_blind_low_level_paths() -> None:
    dataset = market()
    with pytest.raises(ValueError, match="requires the age-aware observation"):
        PPOTradingEnv(
            dataset,
            feature_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.25,
            minimum_hold_bars=1,
        )

    with pytest.raises(ValueError, match="requires the age-aware observation"):
        PPOIntentStrategy(
            _FixedPolicy(),
            feature_indices=(0,),
            minimum_hold_bars=1,
        )

    with pytest.raises(ValueError, match="requires the age-aware observation"):
        fit_ppo_strategy(
            dataset,
            feature_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.25,
            total_timesteps=16,
            minimum_hold_bars=1,
        )

    strategy = PPOIntentStrategy(
        _FixedPolicy(),
        feature_indices=(0,),
    )
    with pytest.raises(ValueError, match="requires the age-aware observation"):
        run_single_symbol_replay(
            dataset,
            strategy,
            start_index=0,
            stop_index=3,
            gross_budget=0.25,
            minimum_hold_bars=1,
        )
