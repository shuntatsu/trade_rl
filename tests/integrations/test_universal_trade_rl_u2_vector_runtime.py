from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest

from tests.rl.test_universal_trade_u2_environment import U2EnvironmentFixture
from tests.rl.test_universal_trade_u2_environment_factory import _construct_factory
from trade_rl.integrations.sb3_environment import (
    _build_training_environment,
    _filtered_environment_factory,
)
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment

pytest_plugins = ("tests.rl.test_universal_trade_u2_environment",)


def _active_u1_environment(
    worker: EpisodeRoutedSingleInstrumentEnv,
) -> UniversalTradeEnvironment:
    environment = worker._active_environment
    if not isinstance(environment, UniversalTradeEnvironment):
        raise AssertionError("U2 worker must have one active U1 environment")
    return environment


def _prime_last_decision(worker: EpisodeRoutedSingleInstrumentEnv) -> None:
    base = _active_u1_environment(worker).base_env
    if base.end_index - base.current_index <= base.decision_bars:
        raise AssertionError("U2 timeout oracle requires a nonterminal active episode")
    base.current_index = base.end_index - base.decision_bars


def test_u2_in_process_vector_accepts_sb3_member_seed_offsets(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    factory, _closure, _locators, _source_loads, _child_builds = _construct_factory(
        u2_environment_fixture
    )
    vector = _build_training_environment(
        _filtered_environment_factory(factory),
        8,
        subprocesses=False,
    )
    try:
        member_seed = factory.run_seed
        assert vector.seed(member_seed) == [member_seed + index for index in range(8)]

        vector.reset()

        assert len(vector.envs) == 8
        for index, filtered in enumerate(vector.envs):
            worker = filtered.unwrapped
            assert isinstance(worker, EpisodeRoutedSingleInstrumentEnv)
            assert worker.run_seed == member_seed
            assert worker.environment_index == index
            assert worker.canonical_probe_seed == member_seed + index
            assert worker.environment_digest == factory.environment_generation_digest
    finally:
        vector.close()


def test_u2_timeout_preserves_terminal_observation_and_raw_economic_reward(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    direct_factory, *_ = _construct_factory(u2_environment_fixture)
    vector_factory, *_ = _construct_factory(u2_environment_fixture)
    direct = direct_factory()
    vector = _build_training_environment(
        _filtered_environment_factory(vector_factory),
        8,
        subprocesses=False,
    )
    try:
        member_seed = direct_factory.run_seed
        direct.reset(seed=member_seed)
        assert vector.seed(member_seed) == [member_seed + index for index in range(8)]
        vector.reset()

        vector_worker0 = vector.envs[0].unwrapped
        assert isinstance(vector_worker0, EpisodeRoutedSingleInstrumentEnv)
        assert direct.active_episode_binding.digest == vector_worker0.active_episode_binding.digest

        warmup_action = np.asarray([1.0], dtype=np.float32)
        _observation, _reward, terminated, truncated, _info = direct.step(warmup_action)
        assert terminated is False
        assert truncated is False

        vector_actions = np.zeros((8, 1), dtype=np.float32)
        vector_actions[0, 0] = 1.0
        _vector_observation, _vector_rewards, vector_dones, _vector_infos = vector.step(
            vector_actions
        )
        assert not bool(vector_dones[0])

        _prime_last_decision(direct)
        _prime_last_decision(vector_worker0)

        before = float(direct.hybrid.portfolio_value)
        next_direct, raw_reward, terminated, truncated, _direct_info = direct.step(
            warmup_action
        )
        after = float(direct.hybrid.portfolio_value)
        vector_observation, vector_rewards, vector_dones, vector_infos = vector.step(
            vector_actions
        )

        assert terminated is False
        assert truncated is True
        assert bool(vector_dones[0]) is True
        assert vector_infos[0]["TimeLimit.truncated"] is True

        terminal = vector_infos[0]["terminal_observation"]
        assert isinstance(next_direct, Mapping)
        assert isinstance(terminal, Mapping)
        assert set(terminal) == set(next_direct)
        for key in next_direct:
            np.testing.assert_allclose(
                np.asarray(terminal[key]),
                np.asarray(next_direct[key]),
                rtol=0.0,
                atol=0.0,
            )

        expected_reward = 100.0 * math.log(after / before)
        assert raw_reward == pytest.approx(expected_reward, abs=1e-10)
        assert float(vector_rewards[0]) == pytest.approx(raw_reward, abs=1e-10)

        terminal_binding = vector_infos[0]["instrument_episode_binding"]
        reset_binding = vector.reset_infos[0]["instrument_episode_binding"]
        assert terminal_binding["completed_episode_count"] == 0
        assert reset_binding["completed_episode_count"] == 1
        assert vector_worker0.active_episode_binding.completed_episode_count == 1

        assert isinstance(vector_observation, Mapping)
        assert set(vector_observation) == set(next_direct)
    finally:
        direct.close()
        vector.close()
