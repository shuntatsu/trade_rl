from __future__ import annotations

from tests.rl.test_universal_trade_u2_environment import U2EnvironmentFixture
from tests.rl.test_universal_trade_u2_environment_factory import _construct_factory
from trade_rl.integrations.sb3_environment import (
    _build_training_environment,
    _filtered_environment_factory,
)
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv

pytest_plugins = ("tests.rl.test_universal_trade_u2_environment",)


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
